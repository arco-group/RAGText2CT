import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from monai.data import decollate_batch, MetaTensor
from monai.networks.utils import copy_model_state
from monai.transforms import LoadImaged, Orientationd, EnsureTyped, Compose
import faiss
import numpy as np
import nibabel as nib
from monai.transforms import SaveImage
from monai.utils import RankFilter

sys.path.append('scripts')
sys.path.append('./')
from scripts.sample import check_input, ldm_conditional_sample_one_image
from scripts.utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "unet_state_dict", "controlnet_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _load_rag_index(embedding_path: str, paths_path: str):
    rag_embeddings = np.load(embedding_path).astype("float32")
    with open(paths_path, "r") as f:
        rag_paths = json.load(f)
    if len(rag_embeddings) != len(rag_paths):
        raise ValueError(f"Mismatch between embeddings ({len(rag_embeddings)}) and paths ({len(rag_paths)}).")
    faiss.normalize_L2(rag_embeddings)
    index = faiss.IndexFlatIP(rag_embeddings.shape[1])
    index.add(rag_embeddings)
    return index, rag_paths


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_rag_controlnet_eval.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_rag_rflow.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        default="./configs/config_rag_controlnet.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-i", "--index", default=None, type=int, help="number of samples")

    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # Step 1: set data loader
    val_loader, _ = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_infer["batch_size_val"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        args=args,
        phase = "validation",
        index = args.index
    )

    # Step 2: define AE, diffusion model and controlnet
    # define AE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")
        autoencoder_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
        autoencoder.load_state_dict(_extract_state_dict(autoencoder_ckpt))
        logger.info(f"Load trained diffusion model from {args.trained_autoencoder_path}.")
    else:
        logger.info("trained autoencoder model is not loaded.")

    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)

    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(_extract_state_dict(diffusion_model_ckpt))
        scale_factor = diffusion_model_ckpt["scale_factor"] if isinstance(diffusion_model_ckpt, dict) and "scale_factor" in diffusion_model_ckpt else 1.0
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(_extract_state_dict(torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)), strict=True)
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("trained controlnet is not loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()
    
    iterator = iter(val_loader)
    index, rag_paths = _load_rag_index(args.rag_embeddings_path, args.rag_paths_path)

    transforms = Compose([
        LoadImaged(keys=["label"], image_only=True, ensure_channel_first=True),
        Orientationd(keys=["label"], axcodes="RAS"),
        EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
    ])

    for num in range(len(iterator)):
        batch = next(iterator)
        # get label mask
        labels = batch["label"].to(device)
        # get corresponding conditions
        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        else:
            top_region_index_tensor = None
            bottom_region_index_tensor = None
        spacing_tensor = batch["spacing"].to(device)
        modality_tensor = args.controlnet_infer["modality"] * torch.ones((len(labels),), dtype=torch.long).to(device)
        
        out_spacing = tuple((batch["spacing"].squeeze().numpy() / 100).tolist())
        # get target dimension
        # dim = batch["dim"]
        dim = [512, 512, 128]
        output_size = (dim[0], dim[1], dim[2])
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
        # check if output_size and out_spacing are valid.
        check_input(None, None, None, output_size, out_spacing, None)

        cond = batch["cond"].to(device)
        if len(cond.shape) == 4:
            cond = cond.squeeze(1)  # Rimuove la seconda dimensione, risultando in (4, 1, 768)
        use_cfg = args.controlnet_infer['use_cfg']
        guidance_scale = args.controlnet_infer['guidance_scale'] if use_cfg else 1.0

        cond_np = cond.squeeze(0).cpu().numpy().astype("float32")
        faiss.normalize_L2(cond_np)

        _, I = index.search(cond_np, k=1)
        retrieved_paths = [rag_paths[i] for i in I[:, 0]]
        mask_path = retrieved_paths[0]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(args.rag_mask_base_dir, mask_path)

        data_dict = {"label": mask_path}

        transformed = transforms(data_dict)
        labels = transformed["label"].unsqueeze(0)

        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_images, _ = ldm_conditional_sample_one_image(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.controlnet_infer["num_inference_steps"],
            autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
            cond=cond,
        )

        labels = decollate_batch(batch)[0]["label"]
        labels.meta["filename_or_obj"] = batch['filename'][0].split('/')[-1]
        synthetic_images = MetaTensor(synthetic_images.squeeze(0), meta=labels.meta)

        output_prefix = batch['filename'][0].replace(args.embedding_base_dir, '').replace(f'_impression_{args.report_encoder_model}.npy', '.nii.gz')

        output_path = "{0}/{1}{2}".format(args.output_dir, args.exp_name, output_prefix)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        mask_txt_path = os.path.join(
            output_dir,
            os.path.basename(output_path).replace(".nii.gz", "") + "_mask_path.txt",
        )
        with open(mask_txt_path, "w") as mask_txt:
            mask_txt.write(mask_path + "\n")

        img_saver = SaveImage(
            output_dir=output_dir,
            output_postfix=os.path.basename(output_path).replace(os.path.basename(output_path), ''),
            separate_folder=False,
        )
        img_saver(synthetic_images)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
