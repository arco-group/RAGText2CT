from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import ReconModel, check_input
from scripts.utils import define_instance, dynamic_infer


def load_filenames(data_list_path: str) -> list:
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
        filenames_raw = json_data["validation"] if "validation" in json_data else json_data["training"]
    return [_item["image"] for _item in filenames_raw]


def set_random_seed(seed: int) -> int:
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def _load_npy(path):
    return torch.tensor(np.load(path), dtype=torch.float32)


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    
    if "unet_state_dict" in checkpoint.keys():
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
        scale_factor = checkpoint["scale_factor"]

    else:
        unet.load_state_dict(checkpoint, strict=True)
        scale_factor = 1.0287

    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")
    logger.info(f"scale_factor -> {scale_factor}.")
    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device: torch.device):
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones((len(spacing_tensor)), dtype=torch.long).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor


def run_inference(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    clip_model: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
    impression_text: str
) -> np.ndarray:
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    noise = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    top_region_index_tensor = None if not args.diffusion_unet_def["include_top_region_index_input"] else top_region_index_tensor
    bottom_region_index_tensor = None if not args.diffusion_unet_def["include_top_region_index_input"] else bottom_region_index_tensor
    spacing_tensor = None if not args.diffusion_unet_def["include_top_region_index_input"] else spacing_tensor

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")

    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    with torch.no_grad():
        impression = clip_model([impression_text], "encode_text").to(device)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    use_cfg = args.use_cfg
    guidance_scale = args.guidance_scale if use_cfg else 1.0

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }

            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )

            if include_modality:
                unet_inputs.update({"class_labels": modality_tensor})

            if use_cfg:
                unet_inputs_no_text = unet_inputs.copy()
                unet_inputs.update({"context": impression})
                unet_inputs_no_text.update({"context": torch.zeros_like(impression, device=device)})

                model_output_uncond = unet(**unet_inputs_no_text)
                model_output_cond = unet(**unet_inputs)
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            else:
                unet_inputs.update({"context": impression})
                model_output = unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image)
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t)

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)

        data = synthetic_images.squeeze().cpu().detach().numpy()
        a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
        return np.int16(data)


def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
    resize: int = 512,
) -> None:
    if resize != 512:
        from monai.transforms import Resized

        resize_transform = Resized(keys="image", spatial_size=(resize, resize), mode="trilinear")
        input_data = {"image": np.transpose(data, (2, 1, 0))}
        output = resize_transform(input_data)
        data = np.transpose(output["image"].numpy(), (2, 1, 0))

    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, clip_weights: str, reports: list) -> None:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank if args.diffusion_unet_inference["random_seed"] else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    cfgm = model_cfg_bank()("clip_3D")
    clip = get_model()(cfgm).to(device)
    clip_weights = Path(clip_weights)
    clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=True)
    clip.eval()
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)

    n = 0
    for report in reports:

        if local_rank == 0:
            logger.info(f"[config] random_seed -> {random_seed}.")
            logger.info(f"[config] output_size -> {output_size}.")
            logger.info(f"[config] out_spacing -> {out_spacing}.")
            logger.info(f"[config] impression -> {report}")

        data = run_inference(
            args,
            device,
            autoencoder,
            unet,
            clip,
            scale_factor,
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            modality_tensor,
            output_size,
            divisor,
            logger,
            impression_text=report,
        )

        output_path = f"./demo/{n}.nii.gz"
        save_image(data, output_size, out_spacing, output_path, logger, resize=512)
        n += 1

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference (demo)")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_diff_model_eval.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_rflow.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )
    parser.add_argument("--clip_weights", type=str, default="./models/CLIP3D_Finding_Impression_30ep.pt")

    example_report = [
        "Findings: Trachea and main bronchi are patent. Mild cardiomegaly. No pericardial effusion. No enlarged mediastinal or hilar lymph nodes. Linear atelectasis in the lower lobes. No focal consolidation. ",
        "Impression: Mild cardiomegaly and bibasilar atelectasis without focal consolidation."
    ]

    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus, args.clip_weights, example_report)
