from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import faiss
import nibabel as nib
import numpy as np
import torch
from monai.networks.utils import copy_model_state
from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import check_input, ldm_conditional_sample_one_image
from scripts.utils import define_instance


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "unet_state_dict", "controlnet_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-case RAGText2CT demo")
    parser.add_argument("--report", type=str, required=True, help="Input radiology report text.")
    parser.add_argument("--ct", type=str, default="ct.nii.gz", help="Reference CT path for metadata only.")
    parser.add_argument("--mask", type=str, default="mask.nii.gz", help="Retrieved mask path.")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="hf_ragtext2ct/models",
        help="Directory containing autoencoder, unet, clip and controlnet weights.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/environment_rag_controlnet_eval.json",
        help="Environment config for RAG inference.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/config_rag_controlnet.json",
        help="ControlNet train/infer config.",
    )
    parser.add_argument(
        "--model-def",
        type=str,
        default="configs/config_rag_rflow.json",
        help="Model definition config.",
    )
    parser.add_argument("--output", type=str, default="predictions/rag_demo_single.nii.gz", help="Output NIfTI path.")
    parser.add_argument("--num-gpus", type=int, default=1)
    return parser.parse_args()


def resolve_weight_paths(args: argparse.Namespace, weights_dir: Path) -> None:
    args.trained_autoencoder_path = str(weights_dir / "autoencoder_epoch273.pt")
    args.trained_diffusion_path = str(weights_dir / "unet_rflow_200ep.pt")
    args.trained_controlnet_path = str(weights_dir / "controlnet_rag_best.pt")
    args.clip_weights = str(weights_dir / "CLIP3D_Finding_Impression_30ep.pt")


def get_diffusion_inference(args: argparse.Namespace) -> dict:
    return getattr(
        args,
        "diffusion_unet_inference",
        {
            "dim": [512, 512, 128],
            "spacing": [0.75, 0.75, 3.0],
            "top_region_index": [0, 1, 0, 0],
            "bottom_region_index": [0, 1, 0, 0],
            "random_seed": 0,
            "num_inference_steps": 30,
            "modality": 1,
        },
    )


def load_clip(device: torch.device, clip_weights: str):
    cfgm = model_cfg_bank()("clip_3D")
    clip = get_model()(cfgm)
    clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=True)
    clip.to(device).eval()
    return clip


def load_mask(mask_path: str, device: torch.device) -> torch.Tensor:
    transforms = Compose(
        [
            LoadImaged(keys=["label"], image_only=True, ensure_channel_first=True),
            Orientationd(keys=["label"], axcodes="RAS"),
            EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
        ]
    )
    transformed = transforms({"label": mask_path})
    return transformed["label"].unsqueeze(0).to(device)


def encode_report_with_singleton_rag(clip, report: str, mask_path: str) -> tuple[torch.Tensor, str]:
    with torch.no_grad():
        embedding = clip([report], "encode_text").squeeze(0).cpu().numpy().astype("float32")

    rag_embeddings = embedding.copy()
    faiss.normalize_L2(rag_embeddings)
    index = faiss.IndexFlatIP(rag_embeddings.shape[1])
    index.add(rag_embeddings)

    query = embedding.copy()
    faiss.normalize_L2(query)
    _, retrieved = index.search(query, k=1)
    assert retrieved[0, 0] == 0

    cond = torch.tensor(embedding).unsqueeze(0)
    return cond, mask_path


def main() -> None:
    cli_args = parse_args()
    args = load_config(cli_args.env_config, cli_args.model_config, cli_args.model_def)
    local_rank, world_size, device = initialize_distributed(cli_args.num_gpus)
    logger = setup_logging("rag_demo_single")
    logger.info(f"Using device {device}")

    weights_dir = Path(cli_args.weights_dir)
    resolve_weight_paths(args, weights_dir)

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(_extract_state_dict(autoencoder_ckpt), strict=True)
    autoencoder.eval()

    unet = define_instance(args, "diffusion_unet_def").to(device)
    diffusion_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
    if isinstance(diffusion_ckpt, dict) and "scale_factor" in diffusion_ckpt:
        scale_factor = diffusion_ckpt["scale_factor"]
    else:
        scale_factor = 1.0
    unet.load_state_dict(_extract_state_dict(diffusion_ckpt), strict=True)
    unet.eval()

    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)
    controlnet.load_state_dict(_extract_state_dict(controlnet_ckpt), strict=True)
    controlnet.eval()

    noise_scheduler = define_instance(args, "noise_scheduler")
    clip = load_clip(device, args.clip_weights)

    cond, retrieved_mask_path = encode_report_with_singleton_rag(clip, cli_args.report, cli_args.mask)
    mask = load_mask(retrieved_mask_path, device)

    diffusion_unet_inference = get_diffusion_inference(args)
    output_size = tuple(diffusion_unet_inference["dim"])
    out_spacing = tuple(diffusion_unet_inference["spacing"])
    check_input(None, None, None, output_size, out_spacing, None)

    top_region_index_tensor = torch.tensor(
        np.array(diffusion_unet_inference["top_region_index"]).astype(float) * 1e2,
        dtype=torch.float16,
        device=device,
    )[None]
    bottom_region_index_tensor = torch.tensor(
        np.array(diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2,
        dtype=torch.float16,
        device=device,
    )[None]
    spacing_tensor = torch.tensor(
        np.array(diffusion_unet_inference["spacing"]).astype(float) * 1e2,
        dtype=torch.float16,
        device=device,
    )[None]
    modality_tensor = args.controlnet_infer["modality"] * torch.ones((1,), dtype=torch.long, device=device)
    latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)

    synthetic_images, _ = ldm_conditional_sample_one_image(
        autoencoder=autoencoder,
        diffusion_unet=unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        device=device,
        combine_label_or=mask,
        top_region_index_tensor=top_region_index_tensor,
        bottom_region_index_tensor=bottom_region_index_tensor,
        spacing_tensor=spacing_tensor,
        modality_tensor=modality_tensor,
        latent_shape=latent_shape,
        output_size=output_size,
        noise_factor=1.0,
        num_inference_steps=args.controlnet_infer.get("num_inference_steps", diffusion_unet_inference["num_inference_steps"]),
        autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
        autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
        use_cfg=args.controlnet_infer["use_cfg"],
        guidance_scale=args.controlnet_infer["guidance_scale"],
        cond=cond.to(device),
    )

    output_path = Path(cli_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = synthetic_images.squeeze().detach().cpu().numpy().astype(np.int16)
    affine = np.diag([out_spacing[0], out_spacing[1], out_spacing[2], 1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(output_path))

    meta_path = output_path.with_suffix("").with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "report": cli_args.report,
                "ct": cli_args.ct,
                "retrieved_mask": retrieved_mask_path,
                "output": str(output_path),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved demo output to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
