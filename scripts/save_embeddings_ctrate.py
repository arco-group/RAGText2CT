"""
Encode CT-RATE reports into text embeddings using the CLIP3D checkpoint.

- Reads volume lists from the train/validation JSON files (relative paths).
- Looks up corresponding reports from CSVs (columns: VolumeName, Findings_EN, Impressions_EN).
- Saves embeddings under the embedding base directory, matching the convention
  expected by the diffusion pipeline: <embedding_base>/<volume>_impression_<report_encoder_model>.npy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model


def load_volume_list(json_path: Path) -> list[str]:
    with open(json_path, "r") as file:
        data = json.load(file)
    key = "training" if "training" in data else "validation"
    return [_item["image"] for _item in data.get(key, [])]


def strip_nii_gz(path: str) -> str:
    return path[:-7] if path.endswith(".nii.gz") else Path(path).with_suffix("").as_posix()


def build_embedding_path(image_path: str, data_base_dir: Path, embedding_base_dir: Path, encoder_name: str) -> Path:
    img_path = Path(image_path)
    if img_path.is_absolute():
        try:
            rel = img_path.relative_to(data_base_dir)
        except ValueError:
            rel = img_path.name
    else:
        rel = img_path
    base = strip_nii_gz(rel.as_posix())
    return embedding_base_dir / f"{base}_impression_{encoder_name}.npy"


def main(args: argparse.Namespace) -> None:
    data_base_dir = Path(args.data_base_dir)
    embedding_base_dir = Path(args.embedding_base_dir)
    encoder_name = args.report_encoder_model

    train_images = load_volume_list(Path(args.train_json))
    val_images = load_volume_list(Path(args.val_json))

    reports_train = pd.read_csv(args.train_reports).set_index("VolumeName")
    reports_val = pd.read_csv(args.val_reports).set_index("VolumeName")

    cfgm = model_cfg_bank()("clip_3D")
    clip = get_model()(cfgm)
    clip_weights = Path(args.clip_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=True)
    clip.to(device)
    clip.eval()

    def encode_impressions(image_list: list[str], reports_df: pd.DataFrame):
        for img in tqdm(image_list, desc="encoding reports"):
            volume_name = Path(img).name
            if volume_name not in reports_df.index:
                continue
            findings = str(reports_df.loc[volume_name, "Findings_EN"])
            impressions = str(reports_df.loc[volume_name, "Impressions_EN"])
            text = f"Findings: {findings} Impression: {impressions}"

            emb_path = build_embedding_path(img, data_base_dir, embedding_base_dir, encoder_name)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            if emb_path.exists() and not args.overwrite:
                continue

            with torch.no_grad():
                embedding = clip([text], "encode_text").squeeze(0).cpu().numpy().astype(np.float32)
            np.save(emb_path, embedding)

    encode_impressions(train_images, reports_train)
    encode_impressions(val_images, reports_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode CT-RATE reports to embeddings")
    parser.add_argument("--train_json", type=str, default="data/train_data_volumes.json")
    parser.add_argument("--val_json", type=str, default="data/validation_data_volumes.json")
    parser.add_argument("--train_reports", type=str, default="data/train_reports.csv")
    parser.add_argument("--val_reports", type=str, default="data/validation_reports.csv")
    parser.add_argument("--data_base_dir", type=str, default="dataset")
    parser.add_argument("--embedding_base_dir", type=str, default="./embeddings")
    parser.add_argument("--clip_weights", type=str, default="./models/CLIP3D_Finding_Impression_30ep.pt")
    parser.add_argument("--report_encoder_model", type=str, default="xgem_3D")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing embeddings")
    args = parser.parse_args()
    main(args)
