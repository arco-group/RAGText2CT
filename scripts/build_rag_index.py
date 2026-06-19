from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_items(json_path: Path) -> list[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    key = "training" if "training" in data else "validation"
    return data[key]


def strip_nii_gz(path: str) -> str:
    return path[:-7] if path.endswith(".nii.gz") else path


def main(args: argparse.Namespace) -> None:
    items = load_items(Path(args.data_list))
    embeddings = []
    mask_paths = []

    for item in items:
        image_path = item["image"]
        base = strip_nii_gz(image_path)
        emb_path = Path(args.embedding_base_dir) / f"{base}_impression_{args.report_encoder_model}.npy"
        if not emb_path.exists():
            continue

        mask_path = item.get("label")
        if not mask_path:
            mask_path = image_path.replace("dataset/", "dataset/ts_seg/ts_total/")
            mask_path = mask_path.replace("/train/", "/train_fixed/").replace("/valid/", "/valid_fixed/")

        embeddings.append(np.load(emb_path).astype(np.float32))
        mask_paths.append(mask_path)

    if not embeddings:
        raise RuntimeError("No embeddings found to build the RAG index.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "impression_embeddings.npy", np.stack(embeddings, axis=0))
    with open(out_dir / "impression_paths.json", "w") as f:
        json.dump(mask_paths, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retrieval files for RAG Text2CT")
    parser.add_argument("--data_list", type=str, default="dataset/train_data_volumes.json")
    parser.add_argument("--embedding_base_dir", type=str, default="./embeddings")
    parser.add_argument("--report_encoder_model", type=str, default="xgem_3D")
    parser.add_argument("--output_dir", type=str, default="./retrieval")
    main(parser.parse_args())
