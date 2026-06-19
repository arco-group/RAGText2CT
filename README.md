# 🎉🎉🎉 MICCAI 2026 ACCEPTED: RAGText2CT 🎉🎉🎉
# 👉 Code release for **[Retrieval-Augmented Anatomical Guidance for Text-to-CT Generation](https://arxiv.org/abs/2603.08305)**

Molino, D., Caruso, C. M., Soda, P., Guarrasi, V. (2026)

This repository extends the original **Text2CT** model with a retrieval-guided anatomical branch. Given a report embedding, the pipeline retrieves a semantically related case and uses its anatomical mask as structural guidance through a 3D `ControlNetMaisi` branch.

## Overview

The release is organized around four stages:

1. Train or reuse the original Text2CT backbone:
   - CLIP3D text encoder
   - 3D autoencoder
   - text-conditioned latent diffusion UNet
2. Precompute report embeddings with `scripts/save_embeddings_ctrate.py`.
3. Build the retrieval bank with `scripts/build_rag_index.py`.
4. Train and run the retrieval-augmented ControlNet with:
   - `scripts/train_controlnet.py`
   - `scripts/infer_controlnet_RAG.py`

## Environment

Python `3.10+` is recommended.

```bash
pip install -r requirements.txt
```

### Weights (place in `models/`)

You can download them from Hugging Face at [Weights](https://huggingface.co/dmolino/RAGText2CT):
```python
from huggingface_hub import snapshot_download

repo_id = "dmolino/RAGText2CT"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir="your_local_path" 
)
```
Place checkpoints in `models/`.

- `autoencoder_epoch273.pt`
- `unet_rflow_200ep.pt`
- `CLIP3D_Finding_Impression_30ep.pt`
- `controlnet_rag_best.pt`

## Data Layout

This release assumes the CT-RATE-style JSON lists already point to image paths like:

```text
dataset/train/...
dataset/valid/...
```

The repo expects:

- `dataset/train_data_volumes.json`
- `dataset/validation_data_volumes.json`
- `dataset/train_reports.csv`
- `dataset/validation_reports.csv`
- CT volumes under `dataset/`
- segmentation masks under a parallel tree such as `dataset/ts_seg/ts_total/...`

Default configs use `data_base_dir: "."`, so JSON paths remain valid as-is.

## Quickstart

### 1. Text embeddings

```bash
python scripts/save_embeddings_ctrate.py \
  --train_json dataset/train_data_volumes.json \
  --val_json dataset/validation_data_volumes.json \
  --train_reports dataset/train_reports.csv \
  --val_reports dataset/validation_reports.csv \
  --data_base_dir . \
  --embedding_base_dir ./embeddings \
  --clip_weights ./models/CLIP3D_Finding_Impression_30ep.pt \
  --report_encoder_model xgem_3D
```

### 2. Latent CT embeddings

```bash
python scripts/diff_model_create_training_data.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1 \
  --index 0
```

### 3. Retrieval bank

```bash
python scripts/build_rag_index.py \
  --data_list dataset/train_data_volumes.json \
  --embedding_base_dir ./embeddings \
  --report_encoder_model xgem_3D \
  --output_dir ./retrieval
```

This produces:

- `retrieval/impression_embeddings.npy`
- `retrieval/impression_paths.json`

The paths file stores mask paths used as anatomical proxies during retrieval.

### 4. Train the RAG ControlNet

```bash
python scripts/train_controlnet.py \
  --environment-file ./configs/environment_rag_controlnet_train.json \
  --config-file ./configs/config_rag_rflow.json \
  --training-config ./configs/config_rag_controlnet.json \
  --gpus 1
```

### 5. Retrieval-augmented inference

```bash
python scripts/infer_controlnet_RAG.py \
  --environment-file ./configs/environment_rag_controlnet_eval.json \
  --config-file ./configs/config_rag_rflow.json \
  --training-config ./configs/config_rag_controlnet.json \
  --gpus 1 \
  --index 0
```

Outputs are written under `predictions/rag/`.

### 6. Single-case demo

For a minimal end-to-end smoke test, place one CT and one mask in the repo root as `ct.nii.gz` and `mask.nii.gz`, then run:

```bash
python scripts/rag_demo_single.py \
  --report "Small right pleural effusion. Mild bibasal atelectatic changes. No focal lung consolidation." \
  --ct ct.nii.gz \
  --mask mask.nii.gz \
  --weights-dir hf_ragtext2ct/models \
  --output predictions/rag_demo_single.nii.gz
```

This demo encodes the report, uses the provided mask as the singleton retrieval target, and writes:

- `predictions/rag_demo_single.nii.gz`
- `predictions/rag_demo_single.json`

## Main Files

- `configs/config_rag_rflow.json`: model definitions for autoencoder, UNet and ControlNet.
- `configs/config_rag_controlnet.json`: ControlNet train/infer hyperparameters.
- `configs/environment_rag_controlnet_train.json`: training paths.
- `configs/environment_rag_controlnet_eval.json`: evaluation paths and retrieval bank paths.
- `scripts/build_rag_index.py`: stacks per-case report embeddings and exports the retrieval mapping.
- `scripts/train_controlnet.py`: ControlNet training.
- `scripts/infer_controlnet_RAG.py`: retrieval-augmented generation.

## Notes

- This repo is intentionally model-focused. Challenge-specific wrappers and `VLM3D` container code are excluded.
- The retrieval bank is currently file-based (`.npy` + `.json`) and loaded into a FAISS inner-product index at inference time.
- The original Text2CT pipeline is still available through the `diff_model_*` scripts.
- For HF downloads of the weights bundle, see `dmolino/RAGText2CT`.

## Citation

```bibtex
@article{Molino2026RAGText2CT,
  title={Retrieval-Augmented Anatomical Guidance for Text-to-CT Generation},
  author={Molino, Daniele and Caruso, Camillo Maria and Soda, Paolo and Guarrasi, Valerio},
  year={2026},
  journal={arXiv preprint arXiv:2603.08305}
}
```
