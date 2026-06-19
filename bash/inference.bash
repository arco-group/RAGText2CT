#!/usr/bin/env bash

# Simple launcher to run diffusion inference (test).
# Edit paths/configs as needed.

python scripts/diff_model_infer.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_eval.json \
  --num_gpus 1 \
  --index 0 \
  --resize 512

# Retrieval-augmented inference:
# python scripts/infer_controlnet_RAG.py \
#   --environment-file ./configs/environment_rag_controlnet_eval.json \
#   --config-file ./configs/config_rag_rflow.json \
#   --training-config ./configs/config_rag_controlnet.json \
#   --gpus 1 \
#   --index 0
