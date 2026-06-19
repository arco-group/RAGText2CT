#!/usr/bin/env bash

# Simple launcher to train the diffusion model.
# Fill in your paths/configs as needed.

python scripts/diff_model_train.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1

# Train the RAG ControlNet after the text-only diffusion model:
# python scripts/train_controlnet.py \
#   --environment-file ./configs/environment_rag_controlnet_train.json \
#   --config-file ./configs/config_rag_rflow.json \
#   --training-config ./configs/config_rag_controlnet.json \
#   --gpus 1
