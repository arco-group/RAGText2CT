#!/usr/bin/env bash

# Extract latent embeddings with the frozen VAE using diff_model_create_training_data.
# Edit paths/configs as needed.

python scripts/diff_model_create_training_data.py \
  --model_def ./configs/config_rflow.json \
  --model_config ./configs/config_diff_model.json \
  --env_config ./configs/environment_diff_model_train.json \
  --num_gpus 1 \
  --index 0
