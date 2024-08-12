#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/train_regular.sh


export HYDRA_CONFIG_NAME="train_contrastive_diffusion_regular.yaml"
python ./src/train.py