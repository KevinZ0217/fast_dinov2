#!/bin/bash

#SBATCH -n 4
#SBATCH -c 6
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH -t 2:00:00
#SBATCH -p gpu-he

PYTHONPATH="robust_dinov2" python dinov2/run/train/train.py \
    --nodes 1 \
    --ngpus 4 \
    --config-file dinov2/configs/resume_150_1K.yaml\
    --output-dir resume_150_1K_gp\
    --max_to_keep 45 \
    --save_frequency 5 \
    train.dataset_path=ImageNet:split=TRAIN:root=ILSVRC2012:extra=imagenet-extra
