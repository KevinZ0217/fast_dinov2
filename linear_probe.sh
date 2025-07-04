#!/bin/bash

#SBATCH -n 2
#SBATCH -c 6
#SBATCH --mem=192G
#SBATCH --gres=gpu:a5000:1
#SBATCH --nodes=1
#SBATCH -t 2:00:00
#SBATCH -p gpu

STEP=$1
OUTPUT=$2

PYTHONPATH="robust_dinov2" python dinov2/run/eval/linear.py \
	--config-file $OUTPUT/config.yaml \
        --pretrained-weights $OUTPUT/eval/training_$STEP/teacher_checkpoint.pth \
        --output-dir $OUTPUT/eval/training_$STEP/linear \
	--train-dataset ImageNet:split=TRAIN:root=ILSVRC2012:extra=imagenet-extra \
        --val-dataset ImageNet:split=VAL:root=imagenet/:extra=imagenet-extra
