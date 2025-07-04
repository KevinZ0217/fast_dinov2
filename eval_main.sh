#!/bin/bash

#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=192G
#SBATCH --gres=gpu:a5500:1
#SBATCH --nodes=1
#SBATCH -t 2:00:00
#SBATCH -p gpu

CORRUPTION=$1
LEVEL=$2
STEP=$3
OUTPUT=$4

PYTHONPATH="/robust_dinov2" python dinov2/run/eval/linear_fast.py \
    --config-file $OUTPUT/config.yaml \
    --pretrained-weights $OUTPUT/eval/training_$STEP/teacher_checkpoint.pth \
    --output-dir $OUTPUT/eval/training_$STEP/corruption/linear_${CORRUPTION}${LEVEL} \
    --classifier-fpath $OUTPUT/eval/training_$STEP/linear/model_final.pth \
    --train-dataset ImageNet:split=TRAIN:root=ILSVRC2012:extra=imagenet-extra \
    --val-dataset ImageNet:split=VAL:root=/ImageNet-C/${CORRUPTION}/${LEVEL}:extra=imagenet-extra
