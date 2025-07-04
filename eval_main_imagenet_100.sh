#!/bin/bash
CORRUPTION=$1
LEVEL=$2
STEP=$3
OUTPUT=$4

python dinov2/run/eval/linear_fast.py \
    --config-file $OUTPUT/config.yaml \
    --pretrained-weights $OUTPUT/eval/training_$STEP/teacher_checkpoint.pth \
    --output-dir $OUTPUT/eval/training_$STEP/corruption/linear_${CORRUPTION}${LEVEL} \
    --classifier-fpath $OUTPUT/eval/training_$STEP/linear/model_final.pth \
    --train-dataset ImageNet:split=TRAIN:root=mini-imagenet:extra=mini-imagenet-extra \
    --val-dataset ImageNet:split=VAL:root=/mini-ImageNet-C/${CORRUPTION}/${LEVEL}:extra=mini-imagenet-extra

