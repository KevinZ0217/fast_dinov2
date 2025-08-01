#!/bin/bash
STEP=$1
OUTPUT=$2

python dinov2/run/eval/linear.py \
	--config-file $OUTPUT/config.yaml \
        --pretrained-weights $OUTPUT/eval/training_$STEP/teacher_checkpoint.pth \
        --output-dir $OUTPUT/eval/training_$STEP/linear \
	--train-dataset ImageNet:split=TRAIN:root=mini-imagenet:extra=-mini-imagenet-extra \
        --val-dataset ImageNet:split=VAL:root=mini-imagenet/:extra=mini-imagenet-extra
