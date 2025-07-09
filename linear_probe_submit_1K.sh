#!/bin/bash

steps=(
249999 237499 224999 212499 199999 187499 174999 162499 149999 137499 124999 112499 99999 87499 74999 62499 49999 37499 24999 12499
)
output="<output folder name from pretraining>"

for step in "${steps[@]}"; do
    echo "Running linear_probe_imagenet_1k.sh with step $step and output $output"
    sh linear_probe_imagenet_1k.sh "$step" "$output"
done
