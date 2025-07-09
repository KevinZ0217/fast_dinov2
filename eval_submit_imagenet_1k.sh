#!/bin/bash

corruptions=(
	brightness elastic_transform frost glass_blur 
	motion_blur shot_noise speckle_noise contrast
	gaussian_blur impulse_noise pixelate snow zoom_blur 
	defocus_blur fog gaussian_noise jpeg_compression saturate spatter
)
levels=(1 2 3 4 5) 
steps=(12499)  # put your checkpoint steps here
output="your_output_folder_for_INET1k"

for step in "${steps[@]}"; do
	for corruption in "${corruptions[@]}"; do
	    for level in "${levels[@]}"; do
	        echo "Running eval_main.sh $corruption $level $step $output"
	        sh eval_main_imagenet_1k.sh $corruption $level $step $output
	    done
	done
done
