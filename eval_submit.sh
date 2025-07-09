#!/bin/bash

last_job_id=""
corruptions=(
	brightness elastic_transform frost glass_blur 
	motion_blur shot_noise speckle_noise contrast
	gaussian_blur impulse_noise pixelate snow zoom_blur 
	defocus_blur fog gaussian_noise jpeg_compression saturate spatter
)
levels=(1 2 3 4 5) 
steps=([the checkpoint step to evaluate(i.e, 12499)])
output=([your_output_folder])
for step in "${steps[@]}"; do
	for corruption in "${corruptions[@]}"; do
	    for level in "${levels[@]}"; do
	    	if [ -z "$last_job_id" ]; then
            		job_id=$(sh --parsable --job-name=${corruption}_${level}_${step} eval_main.sh $corruption $level $step $output)
            	else
            		job_id=$(sh --parsable --dependency=afterok:$last_job_id --job-name=${corruption}_${level}_${step} eval_main.sh $corruption $level $step $output)
        	fi
        
        	last_job_id=$job_id
        	echo "Submitted job $job_id (${corruption}-${level}) depending on $last_job_id"
	    done
	done
done
