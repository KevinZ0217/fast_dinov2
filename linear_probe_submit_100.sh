#!/bin/bash

last_job_id=""
steps=(
249999 237499 224999 212499 199999 187499 174999 162499 149999 137499 124999 112499 99999 87499 74999 62499 49999 37499 24999 12499
)
output=(restart_100)
for step in "${steps[@]}"; do
                if [ -z "$last_job_id" ]; then
                        job_id=$(sh linear_probe_imagenet_100.sh $step $output)
                else
                        job_id=$(sh linear_probe_imagenet_100.sh $step $output)
                fi

                last_job_id=$job_id
                echo "Submitted job $job_id (${step}) depending on $last_job_id"
done
