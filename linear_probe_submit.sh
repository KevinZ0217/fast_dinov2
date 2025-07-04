#!/bin/bash

last_job_id=""
#last_job_id2=""
steps=(
#162499 156249 149999 143749 137499 131249 124999 118749 112499 106249 99999 93749 87499 81249 74999 68749 62499 56249 49999 43749 37499 31249 24999 18749 12499 6249
#249999 231249 224999 218749 212499 206249 199999 193749 187499 181249 174999 168749 162499 156249 149999 143749 137499 131249 124999 118749 112499 106249 99999 93749 87499 81249 68749 62499 56249 49999 43749 37499 31249 24999 12499 9999 7499 6249
#249999 237499 224999 212499 199999 187499 174999 162499 149999 137499 124999 112499 99999 87499 74999 62499 49999 37499 24999 12499)
124999)
output=(resume_150_1K_gp)
for step in "${steps[@]}"; do
                if [ -z "$last_job_id" ]; then
                        job_id=$(sbatch --parsable --job-name=${step} linear_probe2.sh $step $output)
                else
                        job_id=$(sbatch --parsable --dependency=afterok:$last_job_id --job-name=${step} linear_probe2.sh $step $output)
                fi

                last_job_id=$job_id
                echo "Submitted job $job_id (${step}) depending on $last_job_id"
done

