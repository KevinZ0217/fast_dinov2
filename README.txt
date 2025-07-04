# FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed

## Environmental setup
Please follow [DINOv2](https://github.com/facebookresearch/dinov2) for all required packages and environments.






1. In train.sh, change --config-file to dinov2/configs/ssl_112_config.yaml --output-dir path to your own output folder(e.g. /users/username/scratch/robust_dinov2_output/ssl_112_output), "root" path and "extra" path in train.dataset_path=Imagenet to your own dataset path.
2 - training: sbatch train.sh. Check the current directory, a (slurm-"jobid".out) file will be generated. You can go to your output folder to monitor the training process and eta for training to finish. The log file is in (robust_dinov2_output/ssl_112_output/logs/log.txt), if any exception and error are triggered then go to (robust_dinov2_output/"job-id"_0_log.err) to check what error it is.
3 - linear probing: Training will be finished in around 4 hours. Set up linear_probe.sh. If your output folder is (/users/username/scratch/robust_dinov2_output/ssl_112_output), for example, Open linear_probe.sh, 
    - change --config-file to (/users/username/scratch/robust_dinov2_output/$OUTPUT/config.yaml),
    - change --pretrained-weights to (/users/username/scratch/robust_dinov2_output/$OUTPUT/eval/training_$STEP/teacher_checkpoint.pth, 
    - change --output-dir to (/users/username/scratch/robust_dinov2_output/$OUTPUT/eval/training_$STEP/linear)
    - change the "root" and "extra" path for dataset to your own mini-imagenet" dataset directory, like in train.sh.
4 - linear probing: Go to linear_probe_submit.sh. Change the "steps" to the number of steps you want to evaluate((62499, 12499), for example). Assuming your output folder for train.sh is (/users/username/scratch/robust_dinov2_output/ssl_112_output), change output to (ssl_112_output). linear_probe_submit.sh will call linear_probe.sh, and we can do linear probing on multiple checkpoint sequentially - only evaluate the next checkpoint after the previous linear probing is done.
    - run bash linear_probe_submit.sh
5 - testing on mini-imagenet-c. We will use the linear classifier trained on clean image in step 4 to test on mini-imagenet-c. Just like in linear probing, there will be eval_main.sh and eval_submit.sh, where eval_submit.sh will call eval_main.sh. By doing this all the corruption datasets can be tested sequentially without exceeding the CPULIMIT in Oscar. 
    - Go to eval_main.sh. If your output folder in train.sh is (/users/username/scratch/robust_dinov2_output/ssl_112_output), then in every argument, replace whatever before $OUTPUT to be (/users/username/scratch/robust_dinov2_output/) - for example, (/users/{username}/data/{user}/robust_dinov2_output/$OUTPUT/config.yaml) to (/users/usename/scratch/robust_dinov2_output/$OUTPUT/config.yaml). 
    - replace the dataset path by your own dataset path, just like in linear probing and training.
    - replace the --val-dataset's "root" and "extra" with your own imagenet-c dataset path.
6 - testing on mini-imagenet-c. Open eval_submit.sh. In corruptions you will see all kinds of corruptions dataset, with 5 levels. You will need to change the "steps" to be the checkpoint step you want to evaluate (249999, for example), and change the "output" to be your output folder, just like for linear probing. 
    - run bash eval_submit.sh
7. - to restart: we do "restart" in order to do the second stage training. Go to restart.sh file, and do the same change as in train.sh. The major change is in config.yaml - we use dinov2/configs/ssl_resume_150_config_size112-224.yaml, which means we use the 150 epochs of our first stage training checkpoints, and restart training using 224 size images. Go to dinov2/configs/ssl_resume_150_config_size112-224.yaml, in the first line of MODEL: WEIGHTS, change the path to the model_0187499.rank_0.pth to your own path of this checkpoint. it's 0187499 because each epoch has 1250 gradient steps, and 1250*150 = 187500. Then do sbatch restart.sh
