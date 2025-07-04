python dinov2/eval/instance_recog.py 
        --data_root=revisitop/data/ 
        #--test_dataset=roxford5k
        --test_dataset=rparis6k 
        --config-file=pretraining_output/config.yaml 
        --pretrained-weights=pretraining_output/eval/training_step/teacher_checkpoint.pth 
        --noise=identity_0 
        --log_file=log.txt
