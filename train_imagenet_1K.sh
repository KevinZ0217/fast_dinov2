#low resolution training phase for 150 epochs
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_200_size112_1k_config.yaml \
    --output-dir ssl_200_size112_1k_output \
    train.dataset_path=ImageNet:split=TRAIN:root=imagenet:extra=imagenet-extra)

#restart the training using the checkpoint at the 150th epochs
JOB_ID=$(python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/ssl_resume_150_IN1K.yaml \
    --output-dir restart \
    train.dataset_path=ImageNet:split=TRAIN:root=mini-imagenet:extra=mini-imagenet-extra)
