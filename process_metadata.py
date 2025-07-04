from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    mini_imagenet = ImageNet(split=split, root="mini-imagenet", extra="mini-imagenet-extra")
    mini_imagenet.dump_extra()
    imagenet_1k = ImageNet(split=split, root="imagenet", extra="imagenet-extra")
    imagenet_1k.dump_extra()
