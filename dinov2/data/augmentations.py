# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random 
import torch
import cv2
import numpy as np
import math
from PIL import Image, ImageFilter
from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")
class AddPatchGaussianBlur:
    def __init__(
        self,
        patch_size: int,
        max_sigma: float,
        randomize_patch_size: bool,
        randomize_sigma: bool,
        use_opencv: bool = True
    ):
        assert patch_size >= 1 or patch_size == -1, "Patch size must be ≥1 or -1."
        assert max_sigma > 0, "Sigma must be positive."

        self.patch_size = patch_size
        self.max_sigma = max_sigma
        self.randomize_patch_size = randomize_patch_size
        self.randomize_sigma = randomize_sigma
        self.use_opencv = use_opencv

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        x = transforms.ToTensor()(pil_img)
        c, h, w = x.shape

        patch_size, sigma = self._randomize_params(h, w)

        if self.use_opencv:
            blur_fn = self._opencv_blur
        else:
            blur_fn = self._pil_blur

        if patch_size == -1:
            blurred_img = blur_fn(pil_img, sigma)
            return blurred_img
        else:
            h_start, w_start = self._get_patch_start(h, w, patch_size)

            patch_pil = transforms.ToPILImage()(
                x[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
            )
            blurred_patch = blur_fn(patch_pil, sigma)

            merged_img = pil_img.copy()
            merged_img.paste(blurred_patch, (w_start, h_start))
            return merged_img

    def _randomize_params(self, h: int, w: int) -> tuple[int, float]:
        if self.patch_size == -1:
            max_patch = min(h, w)
        else:
            max_patch = self.patch_size

        if self.randomize_patch_size:
            patch_size = random.randint(1, max_patch)
        else:
            patch_size = self.patch_size if self.patch_size != -1 else max_patch

        if self.randomize_sigma:
            sigma = random.uniform(0.1, self.max_sigma)  # 避免sigma=0
        else:
            sigma = self.max_sigma

        return patch_size, sigma

    def _get_patch_start(self, h: int, w: int, patch_size: int) -> tuple[int, int]:
        return (
            random.randint(0, h - patch_size),
            random.randint(0, w - patch_size)
        )

    def _pil_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        radius = max(1, int(sigma * 2))
        return img.filter(ImageFilter.GaussianBlur(radius))

    def _opencv_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        img_np = np.array(img)
        blurred_np = cv2.GaussianBlur(
            img_np,
            ksize=(0, 0),
            sigmaX=sigma
        )
        return Image.fromarray(blurred_np)

class AddPatchGaussian:
    def __init__(
        self,
        patch_size: int,
        max_scale: float,
        randomize_patch_size: bool,
        randomize_scale: bool
    ):
        assert patch_size >= 1 or patch_size == -1, "Patch size must be ≥1 or -1."
        assert 0.0 <= max_scale <= 1.0, "Scale must be in [0, 1]."

        self.patch_size = patch_size
        self.max_scale = max_scale
        self.randomize_patch_size = randomize_patch_size
        self.randomize_scale = randomize_scale

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        # Convert PIL image to tensor
        x = transforms.ToTensor()(pil_img)
        c, h, w = x.shape

        # Randomize parameters
        patch_size, scale = self._randomize_params(h, w)

        if patch_size == -1:
            # Apply noise to the entire image
            noise = torch.randn_like(x) * scale
            x.add_(noise).clamp_(0.0, 1.0)
        else:
            # Get patch coordinates
            h_start, w_start = self._get_patch_start(h, w, patch_size)

            # Apply noise to the selected patch (in-place)
            patch = x[
                :,
                h_start : h_start + patch_size,
                w_start : w_start + patch_size
            ]
            noise = torch.randn_like(patch) * scale
            patch.add_(noise).clamp_(0.0, 1.0)

        # Convert back to PIL
        return transforms.ToPILImage()(x)

    def _randomize_params(self, h: int, w: int) -> tuple[int, float]:
        # Determine patch size
        if self.patch_size == -1:
            max_size = min(h, w)
        else:
            max_size = self.patch_size

        if self.randomize_patch_size:
            patch_size = random.randint(1, max_size)
        else:
            patch_size = self.patch_size if self.patch_size != -1 else max_size

        # Determine scale
        if self.randomize_scale:
            scale = random.uniform(0, self.max_scale)
        else:
            scale = self.max_scale

        return patch_size, scale

    def _get_patch_start(self, h: int, w: int, window_size: int) -> tuple[int, int]:
        # Compute valid start positions
        max_h_start = max(0, h - window_size)
        max_w_start = max(0, w - window_size)

        h_start = random.randint(0, max_h_start)
        w_start = random.randint(0, max_w_start)

        return h_start, w_start

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        current_epoch=0,
        gaussian_patching=False,
        gaussian_blurring=False,
        less_aug=False,
        AA=True
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.current_epoch=current_epoch
        self.gp = gaussian_patching
        self.gb = gaussian_blurring
        self.AA = AA
        self.less_aug = less_aug
        # patch_size_options = [20, 30, 50, 100, 150] 
        # max_scale_options = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

        # selected_patch_size = random.choice(patch_size_options)  #
        # selected_max_scale = random.choice(max_scale_options)
        #self.gaussian_patching = AddPatchGaussian(patch_size=-1, max_scale=0.5,
        #                    randomize_patch_size=True,
        #                    randomize_scale=True)


        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global_AA = transforms.Compose(
            [   
                #transforms.Resize((global_crops_size,global_crops_size),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC,antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_global = transforms.Compose(
            [
                #transforms.Resize((global_crops_size,global_crops_size),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local_AA = transforms.Compose(
            [
                #transforms.Resize((global_crops_size,global_crops_size),interpolation=transforms.InterpolationMode.BICUBIC),
                #transforms.RandomCrop(local_crops_size),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC,antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [
                #transforms.Resize((global_crops_size,global_crops_size),interpolation=transforms.InterpolationMode.BICUBIC),
                #transforms.RandomCrop(local_crops_size),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        
        
        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )
        # remove augmentation that can affect low frequency in first stage training.
        
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])
        
        #self.global_transfo1 = transforms.Compose([color_jittering, self.normalize])
        #self.global_transfo2 = transforms.Compose([color_jittering, self.normalize])
        #self.local_transfo = transforms.Compose([color_jittering, self.normalize])    
    
        self.global_transfo1_less_aug = transforms.Compose([self.normalize])
        self.global_transfo2_less_aug = transforms.Compose([self.normalize])
        self.local_transfo_less_aug = transforms.Compose([self.normalize])       

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def __call__(self, image):
        output = {}
        
        if self.current_epoch < 150:
            max_scale=0.2
        else:
            max_scale=0.5
        self.gaussian_patching_blur = something = AddPatchGaussianBlur(
            patch_size=-1, 
            max_sigma=2.0,
            randomize_patch_size=True,
            randomize_sigma=True)

        self.gaussian_patching = AddPatchGaussian(patch_size=-1, max_scale=max_scale,
                            randomize_patch_size=True,
                            randomize_scale=True)
        dice = np.random.randint(2)
        #gaussian patching
        if self.gp is True:
            image = self.gaussian_patching(image)
        if self.gb is True:
            image = self.gaussian_patching_blur(image)


        # global crops:
        if self.AA is True:
             im1_base = self.geometric_augmentation_global_AA(image)
             im2_base = self.geometric_augmentation_global_AA(image)
        else:
             im1_base = self.geometric_augmentation_global(image)
             im2_base = self.geometric_augmentation_global(image) 
        
        #if self.gp is True:
        #    im1_base = self.gaussian_patching(im1_base)
        #    im2_base = self.gaussian_patching(im2_base)        

        if self.less_aug is True:
            global_crop_1 = self.global_transfo1_less_aug(im1_base)

            global_crop_2 = self.global_transfo2_less_aug(im2_base)
        else:

            global_crop_1 = self.global_transfo1(im1_base)

            global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        if self.AA is True:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local_AA(image)) for _ in range(self.local_crops_number)
            ]
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
