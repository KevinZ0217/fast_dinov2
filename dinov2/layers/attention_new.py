# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("dinov2")
import loralib as lora

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

class LoRAAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 r: int = 4, alpha: float = 32):
        super().__init__()
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(out_features, r))
            self.alpha = alpha
            self.r = r
        else:
            self.A = self.B = None

    def forward(self):
        # returns the weight delta matrix [out_features × in_features]
        return (self.B @ self.A) * (self.alpha / self.r)
# =============================================================================
# LoRA Module: Low-Rank Adaptation for Linear Layers with Uniform requires_grad
# =============================================================================
class LoRALinear(nn.Module):
    """
    Wraps a standard linear layer with LoRA (Low-Rank Adaptation). Instead of freezing the
    base linear weights by setting requires_grad=False, we register hooks that zero out their gradients.
    This ensures all parameters in the module have `requires_grad=True` for FSDP, while effectively
    keeping the base weights unchanged during training.
    
    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        r (int): rank for the low-rank decomposition (default: 4).
        lora_alpha (float): scaling factor for the low-rank update (default: 32).
        bias (bool): whether the underlying linear layer should have a bias.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4, lora_alpha: float = 32, bias: bool = True):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        # Create the original linear layer normally.
        #self.linear = nn.Linear(in_features, out_features, bias=bias)
        base = nn.Linear(in_features, out_features, bias=bias)
        self.register_parameter("weight", base.weight)
        if bias:
            self.register_parameter("bias", base.bias)
        # freeze them
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False
        # IMPORTANT: Ensure all parameters are marked as trainable for FSDP uniformity.
        #self.linear.weight.requires_grad = False
        # Register a hook to zero out gradients so these weights effectively remain frozen.
        #self.linear.weight.register_hook(lambda grad: torch.zeros_like(grad))
        #if bias and self.linear.bias is not None:
        #    self.linear.bias.requires_grad = False
            #self.linear.bias.register_hook(lambda grad: torch.zeros_like(grad))
        #self.linear = nn.Linear(in_features, out_features, bias=bias)
        #for param in self.linear.parameters():
        #    param.requires_grad = False
        # Initialize LoRA parameters if rank > 0.
        #if r > 0:
        #    # Low-rank matrices: A (projecting to a smaller dimension) and B (projecting back)
        #    self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        #    self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        #else:
        #    self.lora_A = None
        #    self.lora_B = None
        self.adapter = LoRAAdapter(in_features, out_features, r, lora_alpha)
    def forward(self, x: Tensor) -> Tensor:
        # always use functional linear so we can swap in a fused weight
        #base_w, base_b = self.linear.weight, self.linear.bias
        w = self.weight
        b = self.bias if hasattr(self, "bias") else None
        #if self.r > 0:
        if self.adapter.r > 0: 
           # delta = B @ A
            # shape: [out_features, in_features]
            weight_delta = self.adapter()
            #weight_delta = (self.lora_B @ self.lora_A) * (self.lora_alpha / self.r)
            # single fused mat‑mul
            return F.linear(x, w + weight_delta, b)
        else:
            return F.linear(x, w, b)
    

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lora_r: int = 64,
        lora_alpha: float = 128,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = LoRALinear(dim, dim * 3, r=lora_r, lora_alpha=lora_alpha, bias=qkv_bias)
        #self.qkv = lora.Linear(dim, 3*dim, r=lora_r,lora_alpha=lora_alpha, bias=qkv_bias)
        #self.qkv = lora.MergedLinear(dim, 3*dim, r=32, enable_lora=[True, False, True], bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
