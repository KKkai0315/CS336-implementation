# type:ignore
import torch
import torch.nn
import os
import regex as re
import pathlib
import math
import pickle
from torch import Tensor
from jaxtyping import Int, Float
from einops import einsum, rearrange
from collections import Counter, defaultdict

class Linear(torch.nn.Module):
    """
    A simple linear layer that can be used in the model.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features)
                , std=std, a=-3*std, b=3*std
            ),
            requires_grad=True
        )
        

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(torch.nn.Module):
    """
    A simple embedding layer that can be used in the model.
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        std = 1.0
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim)
                , std=std, a=-3*std, b=3*std
            ),
            requires_grad=True
        )

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]  

class RMSNorm(torch.nn.Module):
    """
    A simple RMSNorm layer that can be used in the model.
    """
    def __init__(self, 
                d_model: int,
                eps: float = 1e-6,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
                 ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        rms = torch.rsqrt(x.pow(2).mean(-1,keepdim = True) + self.eps)
        x = x * rms
        return (self.weight * x).to(in_dtype)
    
class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return x * torch.sigmoid(x)

class SWiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
    
class RoPE(torch.nn.Module):
    """
    A simple RoPE layer that can be used in the model.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 seq_len: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        assert d_k % 2 == 0

        d = torch.arange(0, d_k, 2).float() / d_k
        freq = 1 / (theta ** d)
        t = torch.arange(seq_len).float()
        freqs = freq.unsqueeze(0)* t.unsqueeze(1)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self,
                x: Float[Tensor, "... seq_len d_model"],
                seq_len: int,
                pos_ids: Int[Tensor, "... seq_len"] | None = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None
                ) -> Float[Tensor, "... seq_len d_model"]:
        if pos_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[pos_ids]
            sin = self.sin_cached[pos_ids]
        x1 = x[...,::2]
        x2 = x[...,1::2]
        while cos.dim() < x1.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        # return torch.cat((x1_new, x2_new), dim=-1)
        result = torch.zeros(x.shape)
        result[...,::2] = x1_new
        result[...,1::2] = x2_new
        return result

