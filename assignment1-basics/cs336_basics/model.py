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

class Softmax(torch.nn.Module):
    """
    A simple Softmax layer that can be used in the model.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x = x - x.max(dim=self.dim, keepdim=True)[0]
        expx = torch.exp(x)
        return expx / expx.sum(dim=self.dim, keepdim=True)
    
class Scaled_Dot_Product_Attention(torch.nn.Module):
    """
    A simple Scaled Dot Product Attention layer that can be used in the model.
    """
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.softmax = Softmax(dim=-1)

    def forward(self, 
                query: Float[Tensor, "... queries d_k"],
                key: Float[Tensor, "... keys d_k"],
                value: Float[Tensor, "... values d_v"],
                mask: Int[Tensor, "... queries keys"] | None = None
                ) -> Float[Tensor, "... queries d_v"]:
        attn = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys")/math.sqrt(torch.tensor(query.shape[-1]))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(attn)
        return einsum(attn, value, "... queries keys, ... keys d_v -> ... queries d_v")
    
class MultiHeadAttention(torch.nn.Module):
    """
    A simple Multi-Head Attention layer that can be used in the model.
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 theta: float = 10000.0,
                 seq_len: int = 512,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads
        
        self.q_proj = Linear(d_model, d_k * num_heads)
        self.k_proj = Linear(d_model, d_k * num_heads)
        self.v_proj = Linear(d_model, d_v * num_heads)
        self.o_proj = Linear(d_v * num_heads, d_model)

        self.positional_encoder = RoPE(theta=theta, d_k=self.d_k, seq_len=seq_len)

        self.attention = Scaled_Dot_Product_Attention()

    def forward(self, 
                x: Float[Tensor, " ... sequence_length d_in"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None
                ) -> Float[Tensor, " ... sequence_length d_out"]:
        *b, sequence_length, d_in = x.shape
        assert d_in == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (
            rearrange(X, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
            for X in (Q, K, V)
        )
        if token_positions is not None:
            token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")
        else:
            token_positions = torch.arange(sequence_length).unsqueeze(0).unsqueeze(0)
        Q = self.positional_encoder(Q, sequence_length, token_positions)
        K = self.positional_encoder(K, sequence_length, token_positions)

        seq = torch.arange(sequence_length)
        qi = seq.view(-1,1) #(seq_len, 1)
        ki = seq.view(1,-1) #(1, seq_len)

        causal_mask = qi >= ki #(seq_len, seq_len)

        for i in range(len(b)+1):
            causal_mask = causal_mask.unsqueeze(0)
            
        attn_output = self.attention(Q, K, V, causal_mask)
        attn_output = rearrange(attn_output, "... heads seq_len d_v -> ... seq_len (heads d_v)")
        attn_output = self.o_proj(attn_output)
        return attn_output
            
class Cross_Entropy_Loss(torch.nn.Module):
    """
    A simple Cross Entropy Loss layer that can be used in the model.
    """
    def __init__(self,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()

    def forward(self,
                inputs: Float[Tensor, "... d_model"], 
                targets: Int[Tensor, "..."])-> Float[Tensor, "..."]:
        """
        Computes the cross-entropy loss between the inputs and targets.
        
        Args:
            inputs (Float[Tensor, "... d_model"]): The input logits.
            targets (Int[Tensor, "..."]): The target indices.
        
        Returns:
            Float[Tensor, "..."]: The average cross-entropy loss across examples.
        """
        log_softmax = inputs - inputs.logsumexp(dim=1, keepdim=True)  # (batch_size, 1)
        target_log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch_size)
        loss = -target_log_probs.mean()
        return loss

        
        