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
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if betas[0] < 0:
            raise ValueError(f"Invalid beta index 0: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid beta index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params,defaults)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # Get parameter-specific state
                state = self.state[p]

                    # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                step_size = group['lr']
                eps = group['eps']

                # Update state
                state['step'] += 1
                grad = p.grad.data

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
                
class TransformerBlock(torch.nn.Module):
    """
    A single Transformer block/layer.
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 theta: float = 10000.0,
                 seq_len: int = 512,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            seq_len=seq_len,
            device=device,
            dtype=dtype
        )
        self.ffn = SWiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        # Pre-norm architecture
        # Apply attention with residual connection
        x_attn = self.attn(self.ln1(x))
        attn_output = x + x_attn
        
        # Apply FFN with residual connection
        x_ffn = self.ffn(self.ln2(attn_output))
        ffn_output = attn_output + x_ffn
        
        return ffn_output


class BasicsTransformerLM(torch.nn.Module):
    """
    A basic Transformer language model implementation.
    """
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float = 10000.0,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        
        # Store configuration
        self.config = {
            k: v for k, v in locals().items() 
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Token embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # Transformer layers
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model)
        
        # Language modeling head
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embedding parameters are excluded.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
        return n_params

    def forward(self, 
                x: Int[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length vocab_size"]:
        """
        Forward pass of the transformer language model.
        
        Args:
            x: Input token IDs of shape (batch_size, sequence_length)
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        # Get token embeddings
        x = self.token_embeddings(x)  # (batch_size, seq_len, d_model)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_len, d_model)
        
        # Final layer norm
        x = self.ln_final(x)  # (batch_size, seq_len, d_model)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits

    @torch.no_grad()
    def generate(self,
                 x: Int[Tensor, "... sequence_length"],
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 eos_token_id: int | None = None) -> Int[Tensor, "... max_new_tokens"]:
        """
        Generate tokens autoregressively.
        
        Args:
            x: Input token IDs to condition on
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: If provided, only sample from top-k tokens
            eos_token_id: If provided, stop generation when this token is generated
            
        Returns:
            Generated token IDs
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_length = x.size(-1)
        
        for _ in range(max_new_tokens):
            # Truncate to context length if needed
            x_cond = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            
            # Get logits
            logits = self.forward(x_cond)  # (batch_size, seq_len, vocab_size)
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax
            softmax = Softmax(dim=-1)
            probs = softmax(next_token_logits)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
                
            # Append to sequence
            x = torch.cat([x, next_token], dim=1)
        
        # Return only the newly generated tokens
        return x[:, original_length:]

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """
        Load a pretrained model from a directory.
        
        Args:
            pretrained_model_path: Path to the directory containing model files
            
        Returns:
            Loaded BasicsTransformerLM model
        """
        import json
        import os
        
        # Load config
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Remove _orig_mod. prefix if present (from compiled models)
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        return model