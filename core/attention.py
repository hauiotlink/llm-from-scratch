"""Attention mechanisms for modern Decoder-Only Large Language Models.

This module implements Grouped-Query Attention (GQA) and DeepSeek
Multi-Head Latent Attention (MLA) with decoupled RoPE, supporting
dynamic KV caches and FlashAttention / SDPA.
"""

from enum import StrEnum
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.kv_cache import DynamicKVCache, DynamicMLACache
from core.layers import RMSNorm, RotaryEmbedding


class AttentionType(StrEnum):
    """Categorical enumeration of supported attention architectures."""

    GQA = "gqa"
    MLA = "mla"


class CausalSelfAttention(nn.Module):
    """Modern Causal Multi-Head / Grouped-Query Attention (GQA).

    Supports RoPE positional embeddings, dynamic KV-caching, and PyTorch
    scaled dot-product attention (FlashAttention-2 dispatch).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes CausalSelfAttention.

        Args:
            hidden_dim: Model hidden representation size (d_model).
            num_heads: Number of Query attention heads.
            num_kv_heads: Number of Key/Value heads (for GQA).
            max_seq_len: Maximum supported context length.
            rope_theta: RoPE base angle frequency parameter.
            bias: Whether linear projection layers include bias.
            device: Target execution device.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(
            hidden_dim, self.num_heads * self.head_dim, bias=bias, device=device
        )
        self.k_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=bias, device=device
        )
        self.v_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=bias, device=device
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[DynamicKVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Executes causal multi-head self-attention.

        Args:
            x: Input activations of shape (batch, seq_len, hidden_dim).
            position_ids: 1D or 2D tensor of token position indices.
            kv_cache: Optional dynamic key-value cache buffer.
            layer_idx: Transformer layer index.

        Returns:
            torch.Tensor: Attention output of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into multi-head format: (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Position Embedding (RoPE)
        cos, sin = self.rotary_emb(v, position_ids)
        cos = cos.unsqueeze(1)  # Broadcast across heads
        sin = sin.unsqueeze(1)
        q = self.rotary_emb.apply_rope(q, cos, sin)
        k = self.rotary_emb.apply_rope(k, cos, sin)

        # 4. Update dynamic KV Cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx=layer_idx)

        # 5. Repeat Key/Value heads if using GQA
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 6. Scaled Dot-Product Attention (SDPA with causal masking)
        # When kv_cache is active and decoding 1 token, is_causal is False
        total_k_len = k.shape[-2]
        is_causal = (seq_len > 1) and (seq_len == total_k_len)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )

        # 7. Reshape and project to output dimension
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        return self.out_proj(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek Multi-Head Latent Attention (MLA) with Decoupled RoPE.

    Compresses Key and Value states into a low-dimensional latent vector
    c_t^(KV) to compress KV Cache VRAM footprint up to 7-8x while preserving
    full multi-head expressive capability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int = 128,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes MultiHeadLatentAttention.

        Args:
            hidden_dim: Hidden dimension size (d_model).
            num_heads: Number of attention heads.
            head_dim: Content head dimension.
            kv_lora_rank: Compressed latent dimension for KV states (d_c).
            qk_rope_head_dim: Decoupled RoPE dimension for positional keys (d_r).
            max_seq_len: Maximum supported sequence length.
            rope_theta: RoPE base angle frequency parameter.
            bias: Whether projection layers include bias.
            device: Target execution device.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # Content and RoPE query projections
        self.q_content_proj = nn.Linear(
            hidden_dim, num_heads * head_dim, bias=bias, device=device
        )
        self.q_rope_proj = nn.Linear(
            hidden_dim, num_heads * qk_rope_head_dim, bias=bias, device=device
        )

        # Low-rank KV compression projections
        self.kv_down_proj = nn.Linear(
            hidden_dim, kv_lora_rank, bias=bias, device=device
        )
        self.kv_norm = RMSNorm(kv_lora_rank, device=device)
        self.k_up_proj = nn.Linear(
            kv_lora_rank, num_heads * head_dim, bias=bias, device=device
        )
        self.v_up_proj = nn.Linear(
            kv_lora_rank, num_heads * head_dim, bias=bias, device=device
        )

        # Decoupled RoPE key projection
        self.k_rope_proj = nn.Linear(
            hidden_dim, qk_rope_head_dim, bias=bias, device=device
        )

        # Output projection
        self.out_proj = nn.Linear(
            num_heads * head_dim, hidden_dim, bias=bias, device=device
        )

        # Rotary embedding for decoupled position features
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mla_cache: Optional[DynamicMLACache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Executes Multi-Head Latent Attention.

        Args:
            x: Input activations (batch, seq_len, hidden_dim).
            position_ids: Position indices tensor.
            mla_cache: Optional dynamic latent KV cache.
            layer_idx: Transformer layer index.

        Returns:
            torch.Tensor: Output activations (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Query projections: Content and RoPE
        q_content = self.q_content_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q_rope = self.q_rope_proj(x).view(
            batch_size, seq_len, self.num_heads, self.qk_rope_head_dim
        ).transpose(1, 2)

        # 2. Key-Value down-projection (Compression)
        c_kv = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, kv_lora_rank)
        k_rope = self.k_rope_proj(x).view(
            batch_size, seq_len, 1, self.qk_rope_head_dim
        ).transpose(1, 2)

        # 3. Apply RoPE on decoupled positional query and key
        cos, sin = self.rotary_emb(x, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rope = self.rotary_emb.apply_rope(q_rope, cos, sin)
        k_rope = self.rotary_emb.apply_rope(k_rope, cos, sin)

        # 4. Latent Cache update: Only store compressed c_kv and k_rope!
        if mla_cache is not None:
            c_kv_unsq = c_kv
            k_rope_sq = k_rope.squeeze(1)
            c_kv_cached, k_rope_cached = mla_cache.update(
                c_kv_unsq, k_rope_sq, layer_idx=layer_idx
            )
            c_kv = c_kv_cached
            k_rope = k_rope_cached.unsqueeze(1)

        total_k_len = c_kv.shape[1]

        # 5. Up-project compressed latent vector to multi-head Keys and Values
        k_content = self.k_up_proj(c_kv).view(
            batch_size, total_k_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(
            batch_size, total_k_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Repeat decoupled RoPE key across all attention heads
        k_rope_expanded = k_rope.repeat(1, self.num_heads, 1, 1)

        # 6. Concatenate content and positional features
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope_expanded], dim=-1)

        # 7. Scaled Dot-Product Attention
        is_causal = (seq_len > 1) and (seq_len == total_k_len)
        scale = 1.0 / math.sqrt(self.head_dim + self.qk_rope_head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=is_causal
        )

        # 8. Project back to hidden representation
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        return self.out_proj(attn_output)

