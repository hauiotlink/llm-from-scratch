"""Configuration dataclasses for Modern Decoder-Only LLMs."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class AttentionType(StrEnum):
    """Categorical enumeration of attention architectures."""

    GQA = "gqa"
    MLA = "mla"


@dataclass
class LLMConfig:
    """Architectural hyperparameters for Modern Decoder-Only LLM.

    Attributes:
        vocab_size: Total vocabulary size.
        hidden_dim: Model hidden dimension (d_model).
        num_layers: Total number of transformer decoder layers.
        num_heads: Number of attention query heads.
        num_kv_heads: Number of Key/Value heads (for GQA).
        intermediate_dim: Dimension of inner SwiGLU feed-forward layer.
        max_seq_len: Maximum supported sequence context length.
        rms_norm_eps: Epsilon parameter for numerical stability in RMSNorm.
        rope_theta: Base frequency for Rotary Position Embeddings.
        attention_type: Architecture type ("gqa" or "mla").
        kv_lora_rank: Compressed latent dimension for MLA (d_c).
        qk_rope_head_dim: Decoupled RoPE dimension for MLA (d_r).
        head_dim: Content head dimension for MLA.
        tie_word_embeddings: Whether to share embedding and LM head weights.
        bias: Whether to use bias in linear projection layers.
    """

    vocab_size: int = 32000
    hidden_dim: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_dim: int = 5632
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_type: str = "mla"
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    head_dim: int = 128
    tie_word_embeddings: bool = False
    bias: bool = False

