"""Core neural network layers for modern Transformer Decoder-Only LLMs.

This module implements RMSNorm, Rotary Position Embeddings (RoPE), and
SwiGLU feed-forward networks strictly adhering to PEP 8 standards.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes activations using root mean square without mean-centering,
    reducing memory operations and computational latency.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes RMSNorm.

        Args:
            dim: Dimension of the input hidden tensor.
            eps: Epsilon value to prevent division by zero.
            device: Target execution device.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies root mean square normalization formula.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Scaled normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RMSNorm.

        Args:
            x: Input activations of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor with matching input shape.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes relative positional information directly into query and key
    representations via 2D vector rotation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes RotaryEmbedding.

        Args:
            dim: Dimension of the head to apply RoPE (must be even).
            max_seq_len: Maximum supported sequence length.
            theta: Base frequency for rotary angles.
            device: Target execution device.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inverted frequency spectrum
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(
        self,
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Constructs cached cosine and sine tables.

        Args:
            seq_len: Target sequence length to cache.
            device: Execution device.
        """
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate freqs to match head dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions of the input tensor.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Half-rotated tensor.
        """
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches precomputed cosine and sine values for given positions.

        Args:
            x: Tensor to match device and dtype (..., seq_len, dim).
            position_ids: 1D or 2D tensor containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (cos, sin) tensors.
        """
        if position_ids.max().item() >= self.cos_cached.shape[0]:
            new_len = max(int(position_ids.max().item()) + 1, self.max_seq_len * 2)
            self._build_cache(new_len, x.device)

        cos = self.cos_cached[position_ids].type_as(x)
        sin = self.sin_cached[position_ids].type_as(x)
        return cos, sin

    def apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Applies 2D rotation to query or key tensor.

        Args:
            x: Query or Key tensor of shape (batch, heads, seq_len, dim).
            cos: Broadcastable cosine tensor.
            sin: Broadcastable sine tensor.

        Returns:
            torch.Tensor: Rotary-embedded tensor.
        """
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements gated linear unit activation:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes SwiGLUFFN.

        Args:
            hidden_dim: Hidden representation dimension (d_model).
            intermediate_dim: Expanded inner dimension (d_ffn).
            bias: Whether to include bias terms in linear layers.
            device: Target execution device.
        """
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=bias, device=device)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=bias, device=device)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes SwiGLU projection.

        Args:
            x: Input activations of shape (batch, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output activations of shape (batch, seq_len, hidden_dim).
        """
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

