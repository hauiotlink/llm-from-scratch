"""Dynamic Key-Value Cache data structures for autoregressive generation.

This module provides dynamic tensor KV-cache implementations supporting
both standard MHA/GQA and DeepSeek Multi-Head Latent Attention (MLA).
"""

from typing import Dict, List, Optional, Tuple

import torch


class DynamicKVCache:
    """Dynamic tensor storage for Key-Value states across decoder layers.

    Enables efficient autoregressive decoding by accumulating past K, V
    tensors dynamically without full recomputation of historical tokens.
    """

    def __init__(self) -> None:
        """Initializes empty dynamic key and value cache dictionaries."""
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new key and value states to the specified layer cache.

        Args:
            key_states: New key tensor (batch, heads, new_len, head_dim).
            value_states: New value tensor (batch, heads, new_len, head_dim).
            layer_idx: Zero-based index of the transformer layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Full accumulated key and value.
        """
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the current sequence length stored in the cache.

        Args:
            layer_idx: Layer index to inspect.

        Returns:
            int: Number of cached tokens.
        """
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reset(self) -> None:
        """Clears all cached tensors to release GPU memory."""
        self.key_cache.clear()
        self.value_cache.clear()

    def get_memory_bytes(self) -> int:
        """Computes the total memory consumed by cached tensors in bytes.

        Returns:
            int: Memory consumption in bytes.
        """
        total_bytes = 0
        for k, v in zip(self.key_cache.values(), self.value_cache.values()):
            total_bytes += k.element_size() * k.nelement()
            total_bytes += v.element_size() * v.nelement()
        return total_bytes


class DynamicMLACache:
    """Dynamic latent cache for DeepSeek Multi-Head Latent Attention (MLA).

    Instead of storing full uncompressed key and value vectors, stores only
    the compressed latent vector c_t^(KV) and decoupled RoPE key k_t^R.
    """

    def __init__(self) -> None:
        """Initializes empty dynamic latent and rope cache dictionaries."""
        self.latent_cache: Dict[int, torch.Tensor] = {}
        self.rope_cache: Dict[int, torch.Tensor] = {}

    def update(
        self,
        latent_states: torch.Tensor,
        rope_key_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends compressed latent states and rope keys to the cache.

        Args:
            latent_states: Compressed KV tensor (batch, seq_len, kv_lora_rank).
            rope_key_states: RoPE key tensor (batch, seq_len, qk_rope_head_dim).
            layer_idx: Index of the transformer layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Full accumulated latent and rope keys.
        """
        if layer_idx not in self.latent_cache:
            self.latent_cache[layer_idx] = latent_states
            self.rope_cache[layer_idx] = rope_key_states
        else:
            self.latent_cache[layer_idx] = torch.cat(
                [self.latent_cache[layer_idx], latent_states], dim=1
            )
            self.rope_cache[layer_idx] = torch.cat(
                [self.rope_cache[layer_idx], rope_key_states], dim=1
            )

        return self.latent_cache[layer_idx], self.rope_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the current sequence length in the latent cache.

        Args:
            layer_idx: Layer index to inspect.

        Returns:
            int: Number of cached tokens.
        """
        if layer_idx not in self.latent_cache:
            return 0
        return self.latent_cache[layer_idx].shape[1]

    def reset(self) -> None:
        """Clears all cached latent tensors to release GPU memory."""
        self.latent_cache.clear()
        self.rope_cache.clear()

    def get_memory_bytes(self) -> int:
        """Computes total memory in bytes consumed by latent cached tensors.

        Returns:
            int: Memory consumption in bytes.
        """
        total_bytes = 0
        for c, r in zip(self.latent_cache.values(), self.rope_cache.values()):
            total_bytes += c.element_size() * c.nelement()
            total_bytes += r.element_size() * r.nelement()
        return total_bytes

