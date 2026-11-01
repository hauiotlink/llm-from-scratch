"""Full architecture implementation of Modern Decoder-Only LLM.

Integrates Pre-RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU,
Grouped-Query Attention (GQA), DeepSeek Multi-Head Latent Attention (MLA),
and Dynamic KV Caches for fast autoregressive generation.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import AttentionType, LLMConfig
from core.attention import CausalSelfAttention, MultiHeadLatentAttention
from core.kv_cache import DynamicKVCache, DynamicMLACache
from core.layers import RMSNorm, SwiGLUFFN


class TransformerDecoderBlock(nn.Module):
    """Single modern Transformer Decoder block with Pre-RMSNorm."""

    def __init__(
        self,
        config: LLMConfig,
        layer_idx: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes TransformerDecoderBlock.

        Args:
            config: Architectural configuration object.
            layer_idx: Index of this layer in the stack.
            device: Target execution device.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(
            config.hidden_dim, eps=config.rms_norm_eps, device=device
        )

        if config.attention_type == AttentionType.MLA:
            self.attn = MultiHeadLatentAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                bias=config.bias,
                device=device,
            )
        else:
            self.attn = CausalSelfAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                bias=config.bias,
                device=device,
            )

        self.ffn_norm = RMSNorm(
            config.hidden_dim, eps=config.rms_norm_eps, device=device
        )
        self.ffn = SwiGLUFFN(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            bias=config.bias,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Optional[DynamicKVCache | DynamicMLACache] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder block with residual highway.

        Args:
            x: Input hidden states (batch, seq_len, hidden_dim).
            position_ids: Position indices tensor.
            cache: Optional dynamic KV or MLA cache.

        Returns:
            torch.Tensor: Output hidden states (batch, seq_len, hidden_dim).
        """
        # Pre-RMSNorm + Attention + Residual
        normed_attn_in = self.attn_norm(x)
        if isinstance(self.attn, MultiHeadLatentAttention):
            mla_cache = cache if isinstance(cache, DynamicMLACache) else None
            attn_out = self.attn(
                normed_attn_in,
                position_ids,
                mla_cache=mla_cache,
                layer_idx=self.layer_idx,
            )
        else:
            kv_cache = cache if isinstance(cache, DynamicKVCache) else None
            attn_out = self.attn(
                normed_attn_in,
                position_ids,
                kv_cache=kv_cache,
                layer_idx=self.layer_idx,
            )
        x = x + attn_out

        # Pre-RMSNorm + SwiGLU + Residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ModernDecoderOnlyLLM(nn.Module):
    """Complete Modern Transformer Decoder-Only Large Language Model."""

    def __init__(
        self,
        config: LLMConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes ModernDecoderOnlyLLM.

        Args:
            config: Model configuration dataclass.
            device: Target execution device.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_dim, device=device
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(config, layer_idx=i, device=device)
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_dim, eps=config.rms_norm_eps, device=device
        )
        self.lm_head = nn.Linear(
            config.hidden_dim, config.vocab_size, bias=False, device=device
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        """Enables activation gradient checkpointing to save VRAM."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disables activation gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[DynamicKVCache | DynamicMLACache] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing token logits and cross-entropy loss.

        Args:
            input_ids: Input token ID tensor of shape (batch, seq_len).
            position_ids: Optional position tensor (batch, seq_len).
            cache: Optional dynamic KV or MLA cache.
            labels: Optional ground-truth target tokens for training.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (logits, loss).
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            past_len = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(
                past_len,
                past_len + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # 1. Embedding lookup
        hidden_states = self.embed_tokens(input_ids)

        # 2. Iterate through all decoder blocks
        for layer in self.layers:
            if self.training and self.gradient_checkpointing and cache is None:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states, position_ids=position_ids, cache=cache
                )

        # 3. Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # 4. Project to vocabulary logits
        logits = self.lm_head(hidden_states)

        # 5. Compute cross-entropy loss if labels are provided
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generates continuation tokens.

        Args:
            input_ids: Prefix prompt tensor of shape (batch, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature scaling factor.
            top_k: Top-k filtering threshold.
            top_p: Top-p (nucleus) cumulative probability threshold.
            eos_token_id: Token ID marking end of sequence.
            use_cache: Whether to use dynamic KV/MLA cache.

        Returns:
            torch.Tensor: Generated token sequence including prompt.
        """
        generated = input_ids.clone()
        cache = None

        if use_cache:
            if self.config.attention_type == AttentionType.MLA:
                cache = DynamicMLACache()
            else:
                cache = DynamicKVCache()

            # Prefill phase: feed the full prompt once to initialize cache
            logits, _ = self(input_ids, cache=cache)
            next_token_logits = logits[:, -1, :]
        else:
            next_token_logits = None

        for _ in range(max_new_tokens):
            if not use_cache or next_token_logits is None:
                # Without cache: feed entire history (O(L^2) cost)
                logits, _ = self(generated, cache=None)
                next_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            if temperature > 0.0:
                scaled_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        scaled_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if use_cache and cache is not None:
                # Decode phase: feed ONLY the single newly generated token
                logits, _ = self(next_token, cache=cache)
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = None

        return generated

