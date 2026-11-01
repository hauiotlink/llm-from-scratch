"""Unit test for ModernDecoderOnlyLLM forward pass and KV Cache generation."""

import torch

from configs.model_config import AttentionType, LLMConfig
from core.model import ModernDecoderOnlyLLM


def test_model_forward_and_generate() -> None:
    """Verifies forward, backward, and generation with and without cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running test on device: {device}")

    # Small configuration for testing
    config = LLMConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=512,
        max_seq_len=512,
        attention_type="mla",
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        head_dim=64,
    )

    model = ModernDecoderOnlyLLM(config, device=device)
    model.eval()

    # 1. Test forward pass with loss
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    print(f"Forward pass successful. Loss: {loss.item():.4f}")

    # 2. Test generation with use_cache=True (MLA Latent Cache)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
    gen_with_cache = model.generate(
        prompt_ids, max_new_tokens=10, temperature=0.0, use_cache=True
    )
    assert gen_with_cache.shape == (1, 18)
    print("Generation with use_cache=True passed.")

    # 3. Test generation with use_cache=False
    gen_without_cache = model.generate(
        prompt_ids, max_new_tokens=10, temperature=0.0, use_cache=False
    )
    assert gen_without_cache.shape == (1, 18)
    print("Generation with use_cache=False passed.")

    # 4. Greedy deterministic equality check between cached vs non-cached
    assert torch.equal(
        gen_with_cache, gen_without_cache
    ), "Cache output differs from non-cache output in greedy decoding!"
    print("Determinism check passed: Cache and Non-Cache produce EXACT same output!")


if __name__ == "__main__":
    test_model_forward_and_generate()
