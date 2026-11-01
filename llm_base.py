"""End-to-end demonstration and profiling of Modern Decoder-Only LLM.

Executes all 3 mandatory capstone project core criteria:
1. Byte-level BPE Tokenizer from scratch with 0% OOV.
2. Modern Decoder-Only Architecture with DeepSeek Multi-Head Latent Attention.
3. Dynamic KV Cache profiling across L in {128, 512, 1024, 2048}.
"""

import os
import random
from typing import List

import torch

from configs.model_config import AttentionType, LLMConfig
from core.model import ModernDecoderOnlyLLM
from core.tokenizer import ByteLevelBPETokenizer, SpecialToken
from profiling.benchmark_cache import run_kv_cache_benchmark


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Executes the end-to-end LLM demonstration and profiling suite."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on target device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Byte-level BPE Tokenizer Demonstration
    print("\n" + "=" * 60)
    print("1. DEMONSTRATING BYTE-LEVEL BPE TOKENIZER FROM SCRATCH")
    print("=" * 60)

    sample_corpus = [
        "Xin chào thế giới! Đây là đồ án xây dựng mô hình ngôn ngữ lớn từ số 0.",
        "DeepSeek Multi-Head Latent Attention (MLA) nén KV Cache hiệu quả cao.",
        "Trí tuệ nhân tạo và mô hình ngôn ngữ lớn đang phát triển thần tốc.",
        "<|think|> Để tối ưu bộ nhớ VRAM, ta áp dụng nén Low-Rank. </|think|> Đáp án là 42 🚀.",
    ]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(sample_corpus, vocab_size=350, min_frequency=1)
    print(f"Trained vocabulary size: {tokenizer.vocab_size} tokens")

    test_prompt = "Xin chào DeepSeek MLA! 🚀 <|think|> Suy luận </|think|>"
    encoded_ids = tokenizer.encode(test_prompt)
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Original Text: {test_prompt}")
    print(f"Encoded IDs:   {encoded_ids}")
    print(f"Decoded Text:  {decoded_text}")
    assert test_prompt == decoded_text
    print("Tokenizer lossless roundtrip verified!")

    # 2. Modern Decoder-Only LLM Initialization
    print("\n" + "=" * 60)
    print("2. INITIALIZING MODERN DECODER-ONLY LLM (DEEPSEEK MLA)")
    print("=" * 60)

    config = LLMConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        intermediate_dim=512,
        max_seq_len=2048,
        attention_type=AttentionType.MLA,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        head_dim=64,
    )

    model = ModernDecoderOnlyLLM(config, device=device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} parameters")

    # 3. Deterministic Generation Check (Cache ON vs. Cache OFF)
    print("\n" + "=" * 60)
    print("3. VERIFYING DYNAMIC KV CACHE MATHEMATICAL EQUALITY")
    print("=" * 60)

    dummy_prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    cached_out = model.generate(
        dummy_prompt, max_new_tokens=10, temperature=0.0, use_cache=True
    )
    uncached_out = model.generate(
        dummy_prompt, max_new_tokens=10, temperature=0.0, use_cache=False
    )

    assert torch.equal(cached_out, uncached_out)
    print("Mathematical determinism verified: Cache ON == Cache OFF (100% Match)!")

    # 4. Mandatory Profiling Suite for L in {128, 512, 1024, 2048}
    print("\n" + "=" * 60)
    print("4. RUNNING MANDATORY PROFILING SUITE (L in {128, 512, 1024, 2048})")
    print("=" * 60)

    chart_path = "profiling/kv_cache_benchmark.png"
    run_kv_cache_benchmark(
        config=config,
        context_lengths=(128, 512, 1024, 2048),
        num_gen_tokens=16,
        output_image_path=chart_path,
        device=device,
    )
    print(f"\nAll tasks completed. Visual profiling report saved to: {chart_path}")


if __name__ == "__main__":
    main()

