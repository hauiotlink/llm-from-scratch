"""Benchmarking and profiling suite for LLM Key-Value (KV) Cache.

Measures and visualizes First Token Latency (TTFT), Next Token Latency
(TPOT), generation throughput (tokens/s), and peak VRAM allocation
with KV Cache On vs. Off across context lengths L in {128, 512, 1024, 2048}.
"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.model_config import LLMConfig
from core.model import ModernDecoderOnlyLLM


def profile_single_configuration(
    model: ModernDecoderOnlyLLM,
    prompt_len: int,
    num_gen_tokens: int = 32,
    use_cache: bool = True,
    device: torch.device | str = "cpu",
) -> Tuple[float, float, float, float]:
    """Profiles TTFT, TPOT, throughput, and VRAM for a single setup.

    Args:
        model: Transformer model to profile.
        prompt_len: Length of input prompt tokens (L).
        num_gen_tokens: Number of tokens to autoregressively decode.
        use_cache: Whether dynamic KV/MLA cache is enabled.
        device: Device where model is executed.

    Returns:
        Tuple[float, float, float, float]:
            - First Token Latency (TTFT) in milliseconds.
            - Next Token Latency (TPOT) in milliseconds.
            - Generation Throughput in tokens per second.
            - Peak VRAM memory in megabytes (MB).
    """
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Generate random prompt tokens
    prompt_ids = torch.randint(
        0, model.config.vocab_size, (1, prompt_len), device=device
    )

    # Warmup pass
    with torch.inference_mode():
        _ = model.generate(prompt_ids[:, :16], max_new_tokens=2, use_cache=use_cache)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)

    # 1. Measure First Token Latency (TTFT - Prefill)
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(prompt_ids, max_new_tokens=1, use_cache=use_cache)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Measure Total Generation Time for num_gen_tokens
    t_start = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(
            prompt_ids, max_new_tokens=num_gen_tokens, use_cache=use_cache
        )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time_s = time.perf_counter() - t_start

    # Next Token Latency: Average latency per generated token after prefill
    tpot_ms = (total_time_s * 1000.0) / num_gen_tokens
    throughput_tokens_per_sec = num_gen_tokens / total_time_s

    # 3. Measure Peak VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    return ttft_ms, tpot_ms, throughput_tokens_per_sec, peak_vram_mb


def run_kv_cache_benchmark(
    config: LLMConfig,
    context_lengths: Tuple[int, ...] = (128, 512, 1024, 2048),
    num_gen_tokens: int = 32,
    output_image_path: str = "profiling/kv_cache_benchmark.png",
    device: Optional[torch.device | str] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Runs complete profiling suite across context lengths L in {128..2048}.

    Args:
        config: Model configuration dataclass.
        context_lengths: Sequence lengths to profile.
        num_gen_tokens: Number of continuation tokens to decode.
        output_image_path: File path to save generated plots.
        device: Target execution device.

    Returns:
        Dict[str, Dict[str, List[float]]]: Dictionary of collected metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing model for profiling on {device}...")
    model = ModernDecoderOnlyLLM(config, device=device)
    model.eval()

    results: Dict[str, Dict[str, List[float]]] = {
        "cache_on": {"ttft": [], "tpot": [], "throughput": [], "vram": []},
        "cache_off": {"ttft": [], "tpot": [], "throughput": [], "vram": []},
    }

    print("\n" + "=" * 70)
    print("STARTING KV CACHE PROFILING EXPERIMENT")
    print("=" * 70)

    for prompt_len in context_lengths:
        print(f"\nEvaluating Context Length L = {prompt_len}...")

        # Cache ON
        ttft_on, tpot_on, tp_on, vram_on = profile_single_configuration(
            model,
            prompt_len=prompt_len,
            num_gen_tokens=num_gen_tokens,
            use_cache=True,
            device=device,
        )
        results["cache_on"]["ttft"].append(ttft_on)
        results["cache_on"]["tpot"].append(tpot_on)
        results["cache_on"]["throughput"].append(tp_on)
        results["cache_on"]["vram"].append(vram_on)
        print(
            f"  [Cache ON ] TTFT: {ttft_on:.2f}ms | TPOT: {tpot_on:.2f}ms | "
            f"Speed: {tp_on:.2f} tok/s | VRAM: {vram_on:.2f}MB"
        )

        # Cache OFF
        ttft_off, tpot_off, tp_off, vram_off = profile_single_configuration(
            model,
            prompt_len=prompt_len,
            num_gen_tokens=num_gen_tokens,
            use_cache=False,
            device=device,
        )
        results["cache_off"]["ttft"].append(ttft_off)
        results["cache_off"]["tpot"].append(tpot_off)
        results["cache_off"]["throughput"].append(tp_off)
        results["cache_off"]["vram"].append(vram_off)
        print(
            f"  [Cache OFF] TTFT: {ttft_off:.2f}ms | TPOT: {tpot_off:.2f}ms | "
            f"Speed: {tp_off:.2f} tok/s | VRAM: {vram_off:.2f}MB"
        )

    # Plot results
    _plot_profiling_results(
        context_lengths=context_lengths,
        results=results,
        output_image_path=output_image_path,
    )
    return results


def _plot_profiling_results(
    context_lengths: Tuple[int, ...],
    results: Dict[str, Dict[str, List[float]]],
    output_image_path: str,
) -> None:
    """Generates a 4-panel comparison figure of profiling metrics.

    Args:
        context_lengths: Evaluated context lengths.
        results: Dictionary containing measured metrics.
        output_image_path: File path to save output image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    x = list(context_lengths)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "KV Cache Profiling: Cache ON (O(L)) vs. Cache OFF (O(L²))",
        fontsize=16,
        fontweight="bold",
    )

    # 1. TTFT (First Token Latency)
    ax1 = axes[0, 0]
    ax1.plot(x, results["cache_on"]["ttft"], "o-", label="Cache ON", color="#2ca02c")
    ax1.plot(x, results["cache_off"]["ttft"], "s--", label="Cache OFF", color="#d62728")
    ax1.set_title("First Token Latency (TTFT)", fontweight="bold")
    ax1.set_xlabel("Context Length (L)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. TPOT (Next Token Latency)
    ax2 = axes[0, 1]
    ax2.plot(x, results["cache_on"]["tpot"], "o-", label="Cache ON", color="#2ca02c")
    ax2.plot(x, results["cache_off"]["tpot"], "s--", label="Cache OFF", color="#d62728")
    ax2.set_title("Next Token Latency (TPOT)", fontweight="bold")
    ax2.set_xlabel("Context Length (L)")
    ax2.set_ylabel("Latency per Token (ms)")
    ax2.set_xticks(x)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. Generation Throughput (Tokens/sec)
    ax3 = axes[1, 0]
    ax3.plot(x, results["cache_on"]["throughput"], "o-", label="Cache ON", color="#2ca02c")
    ax3.plot(x, results["cache_off"]["throughput"], "s--", label="Cache OFF", color="#d62728")
    ax3.set_title("Generation Speed (Tokens/s)", fontweight="bold")
    ax3.set_xlabel("Context Length (L)")
    ax3.set_ylabel("Throughput (Tokens/second)")
    ax3.set_xticks(x)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Peak VRAM Footprint
    ax4 = axes[1, 1]
    ax4.plot(x, results["cache_on"]["vram"], "o-", label="Cache ON", color="#2ca02c")
    ax4.plot(x, results["cache_off"]["vram"], "s--", label="Cache OFF", color="#d62728")
    ax4.set_title("Peak VRAM Footprint", fontweight="bold")
    ax4.set_xlabel("Context Length (L)")
    ax4.set_ylabel("Memory (MB)")
    ax4.set_xticks(x)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"\nProfiling chart saved successfully to: {output_image_path}")
