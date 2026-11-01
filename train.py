"""Pre-training script for Modern Decoder-Only LLM with DeepSeek MLA.

Supports token packing, BF16 mixed-precision, cosine learning rate
scheduling with warmup, gradient accumulation, and checkpoint saving.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.model_config import AttentionType, LLMConfig
from core.model import ModernDecoderOnlyLLM
from core.tokenizer import ByteLevelBPETokenizer, SpecialToken
from data.dataset import PackedTokenDataset, create_pretraining_dataloader


def get_cosine_schedule_with_warmup_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Computes the scheduled learning rate with linear warmup and cosine decay.

    Args:
        step: Current training iteration step.
        warmup_steps: Total warmup steps.
        max_steps: Total training steps.
        peak_lr: Maximum learning rate after warmup.
        min_lr: Final minimum learning rate.

    Returns:
        float: Computed learning rate for current step.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: LLMConfig,
    checkpoint_dir: str,
) -> str:
    """Saves model weights, optimizer state, and training metadata.

    Args:
        model: Transformer model being trained.
        optimizer: PyTorch optimizer instance.
        step: Current global training step.
        loss: Most recent training loss value.
        config: Model configuration dataclass.
        checkpoint_dir: Target directory path for checkpoints.

    Returns:
        str: Absolute file path of the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"llm_step_{step:06d}.pt"
    )
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f" Saved checkpoint: {checkpoint_path} (Loss: {loss:.4f})")
    return checkpoint_path


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the most recent checkpoint file based on step index.

    Args:
        checkpoint_dir: Directory containing saved .pt checkpoints.

    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if empty.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("llm_step_") and f.endswith(".pt")
    ]
    if not ckpts:
        return None
    return max(ckpts)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> int:
    """Resumes training state from a saved checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: Model instance to load weights into.
        optimizer: Optional optimizer to restore states.
        device: Device to place loaded tensors.

    Returns:
        int: The step index where checkpoint was saved.
    """
    print(f" Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_step = checkpoint.get("step", 0)
    print(f" Successfully restored model & optimizer states at Step {resumed_step}")
    return resumed_step


def train_llm(
    model: ModernDecoderOnlyLLM,
    dataloader: DataLoader,
    tokenizer: ByteLevelBPETokenizer,
    max_steps: int = 200,
    grad_accum_steps: int = 4,
    peak_lr: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 20,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    save_interval: int = 100,
    eval_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
    device: Optional[torch.device | str] = None,
) -> List[float]:
    """Runs LLM pre-training loop with mixed precision and gradient accumulation.

    Args:
        model: ModernDecoderOnlyLLM instance.
        dataloader: PyTorch DataLoader yielding packed token batches.
        tokenizer: Configured ByteLevelBPETokenizer for generation testing.
        max_steps: Total training steps.
        grad_accum_steps: Steps to accumulate before optimizer update.
        peak_lr: Target peak learning rate.
        min_lr: Minimum learning rate floor.
        warmup_steps: Number of initial linear warmup steps.
        weight_decay: L2 regularization factor.
        grad_clip: Maximum gradient norm threshold.
        save_interval: Frequency of saving checkpoints.
        eval_interval: Frequency of running qualitative text generation.
        checkpoint_dir: Directory for storing model checkpoints.
        resume: Whether to automatically resume from latest saved checkpoint.
        device: Target execution device.

    Returns:
        List[float]: History of recorded training loss values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    # Use fused AdamW kernel on CUDA when available
    use_fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=use_fused,
    )

    # Auto-resume from latest checkpoint if available
    start_step = 0
    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            start_step = load_checkpoint(
                latest_ckpt, model=model, optimizer=optimizer, device=device
            )

    data_iter = iter(dataloader)
    loss_history: List[float] = []
    accum_loss = 0.0
    start_time = time.perf_counter()
    tokens_processed = 0

    print("\n" + "=" * 65)
    print(f"STARTING LLM PRE-TRAINING (STEPS {start_step + 1} -> {max_steps} ON {device})")
    print(f"Effective Batch Size: {dataloader.batch_size * grad_accum_steps}")
    print(f"Checkpoints saved to: {checkpoint_dir} every {save_interval} steps")
    print("=" * 65)

    for step in range(start_step + 1, max_steps + 1):
        # Update scheduled learning rate
        current_lr = get_cosine_schedule_with_warmup_lr(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            peak_lr=peak_lr,
            min_lr=min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = batch.to(device)
            tokens_processed += batch.numel()

            # Autocast BF16 on CUDA
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(input_ids=batch, labels=batch)
                    loss = loss / grad_accum_steps
            else:
                _, loss = model(input_ids=batch, labels=batch)
                loss = loss / grad_accum_steps

            loss.backward()
            step_loss += loss.item()

        # Clip gradients to prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        accum_loss += step_loss
        loss_history.append(step_loss)

        # Logging
        if step % 10 == 0 or step == 1:
            elapsed = time.perf_counter() - start_time
            throughput = tokens_processed / elapsed if elapsed > 0 else 0
            print(
                f"Step {step:04d}/{max_steps:04d} | "
                f"Loss: {step_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {throughput:,.0f} tok/s"
            )

        # Periodic Text Generation Inspection
        if step % eval_interval == 0:
            model.eval()
            prompt = "Once upon a time"
            prompt_ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                sample_out = model.generate(
                    prompt_ids,
                    max_new_tokens=25,
                    temperature=0.8,
                    top_k=20,
                    use_cache=True,
                )
            generated_str = tokenizer.decode(sample_out[0].tolist())
            print(f"\n--- [Step {step} Sample Generation] ---")
            print(f"'{generated_str}'\n" + "-" * 40)
            model.train()

        # Save checkpoint
        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                loss=step_loss,
                config=model.config,
                checkpoint_dir=checkpoint_dir,
            )

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print(f"PRE-TRAINING FINISHED in {total_time:.2f}s!")
    print(f"Final Step Loss: {loss_history[-1]:.4f}")
    print(f"Total Tokens Processed: {tokens_processed:,}")
    print("=" * 65)
    return loss_history


def main() -> None:
    """Prepares dataset and launches pre-training smoke test on RTX 5060 Ti."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Training Device: {device}")

    # 1. Load trained tokenizer
    tokenizer_path = "data/tokenizer_tinystories.json"
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    print(f"Loaded Tokenizer with {tokenizer.vocab_size} tokens")

    # 2. Configure model architecture
    config = LLMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        intermediate_dim=768,
        max_seq_len=256,
        attention_type=AttentionType.MLA,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        head_dim=32,
    )

    model = ModernDecoderOnlyLLM(config, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created LLM Model: {total_params:,} parameters")

    # 3. Load dataset stream and build packed dataloader
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
    dataloader = create_pretraining_dataloader(
        text_stream=ds,
        tokenizer=tokenizer,
        batch_size=16,
        seq_len=config.max_seq_len,
        max_samples=5000,
    )

    # 4. Launch pre-training (local smoke test: 150 steps)
    train_llm(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        max_steps=150,
        grad_accum_steps=2,
        peak_lr=6e-4,
        min_lr=6e-5,
        warmup_steps=15,
        save_interval=75,
        eval_interval=50,
        checkpoint_dir="checkpoints",
        device=device,
    )


if __name__ == "__main__":
    main()

