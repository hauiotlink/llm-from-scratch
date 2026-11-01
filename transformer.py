"""Transformer architectures, Vision Transformer (ViT), and utilities.

This module provides complete implementations of the Transformer
architecture (Encoder, Decoder, Seq2Seq) and Vision Transformer
(ViT) from scratch using PyTorch, strictly adhering to AI research
coding standards and PEP 8 conventions.
"""

import os
import random
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Random seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> str:
    """Preprocesses input text by removing URLs, HTML tags,
    punctuation, and emojis.

    Args:
        text: Raw input text string.

    Returns:
        str: Cleaned, lowercased, and whitespace-normalized text.
    """
    # 1. Remove URLs
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    text = url_pattern.sub(" ", text)

    # 2. Remove HTML Tags
    html_pattern = re.compile(r"<[^>]+>")
    text = html_pattern.sub(" ", text)

    # 3. Remove punctuations and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # 4. Remove Emojis & special unicode symbols
    emoji_pattern = re.compile(
        "["
        "😀-🙏"  # Emoticons
        "🌀-🗿"  # Symbols & pictographs
        "🚀-🛿"  # Transport & map symbols
        "🇠-🇿"  # Flags (iOS)
        "🇲-🇴"  # Macau flag
        "🇦-🇿"  # Flags
        "✂-➰"
        "Ⓜ-🉑"
        "🤦-🤷"
        "🇲🇴😠‍"
        "♀-♂"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # 5. Normalize whitespace and lowercase
    text = " ".join(text.split())
    return text.lower()


class TokenAndPositionEmbedding(nn.Module):
    """Combines token embeddings with learnable positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_length: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes TokenAndPositionEmbedding.

        Args:
            vocab_size: Total vocabulary size.
            embed_dim: Dimension of token and position embeddings.
            max_length: Maximum sequence length supported.
            device: Execution device (e.g. cuda or cpu).
        """
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining token and position embeddings.

        Args:
            x: Input tensor of token indices of shape
                (batch_size, seq_len).

        Returns:
            torch.Tensor: Combined tensor of shape
                (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len = x.size()
        positions = (
            torch.arange(0, seq_len, device=self.device)
            .expand(batch_size, seq_len)
        )
        output1 = self.word_emb(x)
        output2 = self.pos_emb(positions)
        return output1 + output2


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with MultiheadAttention and FeedForward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initializes TransformerEncoderBlock.

        Args:
            embed_dim: Feature embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimension of the feed-forward layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True),
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs self-attention and feed-forward pass.

        Args:
            query: Query tensor of shape
                (batch_size, seq_len, embed_dim).
            key: Key tensor of shape
                (batch_size, seq_len, embed_dim).
            value: Value tensor of shape
                (batch_size, seq_len, embed_dim).
            attn_mask: Optional attention mask tensor.

        Returns:
            torch.Tensor: Encoder block output of shape
                (batch_size, seq_len, embed_dim).
        """
        attn_output, _ = self.attn(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2


class TransformerEncoder(nn.Module):
    """Transformer Encoder stacking embedding and multiple encoder blocks."""

    def __init__(
        self,
        src_vocab_size: int,
        embed_dim: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes TransformerEncoder.

        Args:
            src_vocab_size: Source vocabulary size.
            embed_dim: Feature embedding dimension.
            max_length: Maximum sequence length.
            num_layers: Number of encoder layers.
            num_heads: Number of attention heads per layer.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout probability.
            device: Execution device.
        """
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            device=device,
        )
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes token input sequences into contextual representations.

        Args:
            x: Source input token ids of shape (batch_size, seq_len).
            mask: Optional attention mask.

        Returns:
            torch.Tensor: Encoded representations of shape
                (batch_size, seq_len, embed_dim).
        """
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, output, output, attn_mask=mask)
        return output


class TransformerDecoderBlock(nn.Module):
    """Decoder block with Masked Self-Attention and Cross-Attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initializes TransformerDecoderBlock.

        Args:
            embed_dim: Feature embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True),
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for decoder block.

        Args:
            x: Target representations of shape
                (batch_size, tgt_seq_len, embed_dim).
            enc_output: Encoder outputs of shape
                (batch_size, src_seq_len, embed_dim).
            src_mask: Mask for cross-attention.
            tgt_mask: Causal mask for target self-attention.

        Returns:
            torch.Tensor: Decoder block output of shape
                (batch_size, tgt_seq_len, embed_dim).
        """
        # 1. Masked Target Self-Attention
        attn_output, _ = self.attn(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(x + attn_output)

        # 2. Cross-Attention
        attn_output, _ = self.cross_attn(
            out_1, enc_output, enc_output, attn_mask=src_mask
        )
        attn_output = self.dropout_2(attn_output)
        out_2 = self.layernorm_2(out_1 + attn_output)

        # 3. Feed-Forward
        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output)
        out_3 = self.layernorm_3(out_2 + ffn_output)
        return out_3


class TransformerDecoder(nn.Module):
    """Transformer Decoder stacking embedding and multiple decoder blocks."""

    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes TransformerDecoder.

        Args:
            tgt_vocab_size: Target vocabulary size.
            embed_dim: Feature embedding dimension.
            max_length: Maximum sequence length.
            num_layers: Number of decoder layers.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout probability.
            device: Execution device.
        """
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            device=device,
        )
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes target representations conditioned on encoder outputs.

        Args:
            x: Target token ids of shape (batch_size, tgt_seq_len).
            enc_output: Encoded source tensor of shape
                (batch_size, src_seq_len, embed_dim).
            src_mask: Optional mask for cross-attention.
            tgt_mask: Optional causal mask for target self-attention.

        Returns:
            torch.Tensor: Decoded tensor of shape
                (batch_size, tgt_seq_len, embed_dim).
        """
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, tgt_mask)
        return output


class Transformer(nn.Module):
    """Standard sequence-to-sequence Transformer architecture."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes the Transformer model."""
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            device=device,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            device=device,
        )
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_mask(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates source attention mask and target causal mask.

        Args:
            src: Source token tensor of shape (batch_size, src_seq_len).
            tgt: Target token tensor of shape (batch_size, tgt_seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (src_mask, tgt_mask).
        """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros(
            (src_seq_len, src_seq_len),
            device=self.device,
            dtype=torch.bool,
        )

        tgt_mask = (
            torch.triu(
                torch.ones(
                    (tgt_seq_len, tgt_seq_len),
                    device=self.device,
                )
            )
            == 1
        ).transpose(0, 1)

        tgt_mask = (
            tgt_mask.float()
            .masked_fill(tgt_mask == 0, float("-inf"))
            .masked_fill(tgt_mask == 1, float(0.0))
        )

        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Performs full forward pass for sequence transduction.

        Args:
            src: Source token tensor of shape (batch_size, src_seq_len).
            tgt: Target token tensor of shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Logits of shape
                (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output


class TransformerEncoderCls(nn.Module):
    """Transformer Encoder model adapted for text classification."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
        num_classes: int = 2,
    ) -> None:
        """Initializes TransformerEncoderCls.

        Args:
            vocab_size: Size of vocabulary.
            max_length: Maximum sequence length.
            num_layers: Number of encoder layers.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward network hidden dimension.
            dropout: Dropout probability.
            device: Target device.
            num_classes: Number of target classification classes.
        """
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            device=device,
        )
        self.pooling = nn.AvgPool1d(kernel_size=max_length)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes tokens, performs pooling, and predicts class logits.

        Args:
            x: Input token indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        output = self.encoder(x)
        # Permute (batch, seq_len, embed_dim) to (batch, embed_dim, seq_len)
        output = self.pooling(output.permute(0, 2, 1)).squeeze(-1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


class PatchPositionEmbedding(nn.Module):
    """Projects image patches to embeddings and adds position vectors."""

    def __init__(
        self,
        image_size: int = 224,
        embed_dim: int = 512,
        patch_size: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes PatchPositionEmbedding.

        Args:
            image_size: Height and width of input square images.
            embed_dim: Embedding dimension for each patch.
            patch_size: Square patch size.
            device: Target device.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = embed_dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_patches, embed_dim)
        )
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image patches and adds positional embeddings.

        Args:
            x: Image tensor of shape
                (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Patch embeddings of shape
                (batch_size, num_patches, embed_dim).
        """
        x = self.conv1(x)  # Shape: (batch_size, embed_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, embed_dim, patches)
        x = x.permute(0, 2, 1)  # (batch, patches, embed_dim)
        x = x + self.positional_embedding.to(self.device)
        return x


class VisionTransformerCls(nn.Module):
    """Vision Transformer (ViT) implemented from scratch."""

    def __init__(
        self,
        image_size: int = 224,
        embed_dim: int = 512,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
        num_classes: int = 10,
        patch_size: int = 16,
    ) -> None:
        """Initializes VisionTransformerCls.

        Args:
            image_size: Input image dimension.
            embed_dim: Feature embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            dropout: Dropout probability.
            device: Execution device.
            num_classes: Number of output categories.
            patch_size: Spatial dimensions of each patch.
        """
        super().__init__()
        self.embd_layer = PatchPositionEmbedding(
            image_size=image_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            device=device,
        )
        self.transformer_layer = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs patch extraction, self-attention, and class prediction.

        Args:
            x: Input image tensor of shape
                (batch_size, 3, image_size, image_size).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        output = self.embd_layer(x)
        output = self.transformer_layer(output, output, output)
        output = output[:, 0, :]  # First patch representation
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    log_interval: int = 50,
) -> Tuple[float, float]:
    """Runs a single training epoch over the given dataloader.

    Args:
        model: PyTorch model instance to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        train_dataloader: Training DataLoader.
        device: Target execution device.
        epoch: Current epoch index.
        log_interval: Batch logging interval frequency.

    Returns:
        Tuple[float, float]: (epoch_accuracy, epoch_loss).
    """
    model.train()
    total_acc = 0
    total_count = 0
    losses: List[float] = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        total_acc += int((predictions.argmax(1) == labels).sum().item())
        total_count += int(labels.size(0))

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            acc = total_acc / total_count if total_count > 0 else 0.0
            print(
                f"| epoch {epoch:3d} | {idx:5d}/{len(train_dataloader):5d} batches "
                f"| accuracy {acc:8.3f} | elapsed {elapsed:5.2f}s"
            )

    epoch_acc = total_acc / total_count if total_count > 0 else 0.0
    epoch_loss = sum(losses) / len(losses) if losses else 0.0
    return epoch_acc, epoch_loss


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    criterion: nn.Module,
    valid_dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model over the validation or test dataloader.

    Args:
        model: PyTorch model to evaluate.
        criterion: Loss function.
        valid_dataloader: Validation or test DataLoader.
        device: Target execution device.

    Returns:
        Tuple[float, float]: (eval_accuracy, eval_loss).
    """
    model.eval()
    total_acc = 0
    total_count = 0
    losses: List[float] = []

    for inputs, labels in valid_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        total_acc += int((predictions.argmax(1) == labels).sum().item())
        total_count += int(labels.size(0))

    eval_acc = total_acc / total_count if total_count > 0 else 0.0
    eval_loss = sum(losses) / len(losses) if losses else 0.0
    return eval_acc, eval_loss


def train_model(
    model: nn.Module,
    model_name: str,
    save_model_dir: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Runs training and evaluation loop, saving the best checkpoint.

    Args:
        model: PyTorch model to train.
        model_name: Base filename identifier for checkpoints.
        save_model_dir: Directory where checkpoints are saved.
        optimizer: Optimizer instance.
        criterion: Loss function.
        train_dataloader: Training DataLoader.
        valid_dataloader: Validation DataLoader.
        num_epochs: Number of training epochs.
        device: Execution device.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: (best_model, metrics_history).
    """
    os.makedirs(save_model_dir, exist_ok=True)
    train_accs: List[float] = []
    train_losses: List[float] = []
    eval_accs: List[float] = []
    eval_losses: List[float] = []
    times: List[float] = []
    best_loss_eval = float("inf")

    checkpoint_path = os.path.join(save_model_dir, f"{model_name}.pt")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        train_acc, train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            device=device,
            epoch=epoch,
        )
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        eval_acc, eval_loss = evaluate_epoch(
            model=model,
            criterion=criterion,
            valid_dataloader=valid_dataloader,
            device=device,
        )
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            torch.save(model.state_dict(), checkpoint_path)

        elapsed = time.time() - epoch_start_time
        times.append(elapsed)

        print("-" * 59)
        print(
            f"| End of epoch {epoch:3d} | Time: {elapsed:5.2f}s "
            f"| Train Accuracy {train_acc:8.3f} | Train Loss {train_loss:8.3f} "
            f"| Valid Accuracy {eval_acc:8.3f} | Valid Loss {eval_loss:8.3f}"
        )
        print("-" * 59)

    # Load best checkpoint before returning
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics = {
        "train_accuracy": train_accs,
        "train_loss": train_losses,
        "valid_accuracy": eval_accs,
        "valid_loss": eval_losses,
        "time": times,
    }
    return model, metrics


def plot_result(
    num_epochs: int,
    train_accs: List[float],
    eval_accs: List[float],
    train_losses: List[float],
    eval_losses: List[float],
) -> None:
    """Plots training and validation accuracy and loss curves.

    Args:
        num_epochs: Total number of epochs plotted.
        train_accs: List of training accuracies per epoch.
        eval_accs: List of validation accuracies per epoch.
        train_losses: List of training losses per epoch.
        eval_losses: List of validation losses per epoch.
    """
    epochs = list(range(1, num_epochs + 1))
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy Curve")
    axs[0].legend()

    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss Curve")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
