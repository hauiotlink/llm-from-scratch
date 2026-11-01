"""Data loading, streaming, and token packing pipeline for LLM Pre-training.

Implements token packing to concatenate variable-length documents into
dense fixed-length sequences without padding waste.
"""

from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from core.tokenizer import ByteLevelBPETokenizer, SpecialToken


class PackedTokenDataset(IterableDataset):
    """Iterable Dataset with Token Packing for dense sequence training.

    Concatenates tokenized texts separated by EOS tokens and partitions
    them into non-overlapping chunks of fixed sequence length (seq_len),
    eliminating padding overhead and maximizing GPU compute utilization.
    """

    def __init__(
        self,
        text_stream: Iterable[Dict[str, Any] | str],
        tokenizer: ByteLevelBPETokenizer,
        seq_len: int = 512,
        text_key: str = "text",
        eos_token_id: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initializes PackedTokenDataset.

        Args:
            text_stream: Stream or list of text documents or dicts.
            tokenizer: Pre-trained ByteLevelBPETokenizer instance.
            seq_len: Target sequence length for each packed training sample.
            text_key: Key name in dict samples containing the raw text.
            eos_token_id: Token ID used to delimit separate documents.
            max_samples: Optional limit on total input documents to process.
        """
        super().__init__()
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key

        if eos_token_id is None:
            self.eos_token_id = tokenizer.vocab.get(
                SpecialToken.EOS.value, 0
            )
        else:
            self.eos_token_id = eos_token_id

        self.max_samples = max_samples

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yields dense token tensors of exact shape (seq_len,).

        Yields:
            torch.Tensor: Long tensor containing packed token sequence.
        """
        buffer: List[int] = []
        sample_count = 0

        for item in self.text_stream:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            if isinstance(item, dict):
                text = item.get(self.text_key, "")
            else:
                text = str(item)

            if not text.strip():
                continue

            # Tokenize document and append EOS boundary delimiter
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_token_id)
            buffer.extend(tokens)
            sample_count += 1

            # Drain buffer in slices of exact length seq_len
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield torch.tensor(chunk, dtype=torch.long)


def create_pretraining_dataloader(
    text_stream: Iterable[Dict[str, Any] | str],
    tokenizer: ByteLevelBPETokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    text_key: str = "text",
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Creates a PyTorch DataLoader wrapping the packed token stream.

    Args:
        text_stream: Iterable stream of documents.
        tokenizer: Configured ByteLevelBPETokenizer.
        batch_size: Micro-batch size per iteration.
        seq_len: Sequence length per sample.
        text_key: Field key containing raw text string.
        max_samples: Optional limit on total documents read.

    Returns:
        DataLoader: PyTorch DataLoader yielding batches of shape (batch, seq_len).
    """
    dataset = PackedTokenDataset(
        text_stream=text_stream,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_key=text_key,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

