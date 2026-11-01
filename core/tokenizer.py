"""Byte-level Byte-Pair Encoding (BBPE) Tokenizer from scratch.

This module provides a production-grade Byte-level BPE tokenizer
implementation adhering strictly to PEP 8, full type annotations,
and Google docstring style conventions.
"""

from enum import StrEnum
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import regex


class SpecialToken(StrEnum):
    """Enumeration of standard LLM special tokens."""

    PAD = "<|pad|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    UNK = "<|unk|>"
    THINK = "<|think|>"
    END_THINK = "</|think|>"


def bytes_to_unicode() -> Dict[int, str]:
    """Returns a reversible mapping between UTF-8 bytes and unicode strings.

    Avoids mapping to whitespace and control characters so tokens are
    fully printable and inspectable in logs.

    Returns:
        Dict[int, str]: Mapping from byte integer (0-255) to character.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteLevelBPETokenizer:
    """Byte-level Byte-Pair Encoding (BBPE) Tokenizer.

    Trained directly on UTF-8 byte representations to guarantee zero
    Out-Of-Vocabulary (OOV) tokens across any language or emojis.
    """

    # GPT-2 / LLaMA style regex splitting pattern
    SPLIT_PATTERN = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Initializes ByteLevelBPETokenizer.

        Args:
            vocab: Mapping from token string to unique integer ID.
            merges: Ordered list of byte-pair merge operations.
            special_tokens: List of special control tokens.
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        if special_tokens is None:
            self.special_tokens = [t.value for t in SpecialToken]
        else:
            self.special_tokens = special_tokens

        self.vocab: Dict[str, int] = vocab if vocab is not None else {}
        self.merges: List[Tuple[str, str]] = merges if merges is not None else []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in self.vocab.items()
        }

    @property
    def vocab_size(self) -> int:
        """Returns the total number of tokens in the vocabulary."""
        return len(self.vocab)

    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Extracts all adjacent character pairs from a tuple-encoded word.

        Args:
            word: Tuple of current sub-token strings.

        Returns:
            Set[Tuple[str, str]]: Set of adjacent sub-token pairs.
        """
        pairs: Set[Tuple[str, str]] = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def train(
        self,
        texts: List[str],
        vocab_size: int = 16384,
        min_frequency: int = 2,
    ) -> None:
        """Trains the Byte-level BPE tokenizer on a list of texts.

        Args:
            texts: List of raw training documents or sentences.
            vocab_size: Target vocabulary size.
            min_frequency: Minimum pair frequency threshold to merge.
        """
        # 1. Initialize base vocabulary with special tokens and 256 bytes
        self.vocab = {}
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)

        for b in range(256):
            char_token = self.byte_encoder[b]
            if char_token not in self.vocab:
                self.vocab[char_token] = len(self.vocab)

        # 2. Count word frequencies after regex pre-tokenization
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            words = regex.findall(self.SPLIT_PATTERN, text)
            for word in words:
                byte_tokens = tuple(
                    self.byte_encoder[b] for b in word.encode("utf-8")
                )
                word_freqs[byte_tokens] = word_freqs.get(byte_tokens, 0) + 1

        # 3. Iterative BPE pair merging
        num_merges = vocab_size - len(self.vocab)
        self.merges = []

        for _ in range(num_merges):
            pair_counts: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_frequency:
                break

            # Create new merged token
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            # Update word frequencies with merged pair
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word: List[str] = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def _bpe(self, token: str) -> List[str]:
        """Applies learned BPE merge rules greedily to a single word token.

        Args:
            token: Unicode string of byte-encoded characters for one word.

        Returns:
            List[str]: List of merged sub-token strings.
        """
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return list(word)

        while True:
            # Find the pair with the lowest merge rank
            min_pair: Optional[Tuple[str, str]] = None
            min_rank = float("inf")
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_pair not in self.bpe_ranks:
                break

            first, second = min_pair
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        return list(word)

    def encode(
        self,
        text: str,
        allowed_special: Optional[Set[str]] = None,
    ) -> List[int]:
        """Encodes an input text string into a list of token IDs.

        Args:
            text: Raw input string to encode.
            allowed_special: Set of special tokens allowed in the text.

        Returns:
            List[int]: List of corresponding token IDs.
        """
        if allowed_special is None:
            allowed_special = set(self.special_tokens)

        # Regex split with special tokens handling
        special_pattern = (
            "(" + "|".join(regex.escape(s) for s in allowed_special) + ")"
            if allowed_special
            else None
        )

        tokens: List[int] = []
        if special_pattern:
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in allowed_special:
                tokens.append(self.vocab[part])
                continue

            words = regex.findall(self.SPLIT_PATTERN, part)
            for word in words:
                byte_encoded = "".join(
                    self.byte_encoder[b] for b in word.encode("utf-8")
                )
                bpe_tokens = self._bpe(byte_encoded)
                for bpe_token in bpe_tokens:
                    token_id = self.vocab.get(
                        bpe_token, self.vocab.get(SpecialToken.UNK.value, 0)
                    )
                    tokens.append(token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a UTF-8 text string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            str: Decoded natural language text.
        """
        byte_list: List[int] = []
        for idx in token_ids:
            token = self.inverse_vocab.get(idx, "")
            if token in self.special_tokens:
                # Append special token string encoded in UTF-8
                byte_list.extend(token.encode("utf-8"))
            else:
                for char in token:
                    if char in self.byte_decoder:
                        byte_list.append(self.byte_decoder[char])

        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        """Saves the tokenizer vocabulary and merge table to a JSON file.

        Args:
            filepath: Destination path of the JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ByteLevelBPETokenizer":
        """Loads a tokenizer from a saved JSON configuration file.

        Args:
            filepath: Path to the saved tokenizer JSON file.

        Returns:
            ByteLevelBPETokenizer: Instantiated and configured tokenizer.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = [tuple(pair) for pair in data["merges"]]
        return cls(
            vocab=data["vocab"],
            merges=merges,
            special_tokens=data.get("special_tokens"),
        )

