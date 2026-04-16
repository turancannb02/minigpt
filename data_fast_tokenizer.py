"""
Fast BPE tokenizer using the tokenizers library (Rust-backed).
This is ~100x faster than pure Python implementations.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class FastBPETokenizer:
    """Fast Byte-level BPE tokenizer using Rust-backed tokenizers library."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._build_tokenizer()

    def _build_tokenizer(self):
        """Build a byte-level BPE tokenizer."""
        self.tokenizer = Tokenizer(models.BPE(unk_token=None))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, texts: List[str], verbose: bool = True):
        """Train BPE on a list of texts."""
        from tqdm import tqdm

        # Special tokens for byte-level BPE
        special_tokens = []

        # Trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
            show_progress=verbose,
        )

        # Train
        self.tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
            length=len(texts)
        )

    @property
    def vocab(self):
        """Get vocabulary dict (bytes -> id)."""
        return {bytes([k]) if isinstance(k, int) else k.encode('utf-8') if isinstance(k, str) else k: v
                for k, v in self.tokenizer.get_vocab().items()}

    @property
    def inverse_vocab(self):
        """Get inverse vocabulary (id -> bytes)."""
        return {v: k for k, v in self.vocab.items()}

    @property
    def merges(self):
        """Get merge rules (not used in fast tokenizer, kept for compatibility)."""
        return []

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.tokenizer:
            return list(bytes(text, 'utf-8'))
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.tokenizer:
            return bytes(tokens).decode('utf-8', errors='replace')
        return self.tokenizer.decode(tokens)

    def save(self, path: str):
        """Save tokenizer to file."""
        # Save the actual tokenizer
        json_path = str(path).replace('.pkl', '.json')
        self.tokenizer.save(json_path)

        # Also save a pickle with metadata for compatibility
        data = {
            'vocab_size': self.vocab_size,
            'tokenizer_path': json_path,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.vocab_size = data['vocab_size']
        tokenizer._build_tokenizer()

        # Load the actual tokenizer from JSON
        json_path = data.get('tokenizer_path', str(path).replace('.pkl', '.json'))
        if Path(json_path).exists():
            tokenizer.tokenizer = Tokenizer.from_file(json_path)
        else:
            # Fallback to old pickle format
            raise FileNotFoundError(f"Tokenizer JSON not found at {json_path}")

        return tokenizer


# Alias for compatibility
BPETokenizer = FastBPETokenizer


def prepare_dataset(
    vocab_size: int = 512,
    context_length: int = 256,
    train_split: float = 0.9,
    data_dir: str = "data",
    force_retokenize: bool = False,
    max_train_samples: int = None,
    max_val_samples: int = None
):
    """
    Prepare TinyStories dataset with FAST BPE tokenization.
    Uses Rust-backed tokenizers for 100x speedup.
    """
    from data import TinyStoriesDataset, download_tinystories

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    tokenizer_path = data_path / "tokenizer.pkl"
    train_tokens_path = data_path / "train_tokens.npy"
    val_tokens_path = data_path / "val_tokens.npy"

    # Check if we can load cached data
    if not force_retokenize and all(p.exists() for p in [tokenizer_path, train_tokens_path, val_tokens_path]):
        print("Loading cached data...", flush=True)
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded (vocab_size={len(tokenizer.vocab)})", flush=True)

        train_dataset = TinyStoriesDataset(str(train_tokens_path), context_length, max_train_samples)
        val_dataset = TinyStoriesDataset(str(val_tokens_path), context_length, max_val_samples)

        print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}", flush=True)
        return train_dataset, val_dataset, tokenizer

    # Download dataset
    dataset = download_tinystories(data_dir)

    # Extract all texts for training tokenizer
    print("Extracting texts for tokenizer training...", flush=True)
    train_texts = dataset['train']['text'][:10000]

    # Train tokenizer
    print(f"Training FAST BPE tokenizer with vocab_size={vocab_size}...", flush=True)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_texts, verbose=True)

    # Save tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}", flush=True)

    # Tokenize full dataset using the fast tokenizer
    print("Tokenizing full dataset with FAST tokenizer...", flush=True)
    from tqdm import tqdm

    all_tokens = []
    for story in tqdm(dataset['train']['text'], desc="Tokenizing"):
        tokens = tokenizer.encode(story)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Split into train/val
    split_idx = int(len(all_tokens) * train_split)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save tokens
    np.save(train_tokens_path, train_tokens)
    np.save(val_tokens_path, val_tokens)

    print(f"Tokenized {len(all_tokens):,} total tokens", flush=True)
    print(f"Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens", flush=True)
    print(f"Vocab size: {len(tokenizer.vocab)}", flush=True)

    train_dataset = TinyStoriesDataset(str(train_tokens_path), context_length, max_train_samples)
    val_dataset = TinyStoriesDataset(str(val_tokens_path), context_length, max_val_samples)

    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    # Test the fast tokenizer
    train_ds, val_ds, tokenizer = prepare_dataset(
        vocab_size=512,
        context_length=256,
        force_retokenize=True
    )

    print(f"\nDataset stats:")
    print(f"  Training samples: {len(train_ds):,}")
    print(f"  Validation samples: {len(val_ds):,}")
    print(f"  Vocabulary size: {len(tokenizer.vocab)}")

    # Test encoding/decoding
    test_text = "Once upon a time, there was a little girl."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest encode/decode:")
    print(f"  Original: {test_text}")
    print(f"  Encoded ({len(encoded)} tokens): {encoded}")
    print(f"  Decoded:  {decoded}")
