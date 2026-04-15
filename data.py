"""
Data processing for MiniGPT.
Implements BPE tokenization and handles the TinyStories dataset.
"""

import os
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import heapq

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        # token bytes -> id
        self.vocab = {}
        # id -> token bytes
        self.inverse_vocab = {}
        # list of (byte_pair, merged_id) for encoding
        self.merges = []  # [(bytes(a, b), merged_id), ...]
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize with single byte tokens."""
        for i in range(256):
            self.vocab[bytes([i])] = i
            self.inverse_vocab[i] = bytes([i])

    def _get_pair_counts(self, token_ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Count frequencies of adjacent token-id pairs across all sequences."""
        pairs = defaultdict(int)
        for ids in token_ids_list:
            for i in range(len(ids) - 1):
                pairs[(ids[i], ids[i + 1])] += 1
        return pairs

    def train(self, texts: List[str], verbose: bool = True):
        """Train BPE on a list of texts."""
        # Convert all texts to lists of byte token ids
        token_ids_list = [list(bytes(text, 'utf-8')) for text in texts]

        num_merges = self.vocab_size - 256
        if num_merges <= 0:
            return

        iterator = range(num_merges)
        if verbose:
            iterator = tqdm(iterator, desc="Training BPE")

        for _ in iterator:
            # Count pairs
            pairs = self._get_pair_counts(token_ids_list)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Create new merged token
            new_id = len(self.vocab)
            merged_bytes = self.inverse_vocab[best_pair[0]] + self.inverse_vocab[best_pair[1]]
            self.vocab[merged_bytes] = new_id
            self.inverse_vocab[new_id] = merged_bytes
            self.merges.append((best_pair, new_id))

            # Replace all occurrences of the pair in the token lists
            for i in range(len(token_ids_list)):
                token_ids_list[i] = self._merge_pair(token_ids_list[i], best_pair, new_id)

    def _merge_pair(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace all occurrences of pair with new_id in a list of token ids."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using fast priority queue BPE encoding."""
        if not self.merges:
            return list(bytes(text, 'utf-8'))

        ids = list(bytes(text, 'utf-8'))
        if not ids:
            return []

        # Build lookup tables
        pair_priority = {pair: i for i, (pair, _) in enumerate(self.merges)}
        pair_to_new_id = {pair: new_id for pair, new_id in self.merges}

        # Special marker for deleted positions
        DELETED = -1

        # Use a linked list approach: each node points to the next active node
        next_pos = list(range(1, len(ids) + 1))
        next_pos[-1] = -1  # Last node points to nothing

        # Use heap for efficient pair selection
        # Heap entries: (priority, position)
        heap = []

        # Initial population of heap with adjacent pairs
        pos = 0
        while pos != -1:
            next_pos_val = next_pos[pos]
            if next_pos_val != -1:
                pair = (ids[pos], ids[next_pos_val])
                if pair in pair_priority:
                    heapq.heappush(heap, (pair_priority[pair], pos))
            pos = next_pos_val

        # Track whether each position has been modified
        version = [0] * len(ids)
        current_version = 0

        while heap:
            priority, pos = heapq.heappop(heap)

            # Skip if this position was deleted
            if ids[pos] == DELETED:
                continue

            # Get the next active position
            next_pos_val = next_pos[pos]
            if next_pos_val == -1:
                continue

            # Check if the pair at this position matches what we expected
            pair = (ids[pos], ids[next_pos_val])
            if pair not in pair_priority:
                continue

            # Get the priority of the current pair at this position
            if pair_priority[pair] != priority:
                continue

            # Merge the pair
            new_id = pair_to_new_id[pair]

            # Find previous position
            prev_pos = -1
            temp = 0
            while temp != -1 and temp != pos:
                prev_pos = temp
                temp = next_pos[temp]

            # Update the linked list: skip over the merged element
            next_pos[pos] = next_pos[next_pos_val]
            ids[pos] = new_id
            ids[next_pos_val] = DELETED
            current_version += 1

            # Add new pair to the left (prev_pos, pos)
            if prev_pos != -1:
                left_pair = (ids[prev_pos], ids[pos])
                if left_pair in pair_priority:
                    heapq.heappush(heap, (pair_priority[left_pair], prev_pos))

            # Add new pair to the right (pos, next_pos[pos])
            right_next = next_pos[pos]
            if right_next != -1:
                right_pair = (ids[pos], ids[right_next])
                if right_pair in pair_priority:
                    heapq.heappush(heap, (pair_priority[right_pair], pos))

        # Reconstruct the list by following the linked list
        result = []
        pos = 0
        while pos != -1:
            result.append(ids[pos])
            pos = next_pos[pos]

        return result

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        byte_values = []
        for token in tokens:
            if token in self.inverse_vocab:
                byte_values.extend(self.inverse_vocab[token])
        return bytes(byte_values).decode('utf-8', errors='replace')

    def save(self, path: str):
        """Save tokenizer to file."""
        # Convert bytes keys to latin1 strings for serialization
        vocab_ser = {}
        for k, v in self.vocab.items():
            vocab_ser[k.decode('latin1')] = v

        data = {
            'vocab_size': self.vocab_size,
            'vocab': vocab_ser,
            'merges': self.merges,
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
        tokenizer.vocab = {k.encode('latin1'): v for k, v in data['vocab'].items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = data['merges']
        return tokenizer


class TinyStoriesDataset(Dataset):
    """PyTorch Dataset for TinyStories using memory-mapped numpy."""

    def __init__(self, tokens_path: str, context_length: int, max_samples: int = None):
        # Use memory mapping to avoid loading full file into RAM
        self.tokens = np.load(tokens_path, mmap_mode='r')
        self.context_length = context_length
        self._len = len(self.tokens) - self.context_length - 1
        if max_samples is not None:
            self._len = min(self._len, max_samples)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def download_tinystories(cache_dir: str = "data") -> str:
    """Download TinyStories dataset from HuggingFace."""
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    return dataset


def prepare_dataset(
    vocab_size: int = 512,
    context_length: int = 256,
    train_split: float = 0.9,
    data_dir: str = "data",
    force_retokenize: bool = False,
    max_train_samples: int = None,
    max_val_samples: int = None
) -> Tuple[TinyStoriesDataset, TinyStoriesDataset, BPETokenizer]:
    """
    Prepare TinyStories dataset with BPE tokenization.

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Trained BPE tokenizer
    """
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
    print(f"Training BPE tokenizer with vocab_size={vocab_size}...", flush=True)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_texts, verbose=True)

    # Save tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}", flush=True)

    # Tokenize full dataset
    print("Tokenizing full dataset...", flush=True)
    all_tokens = []
    for story in tqdm(dataset['train']['text']):
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
    # Test the data pipeline
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
