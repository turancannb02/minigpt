"""
GPT-style Transformer model implementation from scratch.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 100
    context_length: int = 256
    embedding_dim: int = 192
    num_heads: int = 3
    num_layers: int = 3
    dropout: float = 0.1
    feed_forward_dim: int = None  # Defaults to 4 * embedding_dim

    def __post_init__(self):
        if self.feed_forward_dim is None:
            self.feed_forward_dim = 4 * self.embedding_dim


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        # Q, K, V projections
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        # Output projection
        self.out = nn.Linear(config.embedding_dim, config.embedding_dim)

        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.context_length, config.context_length))
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.embedding_dim, dim=2)  # Each: (B, T, C)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.out(y)
        y = self.residual_dropout(y)

        return y


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm and language model head
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying - share weights between token embedding and lm_head
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional, for training)

        Returns:
            loss: cross-entropy loss (if targets provided)
            logits: (B, T, vocab_size) tensor of logits
        """
        B, T = idx.shape
        assert T <= self.config.context_length, f"Sequence length {T} exceeds context length {self.config.context_length}"

        # Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = self.dropout(tok_emb + pos_emb)

        # Forward through transformer blocks
        x = self.blocks(x)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate new tokens.

        Args:
            idx: (B, T) tensor of context tokens
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: if set, only sample from top k tokens
            top_p: if set, use nucleus sampling

        Returns:
            (B, T + max_new_tokens) tensor of generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Optional: top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Optional: top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def get_model_config(vocab_size: int, model_size: str = "tiny") -> GPTConfig:
    """
    Get a predefined model configuration.

    Args:
        vocab_size: Size of the vocabulary
        model_size: One of "tiny", "small", or "medium"

    Returns:
        GPTConfig instance
    """
    configs = {
        "tiny": GPTConfig(
            vocab_size=vocab_size,
            context_length=256,
            embedding_dim=192,
            num_heads=3,
            num_layers=3,
            dropout=0.1,
        ),
        "small": GPTConfig(
            vocab_size=vocab_size,
            context_length=512,
            embedding_dim=384,
            num_heads=6,
            num_layers=6,
            dropout=0.1,
        ),
        "medium": GPTConfig(
            vocab_size=vocab_size,
            context_length=1024,
            embedding_dim=512,
            num_heads=8,
            num_layers=8,
            dropout=0.1,
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(configs.keys())}")

    return configs[model_size]


if __name__ == "__main__":
    # Test the model
    config = get_model_config(vocab_size=100, model_size="tiny")
    model = GPT(config)

    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    B, T = 2, 32
    x = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(x, targets=x)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    idx = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(idx, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
