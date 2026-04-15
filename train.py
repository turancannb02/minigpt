"""
Training script for MiniGPT.
Supports MPS (Apple Silicon), CUDA, and CPU.
"""

import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import prepare_dataset
from model import GPT, GPTConfig, get_model_config


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def cosine_lr_schedule(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """Training loop for MiniGPT."""

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 10000,
        warmup_steps: int = 100,
        eval_interval: int = 500,
        eval_batches: int = 50,
        checkpoint_dir: str = "checkpoints",
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches  # Limit eval iterations for speed
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.gradient_clip = gradient_clip
        self.initial_lr = learning_rate

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Training state
        self.step = 0
        self.train_losses = []
        self.val_losses = []

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on a subset of the validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            if num_batches >= self.eval_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()
        return avg_loss

    def train(self):
        """Main training loop."""
        self.model.train()

        print(f"Starting training (max_steps={self.max_steps})...", flush=True)

        train_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            logits, loss = self.model(x, targets=y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Update weights
            self.optimizer.step()

            # Update learning rate
            lr = cosine_lr_schedule(
                self.step,
                self.warmup_steps,
                self.max_steps,
                self.initial_lr,
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Log
            self.train_losses.append(loss.item())

            if self.step % 10 == 0:
                print(f"Step {self.step}/{self.max_steps} | loss={loss.item():.4f} | lr={lr:.2e}", flush=True)

            # Evaluation
            if (self.step + 1) % self.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"=== Step {self.step + 1} | val_loss={val_loss:.4f} ===", flush=True)
                self.save_checkpoint(f"step_{self.step + 1}.pt")

            self.step += 1

        print("Training completed!", flush=True)
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.model.config,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}", flush=True)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Checkpoint loaded from {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MiniGPT")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium"],
                       help="Model size configuration")
    parser.add_argument("--vocab-size", type=int, default=512,
                       help="Vocabulary size for BPE tokenizer (must be > 256)")
    parser.add_argument("--context-length", type=int, default=256,
                       help="Context length for the model")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps for learning rate schedule")
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Evaluation interval in steps")
    parser.add_argument("--eval-batches", type=int, default=50,
                       help="Number of batches to use for validation")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--force-retokenize", action="store_true",
                       help="Force re-tokenization of dataset")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint file")
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="Limit training dataset size (for testing/low-memory)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                       help="Limit validation dataset size (for testing/low-memory)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Get device
    device = get_device()
    print(f"Using device: {device}", flush=True)

    # Prepare dataset
    print("Preparing dataset...", flush=True)
    train_dataset, val_dataset, tokenizer = prepare_dataset(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        force_retokenize=args.force_retokenize,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    print(f"Dataset prepared! Train: {len(train_dataset):,}, Val: {len(val_dataset):,}", flush=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Create model
    config = get_model_config(vocab_size=len(tokenizer.vocab), model_size=args.model_size)
    config.context_length = args.context_length
    model = GPT(config)

    print(f"Model parameters: {model.get_num_params():,}", flush=True)
    print(f"Model config: {config}", flush=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir,
        gradient_clip=args.gradient_clip,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed / 60:.1f} minutes", flush=True)
    print(f"Final train loss: {trainer.train_losses[-1]:.4f}", flush=True)
    if trainer.val_losses:
        print(f"Final val loss: {trainer.val_losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
