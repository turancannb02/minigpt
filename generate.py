"""
Text generation script for MiniGPT.
Loads a trained checkpoint and generates text.
"""

import argparse
from pathlib import Path

import torch

from data_fast_tokenizer import BPETokenizer
from model import GPT


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config
    config = checkpoint['config']

    # Create model
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {model.get_num_params():,} parameters")
    print(f"Trained for {checkpoint['step']} steps")
    if 'val_losses' in checkpoint and checkpoint['val_losses']:
        print(f"Final val loss: {checkpoint['val_losses'][-1]:.4f}")

    return model, config


def load_tokenizer(data_dir: str = "data"):
    """Load the trained tokenizer."""
    tokenizer_path = Path(data_dir) / "tokenizer.pkl"
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    return tokenizer


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt."""
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        generated_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode
    generated_tokens = generated_idx[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def interactive_mode(model, tokenizer, device):
    """Interactive text generation mode."""
    print("\n" + "="*50)
    print("Interactive Generation Mode")
    print("="*50)
    print("Commands:")
    print("  'quit' or 'exit' - Exit")
    print("  'params' - Show current parameters")
    print("  'temp <value>' - Set temperature (e.g., 'temp 0.8')")
    print("  'tokens <value>' - Set max tokens (e.g., 'tokens 200')")
    print("  Otherwise, enter your prompt and press Enter")
    print("="*50 + "\n")

    temperature = 0.8
    max_tokens = 100
    top_k = None
    top_p = None

    while True:
        user_input = input("\nPrompt> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'params':
            print(f"Current parameters:")
            print(f"  Temperature: {temperature}")
            print(f"  Max tokens: {max_tokens}")
            print(f"  Top-k: {top_k}")
            print(f"  Top-p: {top_p}")
            continue

        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature value")
            continue

        if user_input.lower().startswith('tokens '):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Invalid token count")
            continue

        # Generate text
        print(f"\nGenerating (temp={temperature}, tokens={max_tokens})...")
        generated = generate_text(
            model, tokenizer, user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        print("\n" + "-" * 50)
        print(generated)
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., checkpoints/final.pt)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt text for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling (if set, only sample from k most likely tokens)")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Nucleus sampling (if set, sample from smallest set of tokens with cumulative probability >= p)")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing tokenizer.pkl")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data_dir)

    if args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

        print("\n" + "="*50)
        print("Generated Text:")
        print("="*50)
        print(generated)
        print("="*50)


if __name__ == "__main__":
    main()
