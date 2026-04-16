"""
FastAPI inference server for MiniGPT.
Provides a REST API for text generation.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_fast_tokenizer import BPETokenizer
from model import GPT


# Request/response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(100, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling")


class GenerateResponse(BaseModel):
    text: str = Field(..., description="Generated text including prompt")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class HealthResponse(BaseModel):
    status: str
    model_params: int
    vocab_size: int
    device: str


class MiniGPTServer:
    """Server for MiniGPT inference."""

    def __init__(self, checkpoint_path: str, data_dir: str = "data"):
        # Get device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.model = GPT(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load tokenizer
        tokenizer_path = Path(data_dir) / "tokenizer.pkl"
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        print(f"Model loaded: {self.model.get_num_params():,} parameters")
        print(f"Device: {self.device}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, int]:
        """Generate text from a prompt."""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        prompt_length = len(tokens)
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Generate
        generated_idx = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        generated_tokens = generated_idx[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        tokens_generated = len(generated_tokens) - prompt_length

        return generated_text, tokens_generated


# Create FastAPI app
app = FastAPI(
    title="MiniGPT API",
    description="Text generation API for MiniGPT",
    version="1.0.0"
)

# Global server instance (initialized on startup)
server: Optional[MiniGPTServer] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global server
    # These will be set via command line args before uvicorn runs
    # For now, use defaults
    checkpoint_path = "checkpoints/final.pt"
    data_dir = "data"
    server = MiniGPTServer(checkpoint_path, data_dir)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "MiniGPT API",
        "endpoints": {
            "/health": "Check server health",
            "/generate": "Generate text (POST)",
            "/docs": "API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_params=server.model.get_num_params(),
        vocab_size=len(server.tokenizer.vocab),
        device=str(server.device)
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text, tokens_generated = server.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        return GenerateResponse(
            text=text,
            prompt=request.prompt,
            tokens_generated=tokens_generated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Run MiniGPT inference server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing tokenizer.pkl")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")

    args = parser.parse_args()

    # Initialize server with command line args
    global server
    server = MiniGPTServer(args.checkpoint, args.data_dir)

    # Run uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
