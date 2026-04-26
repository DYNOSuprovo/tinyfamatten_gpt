import os
import torch
import torch.nn.functional as F
import tiktoken
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# Import our model
from family_attn_gpt import FamilyAttnGPT, GPTConfig

app = FastAPI(title="FamilyAttn Generation API")

# Configure CORS so the frontend can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading model on {device}...")

    # Same config used in train.py
    cfg = GPTConfig(
        vocab_size = 50257,
        seq_len    = 256,
        n_layers   = 6,
        n_heads    = 8,
        d_model    = 512,
        dropout    = 0.0 # No dropout needed for generation
    )
    
    model = FamilyAttnGPT(cfg)
    
    # Load our checkpoint
    ckpt_path = Path(r"D:\research\family_attn\checkpoints\latest.pt")
    if not ckpt_path.exists():
        raise RuntimeError("Could not find latest.pt! Please train the model first.")
        
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Remove '_orig_mod.' prefix if it exists (for compatibility with compiled saves)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
        
    model.to(device)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 200

class GenerateResponse(BaseModel):
    generated_text: str

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text_endpoint(req: GenerateRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    # Encode the starting text
    idx = torch.tensor(tokenizer.encode(req.prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tokens = []

    with torch.no_grad():
        for _ in range(req.max_tokens):
            # Crop to context length if needed
            idx_cond = idx if idx.size(1) <= model.cfg.seq_len else idx[:, -model.cfg.seq_len:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            
            # Get logits for the last token only
            logits = logits[:, -1, :] / req.temperature
            
            # Top-k sampling
            if req.top_k is not None:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            generated_tokens.append(idx_next.item())
            
            if idx_next.item() == tokenizer.eot_token:
                break
                
    # Decode the newly generated tokens
    new_text = tokenizer.decode(generated_tokens)
    
    # We return the original prompt + the new text, matching typical generation behavior
    full_text = req.prompt + new_text
    return GenerateResponse(generated_text=full_text)

# Mount static folder at root AFTER defining API routes
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
