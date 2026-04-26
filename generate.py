import os
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

# Import our model
from family_attn_gpt import FamilyAttnGPT, GPTConfig

def generate_text(model, text, tokenizer, max_new_tokens=100, temperature=0.8, top_k=200, device="cuda"):
    model.eval()
    
    # Encode the starting text
    idx = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n[Prompt]: {text}")
    print("[Generating]: ", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            idx_cond = idx if idx.size(1) <= model.cfg.seq_len else idx[:, -model.cfg.seq_len:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            
            # Get logits for the last token only
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Print token as it generates
            word = tokenizer.decode([idx_next.item()])
            print(word, end="", flush=True)

            if idx_next.item() == tokenizer.eot_token:
                break
    print("\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading on {device}...")

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
        print("Could not find latest.pt! Did you train yet?")
        exit()
        
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle the "model" key from our train.py save format
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Let's generate a few stories
    prompts = [
        "Once upon a time, there was a little dog named",
        "Timmy wanted to play outside, but",
        "The big red car went",
    ]
    
    for prompt in prompts:
        generate_text(model, prompt, tokenizer, max_new_tokens=150, temperature=0.8)
