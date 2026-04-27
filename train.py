"""
train.py — Smart Training Script for FamilyAttn NanoGPT
=========================================================
Features:
  - Chunked streaming CSV loader (no full-file OOM)
  - Mixed-precision training (fp16 AMP)
  - Cosine LR schedule with linear warmup
  - Checkpoint save / auto-resume
  - Validation loss every N steps
  - JSON log file (loss, div, stab per step)
  - Gradient clipping + AdamW
  - tqdm progress bars
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, IterableDataset

# ── make sure the module directory is on path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from family_attn_gpt import FamilyAttnGPT, GPTConfig, compute_identity_loss

# ─────────────────────────────────────────────────────────────────────────────
# Memmap Dataset — Ultra-fast zero-copy binary streaming
# ─────────────────────────────────────────────────────────────────────────────

class MemmapDataset(Dataset):
    """
    Reads pre-tokenized uint16 tokens from a flat binary file using memory-mapping.
    This bypasses all CPU tokenization bottlenecks and allows zero-copy,
    instant-seek random access.
    """
    def __init__(self, bin_path: str, seq_len: int, split: str = "train"):
        import os
        self.seq_len = seq_len
        self.bin_path = bin_path
        
        # Calculate how many full sequences we can extract
        file_size = os.path.getsize(bin_path)
        self.total_tokens = file_size // 2 # uint16 is 2 bytes
        # -1 because we need seq_len + 1 tokens for (x, y)
        self.length = max(0, (self.total_tokens - 1) // self.seq_len)
        
        self.data = None # Will instantiate lazily per-worker
        
        print(f"[{split}] File size {file_size/1024/1024:.2f}MB, {self.total_tokens:,} tokens ({self.length:,} batches)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import numpy as np
        if self.data is None:
            self.data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
            
        # We index purely by chunk start to keep it simple and deterministic
        start = idx * self.seq_len
        # Read seq_len + 1 tokens
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        
        # Convert to torch tensor
        chunk_t = torch.from_numpy(chunk)
        
        # Split into inputs and targets
        return chunk_t[:-1], chunk_t[1:]


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scaler, step: int, best_val: float):
    # Always save the uncompiled/unwrapped model's state dict
    raw_model = model
    if hasattr(model, "module"): raw_model = model.module # DataParallel
    if hasattr(raw_model, "_orig_mod"): raw_model = raw_model._orig_mod # torch.compile
    
    torch.save({
        "step":       step,
        "best_val":   best_val,
        "model":      raw_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scaler":     scaler.state_dict(),
    }, path)
    print(f"  [ckpt] Saved  => {path}  (step {step})")


def load_checkpoint(path: str, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    state_dict = ckpt["model"]
    # Remove wrappers prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        k = k.replace("module.", "")
        new_state_dict[k] = v
            
    # Load into the raw model
    raw_model = model
    if hasattr(model, "module"): raw_model = model.module
    if hasattr(raw_model, "_orig_mod"): raw_model = raw_model._orig_mod
    
    raw_model.load_state_dict(new_state_dict)
    
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt and ckpt["scaler"]:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"  [ckpt] Warning: Could not load scaler state (likely from a disabled run): {e}")
        
    print(f"  [ckpt] Resumed from step {ckpt['step']}")
    return ckpt['step'], ckpt['best_val']


# ─────────────────────────────────────────────────────────────────────────────
# Validation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device_type="cuda", enabled=(device == "cuda")):
            logits, _ = model(x, compute_aux=False)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"\nDevice     : {device}")
    if device == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size  = 50257,
        seq_len     = args.seq_len,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
        d_model     = args.d_model,
        d_head      = args.d_model // args.n_heads,
        d_phi       = 64,
        d_id        = 64,
        dropout     = args.dropout,
        ema_decay   = 0.99,
    )
    print(f"Config     : {cfg}")

    # ── Torch Optimizations ───────────────────────────────────────────────────
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        # Speed up matmuls on Ampere+ GPUs (RTX 30/40 series)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # ── Datasets & Loaders ────────────────────────────────────────────────────
    
    # Use MemmapDataset for extreme performance.
    train_ds = MemmapDataset(args.train_bin, seq_len=args.seq_len, split="train")
    
    # Enable multiple workers for prefetching
    workers = args.workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,               # Avoid random page faults, rely on sequential OS caching
        num_workers=workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(workers > 0), # Keep workers alive across epochs
        prefetch_factor=(2 if workers > 0 else None),
        drop_last=True
    )

    print("Loading validation set...")
    val_ds = MemmapDataset(args.val_bin, seq_len=args.seq_len, split="val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FamilyAttnGPT(cfg).to(device)
    
    # Multi-GPU support
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Disable compile on Windows due to Triton requirement
    if device == "cuda" and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("Model compilation enabled (torch.compile)")
        except Exception as e:
            print(f"Could not compile model, falling back to eager: {e}")
    else:
        print("Model compilation skipped (not supported/Windows)")
        
    print(f"Parameters : {model.num_params():,}")

    # ── Optimizer + AMP Scaler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.max_lr,
        weight_decay = 0.1,
        betas        = (0.9, 0.95),
    )
    
    # Determine auto-precision type
    pt_dtype = torch.float16
    # Only use bf16 on Ampere+ (Compute 8.0)
    if device == "cuda" and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            pt_dtype = torch.bfloat16
            print("Using BFloat16 precision (Ampere+ detected)")
        else:
            print(f"Using Float16 precision (Compute {major}.{minor} detected)")
    else:
        print("Using standard precision (CPU fallback)")

    scaler = GradScaler("cuda", enabled=(device == "cuda" and pt_dtype == torch.float16))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    ckpt_dir  = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    start_step, best_val = 0, float("inf")
    if args.resume and ckpt_path.exists():
        start_step, best_val = load_checkpoint(str(ckpt_path), model, optimizer, scaler)

    # ── Log file ─────────────────────────────────────────────────────────────
    log_path = ckpt_dir / "training_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nStarting training — {args.max_steps} steps, batch {args.batch_size}, seq {args.seq_len}")
    print("=" * 70)

    model.train()
    step       = start_step
    loader_iter = iter(train_loader)
    t0         = time.time()

    while step < args.max_steps:
        # ── Dynamic LR ───────────────────────────────────────────────────────
        lr = get_lr(step, args.lr_warmup, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        # ── Gradient Accumulation Loop ────────────────────────────────────────
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # ── Forward (AMP) ─────────────────────────────────────────────────
            compute_iid = (micro_step == args.grad_accum - 1)
            with autocast(device_type=device, dtype=pt_dtype, enabled=(device != "cpu")):
                logits, aux = model(x, compute_aux=compute_iid)
                loss_task   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                if compute_iid:
                    gamma       = min(1.0, step / args.iid_warmup)
                    loss_iid, mean_div, mean_stab = compute_identity_loss(aux, gamma)
                    loss        = (loss_task / args.grad_accum) + loss_iid
                    total_unscaled_loss = loss_task.item() + loss_iid.item()
                else:
                    loss        = loss_task / args.grad_accum
            
            accum_loss += loss_task.item() / args.grad_accum
            
            if device == "cpu":
                loss.backward()
            else:
                scaler.scale(loss).backward()

        # ── Step ─────────────────────────────────────────────────────────────
        if device != "cpu":
            scaler.unscale_(optimizer)
            
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if device == "cpu":
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            dt   = (time.time() - t0) / args.log_every
            t0   = time.time()
            tok_s = int(args.batch_size * args.seq_len / dt)

            log = {
                "step":       step,
                "loss":       round(total_unscaled_loss, 5),
                "loss_task":  round(loss_task.item(), 5),
                "loss_iid":   round(loss_iid.item(), 5),
                "div":        round(mean_div.item(), 5),
                "stab":       round(mean_stab.item(), 7),
                "lr":         round(lr, 7),
                "gamma":      round(gamma, 4),
                "tok_s":      tok_s,
            }
            log_file.write(json.dumps(log) + "\n")
            log_file.flush()

            print(
                f"[{step:>6}/{args.max_steps}] "
                f"loss={log['loss']:.4f}  task={log['loss_task']:.4f}  "
                f"iid={log['loss_iid']:+.4f}  div={log['div']:.4f}  "
                f"stab={log['stab']:.6f}  lr={lr:.2e}  "
                f"{tok_s:,} tok/s"
            )

        # ── Validation ────────────────────────────────────────────────────────
        if step % args.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            is_best  = val_loss < best_val
            if is_best:
                best_val = val_loss
                save_checkpoint(str(best_path), model, optimizer, scaler, step, best_val)
            print(
                f"  [val] step={step}  val_loss={val_loss:.4f}  "
                f"best={best_val:.4f}  {'<-- NEW BEST' if is_best else ''}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % args.ckpt_every == 0:
            save_checkpoint(str(ckpt_path), model, optimizer, scaler, step, best_val)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(str(ckpt_path), model, optimizer, scaler, step, best_val)
    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Log saved to: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train FamilyAttn GPT on TinyStories")

    # Data
    p.add_argument("--train_bin", default=r"D:\research\train.bin")
    p.add_argument("--val_bin",   default=r"D:\research\validation.bin")

    # Model
    p.add_argument("--seq_len",   type=int,   default=256)
    p.add_argument("--n_layers",  type=int,   default=6)
    p.add_argument("--n_heads",   type=int,   default=8)
    p.add_argument("--d_model",   type=int,   default=512)
    p.add_argument("--dropout",   type=float, default=0.1)

    # Training
    p.add_argument("--batch_size",type=int,   default=32)
    p.add_argument("--max_steps", type=int,   default=50_000)
    p.add_argument("--max_lr",    type=float, default=3e-4)
    p.add_argument("--min_lr",    type=float, default=3e-5)
    p.add_argument("--lr_warmup", type=int,   default=1_000)
    p.add_argument("--iid_warmup",type=int,   default=1_000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers",   type=int,   default=0)   # Use 0 workers by default to prevent Windows DataLoader issues

    # Intervals
    p.add_argument("--log_every", type=int,   default=50)
    p.add_argument("--val_every", type=int,   default=500)
    p.add_argument("--ckpt_every",type=int,   default=1_000)

    # Checkpointing
    p.add_argument("--ckpt_dir",  default=r"D:\research\family_attn\checkpoints")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from latest checkpoint if it exists")
    p.add_argument("--device",    default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device to use for training")
    p.add_argument("--grad_accum", type=int,  default=1,
                   help="Number of gradient accumulation steps")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
