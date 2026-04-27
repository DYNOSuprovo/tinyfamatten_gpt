"""
FamilyAttn NanoGPT — Full Implementation
=========================================
Prompts 1-10: Base GPT → FamilyAttn attention → Training Loop → TinyStories Loader

Architecture:
  - Standard token + positional embeddings
  - Causal multi-head self-attention (Prompt 1-2)
  - Per-head identity vectors + lateral communication (Prompt 3-4)
  - Behavioral fingerprints (Prompt 5)
  - EMA-based stability (Prompt 6)
  - Vectorized Jensen-Shannon divergence (Prompt 7)
  - Combined forward pass (Prompt 8)
  - Training loop with identity loss (Prompt 9)
  - TinyStories DataLoader (Prompt 10)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int   = 50257      # GPT-2 tokenizer default
    seq_len:    int   = 256        # context window
    n_layers:   int   = 6          # transformer depth
    n_heads:    int   = 8          # attention heads
    d_model:    int   = 512        # embedding dimension
    d_head:     int   = 64         # per-head dim  (d_model // n_heads)
    d_phi:      int   = 64         # fingerprint projection dim
    d_id:       int   = 64         # identity vector dim  (same as d_head)
    mlp_ratio:  float = 4.0        # MLP hidden width multiplier
    dropout:    float = 0.1
    ema_decay:  float = 0.99       # EMA decay for stability buffer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Embeddings  (Prompt 1)
# ─────────────────────────────────────────────────────────────────────────────

class Embeddings(nn.Module):
    """Token + learned positional embeddings."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N)  →  (B, N, d_model)"""
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)  # (1, N)
        return self.drop(self.tok_emb(x) + self.pos_emb(positions))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FamilyAttn Multi-Head Attention  (Prompts 2 → 8)
# ─────────────────────────────────────────────────────────────────────────────

class FamilyAttention(nn.Module):
    """
    Multi-head causal self-attention with:
      • Per-head outputs before concat   (Prompt 2)
      • Learnable, normalised id vectors (Prompt 3)
      • Lateral head communication       (Prompt 4)
      • Behavioral fingerprints          (Prompt 5)
      • EMA stability                    (Prompt 6)
      • Vectorised JS divergence         (Prompt 7)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.H      = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_phi  = cfg.d_phi
        self.d_id   = cfg.d_id

        # ── Standard QKV projection ──────────────────────────────────────────
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # ── Prompt 3 — Learnable identity vectors (H, d_id) ─────────────────
        self.id_vecs = nn.Parameter(torch.randn(self.H, self.d_id))

        # ── Prompt 4 — Lateral gate (scalar, init small → sigmoid ≈ 0.12) ───
        self.lateral_gate = nn.Parameter(torch.tensor(-2.0))

        # ── Prompt 5 — Fingerprint projection + LN ───────────────────────────
        self.phi_proj = nn.Linear(self.d_head, cfg.d_phi, bias=False)
        self.phi_norm = nn.LayerNorm(cfg.d_phi)

        # ── Prompt 6 — EMA buffer (non-trainable) ────────────────────────────
        self.ema_decay = cfg.ema_decay
        self.register_buffer(
            "ema_buffer",
            torch.zeros(self.H, cfg.d_phi),   # (H, d_phi) — running mean
        )
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool))

        # Causal mask (registered so it moves with .to(device))
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
            .view(1, 1, cfg.seq_len, cfg.seq_len),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalised_id_vecs(self) -> torch.Tensor:
        """Return L2-normalised identity vectors  (H, d_id)."""
        return F.normalize(self.id_vecs, p=2, dim=-1)

    def _lateral_mix(self, h: torch.Tensor) -> torch.Tensor:
        """
        Prompt 4 — Lateral communication between heads.

        h : (B, N, H, d_head)
        Returns mixed h : (B, N, H, d_head)
        """
        id_n = self._normalised_id_vecs()          # (H, d_id)

        # Similarity matrix  (H, H)
        sim   = id_n @ id_n.T                      # (H, H)
        e_mat = F.softmax(sim, dim=-1)             # (H, H)  row-softmax over heads

        # Mix: for each head i,  mix_i = sum_j e_ij * h_j
        # h : (B, N, H, d_head) → treat H as "sequence" for batched matmul
        # mixed[b, n, i, :] = sum_j e[i,j] * h[b, n, j, :]
        mixed = torch.matmul(e_mat, h)  # (B, N, H, d_head)

        # Gated residual
        gate_val = torch.sigmoid(self.lateral_gate)            # small at init
        h_updated = h + gate_val * mixed
        return h_updated

    def _fingerprint(self, h: torch.Tensor) -> torch.Tensor:
        """
        Prompt 5 — Behavioral fingerprint.

        h : (B, N, H, d_head)
        Returns phi : (B, H, d_phi)
        """
        # Mean pool over sequence dimension N
        h_pooled = h.mean(dim=1)                   # (B, H, d_head)
        phi = self.phi_norm(self.phi_proj(h_pooled))  # (B, H, d_phi)
        return phi

    def _stability(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Prompt 6 — EMA stability metric.

        phi : (B, H, d_phi)
        Returns stab : (H,)  — mean squared diff vs. EMA (averaged over batch)
        """
        phi_mean = phi.detach().mean(dim=0)        # (H, d_phi)  — batch avg

        with torch.no_grad():
            if not self.ema_initialized:
                self.ema_buffer.copy_(phi_mean)
                self.ema_initialized.fill_(True)

            ema_new = (
                self.ema_decay * self.ema_buffer
                + (1.0 - self.ema_decay) * phi_mean
            )
            self.ema_buffer.copy_(ema_new)

        diff = phi_mean - self.ema_buffer          # (H, d_phi)
        stab = (diff ** 2).mean(dim=-1)            # (H,)
        return stab

    def _js_divergence(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Prompt 7 — Vectorised pairwise Jensen-Shannon divergence between heads.

        phi : (B, H, d_phi)
        Returns div : (H,)  — mean JS divergence of each head vs. all others
        """
        # Average over batch → (H, d_phi)
        phi_mean = phi.mean(dim=0)                 # (H, d_phi)

        # Convert to probability distributions  (H, d_phi)
        p = F.softmax(phi_mean, dim=-1)            # (H, d_phi)

        # Pairwise mixture: M[i,j] = 0.5*(p[i] + p[j])
        # Expand: p_i (H,1,d_phi)  p_j (1,H,d_phi)
        p_i = p.unsqueeze(1)                       # (H, 1, d_phi)
        p_j = p.unsqueeze(0)                       # (1, H, d_phi)
        M   = 0.5 * (p_i + p_j)                   # (H, H, d_phi)

        # Clamp to avoid log(0)
        eps = 1e-8
        p_i_s = p_i.clamp(min=eps)
        p_j_s = p_j.clamp(min=eps)
        M_s   = M.clamp(min=eps)

        # KL(P || M) = sum P * log(P/M)
        kl_i = (p_i_s * (p_i_s.log() - M_s.log())).sum(dim=-1)  # (H, H)
        kl_j = (p_j_s * (p_j_s.log() - M_s.log())).sum(dim=-1)  # (H, H)

        js_mat = 0.5 * (kl_i + kl_j)              # (H, H)  symmetric JS

        # Per-head divergence = mean JS vs. all other heads (exclude diagonal)
        H = self.H
        mask = (~torch.eye(H, dtype=torch.bool, device=phi.device))
        div  = js_mat[mask].view(H, H - 1).mean(dim=-1)          # (H,)

        return div

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, compute_aux: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x : (B, N, d_model)

        Returns
        -------
        out  : (B, N, d_model)   — final projected attention output
        info : dict with keys
                 "head_out"   (B, N, H, d_head)
                 "id_vecs"    (H, d_id)           normalised
                 "phi"        (B, H, d_phi)
                 "div"        (H,)
                 "stab"       (H,)
        """
        B, N, D = x.shape
        H       = self.H
        d_h     = self.d_head

        # ── QKV ──────────────────────────────────────────────────────────────
        qkv = self.qkv(x)                          # (B, N, 3*D)
        q, k, v = qkv.split(D, dim=-1)            # each (B, N, D)

        # Reshape → (B, H, N, d_head)
        def split_heads(t):
            return t.view(B, N, H, d_h).transpose(1, 2)  # (B, H, N, d_head)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ── Scaled dot-product with causal mask ──────────────────────────────
        scale  = math.sqrt(d_h)
        scores = (q @ k.transpose(-2, -1)) / scale    # (B, H, N, N)
        mask   = self.causal_mask[:, :, :N, :N]       # (1, 1, N, N)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))  # (B, H, N, N)

        # ── Per-head output before concat  (Prompt 2) ────────────────────────
        h = (weights @ v)                          # (B, H, N, d_head)
        h = h.transpose(1, 2)                      # (B, N, H, d_head)

        # ── Lateral communication  (Prompt 4) ────────────────────────────────
        h = self._lateral_mix(h)                   # (B, N, H, d_head)

        # ── Fingerprint  (Prompt 5) ──────────────────────────────────────────
        # ── Stability  (Prompt 6) ────────────────────────────────────────────
        # ── JS Divergence  (Prompt 7) ────────────────────────────────────────
        # ── Final projection  (Prompt 8) ─────────────────────────────────────
        h_cat = h.reshape(B, N, H * d_h)          # (B, N, d_model)
        out   = self.resid_drop(self.proj(h_cat)) # (B, N, d_model)

        info = {
            "head_out": h,                         # (B, N, H, d_head)
            "id_vecs":  self._normalised_id_vecs(),# (H, d_id)
        }

        if compute_aux:
            phi  = self._fingerprint(h)                # (B, H, d_phi)
            stab = self._stability(phi)                # (H,)
            div  = self._js_divergence(phi)            # (H,)
            
            info.update({
                "phi":  phi,
                "div":  div,
                "stab": stab,
            })
        else:
            info.update({
                "phi":  None,
                "div":  torch.zeros(self.H, device=x.device),
                "stab": torch.zeros(self.H, device=x.device),
            })

        return out, info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MLP Block  (Prompt 1)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Transformer Block  (Prompt 1)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LN transformer block with FamilyAttention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = FamilyAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(
        self, x: torch.Tensor, compute_aux: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_out, info = self.attn(self.ln1(x), compute_aux=compute_aux)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPT Model  (Prompt 1 + 8)
# ─────────────────────────────────────────────────────────────────────────────

class FamilyAttnGPT(nn.Module):
    """
    Full NanoGPT-style model with FamilyAttention in every layer.

    Forward returns
    ---------------
    logits : (B, N, vocab_size)
    aux    : list of per-layer info dicts  (each with div, stab, etc.)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = Embeddings(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (standard GPT practice)
        self.head.weight = self.embed.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, compute_aux: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """x : (B, N)  →  logits (B, N, vocab_size), aux list"""
        h   = self.embed(x)
        aux = []
        for block in self.blocks:
            h, info = block(h, compute_aux=compute_aux)
            aux.append(info)
        logits = self.head(self.ln_f(h))
        return logits, aux

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Training Loop  (Prompt 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_identity_loss(
    aux: list,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate div + stab across all layers, compute identity loss.

    loss_iid = -0.1 * mean_div + 0.05 * mean_stab
    Warmed-up by gamma ∈ [0, 1].

    Returns (loss_iid, mean_div, mean_stab)
    """
    all_div  = torch.stack([info["div"]  for info in aux]).mean(dim=0)  # (H,)
    all_stab = torch.stack([info["stab"] for info in aux]).mean(dim=0)  # (H,)

    mean_div  = all_div.mean()
    mean_stab = all_stab.mean()

    loss_iid = gamma * (-0.1 * mean_div + 0.05 * mean_stab)
    return loss_iid, mean_div, mean_stab


def train(
    model: FamilyAttnGPT,
    loader: DataLoader,
    n_steps:      int   = 10_000,
    lr:           float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip:    float = 1.0,
    warmup_iid:   int   = 1000,
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu",
    log_every:    int   = 100,
) -> list:
    """
    Training loop for next-token language modelling.

    Returns
    -------
    logs : list of dicts  {step, loss_total, loss_task, div, stab, gamma}
    """
    model.to(device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    loader_iter = iter(loader)
    logs        = []

    for step in range(1, n_steps + 1):

        # ── Fetch batch ───────────────────────────────────────────────────────
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # ── Forward ───────────────────────────────────────────────────────────
        logits, aux = model(x)                     # (B, N, V)

        # Task loss — next-token prediction
        loss_task = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

        # Identity loss warmup  γ = min(1.0, step / warmup_iid)
        gamma    = min(1.0, step / warmup_iid)
        loss_iid, mean_div, mean_stab = compute_identity_loss(aux, gamma)

        loss = loss_task + loss_iid

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0 or step == 1:
            log = {
                "step":       step,
                "loss_total": loss.item(),
                "loss_task":  loss_task.item(),
                "loss_iid":   loss_iid.item(),
                "div":        mean_div.item(),
                "stab":       mean_stab.item(),
                "gamma":      gamma,
            }
            logs.append(log)
            print(
                f"[{step:>6}] total={log['loss_total']:.4f}  "
                f"task={log['loss_task']:.4f}  "
                f"iid={log['loss_iid']:.4f}  "
                f"div={log['div']:.4f}  "
                f"stab={log['stab']:.6f}  "
                f"γ={log['gamma']:.3f}"
            )

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TinyStories DataLoader  (Prompt 10)
# ─────────────────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset for TinyStories CSV files.

    Expects a CSV with a "text" column.
    Returns (x, y) pairs for next-token prediction.
    """

    def __init__(
        self,
        csv_path:   str,
        seq_len:    int,
        tokenizer=None,
        max_stories: Optional[int] = None,
    ):
        import pandas as pd
        from transformers import GPT2Tokenizer

        self.seq_len = seq_len

        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer

        df = pd.read_csv(csv_path)
        stories = df["text"].dropna().tolist()
        if max_stories:
            stories = stories[:max_stories]

        eos = self.tokenizer.eos_token or ""

        # Tokenize all stories, separate with EOS
        all_ids: list[int] = []
        for story in stories:
            ids = self.tokenizer.encode(story + eos)
            all_ids.extend(ids)

        self.data = torch.tensor(all_ids, dtype=torch.long)

    def __len__(self) -> int:
        # Number of non-overlapping (seq_len+1)-length chunks
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        # Pad if the last chunk is short
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        x = chunk[:-1]   # (seq_len,)
        y = chunk[1:]    # (seq_len,)
        return x, y


def make_tinystories_loader(
    csv_path:    str,
    seq_len:     int   = 256,
    batch_size:  int   = 16,
    num_workers: int   = 0,
    max_stories: Optional[int] = None,
) -> DataLoader:
    dataset = TinyStoriesDataset(csv_path, seq_len, max_stories=max_stories)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Smoke Tests (run as __main__)
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_tests():
    """Run shape checks for all components."""
    print("=" * 60)
    print("FamilyAttn GPT — Smoke Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    cfg = GPTConfig(
        vocab_size=50257,
        seq_len=64,
        n_layers=2,        # shallow for quick test
        n_heads=8,
        d_model=512,
        d_head=64,
        d_phi=64,
        d_id=64,
    )

    B, N = 2, cfg.seq_len
    x = torch.randint(0, cfg.vocab_size, (B, N)).to(device)

    # ── Test 1: Embeddings ────────────────────────────────────────────────────
    emb = Embeddings(cfg).to(device)
    emb_out = emb(x)
    assert emb_out.shape == (B, N, cfg.d_model), f"Embedding shape mismatch: {emb_out.shape}"
    print(f"[OK] Embeddings       : {tuple(emb_out.shape)}")

    # ── Test 2: FamilyAttention ──────────────────────────────────────────────
    attn = FamilyAttention(cfg).to(device)
    h_in = emb_out
    attn_out, info = attn(h_in)

    assert attn_out.shape      == (B, N, cfg.d_model),              f"attn_out shape: {attn_out.shape}"
    assert info["head_out"].shape == (B, N, cfg.n_heads, cfg.d_head), f"head_out shape: {info['head_out'].shape}"
    assert info["id_vecs"].shape  == (cfg.n_heads, cfg.d_id),         f"id_vecs shape: {info['id_vecs'].shape}"
    assert info["phi"].shape      == (B, cfg.n_heads, cfg.d_phi),     f"phi shape: {info['phi'].shape}"
    assert info["div"].shape      == (cfg.n_heads,),                  f"div shape: {info['div'].shape}"
    assert info["stab"].shape     == (cfg.n_heads,),                  f"stab shape: {info['stab'].shape}"

    print(f"[OK] FamilyAttention")
    print(f"      attn_out  : {tuple(attn_out.shape)}")
    print(f"      head_out  : {tuple(info['head_out'].shape)}")
    print(f"      id_vecs   : {tuple(info['id_vecs'].shape)}")
    print(f"      phi       : {tuple(info['phi'].shape)}")
    print(f"      div       : {tuple(info['div'].shape)}")
    print(f"      stab      : {tuple(info['stab'].shape)}")

    # ── No NaN check ─────────────────────────────────────────────────────────
    for key, val in info.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key}!"
    print(f"[OK] No NaNs in attention info")

    # ── Lateral gate check ───────────────────────────────────────────────────
    gate_val = torch.sigmoid(attn.lateral_gate).item()
    print(f"[OK] Lateral gate (sigmoid) : {gate_val:.4f}  (expect ~0.12 at init -2.0)")

    # ── Similarity matrix shape ───────────────────────────────────────────────
    id_n = attn._normalised_id_vecs()
    sim  = id_n @ id_n.T
    assert sim.shape == (cfg.n_heads, cfg.n_heads), f"sim shape: {sim.shape}"
    print(f"[OK] Similarity matrix : {tuple(sim.shape)}")

    # ── Test 3: Full GPT ─────────────────────────────────────────────────────
    model  = FamilyAttnGPT(cfg).to(device)
    logits, aux = model(x)

    assert logits.shape == (B, N, cfg.vocab_size), f"logits shape: {logits.shape}"
    assert len(aux) == cfg.n_layers
    print(f"\n[OK] FamilyAttnGPT")
    print(f"      logits : {tuple(logits.shape)}")
    print(f"      aux layers : {len(aux)}")
    print(f"      Params : {model.num_params():,}")

    print("\n" + "=" * 60)
    print("All smoke tests passed OK")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_tests()
