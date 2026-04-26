import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\research\family_attn")))
from train import MemmapDataset

print("Creating dataset...")
train_ds = MemmapDataset(r"D:\research\train.bin", seq_len=256, split="train")

print("Creating dataloader...")
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

print("Getting iterator...")
loader_iter = iter(train_loader)

print("Getting first batch...")
try:
    x, y = next(loader_iter)
    print("Success! x shape:", x.shape)
except Exception as e:
    print("Failed:", e)
