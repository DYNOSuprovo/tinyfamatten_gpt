import sys
print("starting")
try:
    from train import ValidationDataset
    print("imported")
    val_ds = ValidationDataset("D:/research/validation.csv", seq_len=256)
    print("loaded", len(val_ds))
except Exception as e:
    print("error", e)
