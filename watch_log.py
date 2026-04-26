import time
import json

log_path = r"D:\research\family_attn\checkpoints\training_log.jsonl"
seen = set()

# Pre-load all existing steps silently
with open(log_path, "r") as f:
    for line in f:
        try:
            d = json.loads(line)
            seen.add(d["step"])
        except:
            pass

last_step = max(seen) if seen else 0
print(f"[Monitor] Tracking from step {last_step} — new steps appear every ~40s", flush=True)
print("-" * 95, flush=True)
print(f"{'Step':>12}  {'Loss':>8}  {'Task':>8}  {'IID':>8}  {'Div':>8}  {'Stab':>8}  {'LR':>10}  {'Tok/s':>7}", flush=True)
print("-" * 95, flush=True)

while True:
    try:
        with open(log_path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d["step"] not in seen:
                        seen.add(d["step"])
                        print(
                            f"[{d['step']:>6}/50000]  "
                            f"{d['loss']:>8.4f}  "
                            f"{d['loss_task']:>8.4f}  "
                            f"{d['loss_iid']:>+8.4f}  "
                            f"{d['div']:>8.5f}  "
                            f"{d['stab']:>8.5f}  "
                            f"{d['lr']:>10.2e}  "
                            f"{d['tok_s']:>7,}",
                            flush=True,
                        )
                except:
                    pass
    except Exception as e:
        print(f"[error] {e}", flush=True)
    time.sleep(5)
