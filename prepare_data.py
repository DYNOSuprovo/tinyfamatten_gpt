import os
import csv
import numpy as np
import tiktoken
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def process_csv_bulletproof(csv_path, bin_path, batch_size=1000):
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token
    
    if os.path.exists(bin_path):
        os.remove(bin_path)
        
    print(f"Processing {csv_path} -> {bin_path}...")
    total_tokens = 0
    
    # Process line by line and stream to disk
    with open(csv_path, 'r', encoding='utf-8') as f_csv, open(bin_path, 'wb') as f_bin:
        reader = csv.DictReader(f_csv)
        
        batch = []
        for row in tqdm(reader, desc="Tokenizing"):
            text = row.get('text')
            if text:
                batch.append(text)
                
            if len(batch) >= batch_size:
                # encode
                encoded = enc.encode_ordinary_batch(batch)
                for ids in encoded:
                    ids.append(eos_id)
                    arr = np.array(ids, dtype=np.uint16)
                    f_bin.write(arr.tobytes())
                    total_tokens += len(arr)
                batch.clear()
        
        # Flush the rest
        if batch:
            encoded = enc.encode_ordinary_batch(batch)
            for ids in encoded:
                ids.append(eos_id)
                arr = np.array(ids, dtype=np.uint16)
                f_bin.write(arr.tobytes())
                total_tokens += len(arr)
            batch.clear()

    print(f"Completed! Total tokens: {total_tokens:,}")
    print(f"Binary file size: {os.path.getsize(bin_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    train_csv = r"D:\research\train.csv"
    val_csv   = r"D:\research\validation.csv"
    train_bin = r"D:\research\train.bin"
    val_bin   = r"D:\research\validation.bin"
    
    if os.path.exists(val_csv):
        process_csv_bulletproof(val_csv, val_bin)
    
    if os.path.exists(train_csv):
        process_csv_bulletproof(train_csv, train_bin)
