#!/usr/bin/env python3
"""Prepare a small sample of OpenWebText (~1M chars, character-level like Shakespeare).

This downloads OpenWebText and samples a small subset for fast iteration,
using character-level tokenization for direct comparison with shakespeare_char.
"""

import os
import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Target size (roughly match Shakespeare's ~1M characters)
TARGET_CHARS = 1_000_000
VAL_FRACTION = 0.1  # 10% for validation

if __name__ == '__main__':
    print("Loading OpenWebText (streaming to sample)...")

    # Use streaming to avoid downloading the whole thing
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Collect documents until we have enough characters
    all_text = []
    doc_count = 0
    total_chars = 0

    print(f"Sampling documents until we reach {TARGET_CHARS:,} characters...")
    for example in tqdm(dataset, desc="Sampling"):
        text = example['text']
        all_text.append(text)
        total_chars += len(text)
        doc_count += 1

        if total_chars >= TARGET_CHARS:
            break

    # Join all text
    data = '\n'.join(all_text)
    print(f"Collected {len(data):,} characters from {doc_count:,} documents")

    # Get all unique characters (like Shakespeare prepare.py)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars[:50]), "..." if len(chars) > 50 else "")
    print(f"vocab size: {vocab_size:,}")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    # Split into train/val
    n = len(data)
    train_data = data[:int(n * (1 - VAL_FRACTION))]
    val_data = data[int(n * (1 - VAL_FRACTION)):]

    # Encode to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Export to bin files
    out_dir = os.path.dirname(os.path.abspath(__file__))
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    # Save meta information (vocab, encoder/decoder)
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_path = os.path.join(out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Wrote {train_path}: {len(train_ids):,} tokens ({os.path.getsize(train_path):,} bytes)")
    print(f"Wrote {val_path}: {len(val_ids):,} tokens ({os.path.getsize(val_path):,} bytes)")
    print(f"Wrote {meta_path}")
    print(f"Vocab size: {vocab_size} (character-level)")
