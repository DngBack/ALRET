#!/usr/bin/env python3
"""
Preprocess and split datasets for ALRET training.
"""

import argparse
import logging
import random
from pathlib import Path
import jsonlines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path):
    """Load JSONL file."""
    with jsonlines.open(path) as reader:
        return list(reader)


def save_jsonl(data, path):
    """Save to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(data)


def split_data(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train/val/test."""
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test split ratios (comma-separated)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Parse splits
    splits = [float(x) for x in args.splits.split(',')]
    train_ratio, val_ratio, test_ratio = splits
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Process harmful data
    harmful_path = input_dir / "advbench_harmful.jsonl"
    if harmful_path.exists():
        logger.info("Processing harmful data...")
        harmful_data = load_jsonl(harmful_path)
        
        train_harm, val_harm, test_harm = split_data(
            harmful_data, train_ratio, val_ratio, args.seed
        )
        
        save_jsonl(train_harm, output_dir / "harmful_train.jsonl")
        save_jsonl(val_harm, output_dir / "harmful_val.jsonl")
        save_jsonl(test_harm, output_dir / "harmful_test.jsonl")
        
        logger.info(f"Harmful: {len(train_harm)} train, {len(val_harm)} val, {len(test_harm)} test")
    
    # Process benign data
    benign_path = input_dir / "alpaca_benign.jsonl"
    if benign_path.exists():
        logger.info("Processing benign data...")
        benign_data = load_jsonl(benign_path)
        
        # Filter out very short outputs
        benign_data = [d for d in benign_data if len(d.get("output", "")) >= 20]
        
        train_benign, val_benign, test_benign = split_data(
            benign_data, train_ratio, val_ratio, args.seed
        )
        
        save_jsonl(train_benign, output_dir / "benign_train.jsonl")
        save_jsonl(val_benign, output_dir / "benign_val.jsonl")
        save_jsonl(test_benign, output_dir / "benign_test.jsonl")
        
        logger.info(f"Benign: {len(train_benign)} train, {len(val_benign)} val, {len(test_benign)} test")
    
    logger.info(f"Preprocessing complete! Data saved to {output_dir}")


if __name__ == "__main__":
    main()
