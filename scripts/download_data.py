#!/usr/bin/env python3
"""
Download datasets for ALRET training.
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
import jsonlines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_advbench(output_dir: Path):
    """Download AdvBench harmful behaviors dataset."""
    logger.info("Downloading AdvBench...")
    
    try:
        dataset = load_dataset("walledai/AdvBench", split="train")
        
        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "instruction": item.get("goal", item.get("prompt", "")),
                "expected": "refusal"
            })
        
        # Save
        output_path = output_dir / "advbench_harmful.jsonl"
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(data)
        
        logger.info(f"Saved {len(data)} AdvBench samples to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to download AdvBench: {e}")


def download_alpaca(output_dir: Path):
    """Download Alpaca-GPT4 dataset."""
    logger.info("Downloading Alpaca-GPT4...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        # Convert to list of dicts
        data = []
        for item in dataset:
            instruction = item["instruction"]
            if item.get("input", ""):
                instruction = f"{instruction}\n\nInput: {item['input']}"
            
            data.append({
                "instruction": instruction,
                "output": item["output"]
            })
        
        # Save
        output_path = output_dir / "alpaca_benign.jsonl"
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(data)
        
        logger.info(f"Saved {len(data)} Alpaca samples to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to download Alpaca: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default="advbench,alpaca_gpt4",
        help="Comma-separated list of datasets to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download requested datasets
    datasets = args.datasets.split(',')
    
    if "advbench" in datasets:
        download_advbench(output_dir)
    
    if "alpaca_gpt4" in datasets or "alpaca" in datasets:
        download_alpaca(output_dir)
    
    logger.info("Download complete!")


if __name__ == "__main__":
    main()
