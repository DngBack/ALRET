#!/usr/bin/env python3
"""
Main training script for ALRET.

Usage:
    python experiments/run_alret.py --config configs/qwen_0.5b.yaml --method alret
    python experiments/run_alret.py --config configs/qwen_0.5b.yaml --method standard_ft
    python experiments/run_alret.py --config configs/qwen_0.5b.yaml --method extended_refusal
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from pathlib import Path

import torch

from src.utils import load_config, set_seed, setup_logging
from src.model_wrapper import ModelWrapper
from src.data_loader import prepare_datasets, create_dataloaders
from src.trainer import ALRETTrainer, StandardFinetuner
from src.metrics import RefusalClassifier

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ALRET model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="alret",
        choices=["alret", "standard_ft", "extended_refusal"],
        help="Training method"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval_only"],
        help="Mode: train or eval only"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from or evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override method if specified
    config["training"]["method"] = args.method
    
    # Override output dir if specified
    if args.output_dir:
        config["logging"]["output_dir"] = args.output_dir
    
    # Setup logging
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    
    setup_logging(
        log_level=config["logging"].get("log_level", "INFO"),
        log_file=str(log_file)
    )
    
    logger.info("="*60)
    logger.info(f"ALRET Training: Method={args.method}, Mode={args.mode}")
    logger.info("="*60)
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Load model
    logger.info("Loading model...")
    model_wrapper = ModelWrapper(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"]["torch_dtype"],
        load_in_8bit=config["model"].get("load_in_8bit", False),
        trust_remote_code=config["model"].get("trust_remote_code", True)
    )
    
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["training"]["batch_size"]
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Mode: eval only
    if args.mode == "eval_only":
        logger.info("Evaluation only mode")
        
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=config["model"]["device"])
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Run evaluation
        from src.evaluator import Evaluator
        
        evaluator = Evaluator(model, tokenizer, config, device=config["model"]["device"])
        
        # Load test data
        from src.data_loader import load_jsonl
        test_data_harm = load_jsonl(config["data"]["test_harmful"])
        test_data_benign = load_jsonl(config["data"]["test_benign"])
        
        # Full evaluation
        results = evaluator.full_evaluation(test_data_harm, test_data_benign)
        
        # Save results
        from src.utils import save_results
        results_path = output_dir / "eval_results.json"
        save_results(results, str(results_path))
        
        logger.info(f"Evaluation complete. Results saved to {results_path}")
        return
    
    # Mode: training
    if args.method == "alret":
        logger.info("Training with ALRET (adversarial low-rank editing)")
        
        # Initialize refusal classifier
        refusal_classifier = RefusalClassifier(
            checkpoint=config["refusal_classifier"].get("checkpoint"),
            device=config["model"]["device"]
        )
        
        # Create trainer
        trainer = ALRETTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            refusal_classifier=refusal_classifier
        )
        
        # Train
        trainer.train(train_loader, val_loader)
        
    elif args.method == "standard_ft":
        logger.info("Training with standard fine-tuning")
        
        trainer = StandardFinetuner(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        trainer.train(train_loader)
        
    elif args.method == "extended_refusal":
        logger.info("Training with extended refusal baseline")
        
        # Use standard fine-tuner with extended refusal data
        # (data should be preprocessed with diverse refusal templates)
        trainer = StandardFinetuner(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        trainer.train(train_loader)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
