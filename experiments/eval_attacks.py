#!/usr/bin/env python3
"""
Evaluation script for attack suite.

Usage:
    python experiments/eval_attacks.py --checkpoint outputs/alret_qwen_0.5b/final.pt \
                                       --config configs/qwen_0.5b.yaml \
                                       --output results/alret.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import torch
from pathlib import Path

from src.utils import load_config, set_seed, setup_logging, save_results
from src.model_wrapper import ModelWrapper
from src.evaluator import Evaluator
from src.data_loader import load_jsonl

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model under attacks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--attack_suite",
        type=str,
        default="all",
        choices=["all", "lora", "directional", "custom"],
        help="Which attack suite to run"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick testing)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    logger.info("="*60)
    logger.info("ALRET Attack Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Attack suite: {args.attack_suite}")
    logger.info("="*60)
    
    # Set seed
    set_seed(42)
    
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
    
    # Load checkpoint
    if args.checkpoint.endswith(".pt"):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config["model"]["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume it's a directory with saved model
        logger.info(f"Loading model from: {args.checkpoint}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=config["model"]["device"]
        )
    
    # Load test data
    logger.info("Loading test data...")
    test_data_harm = load_jsonl(config["data"]["test_harmful"])
    test_data_benign = load_jsonl(config["data"]["test_benign"])
    
    # Limit samples if specified
    if args.num_samples:
        test_data_harm = test_data_harm[:args.num_samples]
        test_data_benign = test_data_benign[:args.num_samples]
    
    logger.info(f"Test harmful: {len(test_data_harm)}")
    logger.info(f"Test benign: {len(test_data_benign)}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=config["model"]["device"]
    )
    
    # Run evaluation
    results = {}
    
    if args.attack_suite == "all":
        logger.info("Running full evaluation suite...")
        results = evaluator.full_evaluation(test_data_harm, test_data_benign)
        
    elif args.attack_suite == "lora":
        logger.info("Running LoRA attack evaluation...")
        
        # Clean evaluation
        results["clean"] = evaluator.evaluate_clean(test_data_harm, test_data_benign)
        
        # LoRA attacks with different ranks
        results["attacks"] = {}
        for rank in config["eval"].get("attack_ranks", [1, 2, 4, 8, 16]):
            attack_config = {
                "type": "lora_weight",
                "rank": rank,
                "inner_steps": 50,
                "inner_lr": 0.1,
                "gamma": 0.5,
                "eta": 0.01
            }
            logger.info(f"Evaluating LoRA rank={rank}...")
            results["attacks"][f"lora_r{rank}"] = evaluator.evaluate_under_attack(
                test_data_harm,
                test_data_benign,
                attack_config
            )
        
        # Tamper-cost curve
        logger.info("Computing tamper-cost curve...")
        results["tamper_cost_curve"] = evaluator.compute_tamper_cost_curve(
            test_data_harm,
            test_data_benign,
            ranks=config["eval"].get("attack_ranks", [1, 2, 4, 8, 16])
        )
        
    elif args.attack_suite == "directional":
        logger.info("Running directional attack evaluation...")
        
        # Clean evaluation
        results["clean"] = evaluator.evaluate_clean(test_data_harm, test_data_benign)
        
        # Directional attack
        attack_config = {
            "type": "directional",
            "num_directions": 3,
            "inner_steps": 50,
            "inner_lr": 0.1
        }
        logger.info("Evaluating directional attack...")
        results["attacks"] = {
            "directional": evaluator.evaluate_under_attack(
                test_data_harm,
                test_data_benign,
                attack_config
            )
        }
    
    # Intrinsic dimension analysis
    logger.info("Analyzing intrinsic dimensionality...")
    results["intrinsic_dim"] = evaluator.analyze_intrinsic_dimension(test_data_harm)
    
    # Save results
    logger.info(f"Saving results to: {args.output}")
    save_results(results, args.output)
    
    # Print summary
    logger.info("="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    
    if "clean" in results:
        clean = results["clean"]
        logger.info(f"Clean - RR: {clean['refusal_rate']:.3f}, "
                   f"Safety: {clean['safety_score']:.3f}, "
                   f"Utility: {clean['benign_utility']:.3f}")
    
    if "attacks" in results:
        for attack_name, attack_results in results["attacks"].items():
            logger.info(f"{attack_name} - RR: {attack_results['refusal_rate_attacked']:.3f}, "
                       f"ASR: {attack_results['attack_success_rate']:.3f}")
    
    if "intrinsic_dim" in results:
        id_results = results["intrinsic_dim"]
        logger.info(f"Intrinsic Dim - PR: {id_results['participation_ratio']:.2f}, "
                   f"ID95: {id_results['intrinsic_dim_95']}")
    
    logger.info("="*60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
