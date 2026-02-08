#!/usr/bin/env python3
"""
Visualization script for ALRET results.

Usage:
    python experiments/visualize_results.py --results results/*.json \
                                            --output figures/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils import load_results

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ALRET results")
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Paths to result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each result file (default: use filenames)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for figures"
    )
    
    return parser.parse_args()


def plot_tamper_cost_curve(
    results_dict: Dict[str, Dict],
    output_path: str,
    format: str = "pdf"
):
    """
    Plot tamper-cost curve: refusal rate vs attack rank.
    
    Args:
        results_dict: Dictionary mapping method names to result dicts
        output_path: Path to save figure
        format: Output format
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for method_name, results in results_dict.items():
        if "tamper_cost_curve" not in results:
            continue
        
        curve = results["tamper_cost_curve"]
        ranks = sorted([int(r) for r in curve.keys()])
        refusal_rates = [curve[str(r)] * 100 for r in ranks]
        
        ax.plot(ranks, refusal_rates, marker='o', linewidth=2, label=method_name)
    
    ax.set_xlabel("Attack Rank Budget")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Tamper-Cost Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", format=format, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved tamper-cost curve to {output_path}.{format}")


def plot_participation_ratio(
    results_dict: Dict[str, Dict],
    output_path: str,
    format: str = "pdf"
):
    """
    Plot participation ratio comparison.
    
    Args:
        results_dict: Dictionary mapping method names to result dicts
        output_path: Path to save figure
        format: Output format
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    methods = []
    prs = []
    
    for method_name, results in results_dict.items():
        if "intrinsic_dim" not in results:
            continue
        
        pr = results["intrinsic_dim"].get("participation_ratio", 0)
        methods.append(method_name)
        prs.append(pr)
    
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(methods, prs, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Intrinsic Dimensionality (Participation Ratio)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels if needed
    if len(methods) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", format=format, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved participation ratio plot to {output_path}.{format}")


def plot_safety_utility_frontier(
    results_dict: Dict[str, Dict],
    output_path: str,
    format: str = "pdf"
):
    """
    Plot safety-utility Pareto frontier.
    
    Args:
        results_dict: Dictionary mapping method names to result dicts
        output_path: Path to save figure
        format: Output format
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for method_name, results in results_dict.items():
        # Get clean metrics
        if "clean" not in results:
            continue
        
        clean = results["clean"]
        utility = clean.get("benign_utility", 0) * 100
        safety = clean.get("refusal_rate", 0) * 100
        
        # Plot clean performance
        ax.scatter(utility, safety, s=150, label=f"{method_name} (clean)", 
                  marker='o', alpha=0.8)
        
        # Plot under attack if available
        if "attacks" in results:
            for attack_name, attack_results in results["attacks"].items():
                if "lora_r4" in attack_name or "lora_r8" in attack_name:
                    safety_attacked = attack_results.get("refusal_rate_attacked", 0) * 100
                    utility_attacked = attack_results.get("benign_utility_attacked", 0) * 100
                    
                    # Draw arrow from clean to attacked
                    ax.annotate('', xy=(utility_attacked, safety_attacked), 
                              xytext=(utility, safety),
                              arrowprops=dict(arrowstyle='->', lw=1.5, alpha=0.5))
                    
                    ax.scatter(utility_attacked, safety_attacked, s=100, 
                             marker='x', alpha=0.8)
    
    ax.set_xlabel("Benign Utility (ROUGE-L %)")
    ax.set_ylabel("Safety (Refusal Rate %)")
    ax.set_title("Safety-Utility Frontier")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", format=format, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved safety-utility frontier to {output_path}.{format}")


def plot_attack_comparison(
    results_dict: Dict[str, Dict],
    output_path: str,
    format: str = "pdf"
):
    """
    Plot attack success rate comparison.
    
    Args:
        results_dict: Dictionary mapping method names to result dicts
        output_path: Path to save figure
        format: Output format
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect data
    methods = list(results_dict.keys())
    attack_types = []
    data = {method: [] for method in methods}
    
    # Get all attack types
    for results in results_dict.values():
        if "attacks" in results:
            for attack_name in results["attacks"].keys():
                if attack_name not in attack_types:
                    attack_types.append(attack_name)
    
    # Collect ASR for each method and attack
    for attack_type in attack_types:
        for method in methods:
            results = results_dict[method]
            if "attacks" in results and attack_type in results["attacks"]:
                asr = results["attacks"][attack_type].get("attack_success_rate", 0) * 100
                data[method].append(asr)
            else:
                data[method].append(0)
    
    # Plot grouped bar chart
    x = np.arange(len(attack_types))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        offset = width * i - (width * len(methods) / 2) + width / 2
        ax.bar(x + offset, data[method], width, label=method, alpha=0.8)
    
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Robustness Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", format=format, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attack comparison to {output_path}.{format}")


def create_summary_table(
    results_dict: Dict[str, Dict],
    output_path: str
):
    """
    Create LaTeX table with summary results.
    
    Args:
        results_dict: Dictionary mapping method names to result dicts
        output_path: Path to save table
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{ALRET Evaluation Results}")
    lines.append("\\label{tab:alret_results}")
    lines.append("\\begin{tabular}{l|ccc|cc}")
    lines.append("\\hline")
    lines.append("Method & RR (clean) & ASR (r=4) & ASR (r=8) & Utility & PR \\\\")
    lines.append("\\hline")
    
    for method_name, results in results_dict.items():
        # Clean RR
        rr_clean = results.get("clean", {}).get("refusal_rate", 0) * 100
        
        # ASR for rank 4 and 8
        asr_r4 = 0
        asr_r8 = 0
        if "attacks" in results:
            asr_r4 = results["attacks"].get("lora_r4", {}).get("attack_success_rate", 0) * 100
            asr_r8 = results["attacks"].get("lora_r8", {}).get("attack_success_rate", 0) * 100
        
        # Utility
        utility = results.get("clean", {}).get("benign_utility", 0)
        
        # PR
        pr = results.get("intrinsic_dim", {}).get("participation_ratio", 0)
        
        lines.append(f"{method_name} & {rr_clean:.1f}\\% & {asr_r4:.1f}\\% & "
                    f"{asr_r8:.1f}\\% & {utility:.3f} & {pr:.2f} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved LaTeX table to {output_path}")


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_dict = {}
    labels = args.labels if args.labels else [Path(p).stem for p in args.results]
    
    for result_path, label in zip(args.results, labels):
        logger.info(f"Loading {result_path} as '{label}'")
        results_dict[label] = load_results(result_path)
    
    # Generate plots
    logger.info("Generating visualizations...")
    
    # Tamper-cost curve
    plot_tamper_cost_curve(
        results_dict,
        str(output_dir / "tamper_cost_curve"),
        format=args.format
    )
    
    # Participation ratio
    plot_participation_ratio(
        results_dict,
        str(output_dir / "participation_ratio"),
        format=args.format
    )
    
    # Safety-utility frontier
    plot_safety_utility_frontier(
        results_dict,
        str(output_dir / "safety_utility_frontier"),
        format=args.format
    )
    
    # Attack comparison
    plot_attack_comparison(
        results_dict,
        str(output_dir / "attack_comparison"),
        format=args.format
    )
    
    # Summary table
    create_summary_table(
        results_dict,
        str(output_dir / "summary_table.tex")
    )
    
    logger.info("="*60)
    logger.info("Visualization complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
