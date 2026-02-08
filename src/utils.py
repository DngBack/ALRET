"""
Utility functions for ALRET training and evaluation.
"""

import os
import random
import logging
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {save_path}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    save_path: str,
    additional_state: Optional[Dict] = None
):
    """Save model checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if additional_state:
        checkpoint.update(additional_state)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path} at step {step}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    step = checkpoint.get("step", 0)
    logger.info(f"Loaded checkpoint from {checkpoint_path} at step {step}")
    
    return checkpoint


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_results(results: Dict, save_path: str):
    """Save evaluation results to JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy/torch types to Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    results = convert_types(results)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {save_path}")


def load_results(results_path: str) -> Dict:
    """Load evaluation results from JSON."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    logger.info(f"Loaded results from {results_path}")
    return results


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_device():
    """Get available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU")
    return device


def estimate_memory_usage(model: torch.nn.Module, batch_size: int, seq_length: int) -> float:
    """Estimate memory usage in GB."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    
    # Rough estimate: activations ~4x parameters for forward pass
    activation_memory = param_memory * 4 * batch_size
    
    # Gradients + optimizer states (AdamW: 2x params)
    training_memory = param_memory * 3
    
    total_memory = param_memory + activation_memory + training_memory
    
    logger.info(f"Estimated memory usage: {total_memory:.2f} GB")
    return total_memory


def print_model_info(model: torch.nn.Module):
    """Print model architecture information."""
    param_counts = get_parameter_count(model)
    
    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    logger.info("=" * 60)


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """Create a unique experiment directory with timestamp."""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return str(exp_dir)
