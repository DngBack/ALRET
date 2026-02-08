"""
Model wrapper for loading and managing LLMs in ALRET.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Optional, Dict, List
import logging
import copy

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for managing LLM model and tokenizer."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        trust_remote_code: bool = True
    ):
        """
        Initialize model wrapper.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load model in 8-bit
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = self._get_torch_dtype(torch_dtype)
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        
        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        logger.info(f"Loaded model: {model_name}")
        self._print_model_info()
    
    def _get_torch_dtype(self, dtype_str: str):
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float": torch.float32,
            "half": torch.float16,
            "bf16": torch.bfloat16
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        return tokenizer
    
    def _load_model(self) -> PreTrainedModel:
        """Load model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if not self.load_in_8bit else "auto",
            load_in_8bit=self.load_in_8bit,
            trust_remote_code=self.trust_remote_code
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        return model
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model dtype: {self.torch_dtype}")
        logger.info(f"Device: {self.device}")
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate text from prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            List of generated texts
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode outputs (only the generated part)
        input_lengths = inputs["input_ids"].shape[1]
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, input_lengths:],
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        return self.tokenizer
    
    def save(self, save_path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved model to {save_path}")
    
    def copy_model(self) -> PreTrainedModel:
        """Create a deep copy of the model for base model preservation."""
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        for param in model_copy.parameters():
            param.requires_grad = False
        logger.info("Created frozen copy of model")
        return model_copy


def get_target_modules(model: PreTrainedModel, target_layers: List[int], module_names: List[str]) -> List[str]:
    """
    Get full module names for target layers and modules.
    
    Args:
        model: The model
        target_layers: List of layer indices to target
        module_names: List of module name patterns (e.g., ["self_attn.o_proj", "mlp.down_proj"])
        
    Returns:
        List of full module names
    """
    full_module_names = []
    
    for layer_idx in target_layers:
        for module_name in module_names:
            # For Qwen2 architecture
            full_name = f"model.layers.{layer_idx}.{module_name}"
            full_module_names.append(full_name)
    
    logger.info(f"Target modules: {full_module_names}")
    return full_module_names


def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """Get a module from model by its full name."""
    parts = module_name.split('.')
    module = model
    
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            logger.warning(f"Module {module_name} not found in model")
            return None
    
    return module


def set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module):
    """Set a module in model by its full name."""
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], new_module)
    logger.debug(f"Set module {module_name}")


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """Get hidden size of the model."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        return model.config.d_model
    else:
        raise ValueError("Could not determine model hidden size")


def get_model_num_layers(model: PreTrainedModel) -> int:
    """Get number of layers in the model."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    else:
        raise ValueError("Could not determine number of layers")


def freeze_model(model: PreTrainedModel):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Froze all model parameters")


def unfreeze_model(model: PreTrainedModel):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfroze all model parameters")
