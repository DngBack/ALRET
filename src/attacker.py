"""
Attacker implementations for ALRET.

Includes:
- LoRA-style low-rank weight attacker
- Directional ablation attacker (activation-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy
import logging
from transformers import PreTrainedModel

from .model_wrapper import get_module_by_name, set_module_by_name

logger = logging.getLogger(__name__)


class LoRAAttacker:
    """
    Low-rank weight attacker using LoRA-style parameterization.
    
    Finds worst-case low-rank weight perturbations to break safety.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        target_modules: List[str],
        rank: int = 4,
        gamma: float = 0.5,
        eta: float = 0.01,
        inner_lr: float = 0.1,
        inner_steps: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize LoRA attacker.
        
        Args:
            model: Target model to attack
            target_modules: List of module names to perturb
            rank: Rank of low-rank perturbation
            gamma: Weight for benign preservation
            eta: Norm regularization penalty
            inner_lr: Learning rate for inner optimization
            inner_steps: Number of inner optimization steps
            device: Device to run on
        """
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.gamma = gamma
        self.eta = eta
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # Initialize low-rank parameters
        self.lora_params = self._initialize_lora_params()
        
        logger.info(f"Initialized LoRA attacker: rank={rank}, modules={len(target_modules)}")
    
    def _initialize_lora_params(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize U and V matrices for each target module."""
        lora_params = {}
        
        for module_name in self.target_modules:
            module = get_module_by_name(self.model, module_name)
            if module is None:
                continue
            
            # Get weight shape
            if hasattr(module, 'weight'):
                weight_shape = module.weight.shape
            else:
                logger.warning(f"Module {module_name} has no weight, skipping")
                continue
            
            # Initialize U and V with small random values
            # Delta = U @ V^T, shape (out_features, in_features)
            U = torch.randn(weight_shape[0], self.rank, device=self.device) * 0.01
            V = torch.randn(weight_shape[1], self.rank, device=self.device) * 0.01
            
            U.requires_grad = True
            V.requires_grad = True
            
            lora_params[module_name] = (U, V)
        
        return lora_params
    
    def reset_params(self):
        """Reset LoRA parameters to small random values."""
        for module_name in self.lora_params:
            U, V = self.lora_params[module_name]
            U.data = torch.randn_like(U) * 0.01
            V.data = torch.randn_like(V) * 0.01
    
    def get_delta(self) -> Dict[str, torch.Tensor]:
        """Compute weight deltas from U @ V^T."""
        deltas = {}
        for module_name, (U, V) in self.lora_params.items():
            delta = torch.matmul(U, V.T)
            deltas[module_name] = delta
        return deltas
    
    def apply_delta(self, model: PreTrainedModel, deltas: Dict[str, torch.Tensor]) -> PreTrainedModel:
        """
        Apply weight deltas to model (functionally, without modifying original).
        
        Args:
            model: Model to perturb
            deltas: Dictionary of weight deltas
            
        Returns:
            Model with applied deltas (original model is unchanged)
        """
        # Note: For efficiency, we modify in-place but save original weights
        original_weights = {}
        
        for module_name, delta in deltas.items():
            module = get_module_by_name(model, module_name)
            if module is None or not hasattr(module, 'weight'):
                continue
            
            # Save original
            original_weights[module_name] = module.weight.data.clone()
            
            # Apply delta
            module.weight.data = module.weight.data + delta
        
        return model, original_weights
    
    def restore_weights(self, model: PreTrainedModel, original_weights: Dict[str, torch.Tensor]):
        """Restore original weights after attack."""
        for module_name, original_weight in original_weights.items():
            module = get_module_by_name(model, module_name)
            if module is not None and hasattr(module, 'weight'):
                module.weight.data = original_weight
    
    def compute_norm_penalty(self) -> torch.Tensor:
        """Compute Frobenius norm penalty for all deltas."""
        total_norm = 0.0
        for module_name, (U, V) in self.lora_params.items():
            delta = torch.matmul(U, V.T)
            total_norm = total_norm + torch.norm(delta, p='fro') ** 2
        return total_norm
    
    def find_worst_case_edit(
        self,
        model: PreTrainedModel,
        batch_harm: Dict[str, torch.Tensor],
        batch_benign: Dict[str, torch.Tensor],
        loss_fn_harm,
        loss_fn_benign
    ) -> Dict[str, torch.Tensor]:
        """
        Find worst-case low-rank edit via gradient ascent.
        
        Args:
            model: Model to attack
            batch_harm: Harmful batch data
            batch_benign: Benign batch data
            loss_fn_harm: Function to compute loss on harmful (higher = more compliant)
            loss_fn_benign: Function to compute loss on benign (higher = worse utility)
            
        Returns:
            Dictionary of optimal weight deltas
        """
        # Reset parameters
        self.reset_params()
        
        # Inner optimization loop
        for step in range(self.inner_steps):
            # Get current deltas
            deltas = self.get_delta()
            
            # Apply to model (temporarily)
            model_pert, original_weights = self.apply_delta(model, deltas)
            
            # Compute attacker objective
            # Attacker wants to: maximize compliance on harmful, minimize benign loss
            L_harm = loss_fn_harm(model_pert, batch_harm)  # Higher = more compliant
            L_benign = loss_fn_benign(model_pert, batch_benign)  # Higher = worse utility
            
            # Attacker maximizes: L_harm - gamma * L_benign - eta * norm
            norm_penalty = self.compute_norm_penalty()
            attacker_obj = L_harm - self.gamma * L_benign - self.eta * norm_penalty
            
            # Gradient ascent on U, V
            grads = torch.autograd.grad(
                attacker_obj,
                list(self.lora_params.values()),
                create_graph=False
            )
            
            # Update U, V
            with torch.no_grad():
                for (module_name, (U, V)), (grad_U, grad_V) in zip(
                    self.lora_params.items(), grads
                ):
                    U.data = U.data + self.inner_lr * grad_U
                    V.data = V.data + self.inner_lr * grad_V
            
            # Restore original weights
            self.restore_weights(model_pert, original_weights)
            
            if step % max(1, self.inner_steps // 2) == 0:
                logger.debug(f"Attacker step {step}: obj={attacker_obj.item():.4f}, "
                           f"L_harm={L_harm.item():.4f}, L_benign={L_benign.item():.4f}")
        
        # Return final deltas
        return self.get_delta()


class DirectionalAttacker:
    """
    Directional ablation attacker (activation-based).
    
    Removes/adds directional components in residual stream to disable refusal.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        target_layers: List[int],
        num_directions: int = 3,
        alpha_range: Tuple[float, float] = (0.0, 2.0),
        inner_lr: float = 0.1,
        inner_steps: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize directional attacker.
        
        Args:
            model: Target model
            target_layers: List of layer indices to intervene on
            num_directions: Number of directions per layer
            alpha_range: Range for ablation strength
            inner_lr: Learning rate for inner optimization
            inner_steps: Number of inner optimization steps
            device: Device
        """
        self.model = model
        self.target_layers = target_layers
        self.num_directions = num_directions
        self.alpha_range = alpha_range
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # Get hidden size
        self.hidden_size = model.config.hidden_size
        
        # Initialize direction vectors
        self.directions = self._initialize_directions()
        
        # Hook handles
        self.hooks = []
        
        logger.info(f"Initialized directional attacker: layers={target_layers}, "
                   f"directions={num_directions}")
    
    def _initialize_directions(self) -> Dict[int, List[torch.Tensor]]:
        """Initialize random unit direction vectors for each layer."""
        directions = {}
        
        for layer_idx in self.target_layers:
            layer_directions = []
            for _ in range(self.num_directions):
                direction = torch.randn(self.hidden_size, device=self.device)
                direction = direction / torch.norm(direction)
                direction.requires_grad = True
                layer_directions.append(direction)
            directions[layer_idx] = layer_directions
        
        return directions
    
    def reset_directions(self):
        """Reset directions to random unit vectors."""
        for layer_idx in self.directions:
            for i in range(len(self.directions[layer_idx])):
                direction = torch.randn(self.hidden_size, device=self.device)
                direction = direction / torch.norm(direction)
                direction.requires_grad = True
                self.directions[layer_idx][i] = direction
    
    def apply_intervention(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Apply directional intervention to activations.
        
        h' = h - alpha * <h, v> * v
        
        Args:
            activations: Tensor of shape (batch, seq_len, hidden_size)
            layer_idx: Layer index
            alpha: Ablation strength
            
        Returns:
            Modified activations
        """
        if layer_idx not in self.directions:
            return activations
        
        modified = activations
        
        for direction in self.directions[layer_idx]:
            # Compute projection
            projection = torch.sum(modified * direction, dim=-1, keepdim=True)
            
            # Remove component
            modified = modified - alpha * projection * direction
        
        return modified
    
    def register_hooks(self, model: PreTrainedModel, alpha: float = 1.0):
        """Register forward hooks to apply interventions."""
        self.remove_hooks()
        
        def create_hook(layer_idx):
            def hook(module, input, output):
                # output is tuple (hidden_states, ...)
                hidden_states = output[0] if isinstance(output, tuple) else output
                modified = self.apply_intervention(hidden_states, layer_idx, alpha)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook
        
        # Register hooks on target layers
        for layer_idx in self.target_layers:
            layer = model.model.layers[layer_idx]
            handle = layer.register_forward_hook(create_hook(layer_idx))
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def find_worst_case_edit(
        self,
        model: PreTrainedModel,
        batch_harm: Dict[str, torch.Tensor],
        batch_benign: Dict[str, torch.Tensor],
        loss_fn_harm,
        loss_fn_benign
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Find worst-case directional intervention.
        
        Returns:
            Dictionary of optimal directions per layer
        """
        # Reset directions
        self.reset_directions()
        
        # Optimize directions and alpha
        alpha = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        for step in range(self.inner_steps):
            # Register hooks with current directions and alpha
            self.register_hooks(model, alpha.item())
            
            # Forward pass
            L_harm = loss_fn_harm(model, batch_harm)
            L_benign = loss_fn_benign(model, batch_benign)
            
            # Attacker objective
            attacker_obj = L_harm - 0.5 * L_benign
            
            # Gradient ascent
            all_params = [alpha]
            for layer_directions in self.directions.values():
                all_params.extend(layer_directions)
            
            grads = torch.autograd.grad(
                attacker_obj,
                all_params,
                create_graph=False,
                allow_unused=True
            )
            
            # Update
            with torch.no_grad():
                # Update alpha (keep in range)
                if grads[0] is not None:
                    alpha.data = torch.clamp(
                        alpha + self.inner_lr * grads[0],
                        self.alpha_range[0],
                        self.alpha_range[1]
                    )
                
                # Update directions (keep unit norm)
                grad_idx = 1
                for layer_idx in self.directions:
                    for i in range(len(self.directions[layer_idx])):
                        if grads[grad_idx] is not None:
                            direction = self.directions[layer_idx][i]
                            direction.data = direction + self.inner_lr * grads[grad_idx]
                            direction.data = direction / torch.norm(direction)
                        grad_idx += 1
            
            # Remove hooks
            self.remove_hooks()
            
            logger.debug(f"Directional attacker step {step}: obj={attacker_obj.item():.4f}, "
                       f"alpha={alpha.item():.3f}")
        
        # Return final directions and alpha
        return {
            "directions": self.directions,
            "alpha": alpha.item()
        }


def build_attacker(config: Dict, model: PreTrainedModel):
    """
    Build attacker from configuration.
    
    Args:
        config: ALRET config dictionary
        model: Model to attack
        
    Returns:
        Attacker instance
    """
    attacker_type = config["alret"]["attacker_type"]
    device = config["model"]["device"]
    
    if attacker_type == "lora_weight":
        # Build target module list
        from .model_wrapper import get_target_modules
        target_modules = get_target_modules(
            model,
            config["alret"]["target_layers"],
            config["alret"]["target_modules"]
        )
        
        attacker = LoRAAttacker(
            model=model,
            target_modules=target_modules,
            rank=config["alret"]["rank"],
            gamma=config["alret"]["gamma"],
            eta=config["alret"]["eta"],
            inner_lr=config["alret"]["inner_lr"],
            inner_steps=config["alret"]["inner_steps"],
            device=device
        )
    
    elif attacker_type == "directional":
        attacker = DirectionalAttacker(
            model=model,
            target_layers=config["alret"]["target_layers"],
            num_directions=config["alret"].get("num_directions", 3),
            alpha_range=tuple(config["alret"].get("alpha_range", [0.0, 2.0])),
            inner_lr=config["alret"]["inner_lr"],
            inner_steps=config["alret"]["inner_steps"],
            device=device
        )
    
    else:
        raise ValueError(f"Unknown attacker type: {attacker_type}")
    
    return attacker
