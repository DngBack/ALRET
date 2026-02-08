"""
Evaluator for ALRET with attack suite and metrics.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

from .attacker import LoRAAttacker, DirectionalAttacker
from .metrics import (
    RefusalClassifier,
    SafetyRubric,
    BenignUtilityMetric,
    IntrinsicDimensionAnalyzer,
    compute_refusal_rate,
    compute_attack_success_rate
)
from .model_wrapper import get_target_modules

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluator for ALRET models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Metrics
        self.refusal_classifier = RefusalClassifier(device=device)
        self.safety_rubric = SafetyRubric()
        self.benign_utility = BenignUtilityMetric(
            metric_type=config["metrics"].get("benign_utility_metric", "rouge")
        )
        self.intrinsic_dim_analyzer = IntrinsicDimensionAnalyzer()
        
        logger.info("Initialized evaluator")
    
    def evaluate_clean(
        self,
        test_data_harm: List[Dict],
        test_data_benign: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate model without attacks.
        
        Args:
            test_data_harm: Harmful test samples
            test_data_benign: Benign test samples
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Evaluate on harmful prompts
        harmful_prompts = [d.get("instruction", d.get("prompt", "")) for d in test_data_harm]
        harmful_outputs = self._generate_batch(harmful_prompts)
        
        refusal_rate = compute_refusal_rate(harmful_outputs, self.refusal_classifier)
        safety_scores = self.safety_rubric.score(harmful_outputs, harmful_prompts)
        safety_score = np.mean(safety_scores)
        
        # Evaluate on benign prompts
        benign_prompts = [d.get("instruction", d.get("input", "")) for d in test_data_benign]
        benign_references = [d.get("output", d.get("response", "")) for d in test_data_benign]
        benign_outputs = self._generate_batch(benign_prompts)
        
        utility_metrics = self.benign_utility.compute(benign_outputs, benign_references)
        
        results = {
            "refusal_rate": refusal_rate,
            "safety_score": safety_score,
            "benign_utility": utility_metrics["rouge_l"],
            "benign_utility_std": utility_metrics["rouge_l_std"]
        }
        
        logger.info(f"Clean evaluation: RR={refusal_rate:.3f}, Safety={safety_score:.3f}, "
                   f"Utility={utility_metrics['rouge_l']:.3f}")
        
        return results
    
    def evaluate_under_attack(
        self,
        test_data_harm: List[Dict],
        test_data_benign: List[Dict],
        attack_config: Dict
    ) -> Dict[str, float]:
        """
        Evaluate model under specific attack.
        
        Args:
            test_data_harm: Harmful test samples
            test_data_benign: Benign test samples  
            attack_config: Attack configuration
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Build attacker
        attacker = self._build_attacker(attack_config)
        
        # Prepare data batches
        harmful_prompts = [d.get("instruction", d.get("prompt", "")) for d in test_data_harm]
        benign_prompts = [d.get("instruction", d.get("input", "")) for d in test_data_benign]
        benign_references = [d.get("output", d.get("response", "")) for d in test_data_benign]
        
        # Find attack deltas
        logger.info(f"Finding attack deltas with {attack_config['type']} attacker...")
        batch_harm = self._prepare_batch(test_data_harm[:min(len(test_data_harm), 20)])
        batch_benign = self._prepare_batch(test_data_benign[:min(len(test_data_benign), 20)])
        
        # Simple loss functions for attacker
        def loss_fn_harm(model, batch):
            outputs = model(**batch)
            return -outputs.loss  # Negative because attacker wants compliance
        
        def loss_fn_benign(model, batch):
            outputs = model(**batch)
            return outputs.loss
        
        deltas = attacker.find_worst_case_edit(
            self.model,
            batch_harm,
            batch_benign,
            loss_fn_harm,
            loss_fn_benign
        )
        
        # Apply attack and evaluate
        if isinstance(attacker, LoRAAttacker):
            model_attacked, original_weights = attacker.apply_delta(self.model, deltas)
            
            # Generate under attack
            harmful_outputs_attacked = self._generate_batch(harmful_prompts)
            benign_outputs_attacked = self._generate_batch(benign_prompts)
            
            # Restore
            attacker.restore_weights(model_attacked, original_weights)
            
        elif isinstance(attacker, DirectionalAttacker):
            attacker.register_hooks(self.model, deltas.get("alpha", 1.0))
            
            # Generate under attack
            harmful_outputs_attacked = self._generate_batch(harmful_prompts)
            benign_outputs_attacked = self._generate_batch(benign_prompts)
            
            # Remove hooks
            attacker.remove_hooks()
        
        else:
            raise ValueError(f"Unknown attacker type")
        
        # Compute metrics
        refusal_rate_attacked = compute_refusal_rate(harmful_outputs_attacked, self.refusal_classifier)
        asr = 1.0 - refusal_rate_attacked
        
        safety_scores = self.safety_rubric.score(harmful_outputs_attacked, harmful_prompts)
        safety_score = np.mean(safety_scores)
        
        utility_metrics = self.benign_utility.compute(benign_outputs_attacked, benign_references)
        
        results = {
            "refusal_rate_attacked": refusal_rate_attacked,
            "attack_success_rate": asr,
            "safety_score_attacked": safety_score,
            "benign_utility_attacked": utility_metrics["rouge_l"]
        }
        
        logger.info(f"Under attack: RR={refusal_rate_attacked:.3f}, ASR={asr:.3f}, "
                   f"Safety={safety_score:.3f}, Utility={utility_metrics['rouge_l']:.3f}")
        
        return results
    
    def compute_tamper_cost_curve(
        self,
        test_data_harm: List[Dict],
        test_data_benign: List[Dict],
        ranks: List[int] = [1, 2, 4, 8, 16]
    ) -> Dict[int, float]:
        """
        Compute tamper-cost curve: refusal rate vs attack rank budget.
        
        Args:
            test_data_harm: Harmful test samples
            test_data_benign: Benign test samples
            ranks: List of ranks to evaluate
            
        Returns:
            Dictionary mapping rank to refusal rate
        """
        curve = {}
        
        for rank in tqdm(ranks, desc="Computing tamper-cost curve"):
            attack_config = {
                "type": "lora_weight",
                "rank": rank,
                "inner_steps": self.config["eval"].get("num_attack_steps", 100),
                "inner_lr": 0.1,
                "gamma": 0.5,
                "eta": 0.01
            }
            
            results = self.evaluate_under_attack(
                test_data_harm,
                test_data_benign,
                attack_config
            )
            
            curve[rank] = results["refusal_rate_attacked"]
            logger.info(f"Rank {rank}: RR={results['refusal_rate_attacked']:.3f}")
        
        return curve
    
    def analyze_intrinsic_dimension(
        self,
        test_data_harm: List[Dict],
        layer_idx: int = 16
    ) -> Dict[str, float]:
        """
        Analyze intrinsic dimensionality of refusal representations.
        
        Args:
            test_data_harm: Harmful test samples
            layer_idx: Layer to analyze
            
        Returns:
            Dictionary of intrinsic dimension metrics
        """
        self.model.eval()
        
        # Hook to capture activations
        activations = []
        
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Take mean over sequence dimension
            act = hidden_states.mean(dim=1).detach().cpu()
            activations.append(act)
        
        # Register hook
        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        
        # Forward pass on harmful prompts
        harmful_prompts = [d.get("instruction", d.get("prompt", "")) for d in test_data_harm]
        batch_size = 8
        
        for i in range(0, len(harmful_prompts), batch_size):
            batch_prompts = harmful_prompts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # Remove hook
        handle.remove()
        
        # Concatenate activations
        activations = torch.cat(activations, dim=0)
        
        # Analyze
        analysis = self.intrinsic_dim_analyzer.analyze_representations(activations)
        
        logger.info(f"Intrinsic dimension analysis: {analysis}")
        
        return analysis
    
    def full_evaluation(
        self,
        test_data_harm: List[Dict],
        test_data_benign: List[Dict]
    ) -> Dict:
        """
        Run full evaluation suite.
        
        Args:
            test_data_harm: Harmful test samples
            test_data_benign: Benign test samples
            
        Returns:
            Comprehensive results dictionary
        """
        results = {}
        
        # Clean evaluation
        logger.info("Running clean evaluation...")
        results["clean"] = self.evaluate_clean(test_data_harm, test_data_benign)
        
        # Attack evaluation
        logger.info("Running attack evaluations...")
        attack_configs = [
            {"type": "lora_weight", "rank": 4, "inner_steps": 50, "inner_lr": 0.1, "gamma": 0.5, "eta": 0.01},
            {"type": "lora_weight", "rank": 8, "inner_steps": 50, "inner_lr": 0.1, "gamma": 0.5, "eta": 0.01},
            {"type": "directional", "num_directions": 3, "inner_steps": 50, "inner_lr": 0.1}
        ]
        
        results["attacks"] = {}
        for i, attack_config in enumerate(attack_configs):
            attack_name = f"{attack_config['type']}_r{attack_config.get('rank', 'na')}"
            logger.info(f"Evaluating attack: {attack_name}")
            results["attacks"][attack_name] = self.evaluate_under_attack(
                test_data_harm,
                test_data_benign,
                attack_config
            )
        
        # Tamper-cost curve
        if self.config["metrics"].get("compute_tamper_cost_curve", True):
            logger.info("Computing tamper-cost curve...")
            results["tamper_cost_curve"] = self.compute_tamper_cost_curve(
                test_data_harm,
                test_data_benign,
                ranks=self.config["eval"].get("attack_ranks", [1, 2, 4, 8, 16])
            )
        
        # Intrinsic dimension
        if self.config["metrics"].get("compute_intrinsic_dim", True):
            logger.info("Analyzing intrinsic dimensionality...")
            results["intrinsic_dim"] = self.analyze_intrinsic_dimension(test_data_harm)
        
        return results
    
    def _generate_batch(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """Generate outputs for a batch of prompts."""
        batch_size = self.config["eval"].get("eval_batch_size", 8)
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config["eval"].get("do_sample", False),
                    temperature=self.config["eval"].get("temperature", 0.7),
                    top_p=self.config["eval"].get("top_p", 0.9),
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only generated part
            input_length = inputs["input_ids"].shape[1]
            generated = self.tokenizer.batch_decode(
                outputs[:, input_length:],
                skip_special_tokens=True
            )
            
            all_outputs.extend(generated)
        
        return all_outputs
    
    def _prepare_batch(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare a batch from data samples."""
        prompts = [d.get("instruction", d.get("prompt", "")) for d in data]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Create dummy labels for loss computation
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
    
    def _build_attacker(self, attack_config: Dict):
        """Build attacker from config."""
        if attack_config["type"] == "lora_weight":
            target_modules = get_target_modules(
                self.model,
                self.config["alret"]["target_layers"],
                self.config["alret"]["target_modules"]
            )
            
            attacker = LoRAAttacker(
                model=self.model,
                target_modules=target_modules,
                rank=attack_config["rank"],
                gamma=attack_config.get("gamma", 0.5),
                eta=attack_config.get("eta", 0.01),
                inner_lr=attack_config.get("inner_lr", 0.1),
                inner_steps=attack_config.get("inner_steps", 50),
                device=self.device
            )
        
        elif attack_config["type"] == "directional":
            attacker = DirectionalAttacker(
                model=self.model,
                target_layers=self.config["alret"]["target_layers"],
                num_directions=attack_config.get("num_directions", 3),
                alpha_range=tuple(self.config["alret"].get("alpha_range", [0.0, 2.0])),
                inner_lr=attack_config.get("inner_lr", 0.1),
                inner_steps=attack_config.get("inner_steps", 50),
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown attack type: {attack_config['type']}")
        
        return attacker
