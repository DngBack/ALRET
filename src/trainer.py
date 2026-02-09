"""
ALRET trainer with adversarial low-rank editing training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Dict, Optional, List
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

from .metrics import RefusalClassifier
from .attacker import build_attacker
from .utils import AverageMeter, save_checkpoint

logger = logging.getLogger(__name__)


class ALRETTrainer:
    """Trainer for ALRET adversarial robust training."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict,
        refusal_classifier: Optional[RefusalClassifier] = None
    ):
        """
        Initialize ALRET trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            config: Training configuration
            refusal_classifier: Refusal classifier for harmful loss
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config["model"]["device"]
        
        # Move model to device
        self.model.to(self.device)
        
        # Create base model copy for KL preservation
        self.base_model = self._create_base_model()
        
        # Refusal classifier
        self.refusal_classifier = refusal_classifier or RefusalClassifier(device=self.device)
        
        # Build attacker
        self.attacker = build_attacker(config, model)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.metrics = {
            "loss_total": AverageMeter(),
            "loss_harm": AverageMeter(),
            "loss_benign": AverageMeter(),
            "loss_robust": AverageMeter()
        }
        
        # Wandb
        self.use_wandb = config["logging"].get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity"),
                config=config
            )
        
        logger.info("Initialized ALRET trainer")
    
    def _create_base_model(self) -> PreTrainedModel:
        """Create frozen copy of base model for KL preservation."""
        import copy
        base_model = copy.deepcopy(self.model)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("Created frozen base model copy")
        return base_model
    
    def _create_optimizer(self):
        """Create optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            betas=(0.9, 0.999)
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["training"]["warmup_steps"],
            num_training_steps=self.config["training"]["num_steps"]
        )
    
    def compute_harm_loss(
        self,
        model: PreTrainedModel,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss for harmful samples (encourage refusal).
        
        L_harm = L_refusal_classifier + lambda * L_nll_refusal_template
        
        Args:
            model: Model (possibly perturbed)
            batch: Batch with harmful prompts
            
        Returns:
            Loss tensor
        """
        # Component 1: Generate and classify
        prompts = batch["prompt_text"]
        
        with torch.no_grad():
            outputs_text = model.generate(
                batch["input_ids"][:, :batch["input_ids"].shape[1]//2],  # Just prompt part
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode
            generated = self.tokenizer.batch_decode(
                outputs_text,
                skip_special_tokens=True
            )
        
        # Classify refusal
        refusal_scores = self.refusal_classifier(generated)
        target_refusal = torch.ones_like(refusal_scores)
        L_refusal = F.binary_cross_entropy(refusal_scores, target_refusal)
        
        # Component 2: NLL of refusal template (teacher forcing)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        L_nll = outputs.loss
        
        # Combined loss
        lambda_refusal = self.config["alret"].get("lambda_refusal", 0.5)
        L_harm = L_refusal + lambda_refusal * L_nll
        
        return L_harm
    
    def compute_benign_loss(
        self,
        model: PreTrainedModel,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss for benign samples (encourage helpfulness + KL preservation).
        
        L_benign = L_task + beta * L_kl
        
        Args:
            model: Model (possibly perturbed)
            batch: Batch with benign prompts
            
        Returns:
            Loss tensor
        """
        # Task loss (instruction following)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        L_task = outputs.loss
        
        # KL divergence with base model
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            base_logits = base_outputs.logits.detach()
        
        curr_logits = outputs.logits
        
        # Compute KL(base || current) on valid positions
        mask = batch["labels"] != -100
        
        if mask.sum() > 0:
            base_probs = F.softmax(base_logits[mask], dim=-1)
            curr_log_probs = F.log_softmax(curr_logits[mask], dim=-1)
            L_kl = F.kl_div(curr_log_probs, base_probs, reduction='batchmean')
        else:
            L_kl = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        beta = self.config["alret"].get("beta_kl", 0.1)
        L_benign = L_task + beta * L_kl
        
        return L_benign
    
    def compute_compliance_loss(
        self,
        model: PreTrainedModel,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute compliance loss (for attacker: higher = more compliant).
        
        This is the negative of harm loss.
        """
        # Attacker wants model to comply, so we negate the refusal loss
        L_harm = self.compute_harm_loss(model, batch)
        return -L_harm
    
    def train_step(
        self,
        batch_harm: Dict[str, torch.Tensor],
        batch_benign: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single ALRET training step with inner-outer optimization.
        
        Args:
            batch_harm: Batch of harmful samples
            batch_benign: Batch of benign samples
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move batches to device
        batch_harm = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch_harm.items()}
        batch_benign = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch_benign.items()}
        
        # Step 1: Inner attack - find worst-case edit
        deltas = self.attacker.find_worst_case_edit(
            self.model,
            batch_harm,
            batch_benign,
            loss_fn_harm=self.compute_compliance_loss,
            loss_fn_benign=self.compute_benign_loss
        )
        
        # Step 2: Apply deltas to model
        if hasattr(self.attacker, 'apply_delta'):
            # LoRA attacker
            model_pert, original_weights = self.attacker.apply_delta(self.model, deltas)
        else:
            # Directional attacker (uses hooks)
            model_pert = self.model
            self.attacker.register_hooks(self.model, deltas.get("alpha", 1.0))
        
        # Step 3: Compute robust loss on perturbed model
        L_harm = self.compute_harm_loss(model_pert, batch_harm)
        L_benign = self.compute_benign_loss(model_pert, batch_benign)
        L_robust = L_harm + L_benign
        
        # Step 4: Restore model (remove perturbations)
        if hasattr(self.attacker, 'restore_weights'):
            self.attacker.restore_weights(model_pert, original_weights)
        else:
            self.attacker.remove_hooks()
        
        # Step 5: Backward pass (on original model)
        self.optimizer.zero_grad()
        L_robust.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["training"]["max_grad_norm"]
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        metrics = {
            "loss_total": L_robust.item(),
            "loss_harm": L_harm.item(),
            "loss_benign": L_benign.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }
        
        return metrics
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Separate harmful and benign samples
            is_harmful = batch["is_harmful"]
            
            # Create batch dictionaries with proper indexing
            batch_harm = {}
            batch_benign = {}
            for k, v in batch.items():
                if k != "is_harmful":
                    if isinstance(v, torch.Tensor):
                        batch_harm[k] = v[is_harmful.bool()]
                        batch_benign[k] = v[~is_harmful.bool()]
                    elif isinstance(v, list):
                        # For lists (like prompt_text), index using list comprehension
                        batch_harm[k] = [v[i] for i in range(len(v)) if is_harmful[i].item()]
                        batch_benign[k] = [v[i] for i in range(len(v)) if not is_harmful[i].item()]
                    else:
                        # For other types, just copy the value
                        batch_harm[k] = v
                        batch_benign[k] = v
            
            # Skip if either batch is empty
            if len(batch_harm.get("input_ids", [])) == 0 or len(batch_benign.get("input_ids", [])) == 0:
                continue
            
            # Training step
            metrics = self.train_step(batch_harm, batch_benign)
            
            # Update progress bar
            pbar.set_postfix(metrics)
            
            # Log to wandb
            if self.use_wandb and self.global_step % self.config["training"]["logging_steps"] == 0:
                wandb.log(metrics, step=self.global_step)
            
            # Update global step
            self.global_step += 1
            
            # Save checkpoint
            if self.global_step % self.config["training"]["save_steps"] == 0:
                self.save_checkpoint()
            
            # Check if reached max steps
            if self.global_step >= self.config["training"]["num_steps"]:
                break
        
        self.epoch += 1
    
    def train(self, train_loader, val_loader=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        logger.info("Starting ALRET training...")
        logger.info(f"Total steps: {self.config['training']['num_steps']}")
        
        while self.global_step < self.config["training"]["num_steps"]:
            self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None and self.global_step % self.config["eval"]["eval_steps"] == 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Validation metrics: {val_metrics}")
                
                if self.use_wandb:
                    wandb.log({"val_" + k: v for k, v in val_metrics.items()}, step=self.global_step)
        
        # Final checkpoint
        self.save_checkpoint(final=True)
        logger.info("Training complete!")
    
    def evaluate(self, eval_loader):
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Separate harmful and benign
                is_harmful = batch["is_harmful"]
                
                # Create batch dictionaries with proper indexing
                batch_harm = {}
                batch_benign = {}
                for k, v in batch.items():
                    if k != "is_harmful":
                        if isinstance(v, torch.Tensor):
                            batch_harm[k] = v[is_harmful.bool()].to(self.device)
                            batch_benign[k] = v[~is_harmful.bool()].to(self.device)
                        elif isinstance(v, list):
                            batch_harm[k] = [v[i] for i in range(len(v)) if is_harmful[i].item()]
                            batch_benign[k] = [v[i] for i in range(len(v)) if not is_harmful[i].item()]
                        else:
                            batch_harm[k] = v
                            batch_benign[k] = v
                
                if len(batch_harm.get("input_ids", [])) == 0 or len(batch_benign.get("input_ids", [])) == 0:
                    continue
                
                # Compute losses
                L_harm = self.compute_harm_loss(self.model, batch_harm)
                L_benign = self.compute_benign_loss(self.model, batch_benign)
                loss = L_harm + L_benign
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        return {"eval_loss": total_loss / max(num_batches, 1)}
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        output_dir = Path(self.config["logging"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            checkpoint_path = output_dir / "final.pt"
        else:
            checkpoint_path = output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        save_checkpoint(
            self.model,
            self.optimizer,
            self.global_step,
            str(checkpoint_path),
            additional_state={"epoch": self.epoch}
        )
        
        # Also save model and tokenizer separately
        model_dir = output_dir / "model"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")


class StandardFinetuner:
    """Standard fine-tuning baseline (without adversarial training)."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config["model"]["device"]
        
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["training"]["warmup_steps"],
            num_training_steps=config["training"]["num_steps"]
        )
        
        self.global_step = 0
        
        logger.info("Initialized standard fine-tuner")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Standard training step."""
        self.model.train()
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {"loss": loss.item()}
    
    def train(self, train_loader):
        """Training loop."""
        logger.info("Starting standard fine-tuning...")
        
        pbar = tqdm(total=self.config["training"]["num_steps"])
        
        while self.global_step < self.config["training"]["num_steps"]:
            for batch in train_loader:
                metrics = self.train_step(batch)
                pbar.update(1)
                pbar.set_postfix(metrics)
                
                self.global_step += 1
                
                if self.global_step >= self.config["training"]["num_steps"]:
                    break
        
        pbar.close()
        logger.info("Training complete!")
