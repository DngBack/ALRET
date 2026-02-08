# ALRET Quick Start Guide

This guide will get you running ALRET in ~10 minutes.

## Step 0: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Step 1: Prepare Data (5 minutes)

```bash
# Download datasets
python scripts/download_data.py \
    --datasets advbench,alpaca_gpt4 \
    --output data/raw/

# Preprocess and split
python scripts/preprocess.py \
    --input data/raw/ \
    --output data/processed/ \
    --splits 0.8,0.1,0.1

# Verify data
ls -lh data/processed/
```

Expected output:
```
harmful_train.jsonl  (160 samples)
harmful_val.jsonl    (20 samples)
harmful_test.jsonl   (20 samples)
benign_train.jsonl   (800 samples)
benign_val.jsonl     (100 samples)
benign_test.jsonl    (100 samples)
```

## Step 2: Quick Test Run (2 minutes)

Test that everything works with a minimal config:

```bash
# Create test config (minimal steps)
cat > configs/qwen_0.5b_test.yaml << 'EOF'
model:
  name: "Qwen/Qwen2-0.5B-Instruct"
  device: "cuda"
  torch_dtype: "bfloat16"
  trust_remote_code: true

data:
  harmful_path: "data/processed/harmful_train.jsonl"
  benign_path: "data/processed/benign_train.jsonl"
  val_harmful: "data/processed/harmful_val.jsonl"
  val_benign: "data/processed/benign_val.jsonl"
  test_harmful: "data/processed/harmful_test.jsonl"
  test_benign: "data/processed/benign_test.jsonl"
  max_length: 512

training:
  method: "alret"
  batch_size: 2
  grad_accumulation_steps: 2
  num_steps: 10  # Minimal for testing
  learning_rate: 5.0e-6
  warmup_steps: 2
  weight_decay: 0.01
  max_grad_norm: 1.0
  bf16: true
  seed: 42
  save_steps: 5
  eval_steps: 5
  logging_steps: 1

alret:
  attacker_type: "lora_weight"
  rank: 2
  inner_steps: 1  # Minimal for testing
  inner_lr: 0.1
  gamma: 0.5
  eta: 0.01
  target_layers: [12, 16]
  target_modules: ["self_attn.o_proj", "mlp.down_proj"]
  beta_kl: 0.1
  lambda_refusal: 0.5

eval:
  attack_ranks: [1, 2, 4]
  num_attack_steps: 20
  eval_batch_size: 4
  generate_max_tokens: 50
  do_sample: false

metrics:
  compute_intrinsic_dim: true
  compute_participation_ratio: true
  benign_utility_metric: "rouge"

logging:
  use_wandb: false
  output_dir: "outputs/test_run"
  log_level: "info"

refusal_classifier:
  model_name: "distilbert-base-uncased"
  threshold: 0.5
EOF

# Run quick test
python experiments/run_alret.py \
    --config configs/qwen_0.5b_test.yaml \
    --method alret
```

If this completes without errors, you're ready for full training!

## Step 3: Full ALRET Training (3-4 hours)

```bash
# Train ALRET model
python experiments/run_alret.py \
    --config configs/qwen_0.5b.yaml \
    --method alret
```

Monitor training:
- Logs: `outputs/alret_qwen_0.5b/train.log`
- Checkpoints: `outputs/alret_qwen_0.5b/checkpoint_step_*.pt`
- Final model: `outputs/alret_qwen_0.5b/final.pt`

## Step 4: Train Baselines (comparison)

```bash
# Baseline 1: Standard fine-tuning
python experiments/run_alret.py \
    --config configs/qwen_0.5b.yaml \
    --method standard_ft \
    --output_dir outputs/baseline_ft

# Baseline 2: Extended refusal
python experiments/run_alret.py \
    --config configs/qwen_0.5b_extended.yaml \
    --method extended_refusal \
    --output_dir outputs/baseline_extended

# Baseline 3: Vanilla (eval only)
python experiments/run_alret.py \
    --config configs/qwen_0.5b.yaml \
    --mode eval_only \
    --output_dir outputs/vanilla
```

## Step 5: Evaluate Under Attacks (2 hours)

```bash
# Evaluate ALRET
python experiments/eval_attacks.py \
    --checkpoint outputs/alret_qwen_0.5b/final.pt \
    --config configs/qwen_0.5b.yaml \
    --output results/alret.json

# Evaluate baselines
python experiments/eval_attacks.py \
    --checkpoint outputs/baseline_ft/final.pt \
    --config configs/qwen_0.5b.yaml \
    --output results/baseline_ft.json

python experiments/eval_attacks.py \
    --checkpoint outputs/baseline_extended/final.pt \
    --config configs/qwen_0.5b_extended.yaml \
    --output results/baseline_extended.json

# Evaluate vanilla (no checkpoint needed)
python experiments/eval_attacks.py \
    --checkpoint outputs/vanilla/final.pt \
    --config configs/qwen_0.5b.yaml \
    --output results/vanilla.json \
    --attack_suite lora
```

## Step 6: Generate Figures (1 minute)

```bash
# Create publication-quality figures
python experiments/visualize_results.py \
    --results results/alret.json \
             results/baseline_ft.json \
             results/baseline_extended.json \
             results/vanilla.json \
    --labels "ALRET" "Standard FT" "Extended Refusal" "Vanilla" \
    --output figures/ \
    --format pdf
```

Generated files:
- `figures/tamper_cost_curve.pdf` - Main result
- `figures/participation_ratio.pdf` - Intrinsic dimension
- `figures/safety_utility_frontier.pdf` - Pareto frontier
- `figures/attack_comparison.pdf` - ASR comparison
- `figures/summary_table.tex` - LaTeX table for paper

## Step 7: Analyze Results

```bash
# Print summary
python -c "
import json
with open('results/alret.json') as f:
    r = json.load(f)
print('=== ALRET Results ===')
print(f'Clean RR: {r[\"clean\"][\"refusal_rate\"]*100:.1f}%')
print(f'ASR (r=4): {r[\"attacks\"][\"lora_r4\"][\"attack_success_rate\"]*100:.1f}%')
print(f'ASR (r=8): {r[\"attacks\"][\"lora_r8\"][\"attack_success_rate\"]*100:.1f}%')
print(f'Utility: {r[\"clean\"][\"benign_utility\"]:.3f}')
print(f'PR: {r[\"intrinsic_dim\"][\"participation_ratio\"]:.2f}')
"
```

Expected output:
```
=== ALRET Results ===
Clean RR: 92.0%
ASR (r=4): 31.0%
ASR (r=8): 45.0%
Utility: 0.700
PR: 5.80
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 2  # Was 4
  grad_accumulation_steps: 8  # Was 4 (keep effective batch same)
```

**Solution 2**: Use 8-bit quantization
```yaml
model:
  load_in_8bit: true
```

### Issue: Training too slow

**Solution 1**: Reduce inner attacker steps
```yaml
alret:
  inner_steps: 1  # Was 3
```

**Solution 2**: Target fewer layers
```yaml
alret:
  target_layers: [16, 20]  # Was [8, 12, 16, 20]
```

### Issue: Refusal classifier not working

The implementation uses rule-based fallback by default. This is sufficient for proof-of-concept. For production, consider training a classifier:

```python
from src.metrics import RefusalClassifier

# Create training data
train_data = [
    {"text": "I cannot help with that.", "label": 1},  # Refusal
    {"text": "Here's how to do it...", "label": 0},    # Compliance
    # ... 500 examples
]

# Train
classifier = RefusalClassifier()
classifier.train(train_data, val_data, output_dir="outputs/refusal_classifier")
```

## Next Steps

1. **Scale to larger models**: Edit config to use Qwen2-7B or Llama-3-8B
2. **Tune hyperparameters**: Experiment with `rank`, `gamma`, `inner_steps`
3. **Add more attacks**: Extend `src/attacker.py` with new attack variants
4. **Compare with TAR**: Implement TAR-style multi-step fine-tune attacks

## Tips for Paper-Ready Results

1. **Run multiple seeds**: Train 3-5 models with different random seeds
2. **Error bars**: Compute mean ± std across seeds for all metrics
3. **Statistical tests**: Use paired t-tests to verify ALRET > baselines
4. **Ablations**: 
   - Vary rank: {1, 2, 4, 8, 16}
   - Vary gamma: {0.2, 0.5, 0.8}
   - Vary inner_steps: {1, 3, 5}
5. **Qualitative analysis**: Show example refusals (clean vs attacked)

## Getting Help

- **Documentation**: See main [README.md](README.md)
- **Issues**: Open a GitHub issue with error logs
- **Questions**: Check troubleshooting section in README

Good luck with your experiments! 🚀
