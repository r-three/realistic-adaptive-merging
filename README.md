# The Appeal and Reality of Recycling LoRAs with Adaptive Merging

Code for the paper "The Appeal and Reality of Recycling LoRAs with Adaptive Merging" (ICML 2026).

We conduct the first large-scale evaluation of adaptive merging using nearly 1,000 user-contributed LoRAs from Hugging Face, assessing whether these methods can recycle public LoRAs to improve target-task performance in realistic conditions.

## Setup

```bash
# Install dependencies with uv
uv sync
source .venv/bin/activate

# Extract data
tar -xzf data.tar.gz
```

See [compute_clusters/cluster_setup.md](compute_clusters/cluster_setup.md) for HPC cluster setup.

**Note:** `tasks.yaml` contains local file paths (ending in `.json`) for datasets not on HuggingFace. After extracting `data.tar.gz`, update these paths to match your local directory.

## Repository Structure

```
├── peft_extension/          # Custom PEFT implementations
│   ├── lorahub/             # Adaptive merging (Ours, AdaMerging, LoraHub, π-Tuning)
│   └── arrow/               # Arrow routing baseline
├── jobs/                    # Main job scripts
│   ├── run_tuning.py        # Train LoRA / adaptive merging methods
│   ├── run_evaluation.py    # Evaluate models on tasks
│   ├── run_hub_selection.py # Select experts from LoRA pool
│   ├── run_merging.py       # Simple Averaging / TIES / TSV merging
│   └── run_gradfree_lorahub.py  # Gradient-free LoraHub optimization
├── scripts/                 # Model classes and evaluation utilities
├── utils/                   # Merging, metrics, and logging utilities
├── weight_processing/       # LoRA weight extraction and processing
├── experiments/             # Experiment shell scripts
├── results/                 # Model lists and result tables
├── data.tar.gz              # Task data (62 downstream tasks)
└── tasks.yaml               # Task definitions and prompts
```

## Reproducing Experiments

All experiments use 100 target-task samples (80 train / 20 validation) unless otherwise noted.

### 1. LoRA Pool Preparation

**Cache HuggingFace models:**
```bash
python jobs/cache_hf_repos.py \
    --models_file results/model_lists/refiltered_model_ids_new.txt
```

**Select experts for a target task:**
```bash
python jobs/run_hub_selection.py \
    --model_ids_file results/model_lists/refiltered_model_ids_new.txt \
    --output_file outputs/hub_selection/evaluation_top30/banking77.txt \
    --selection_method evaluation \
    --task banking77 \
    --num_selected 30
```

Selection methods: `evaluation`, `random`, `closeness` (cosine/clamp), `quasi_fim`, `rank`.

### 2. Target-Task LoRA Baseline (Section 5)

```bash
python jobs/run_tuning.py \
    --peft_type lora \
    --task banking77 \
    --rank 64 \
    --lr 3e-4 \
    --step 400 \
    --data_size 100 \
    --save_model
```

### 3. Adaptive Merging — "Ours" (Section 3)

```bash
python jobs/run_tuning.py \
    --peft_type lorahub \
    --task banking77 \
    --expert_list_path outputs/hub_selection/evaluation_top30/banking77.txt \
    --weight_granularity per_module \
    --router_activation leaky_relu \
    --router_weight_init_method target \
    --force_routing 0.0 \
    --lr 5e-2 \
    --step 100 \
    --data_size 100
```

### 4. Other Adaptive Methods

**AdaMerging:**
```bash
python jobs/run_tuning.py \
    --peft_type lorahub \
    --task banking77 \
    --expert_list_path outputs/hub_selection/random_top30/banking77.txt \
    --weight_granularity per_module \
    --router_activation linear \
    --router_weight_init_method equal_weight \
    --lr 5e-2 --step 100 --data_size 100
```

**LoraHub (gradient-free):**
```bash
python jobs/run_gradfree_lorahub.py \
    --task banking77 \
    --expert_list_path outputs/hub_selection/random_top30/banking77.txt \
    --data_size 100
```

**π-Tuning:**
```bash
python jobs/run_tuning.py \
    --peft_type lorahub \
    --task banking77 \
    --expert_list_path outputs/hub_selection/quasi_fim_top20/banking77.txt \
    --pi_tuning \
    --router_activation softmax \
    --weight_granularity per_module \
    --lr 1e-4 --step 100 --data_size 100
```

### 5. Non-Adaptive Merging Baselines (Section 5)

```bash
# Simple Averaging
python jobs/run_merging.py \
    --task banking77 \
    --merging_method aggregate \
    --aggregate_method avg \
    --expert_file_path outputs/hub_selection/random_top30/banking77.txt

# TIES Merging
python jobs/run_merging.py \
    --task banking77 \
    --merging_method ties \
    --density 0.4 \
    --expert_file_path outputs/hub_selection/random_top30/banking77.txt

# TSV Merging
python jobs/run_merging.py \
    --task banking77 \
    --merging_method tsv \
    --tsv_reduced_size 8 \
    --expert_file_path outputs/hub_selection/random_top30/banking77.txt
```

### 6. Arrow Baseline (Section 5)

```bash
python jobs/run_tuning.py \
    --peft_type arrow \
    --task banking77 \
    --expert_list_path outputs/hub_selection/evaluation_top30/banking77.txt \
    --lr 5e-2 --step 100 --data_size 100
```

## Experiment Scripts

Full experiment scripts used in the paper are in `experiments/`:

| Script | Paper Section |
|--------|--------------|
| `lora_baseline.sh` | LoRA baseline (all sections) |
| `run_adaptive_merging.sh` | Main Llama results (Sections 3-5) |
| `run_adaptive_merging_qwen.sh` | Qwen replication (Section G) |
| `run_adaptive_merging_lowdata.sh` | 10-sample budget (Section F) |
| `run_hub_selection_generation.sh` | Expert selection lists |
| `run_lorahub_grad_free.sh` | LoraHub gradient-free |
| `run_arrow.sh` | Arrow baseline |
| `run_baseline_all.sh` | Simple Avg / TIES / TSV |
| `ablation_selection_methods.sh` | Selection ablation (Section E) |
| `ablation_tuning_designs.sh` | Tuning design ablation (Section E) |
| `last_section_exps.sh` | In-house LoRA experiments (Q4) |

## Model Lists

| File | Description |
|------|-------------|
| `results/model_lists/refiltered_model_ids_new.txt` | 960 Llama 3.1 8B-Instruct LoRAs (recycled pool) |
| `results/model_lists/qwen4b_model_ids.txt` | 1956 Qwen3-4B-Instruct-2507 LoRAs |
| `results/model_lists/inhouse_model_ids.txt` | 64 in-house target-task LoRAs (Q4) |

## Citation

```bibtex
@inproceedings{liu2026appeal,
  title={The Appeal and Reality of Recycling LoRAs with Adaptive Merging},
  author={Liu, Haokun and Je, Gyung Hyun and Ciccone, Marco and Xu, Zhenlin and YSS, Prasanth and Raffel, Colin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
