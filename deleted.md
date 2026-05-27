# Pre-Release Cleanup

Files and directories removed before public release. They were not part of the
ICML 2026 paper "The Appeal and Reality of Recycling LoRAs with Adaptive Merging".

## File Inventory

### Top-level files

| File | Purpose | Paper? |
|------|---------|--------|
| `pyproject.toml` | Package config, dependencies | Keep |
| `README.md` | Project documentation | Keep |
| `tasks.yaml` | 62 downstream task definitions (datasets, prompts) | Keep |
| `vares.yaml` | Result visualization/comparison config | Keep |
| `data.tar.gz` | Compressed data archive (remade to include all paper data) | Keep |
| `hf_token.txt` | HuggingFace auth token | Delete (secret) |
| `wandb_token.txt` | W&B auth token | Delete (secret) |

### jobs/

| File | Purpose | Paper? |
|------|---------|--------|
| `run_tuning.py` | Main training script (LoRA, LorahubModel, MOOSE, Arrow) | Yes (needs Moose removal) |
| `run_evaluation.py` | Batch evaluation on tasks, best-checkpoint selection | Yes |
| `run_hub_selection.py` | Expert selection (evaluation, cosine, quasi-FIM, random, rank) | Yes |
| `run_merging.py` | Simple Averaging, TIES, TSV merging pipelines | Yes |
| `run_gradfree_lorahub.py` | Gradient-free LorahubModel optimization (LoraHub method) | Yes |
| `lorahub_load_experts.py` | Load LoRA expert weights/metadata from HuggingFace | Yes |
| `cache_hf_repos.py` | Cache HF datasets/models locally | Yes (utility) |
| `load_model.py` | Minimal model loading test script | Delete |
| `run_compression.py` | Joint diag compression pipeline | Delete |
| `delta_evaluation.py` | Weight delta evaluation | Delete |

### peft_extension/

| File | Purpose | Paper? |
|------|---------|--------|
| `__init__.py` | Registers custom PEFT types by hijacking POLY mapping | Yes (needs Moose removal) |
| `utils.py` | Weight initialization helpers | Yes |

### peft_extension/lorahub/ — Paper's "Ours" method + LoraHub + AdaMerging + pi-Tuning

| File | Purpose | Paper? |
|------|---------|--------|
| `config.py` | LorahubConfig: expert_info_dir, router_activation, weight_granularity, pi-tuning flag | Yes |
| `model.py` | LorahubModel: loads experts, concatenates LoRA A/B, creates routing layers | Yes |
| `layers.py` | LorahubLinear/LorahubModule: learned per-expert routing weights | Yes |

### peft_extension/arrow/ — Arrow baseline

| File | Purpose | Paper? |
|------|---------|--------|
| `config.py` | ArrowConfig: expert_info_dir, top_k, router_temp | Yes |
| `model.py` | ArrowModel: loads experts, computes SVD prototypes | Yes |
| `layers.py` | ArrowLinear: per-token dot-product routing to top-k experts | Yes |

### peft_extension/moose/ — Compression-based merging (NOT in paper)

| File | Purpose | Paper? |
|------|---------|--------|
| `config.py` | MooseConfig: compression_weight_dir, TIES options | Delete |
| `model.py` | MooseModel: loads compressed joint-diag states | Delete |
| `layers.py` | MooseLinearLayer: routing over compressed expert basis | Delete |

### scripts/

| File | Purpose | Paper? |
|------|---------|--------|
| `base_model.py` | Abstract class: task config loading, dataset handling, quantization | Yes |
| `model_builder.py` | Detects model type, returns Llama/Qwen/etc subclass | Yes (needs Mistral/SmolLM removal) |
| `llama_child.py` | Llama 3.1 8B-Instruct subclass with chat template | Yes |
| `qwen_child.py` | Qwen3-4B-Instruct-2507 subclass | Yes |
| `evaluation.py` | evaluate_model: inference, metrics (BLEU, ROUGE, exact match, SQuAD) | Yes |
| `model_closeness_metrics.py` | l2/cosine similarity between LoRA weights (for selection) | Yes |
| `update_task_names.py` | Maps short task names to HF dataset paths | Yes |
| `results_to_csv.py` | Parses experiment logs into CSV tables | Yes (utility) |
| `mistral_child.py` | Mistral 7B subclass | Delete |
| `smollm_child.py` | SmolLM subclass | Delete |
| `base_lorahub.py` | Deprecated LorahubModel (nevergrad-based) | Delete |
| `llama_lorahub.py` | Deprecated Llama-specific LorahubModel | Delete |

### utils/

| File | Purpose | Paper? |
|------|---------|--------|
| `merge_utils.py` | Simple Averaging, TIES, TSV merging implementations | Yes |
| `misc_utils.py` | Name conversion, GPU memory stats, NaN checks, data collators | Yes |
| `calculate_eval_metrics.py` | BLEU, ROUGE, exact match, SQuAD metric functions | Yes |
| `logging_utils.py` | Logger setup with SLURM job ID support | Yes |
| `log_parser.py` | Regex parsers for training log strings | Yes |

### weight_processing/

| File | Purpose | Paper? |
|------|---------|--------|
| `__init__.py` | Exports core functions | Yes |
| `collect_lora_experts.py` | Load PeftConfig, extract/normalize LoRA weights, filter model IDs | Yes |
| `average_lora_deltas.py` | Average weight deltas across experts | Yes |
| `hierarchical_model_id_scan.py` | Segment expert lists for hierarchical selection ablations | Yes |

### joint_diag/ — Joint diagonalization (NOT in paper)

| File | Purpose | Paper? |
|------|---------|--------|
| `__init__.py` | Exports compress_lora_models | Delete |
| `compress.py` | PCA-based compression of expert LoRA weights | Delete |
| `solve.py` | Low-level joint diag solvers (PCA, sparse, gradient, core-space) | Delete |

### experiments/

| File | Purpose | Paper? |
|------|---------|--------|
| `task_info.sh` | Central config: 62 tasks, model_id_file, dataset mappings | Yes |
| `last_section_task_info.sh` | Task/model config for Q4 in-house LoRA experiments | Yes |
| `last_section_exps.sh` | Q4 in-house LoRA cross-evaluation (Section 5, Figure 5) | Yes |
| `lora_baseline.sh` | LoRA fine-tuning baseline with HP sweeps | Yes |
| `lora_eval.sh` | Evaluate LoRA baselines | Yes |
| `run_baseline_all.sh` | Master script for Simple Avg / TIES / TSV baselines | Yes |
| `hub_selection.sh` | Expert selection with multiple methods and top-k values | Yes |
| `hub_selection_all_hf.sh` | Hub selection over all HF LoRAs | Yes |
| `hub_selection_lora_excluded.sh` | Hub selection excluding target-task LoRA | Yes |
| `hub_selection_lowdata.sh` | Hub selection with 10-sample budget (Section F) | Yes |
| `hub_selection_qwen.sh` | Hub selection for Qwen (Section G) | Yes |
| `run_adaptive_merging.sh` | Main Llama adaptive merging (Sections 5, E) | Yes (renamed from diagnostic_lorahub.sh) |
| `run_adaptive_merging_qwen.sh` | Qwen adaptive merging (Section G) | Yes (renamed from diagnostic_lorahub_qwen.sh) |
| `run_adaptive_merging_lowdata.sh` | 10-sample adaptive merging (Section F) | Yes (renamed from diagnostic_lorahub_10.sh) |
| `run_hub_selection_generation.sh` | Generate expert selection lists | Yes (renamed from diagnostic_selection.sh) |
| `run_lorahub_grad_based.sh` | Gradient-based LorahubModel training | Yes |
| `run_lorahub_grad_free.sh` | Gradient-free LorahubModel (LoraHub method) | Yes |
| `run_lorahub_grad_free_10sample.sh` | Gradient-free LorahubModel, 10-sample | Yes |
| `run_lorahub_grad_free_qwen.sh` | Gradient-free LorahubModel for Qwen | Yes |
| `run_arrow.sh` | Arrow routing baseline | Yes |
| `ablation_selection_methods.sh` | Selection method ablation (Section E) | Yes |
| `ablation_tuning_designs.sh` | Tuning design ablation (Section E) | Yes |
| `mass_eval.sh` | Batch evaluation over sharded model lists | Yes |
| `moose_evaluation.sh` | Evaluate MOOSE compressed models | Delete |
| `moose_all_hf_expert.sh` | MOOSE over all HF experts | Delete |
| `moose_all_hf_expert_with_lora.sh` | MOOSE over all HF experts + target LoRA | Delete |
| `moose_tuning_lr_init.sh` | MOOSE LR ablation | Delete |
| `moose_tuning_lr_step.sh` | MOOSE LR/step ablation | Delete |
| `model_compression.sh` | Joint diag compression | Delete |
| `model_soup.sh` | Model soup experiment | Delete |
| `diagnostic_full_search.sh` | Hierarchical segment search (not in paper) | Delete |
| `diagnostic_lora.sh` | LoRA HP ablation: dropout/WD/batch (not in paper) | Delete |
| `temp.py` | Temporary helper | Delete |
| `temp_create_job_summary.py` | Temporary job summary | Delete |

### compute_clusters/

| File | Purpose | Paper? |
|------|---------|--------|
| `cluster_setup.md` | HPC environment setup docs | Keep |
| `start_env.sh` | Env init (CUDA, conda, HF cache) | Keep |
| `submit_*.sbatch` (7 files) | SLURM job templates for 7 clusters | Keep |
| `build_table.py` | Aggregate results into tables | Keep |
| `check_logs.py` | Analyze job logs | Keep |
| `cluster_reports/` (7 files) | Job monitoring logs | Delete |

### notebooks/

| File | Purpose | Paper? |
|------|---------|--------|
| `example_analysis_by_task.ipynb` | Per-task performance analysis | Keep |
| `metric_comparison_clean.ipynb` | Metric comparison across methods | Keep |
| `plot_task_representation.ipynb` | Task representation visualization | Keep |
| `read_hf_target_task_output.ipynb` | HF target task output analysis | Keep |
| `parse_lora_training_result.ipynb` | Parse LoRA training results | Keep |
| `evaluate_expert.ipynb` | Expert evaluation analysis | Keep |
| All others (33 notebooks + 3 tmp JSON) | See deletion list below | Delete |

### results/

| File | Purpose | Paper? |
|------|---------|--------|
| `lorahub/` | LorahubModel experiment results (Llama) | Keep |
| `lorahub_qwen/` | LorahubModel experiment results (Qwen) | Keep |
| `expert_info/` | Expert metadata JSON | Keep |
| `model_lists/inhouse_model_ids.txt` | 64 in-house LoRA IDs (Q4) | Keep |
| `model_lists/qwen4b_model_ids.txt` | 1956 Qwen LoRA IDs | Keep |
| `model_lists/refiltered_model_ids_new.txt` | 960 Llama LoRA IDs (renamed from refiltered_model_ids.txt) | Keep |
| `model_lists/diagnostic_model_ids.txt` | Diagnostic model set | Delete |
| `model_lists/extended_model_ids.txt` | Superseded model set | Delete |
| `model_lists/valid_model_ids_1203_sorted_filtered.txt` | Superseded model set | Delete |
| `model_lists/qwen4b_model_ids_shard_*.txt` (8 files) | Processing shards | Delete |
| `tables/diagnostic_model_evaluation.csv` | Diagnostic results | Delete |
| `tables/` (other CSVs) | Evaluation result caches | Keep |

---

## Deletion List

### Entire Directories

- `.tach/` — Tach cache directory
- `data/` — Extracted data files (contents preserved in `data.tar.gz`)
- `tests/` — Empty directory
- `experiments/archive/` — 13 superseded experiment shell scripts
- `scripts/writing/` — Empty directory
- `compute_clusters/cluster_reports/` — HPC job monitoring logs (7 cluster report files)

### Non-Paper Model / Module Support

- `joint_diag/` — Joint diagonalization / compression module (not in paper)
- `peft_extension/moose/` — Compression-based merging via joint diag (not in paper)
- `scripts/mistral_child.py` — Mistral 7B model class (no Mistral experiments in paper)
- `scripts/smollm_child.py` — SmolLM model class (no SmolLM experiments in paper)
- `scripts/base_lorahub.py` — Deprecated LorahubModel (nevergrad-based)
- `scripts/llama_lorahub.py` — Deprecated Llama-specific LorahubModel

#### Code changes required after removing `joint_diag/` and `peft_extension/moose/`

- `peft_extension/__init__.py` — Remove `MooseConfig`/`MooseModel` imports and all Moose branches
- `jobs/run_tuning.py` — Remove `MooseConfig` import and Moose code path
- `scripts/model_builder.py` — Remove Mistral/SmolLM branches

### Renamed Experiment Scripts

- `experiments/diagnostic_lorahub.sh` → `run_adaptive_merging.sh`
- `experiments/diagnostic_lorahub_qwen.sh` → `run_adaptive_merging_qwen.sh`
- `experiments/diagnostic_lorahub_10.sh` → `run_adaptive_merging_lowdata.sh`
- `experiments/diagnostic_selection.sh` → `run_hub_selection_generation.sh`

### Non-Paper Experiment Scripts

- `experiments/diagnostic_full_search.sh`
- `experiments/diagnostic_lora.sh`
- `experiments/model_compression.sh`
- `experiments/model_soup.sh`
- `experiments/moose_evaluation.sh`
- `experiments/moose_all_hf_expert.sh`
- `experiments/moose_all_hf_expert_with_lora.sh`
- `experiments/moose_tuning_lr_init.sh`
- `experiments/moose_tuning_lr_step.sh`
- `experiments/temp.py`
- `experiments/temp_create_job_summary.py`

### Non-Paper Jobs

- `jobs/run_compression.py` — Compression pipeline (imports `joint_diag`)
- `jobs/delta_evaluation.py` — Delta evaluation (not in paper)
- `jobs/load_model.py` — Minimal model loading test

### Notebooks

#### Non-paper models
- `notebooks/mistral_7b_instruct_test_finetuning.ipynb`
- `notebooks/smollm_350m_instruct_test_finetuning.ipynb`
- `notebooks/smollm_task_vectors_and_grad_metrics_every_sample.ipynb`

#### Development / debugging artifacts
- `notebooks/debugging.ipynb`
- `notebooks/peft_testing.ipynb`
- `notebooks/load_adapter.ipynb`
- `notebooks/batch_generate_inference.ipynb`
- `notebooks/unsloth_finetune_llama318b.ipynb`

#### Early exploration (not in paper)
- `notebooks/llama_3.1_8b_instruct_full_finetuning.ipynb`
- `notebooks/llama_3.1_8b_instruct_finetune_evaluate.ipynb`
- `notebooks/llama_3.1_8b_instruct_finetune_evaluate_fsdp.ipynb`
- `notebooks/llama_3.1_8b_instruct_gradient_analysis.ipynb`
- `notebooks/llama_3.1_8b_instruct_anomalous_eval_aqua_rat.ipynb`
- `notebooks/llama_3.1_8b_instruct_anomalous_eval_medmcqa.ipynb`
- `notebooks/llama_3.1_8b_instruct_anomalous_eval_sciq.ipynb`

#### Analysis not in paper
- `notebooks/model_perplexity_2.ipynb`
- `notebooks/per_layer_norm.ipynb`
- `notebooks/track_memory.ipynb`
- `notebooks/track_model_loss.ipynb`
- `notebooks/task_vector_mnist_ascii.ipynb`
- `notebooks/task_vector_sample_analysis.ipynb`
- `notebooks/task_vector_comparison.ipynb`
- `notebooks/task_vectors_and_grad_metrics.ipynb`
- `notebooks/task_vectors_and_grad_metrics_every_sample.ipynb`
- `notebooks/select_and_store_wrong_examples.ipynb`
- `notebooks/fit_the_model_ahhhh.ipynb`

#### Superseded plotting / result readers
- `notebooks/plot_acc_num_param.ipynb`
- `notebooks/plot_acc_num_param mod.ipynb`
- `notebooks/plot_results_modified.ipynb`
- `notebooks/read_run_results.ipynb`
- `notebooks/read_run_results_2.ipynb`
- `notebooks/read_run_results_2_steps.ipynb`

#### Temporary cache files
- `notebooks/tmp_db_index.json`
- `notebooks/tmp_db_index_avg.json`
- `notebooks/tmp_db_index_conf0.json`

### Non-Paper Result / Data Files

- `results/model_lists/diagnostic_model_ids.txt`
- `results/model_lists/extended_model_ids.txt`
- `results/model_lists/valid_model_ids_1203_sorted_filtered.txt`
- `results/model_lists/qwen4b_model_ids_shard_0.txt`
- `results/model_lists/qwen4b_model_ids_shard_250.txt`
- `results/model_lists/qwen4b_model_ids_shard_500.txt`
- `results/model_lists/qwen4b_model_ids_shard_750.txt`
- `results/model_lists/qwen4b_model_ids_shard_1000.txt`
- `results/model_lists/qwen4b_model_ids_shard_1250.txt`
- `results/model_lists/qwen4b_model_ids_shard_1500.txt`
- `results/model_lists/qwen4b_model_ids_shard_1750.txt`
- `results/tables/diagnostic_model_evaluation.csv`

### Non-Paper Data Files

- `data/polish_sequence_labeling.json` — Not one of the 62 downstream tasks
- `data/subset_idx_polish_sequence_labeling_size100.json` — Subset index for non-paper task
- `data/sports_understanding_raw.json` — Raw/unprocessed duplicate

### Secrets (should never be committed)

- `hf_token.txt`
- `wandb_token.txt`
