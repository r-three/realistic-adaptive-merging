# Cluster Setup

## Setup path
Make sure your cluster has `$SCRATCH` pointing to scratch storage:
```bash
if [ -z "$SCRATCH" ]; then
    if [ -d "/scratch/$USER" ]; then
        export SCRATCH="/scratch/$USER"
    elif [ -L "$HOME/scratch" ]; then
        export SCRATCH="$(readlink -f "$HOME/scratch")"
    else
        echo "Error: Cannot resolve SCRATCH path."
    fi
    echo "export SCRATCH=$SCRATCH" >> ~/.bashrc
    mkdir -p $SCRATCH/.cache/
    rm -rf ~/.cache
    ln -s $SCRATCH/.cache/ ~/.cache
fi
```

## Install uv env
This assumes the repo is at `~/s/code/moose` (where `~/s` is a symlink to scratch).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
module load cuda/12.6   # adjust for your cluster
module load rust

cd ~/s/code/moose
uv sync
source .venv/bin/activate
```

## Prepare offline cache
Create `hf_token.txt` and optionally `wandb_token.txt` under the repo root.
```bash
export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets

if [ -f hf_token.txt ]; then
    huggingface-cli login --token $(cat hf_token.txt)
fi

python jobs/cache_hf_repos.py --repo_file model_and_dataset_to_cache.txt
```

## Submitting jobs
Source `start_env.sh` then use the `submit` function:
```bash
source compute_clusters/start_env.sh

JOB_NAME=debug
COMMAND="python jobs/run_tuning.py \
    --run_name $JOB_NAME \
    --task banking77 \
    --batch_size 1 \
    --lr 1e-4 \
    --data_size 100 \
    --step 100 \
    --rank 32 \
    --use_flash_attention \
    --peft_type lora \
    --save_model"
submit "${JOB_NAME}" "${COMMAND}"
```

Customize `compute_clusters/submit_example.sbatch` with your cluster's partition and account.
