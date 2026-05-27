# Load required modules (adjust for your cluster)
# module load cuda/12.6
# module load rust

cd ~/s/code/moose
source .venv/bin/activate

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
if [ -f ~/hf_token.txt ]; then
    huggingface-cli login --token $(cat ~/hf_token.txt)
fi

if [[ $MANUAL_OFFLINE == 1 ]]; then
    echo "Running in offline mode"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=offline
fi

# Define job submission function for your cluster.
# Adjust the hostname check and sbatch file as needed.
function submit() {
    local job_name="$1"
    local command="$2"
    local time="${3:-02:00:00}"
    sbatch --job-name="$job_name" --time="$time" --output="logs/$job_name.out" --error="logs/$job_name.out" compute_clusters/submit_example.sbatch "$command"
}
