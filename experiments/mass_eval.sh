source experiments/task_info.sh
# source experiments/diagnostic_task_info.sh

# Base model - can be overridden via environment variable
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
echo "Using base model: $BASE_MODEL"

# Submitting jobs and saving job names to a file for later checking
rm -f exps_to_check.txt

function run_exp() {
    # Input args:
    # $1: model_id_file (optional, defaults to $model_id_file from task_info.sh)
    local input_file="${1:-$model_id_file}"

    # Extract start index from filename if it contains _shard_XXX pattern
    local start_idx=0
    if [[ "$input_file" =~ _shard_([0-9]+) ]]; then
        start_idx="${BASH_REMATCH[1]}"
    fi
    echo "Using model file: $input_file (start_idx: $start_idx)"

    # Read model IDs from file
    local model_ids=()
    while IFS= read -r line; do
        model_ids+=("$line")
    done < "$input_file"
    echo "Loaded ${#model_ids[@]} model IDs from $input_file"

    for task in "${tasks[@]}"; do
        echo "Running for task: $task"

        for local_idx in "${!model_ids[@]}"; do
            model_id="${model_ids[$local_idx]}"
            model_idx=$((start_idx + local_idx))
            echo "Model $model_idx: $model_id"
            export model_id=$model_id

            JOB_NAME="hub_100eval_homemade_model${model_idx}_task_${task}"
            OUTPUT_ARTIFACT_PATH="outputs/mass_eval/${JOB_NAME}/job_summary.json"
            COMMAND="python jobs/run_evaluation.py \
                --model_name $model_id \
                --tokenizer_path $BASE_MODEL \
                --run_name $JOB_NAME \
                --task $task \
                --split train+valid \
                --eval_batch_size 4 \
                --max_seq_len 2048 \
                --data_size -100 \
                --use_safetensor \
                --use_flash_attention \
                --output_dir outputs/mass_eval/"
            if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
                echo "Skipping ${JOB_NAME} - already done"
            elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                echo "Skipping ${JOB_NAME} - already running"
            else
                echo $COMMAND
                submit "${JOB_NAME}" "${COMMAND}"
                echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
            fi

            if [ $(squeue -u $USER | wc -l) -gt 300 ]; then
                echo "Waiting for 1m due to high job count"
                sleep 1m
            fi
        done
    done
}

# python jobs/run_evaluation.py --model_name outputs/lora_baseline/lora_baseline_lr3e-4_step400_rank64_glue_mrpc/final_model --run_name hub_100eval_model25_task_xnli_de --task xnli_de --split train+valid --max_seq_len 2048 --data_size -100 --use_safetensor --use_flash_attention --output_dir outputs/mass_eval/

# Usage: bash experiments/mass_eval.sh [model_id_file]
# Example: BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" bash experiments/mass_eval.sh results/model_lists/qwen4b_model_ids_shard_50.txt

# Llama eval
export BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python jobs/cache_hf_repos.py --models_file results/model_lists/refiltered_model_ids.txt
run_exp results/model_lists/refiltered_model_ids.txt

# Qwen3 eval
# killarney (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_0.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# killarney (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_250.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# fir (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_500.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# killarney vulcan fir (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_750.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# rorqual (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_1000.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# killarney (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_1250.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard"
python jobs/load_model.py
run_exp "$model_shard"

# killarney (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_1500.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard" 
python jobs/load_model.py
run_exp "$model_shard"

# killarney (done)
export BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
model_shard="results/model_lists/qwen4b_model_ids_shard_1750.txt"
python jobs/cache_hf_repos.py --models_file "$model_shard" 
python jobs/load_model.py
run_exp "$model_shard"


model_shard="results/model_lists/qwen4b_model_ids.txt"
run_exp "$model_shard"

for retry in {1..3}; do
    run_exp "$1"
    if [ $(squeue -u $USER | wc -l) -eq 1 ]; then
        break
    else
        echo "Waiting for 10m before next retry"
        sleep 10m
    fi
done
