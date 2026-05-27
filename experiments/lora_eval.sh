cd ~/moose/
source experiments/task_info.sh

# Read all model IDs into array first
#file_name=weight_processing/lora_ft_lowest_val_loss_run.csv

for file_name in "weight_processing/lora_ft_lowest_val_loss_run.csv" "weight_processing/lora_ft_highest_val_acc_run.csv"; do

    lora_model_runs=($(tail -n +2 $file_name | cut -d',' -f1))
    datasets=($(tail -n +2 $file_name | cut -d',' -f2))
    subsets=($(tail -n +2 $file_name | cut -d',' -f3))
    echo "Loaded ${#lora_model_runs[@]} model IDs into array"

    # Process each model ID
    for task_idx in ${!lora_model_runs[@]}; do
        dataset=${datasets[$task_idx]}
        subset=${subsets[$task_idx]}
        lora_model_run=${lora_model_runs[$task_idx]}
        echo "Running for model_run: $lora_model_run  dataset: $dataset, subset: $subset"

        JOB_NAME="${lora_model_run}_test_eval"
        MODEL_DIR="/home/jayje/projects/aip-craffel/moose/lora_training"
        OUTPUT_DIR="/home/jayje/projects/aip-craffel/moose/lora_eval"
        OUTPUT_ARTIFACT_PATH="${OUTPUT_DIR}/${JOB_NAME}/job_summary.json"
        COMMAND="python utils/evaluation.py \
            --model_name "${MODEL_DIR}/${lora_model_run}/final_model" \
            --run_name $JOB_NAME \
            --task $subset \
            --split test \
            --batch_size 1 \
            --max_seq_len 2048 \
            --data_size 0 \
            --use_safetensor \
            --use_flash_attention \
	    --output_dir $OUTPUT_DIR"

        if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
            echo "Skipping ${JOB_NAME} - already done"
        else
            echo $COMMAND
            #$COMMAND
        fi
    done
done
