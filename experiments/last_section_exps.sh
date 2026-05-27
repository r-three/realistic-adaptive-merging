source experiments/last_section_task_info.sh

# After finetuning lora baseline and save the models to shared_space/lora_baseline/...
# TODO: add instructions

# After running mass eval on inhourse models
model_ids=()
while IFS= read -r line; do
    model_ids+=("$line")
done < $model_id_file

echo "Loaded ${#model_ids[@]} model IDs into array"

function run_exp() {
    # Input args: 
    # TASKS, MODEL_IDS
    for task in "${tasks[@]}"; do
        echo "Running for task: $task"

        for model_idx in "${!model_ids[@]}"; do
            model_id="${model_ids[$model_idx]}"
            echo "Model $model_idx: $model_id"
            export model_id=$model_id

            JOB_NAME="hub_100eval_inhouse_model${model_idx}_task_${task}"
            OUTPUT_ARTIFACT_PATH="outputs/mass_eval/${JOB_NAME}/job_summary.json"
            COMMAND="python jobs/run_evaluation.py \
                --model_name $model_id \
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

run_exp



# Run selection to get model list

function generate_list() {
    # Input args: 
    # METHOD_NAME, K, METHOD_ARGS

    OUTPUT_FOLDER="outputs/inhouse_selection/${METHOD_NAME}_top${K}"
    mkdir -p ${OUTPUT_FOLDER}

    for task in "${tasks[@]}"; do
        echo "Running for task: $task"
        OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${task}.txt"
        COMMAND="python jobs/run_hub_selection.py \
            --model_ids_file results/model_lists/inhouse_model_ids.txt \
            --output_file ${OUTPUT_ARTIFACT_PATH} \
            --num_selected ${K} \
            --include_target_model \
            --task ${task} \
            --target_model_path shared_space/lora_baseline/lora_baseline_lr3e-4_step400_rank64_${task}/final_model \
            ${METHOD_ARGS}"
        JOB_NAME="inhouse_selection_${METHOD_NAME}_top${K}_${task}"

        if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
            echo "Skipping ${JOB_NAME} - already done"
        elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
            echo "Skipping ${JOB_NAME} - already running"
        else
            echo $COMMAND
            echo "Submitting job: ${JOB_NAME}"
            # submit "${JOB_NAME}" "${COMMAND}"
            eval "${COMMAND}"
            echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
            if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                echo "Waiting for 5m due to high job count"
                sleep 5m
            fi
        fi
    done
}

export METHOD_NAME="evaluation"
export METHOD_ARGS="--selection_method evaluation --evaluation_csv_path last_section_results.csv --job_summary_dir outputs/mass_eval/"
generate_list



export K_VALUES=(5 10 20 30)
export K_SOURCE=50
function shorten_list() {
    # Input args: 
    # METHOD_NAME, K_VALUES, K_SOURCE
    SOURCE_FOLDER="outputs/inhouse_selection/${METHOD_NAME}_top${K_SOURCE}"


    for K in "${K_VALUES[@]}"; do
        OUTPUT_FOLDER="outputs/inhouse_selection/${METHOD_NAME}_top${K}"
        mkdir -p ${OUTPUT_FOLDER}
        echo "Shortening to top ${K} for method: ${METHOD_NAME}"
        # Take the first K lines from each file in OUTPUT_FOLDER with K_MAX
        for file in ${SOURCE_FOLDER}/*.txt; do
            base_filename=$(basename $file)
            head -n ${K} $file > ${OUTPUT_FOLDER}/${base_filename}
            echo "Generated ${OUTPUT_FOLDER}/${base_filename}"
            head -n ${K} $file | wc -l
        done
    done
}

export METHOD_NAME="evaluation"
shorten_list


export K_WINDOW=10
export K_SOURCE=50

function create_specific_skip() {
    # Interface: Pass the desired SKIP value as the first argument

    # Safety check: Ensure a skip value was provided
    if [ -z "$CURRENT_SKIP" ]; then
        echo "Error: Please provide a skip value. Usage: create_specific_skip <number>"
        return 1
    fi

    SOURCE_FOLDER="outputs/inhouse_selection/${METHOD_NAME}_top${K_SOURCE}"
    
    # Folder name includes the specific skip value
    OUTPUT_FOLDER="outputs/inhouse_selection/${METHOD_NAME}_skip${CURRENT_SKIP}_top${K_WINDOW}"
    
    mkdir -p "${OUTPUT_FOLDER}"
    echo "Processing ${METHOD_NAME} at Skip Offset ${CURRENT_SKIP}..."
    echo "Outputting to: ${OUTPUT_FOLDER}"

    for file in "${SOURCE_FOLDER}"/*.txt; do
        base_filename=$(basename "$file")
        
        # --- MATH LOGIC ---
        # Line 1 = Header
        # Line 2 = Data Index 0
        # If Skip = 0, we want Data Index 0 -> Start at Line 2
        # If Skip = 5, we want Data Index 5 -> Start at Line 7
        
        sed_start=$((CURRENT_SKIP + 2))
        sed_end=$((CURRENT_SKIP + K_WINDOW))
        
        # Run sed: 
        # 1. Print header ('1p')
        # 2. Print the specific window range
        sed -n -e '1p' -e "${sed_start},${sed_end}p" "$file" > "${OUTPUT_FOLDER}/${base_filename}"
    done
}

for CURRENT_SKIP in 5 10 15 20; do
    create_specific_skip
done


export STEP=100
export DATA_SIZE=100
export LR_OPTIONS=(5e-2)


rm exps_to_check.txt
function run_exp() {
    # Input args: 
    # SELECTION_METHOD, TASKS, EXTRA_ARGS
    for task in ${tasks[@]}; do
        echo "Running for task: $task"
        INPUT_EXPERT_LIST="outputs/inhouse_selection/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="outputs/lorahub/${SELECTION_METHOD}_${VARIANT_NAME}"
        for lr in "${LR_OPTIONS[@]}"; do
            JOB_NAME="lorahub_${SELECTION_METHOD}_${VARIANT_NAME}_lr${lr}_${task}"
            COMMAND="python jobs/run_tuning.py \
                    --peft_type lorahub \
                    --expert_list_path $INPUT_EXPERT_LIST \
                    --task $task \
                    --batch_size 1 \
                    --lr $lr \
                    --data_size $DATA_SIZE \
                    --step $STEP \
                    --use_flash_attention \
                    --log_and_save_step 10 \
                    --save_model \
                    --output_dir $OUTPUT_DIR \
                    --run_name=${JOB_NAME} \
                    ${EXTRA_ARGS}
                    "

            OUTPUT_ARTIFACT_FILE="${OUTPUT_DIR}/${JOB_NAME}/job_summary.json"
            if [ -f "${OUTPUT_ARTIFACT_FILE}" ]; then
                echo "Skipping ${JOB_NAME} - already done"
            elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                echo "Skipping ${JOB_NAME} - already running"
            else
                echo $JOB_NAME
                echo $COMMAND
                submit "${JOB_NAME}" "${COMMAND}"
                echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
                if [ $(squeue -u $USER | wc -l) -gt 500 ]; then
                    echo "Waiting for 5m due to high job count"
                    sleep 5m
                fi
            fi
        done
        # if [ -d "${EXPERT_MERGED_PATH}" ]; then
        #     rm -rf $EXPERT_MERGED_PATH
        # fi
    done
}

export VARIANT_NAME="just_inhouse"
export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0  --router_activation=leaky_relu"
for SELECTION in "evaluation"; do
    # for K in 5 10 20 30; do
    for K in 10; do
        export SELECTION_METHOD="${SELECTION}_top${K}"
        run_exp
    done
done

export K=10
for SELECTION in "evaluation"; do
    for CURRENT_SKIP in 5 10 15 20; do
        export SELECTION_METHOD="${SELECTION}_skip${CURRENT_SKIP}_top${K}"
        run_exp
    done
done

