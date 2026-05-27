source experiments/task_info.sh

export STEP=400
export DATA_SIZE=-100
# export LR_OPTIONS=(5e-2 1e-2 2e-3)
export LR_OPTIONS=(1e-2)
export TOP_K_OPTIONS=(4)


rm -f exps_to_check.txt
function run_exp() {
    for task in ${tasks[@]}; do
        echo "Running for task: $task"
        INPUT_EXPERT_LIST="shared_space/hub_selection/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="outputs/arrow/${SELECTION_METHOD}_${VARIANT_NAME}"

        for top_k in "${TOP_K_OPTIONS[@]}"; do
            for lr in "${LR_OPTIONS[@]}"; do
                JOB_NAME="arrow_${SELECTION_METHOD}_${VARIANT_NAME}_topk${top_k}_lr${lr}_${task}"
                COMMAND="python jobs/run_tuning.py \
                        --peft_type arrow \
                        --expert_list_path $INPUT_EXPERT_LIST \
                        --task $task \
                        --batch_size 1 \
                        --effective_batch_size 8 \
                        --lr $lr \
                        --data_size $DATA_SIZE \
                        --step $STEP \
                        --use_flash_attention \
                        --log_and_save_step 10 \
                        --save_model \
                        --output_dir $OUTPUT_DIR \
                        --run_name=${JOB_NAME} \
                        --top_k $top_k \
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
                    if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                        echo "Waiting for 5m due to high job count"
                        sleep 5m
                    fi
                fi
            done
        done
    done
}


########## Llama experiments ##########
SELECTION_METHODS=("evaluation_top30" "evaluation_top30_wo_target" "random_top30" "randomseed0_top30" "randomseed1_top30" "random_top30_wo_target" "randomseed0_top30_wo_target" "randomseed1_top30_wo_target")

# Zero-step
export LR_OPTIONS=(1e-6)
export STEP=0
export EXTRA_ARGS=""
for SELECTION_METHOD in "${SELECTION_METHODS[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="zerostep"
    run_exp
done

# Standard (trained)
export LR_OPTIONS=(1e-2)
export STEP=400
export EXTRA_ARGS=""
for SELECTION_METHOD in "${SELECTION_METHODS[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="standard"
    run_exp
done

########## Qwen experiments ##########
QWEN_SELECTION_METHODS=("evaluation_top30" "evaluation_top30_wo_target" "random_top30" "randomseed1_top30" "randomseed2_top30" "random_top30_wo_target" "randomseed1_top30_wo_target" "randomseed2_top30_wo_target")

function run_qwen_exp() {
    for task in ${tasks[@]}; do
        echo "Running Qwen for task: $task"
        INPUT_EXPERT_LIST="shared_space/hub_selection_qwen/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="outputs/arrow_qwen/${SELECTION_METHOD}_${VARIANT_NAME}"

        for top_k in "${TOP_K_OPTIONS[@]}"; do
            for lr in "${LR_OPTIONS[@]}"; do
                JOB_NAME="arrow_qwen_${SELECTION_METHOD}_${VARIANT_NAME}_topk${top_k}_lr${lr}_${task}"
                COMMAND="python jobs/run_tuning.py \
                        --peft_type arrow \
                        --model_name Qwen/Qwen3-4B-Instruct-2507 \
                        --expert_list_path $INPUT_EXPERT_LIST \
                        --task $task \
                        --batch_size 1 \
                        --effective_batch_size 8 \
                        --lr $lr \
                        --data_size $DATA_SIZE \
                        --step $STEP \
                        --use_flash_attention \
                        --log_and_save_step 10 \
                        --save_model \
                        --output_dir $OUTPUT_DIR \
                        --run_name=${JOB_NAME} \
                        --top_k $top_k \
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
                    if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                        echo "Waiting for 5m due to high job count"
                        sleep 5m
                    fi
                fi
            done
        done
    done
}

# Zero-step
export LR_OPTIONS=(1e-6)
export STEP=0
export EXTRA_ARGS=""
for SELECTION_METHOD in "${QWEN_SELECTION_METHODS[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="zerostep"
    run_qwen_exp
done

# Standard (trained)
export LR_OPTIONS=(1e-2)
export STEP=400
export EXTRA_ARGS=""
for SELECTION_METHOD in "${QWEN_SELECTION_METHODS[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="standard"
    run_qwen_exp
done

########## Llama 10-sample experiments ##########
SELECTION_METHODS_10=("evaluation_top30" "evaluation_top30_wo_target" "random_top30" "randomseed0_top30" "randomseed1_top30" "random_top30_wo_target" "randomseed0_top30_wo_target" "randomseed1_top30_wo_target")

function run_10sample_exp() {
    for task in ${tasks[@]}; do
        echo "Running 10-sample for task: $task"
        INPUT_EXPERT_LIST="/home/haokun/scratch/code/moose/shared_space/hub_selection_10/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="outputs/arrow_10sample/${SELECTION_METHOD}_${VARIANT_NAME}"

        for top_k in "${TOP_K_OPTIONS[@]}"; do
            for lr in "${LR_OPTIONS[@]}"; do
                JOB_NAME="arrow_10s_${SELECTION_METHOD}_${VARIANT_NAME}_topk${top_k}_lr${lr}_${task}"
                COMMAND="python jobs/run_tuning.py \
                        --peft_type arrow \
                        --expert_list_path $INPUT_EXPERT_LIST \
                        --task $task \
                        --batch_size 1 \
                        --effective_batch_size 8 \
                        --lr $lr \
                        --data_size -10 \
                        --combine_train_valid \
                        --step $STEP \
                        --use_flash_attention \
                        --log_and_save_step 10 \
                        --save_model \
                        --output_dir $OUTPUT_DIR \
                        --run_name=${JOB_NAME} \
                        --top_k $top_k \
                        --eval_split train+valid,test \
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
                    if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                        echo "Waiting for 5m due to high job count"
                        sleep 5m
                    fi
                fi
            done
        done
    done
}

# Zero-step
export LR_OPTIONS=(1e-6)
export STEP=0
export EXTRA_ARGS=""
for SELECTION_METHOD in "${SELECTION_METHODS_10[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="zerostep"
    run_10sample_exp
done

# Standard (trained)
export LR_OPTIONS=(1e-2)
export STEP=100
export EXTRA_ARGS=""
for SELECTION_METHOD in "${SELECTION_METHODS_10[@]}"; do
    export SELECTION_METHOD
    export VARIANT_NAME="standard"
    run_10sample_exp
done
