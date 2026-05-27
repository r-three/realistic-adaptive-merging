# source experiments/diagnostic_task_info.sh
source experiments/task_info.sh

export STEP=100
export DATA_SIZE=100
export LR_OPTIONS=(5e-2)


# export SELECTION_METHODS=('from_5k_iid' 'from_5k_same_task' 'from_5k_same_task_plus10random')

# tasks=("xnli_en")

rm exps_to_check.txt
function run_exp() {
    # Input args: 
    # SELECTION_METHOD, TASKS, EXTRA_ARGS
    for task in ${tasks[@]}; do
        echo "Running for task: $task"
        INPUT_EXPERT_LIST="outputs/hub_selection/${SELECTION_METHOD}/${task}.txt"
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
                if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
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

for retry in {1..20}; do
    # Exp segment begin

    # Exp segment end
    if [ $(squeue -u $USER | wc -l) -eq 0 ]; then
        echo "All done!"
        break
    else
        echo "Waiting for 10m until next retry"
        sleep 10m
    fi
done

for SELECTION_METHOD in "${SELECTION_METHODS[@]}"; do
    echo "Running for sample method: $SELECTION_METHOD"
    export SELECTION_METHOD
    export VARIANT_NAME="standard"
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=squared"
    run_exp
done


export SELECTION_METHOD="from_5k_same_task"
export VARIANT_NAME="reinit_nontarget_experts"
export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=squared"
run_exp


export SELECTION_METHOD="from_5k_same_task"
export VARIANT_NAME="reinit_all_experts"
export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=squared"
run_exp



export VARIANT_NAME="init0_leakyrelu"
export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0  --router_activation=leaky_relu"
for SELECTION in "evaluation" "random" "abs" "clamp"; do
    for K in 5 10 20 30; do
        export SELECTION_METHOD="${SELECTION}_top${K}"
        run_exp
    done
done

export VARIANT_NAME="adamerge"
export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=linear"

export VARIANT_NAME="reinit_hub"
export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0  --router_activation=leaky_relu --reinit_case=non_target_experts"
for SELECTION in "evaluation" "random"; do
    for K in 5 10 20 30; do
        export SELECTION_METHOD="${SELECTION}_top${K}"
        run_exp
    done
done


for SEED in 0 42; do
    export VARIANT_NAME="reinit_hub_seed${SEED}"
    export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0  --router_activation=leaky_relu --reinit_case=non_target_experts --seed=${SEED}"
    for SELECTION in "evaluation" "random"; do
        for K in 5 10 20 30; do
            export SELECTION_METHOD="${SELECTION}_top${K}"
            run_exp
        done
    done
done

sleep 2h
export VARIANT_NAME="fixed_hubset"
export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0  --router_activation=leaky_relu"
for SEED in 0 1 2 3 4 5 6 7 8 9; do
    export SELECTION_METHOD="randomseed${SEED}_top30"
    run_exp
done

