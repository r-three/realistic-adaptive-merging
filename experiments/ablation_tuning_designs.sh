source experiments/task_info.sh

export step=100
export data_size=100
export num_expert_options=(2 5 10 20)
export lr_options=(1e-1 5e-2 1e-2)


export task_ids=(47 26 36 57 50 40 25 0 13 60 54 55 35 49 41 37 7 17 33 32) # NOTE: replace 14 with cardiffnlp/tweet_eval -> 57
export sample_method='oracle'

rm exps_to_check.txt
function run_exp() {
    for i in ${task_ids[@]}; do
        dataset=${datasets[$i]}
        subset=${subsets[$i]}
        task_key=${task_keys[$i]}
        echo "Running for dataset: $dataset, subset: $subset"

        for num_experts in "${num_expert_options[@]}"; do
            # export WANDB_PROJECT="lorahub_expert${num_experts}_${sample_method}_20_sweep"
            OUTPUT_DIR="outputs/lorahub_weights"
            EXPERT_MERGED_PATH="outputs/lorahub_input/${sample_method}/${task_key}/expert_num${num_experts}"
            # if output artifact does not exist, then do:
            echo "checking $EXPERT_MERGED_PATH"
            if [ ! -d "${EXPERT_MERGED_PATH}" ]; then
                echo "doesn't exist!"
                JOB_NAME="load_experts_${sample_method}_${task_key}_${num_experts}expert"
                COMMAND="python jobs/lorahub_load_experts.py \
                    --hf_expert_list_dir 'shared_space/hub_selection' \
                    --base_model_name 'meta-llama/Llama-3.1-8B-Instruct' \
                    --sample_method $sample_method \
                    --num_experts $num_experts \
                    --save_dir 'outputs/lorahub_input' \
                    --task $subset \
                    --task_key $task_key"
                OUTPUT_ARTIFACT_PATH=${EXPERT_MERGED_PATH}
                if [ -d "${OUTPUT_ARTIFACT_PATH}" ]; then
                    echo "Skipping ${JOB_NAME} - already done"
                elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                    echo "Skipping ${JOB_NAME} - already running"
                else
                    echo $COMMAND
                    submit "${JOB_NAME}" "${COMMAND}"
                    echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
                    if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                        echo "Waiting for 5m due to high job count"
                        sleep 5m
                    fi
                fi
                echo "Skipping tuning, waiting for load to finish"
                continue
            fi

            for lr in "${lr_options[@]}"; do
                JOB_NAME="lorahub_${VARIANT_NAME}_${sample_method}_${task_key}_${num_experts}expert_lr${lr}"
                COMMAND="python moerging_methods/lorahub/step2_run_tuning.py \
                        --model_type lorahub \
                        --sample_method $sample_method \
                        --expert_info_dir $EXPERT_MERGED_PATH \
                        --num_experts $num_experts \
                        --task $subset \
                        --batch_size 1 \
                        --effective_batch_size 8 \
                        --lr $lr \
                        --data_size $data_size \
                        --step $step \
                        --seed 123 \
                        --use_flash_attention \
                        --log_and_save_step 1 \
                        --save_model \
                        --output_dir $OUTPUT_DIR \
                        --run_name=${JOB_NAME} \
                        --moose_directory_path=$(pwd) \
                        ${EXTRA_ARGS}
                        "

                OUTPUT_ARTIFACT_FILE="${OUTPUT_DIR}/${JOB_NAME}/job_summary.json"
                if [ -f "${OUTPUT_ARTIFACT_FILE}" ]; then
                    echo "Skipping ${JOB_NAME} - already done"
                elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                    echo "Skipping ${JOB_NAME} - already running"
                else
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
    done
}

for retry in {1..20}; do
    export task_ids=(47 26 36 57 50 40 25 0 13 60 54 55 35 49 41 37 7 17 33 32) # NOTE: replace 14 with cardiffnlp/tweet_eval -> 57
    export VARIANT_NAME="standard" # (init uniform)
    export EXTRA_ARGS="--router_weight_init_method=uniform --router_activation=squared"
    run_exp
    export VARIANT_NAME="activation_linear" # (n/a)
    export EXTRA_ARGS="--router_weight_init_method=uniform --router_activation=real_linear"
    run_exp
    export VARIANT_NAME="activation_softmax" # (n/a)
    export EXTRA_ARGS="--router_weight_init_method=uniform --router_activation=softmax"
    run_exp
    export VARIANT_NAME="activation_sigmoid" # (n/a)
    export EXTRA_ARGS="--router_weight_init_method=uniform --router_activation=sigmoid"
    run_exp
    export VARIANT_NAME="init_normal" # (init normal)
    export EXTRA_ARGS="--router_weight_init_method=normal --router_activation=squared"
    run_exp
    export VARIANT_NAME="init_equal" # (new standard)
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=squared"
    run_exp
    export VARIANT_NAME="init_target" # (init target)
    export EXTRA_ARGS="--router_weight_init_method=target --router_activation=squared"
    run_exp
    export VARIANT_NAME="activation_linear_equal" # (activation linear)
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=real_linear"
    run_exp
    export VARIANT_NAME="activation_softmax_equal" # (activation softmax)
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=softmax"
    run_exp
    export VARIANT_NAME="activation_sigmoid_equal" # (activation sigmoid)
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=sigmoid"
    run_exp
    export VARIANT_NAME="per_expert"
    export EXTRA_ARGS="--weight_granularity=per_expert"
    run_exp
    export VARIANT_NAME="per_layer"
    export EXTRA_ARGS="--weight_granularity=per_layer"
    run_exp
    export VARIANT_NAME="per_sublayer"
    export EXTRA_ARGS="--weight_granularity=per_sublayer"
    run_exp
    export VARIANT_NAME="per_module" # (standard, sanity check)
    export EXTRA_ARGS="--weight_granularity=per_module"
    run_exp
    export VARIANT_NAME="per_dimension"
    export EXTRA_ARGS="--weight_granularity=per_dimension"
    run_exp
    
    export task_ids=(0 1 2 3 4 5 6 7 8 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)
    export VARIANT_NAME="pi_tuning"
    export EXTRA_ARGS="--pi_tuning --router_activation=softmax"
    run_exp
    export VARIANT_NAME="init_equal" # (new standard)
    export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=squared"
    run_exp


    if [ $(squeue -u $USER | wc -l) -le 1 ]; then
        echo "All done!"
        break
    else
        echo "Waiting for 30m before next check"
        sleep 30m
    fi
done
