source experiments/task_info.sh
k_values=(2 5 10 20 50)

rm exps_to_check.txt
function generate_list() {
    for K in "${k_values[@]}"; do
        OUTPUT_FOLDER=shared_space/hub_selection/${METHOD_NAME}_top${K}
        mkdir -p ${OUTPUT_FOLDER}
        # for idx in "${!datasets[@]}"; do
        for idx in "${task_ids[@]}"; do
            dataset=${datasets[$idx]}
            subset=${subsets[$idx]}
            comma_name=$(echo "$dataset,,${subset}" | tr '/' ',')
            OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${comma_name}.txt"
            COMMAND="python jobs/run_hub_selection.py \
                --model_ids_file weight_processing/refiltered_model_ids.txt \
                --output_file ${OUTPUT_ARTIFACT_PATH} \
                --num_selected ${K} \
                --include_target_model \
                --task ${subset} \
                --target_model_path shared_space/best_lora/${comma_name}/final_model \
                ${METHOD_ARGS}"
            JOB_NAME="hub_selection_${METHOD_NAME}_top${K}_task${idx}"
            if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
                echo "Skipping ${JOB_NAME} - already done"
            elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                echo "Skipping ${JOB_NAME} - already running"
            else
                echo $COMMAND
                submit "${JOB_NAME}" "${COMMAND}"
                echo -e "${JOB_NAME}" >> exps_to_check.txt
            fi            
        done
    done
}

# Evaluation
export METHOD_NAME="evaluation"
export METHOD_ARGS="--selection_method evaluation --evaluation_csv_path evaluation_results_cache.csv"
generate_list

# Random
export METHOD_NAME="random"
export METHOD_ARGS="--selection_method random"
generate_list

# Abs
export METHOD_NAME="abs"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant abs --cosine_aggregate macro"
generate_list

# Clamp
export METHOD_NAME="clamp"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp --cosine_aggregate macro"
generate_list

# Quasi fim
export METHOD_NAME="quasi_fim"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant quasi_fim --cosine_aggregate macro"
generate_list

# Clamp per module
export METHOD_NAME="clamp_pm"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp_pm --cosine_aggregate macro"
for retry in {0..20}; do
    echo "Retry round ${retry}"
    generate_list
    echo "Sleeping for 1 hour before next retry"
    sleep 1h
done
