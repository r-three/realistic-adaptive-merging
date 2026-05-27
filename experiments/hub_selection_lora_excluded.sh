source experiments/task_info.sh
k_values=(2 5 10 20 50)
best_lora_path="~/projects/aip-craffel/moose/best_lora"

function generate_list() {
    for K in "${k_values[@]}"; do
        OUTPUT_FOLDER=~/projects/aip-craffel/moose/hub_selection/${METHOD_NAME}_lora_excluded_top${K}
        mkdir -p ${OUTPUT_FOLDER}
        for idx in "${!datasets[@]}"; do
            DATASET=${datasets[$idx]}
            SUBSET=${subsets[$idx]}
            COMMA_NAME=$(echo "$DATASET,,${SUBSET}" | tr '/' ',')
            OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${COMMA_NAME}.txt"
            COMMAND="python jobs/run_hub_selection.py \
                --model_ids_file weight_processing/refiltered_model_ids.txt \
                --output_file ${OUTPUT_ARTIFACT_PATH} \
                --num_selected ${K} \
                --dataset_name ${DATASET} \
                --task ${SUBSET} \
                --target_model_path /home/jayje/shared_space/best_lora/${COMMA_NAME}/final_model \
                ${METHOD_ARGS}"
            JOB_NAME="hub_selection_${METHOD_NAME}_top${K}_task${idx}"
            if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
                echo "Skipping ${JOB_NAME} - already done"
            elif [ $(squeue -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                echo "Skipping ${JOB_NAME} - already running"
            else
                echo $COMMAND
                # submit "${JOB_NAME}" "${COMMAND}"
                eval $COMMAND
            fi            
        done
    done
}

# Evaluation
#METHOD_NAME="evaluation"
#METHOD_ARGS="--selection_method evaluation --evaluation_csv_path evaluation_results_cache.csv"
#generate_list

# Random
#METHOD_NAME="random"
#METHOD_ARGS="--selection_method random"
#generate_list

# Abs # TODO: this need rerunning, but just exclude for now
#export METHOD_NAME="abs"
#export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant abs --cosine_aggregate macro"
#generate_list

# Clamp
export METHOD_NAME="clamp"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp --cosine_aggregate macro"
generate_list

# Quasi fim
export METHOD_NAME="quasi_fim"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant quasi_fim --cosine_aggregate macro"
generate_list
