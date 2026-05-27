source experiments/task_info.sh
k_values=(20) # (2 5 10 20 50) 
best_lora_path="r-three" # THIS SHOULD BE THE HF PATH FOR TRAINED LORA

function generate_list() {
    for K in "${k_values[@]}"; do
        OUTPUT_FOLDER=outputs/hub_selection_test/${METHOD_NAME}_top${K}
        mkdir -p ${OUTPUT_FOLDER}
        for idx in "${!datasets[@]}"; do
            DATASET=${datasets[$idx]}
            SUBSET=${subsets[$idx]}
	    TASKKEY=${task_keys[$idx]}
            COMMA_NAME=$(echo "$DATASET,,${SUBSET}" | tr '/' ',')
            OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${COMMA_NAME}.txt"
            COMMAND="python jobs/run_hub_selection.py \
                --model_ids_file weight_processing/refiltered_model_ids.txt \
                --output_file ${OUTPUT_ARTIFACT_PATH} \
                --num_selected ${K} \
                --include_target_model \
                --task ${SUBSET} \
                --target_model_path ${best_lora_path}/moose-${TASKKEY} \
                ${METHOD_ARGS}"
            JOB_NAME="hub_selection_${METHOD_NAME}_top${K}_task${idx}"
            
	    $COMMAND

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
METHOD_NAME="evaluation"
METHOD_ARGS="--selection_method evaluation --evaluation_csv_path evaluation_results_cache.csv"
generate_list

# Random
#METHOD_NAME="random"
#METHOD_ARGS="--selection_method random"
#generate_list

# Abs
#export METHOD_NAME="abs"
#export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant abs --cosine_aggregate macro"
#generate_list

# Clamp
#export METHOD_NAME="clamp"
#export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp --cosine_aggregate macro"
#generate_list

# Quasi fim
#export METHOD_NAME="quasi_fim"
#export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant quasi_fim --cosine_aggregate macro"
#generate_list
