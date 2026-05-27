source experiments/task_info.sh
k_values=(2 5 10 20 50)

rm -f exps_to_check.txt

function generate_list() {
    for K in "${k_values[@]}"; do
        OUTPUT_FOLDER=shared_space/hub_selection_qwen/${METHOD_NAME}_top${K}
        mkdir -p ${OUTPUT_FOLDER}
        for task in "${tasks[@]}"; do
            OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${task}.txt"
            COMMAND="python jobs/run_hub_selection.py \
                --model_ids_file results/model_lists/qwen4b_model_ids.txt \
                --output_file ${OUTPUT_ARTIFACT_PATH} \
                --num_selected ${K} \
                --include_target_model \
                --task ${task} \
                --target_model_path r-three/lora_baseline_qwen_lr3e-4_step400_rank64_${task} \
                --job_summary_dir outputs/mass_eval/ \
                --evaluation_csv_path results/tables/evaluation_results_cache_qwen.csv \
                ${METHOD_ARGS}"
            JOB_NAME="hub_selection_qwen_${METHOD_NAME}_top${K}_${task}"
            if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
                echo "Skipping ${JOB_NAME} - already done"
            elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                echo "Skipping ${JOB_NAME} - already running"
            else
                echo $COMMAND
                submit "${JOB_NAME}" "${COMMAND}"
                # eval ${COMMAND}
                echo -e "${JOB_NAME}" >> exps_to_check.txt
            fi
        done
    done
}

k_values=(50)

# Evaluation
METHOD_NAME="evaluation"
METHOD_ARGS="--selection_method evaluation"
generate_list

# # Random
# METHOD_NAME="random"
# METHOD_ARGS="--selection_method random"
# generate_list

# # Abs
# METHOD_NAME="abs"
# METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant abs --cosine_aggregate macro"
# generate_list

# # Clamp
# METHOD_NAME="clamp"
# METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp --cosine_aggregate macro"
# generate_list

# Quasi fim
METHOD_NAME="quasi_fim"
METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant quasi_fim --cosine_aggregate macro"
generate_list

for retry in {1..5}; do
    echo "Retry round ${retry}"
    generate_list
    echo "Sleeping for 1 hour before next retry"
    sleep 1h
done

# Shorten top 50 lists to smaller K values
K_SOURCE=50
K_VALUES=(5 10 20 30)
function shorten_list() {
    SOURCE_FOLDER="shared_space/hub_selection_qwen/${METHOD_NAME}_top${K_SOURCE}"
    for K in "${K_VALUES[@]}"; do
        # With target (first K lines)
        OUTPUT_FOLDER="shared_space/hub_selection_qwen/${METHOD_NAME}_top${K}"
        mkdir -p ${OUTPUT_FOLDER}
        echo "Shortening to top ${K} for method: ${METHOD_NAME}"
        for file in ${SOURCE_FOLDER}/*.txt; do
            base_filename=$(basename $file)
            head -n ${K} $file > ${OUTPUT_FOLDER}/${base_filename}
            echo "Generated ${OUTPUT_FOLDER}/${base_filename}"
        done

        # Without target (skip first line, take K lines from the rest)
        OUTPUT_FOLDER_WO="shared_space/hub_selection_qwen/${METHOD_NAME}_top${K}_wo_target"
        mkdir -p ${OUTPUT_FOLDER_WO}
        echo "Shortening to top ${K} wo_target for method: ${METHOD_NAME}"
        for file in ${SOURCE_FOLDER}/*.txt; do
            base_filename=$(basename $file)
            tail -n +2 $file | head -n ${K} > ${OUTPUT_FOLDER_WO}/${base_filename}
            echo "Generated ${OUTPUT_FOLDER_WO}/${base_filename}"
        done
    done
}

for METHOD_NAME in "evaluation" "quasi_fim" "random"; do
    shorten_list
done
