# source experiments/diagnostic_task_info.sh
source experiments/task_info.sh



function generate_list() {
    # Input args: 
    # METHOD_NAME, K, METHOD_ARGS

    OUTPUT_FOLDER="outputs/hub_selection/${METHOD_NAME}_top${K}"
    mkdir -p ${OUTPUT_FOLDER}

    for task in "${tasks[@]}"; do
        echo "Running for task: $task"
        OUTPUT_ARTIFACT_PATH="${OUTPUT_FOLDER}/${task}.txt"
        COMMAND="python jobs/run_hub_selection.py \
            --model_ids_file results/model_lists/refiltered_model_ids.txt \
            --output_file ${OUTPUT_ARTIFACT_PATH} \
            --num_selected ${K} \
            --include_target_model \
            --task ${task} \
            --target_model_path shared_space/lora_baseline/lora_baseline_lr3e-4_step400_rank64_${task}/final_model \
            ${METHOD_ARGS}"
        JOB_NAME="hub_selection_${METHOD_NAME}_top${K}_${task}"

        if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
            echo "Skipping ${JOB_NAME} - already done"
        elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
            echo "Skipping ${JOB_NAME} - already running"
        else
            echo $COMMAND
            echo "Submitting job: ${JOB_NAME}"
            submit "${JOB_NAME}" "${COMMAND}"
            echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
            if [ $(squeue -u $USER | wc -l) -gt 200 ]; then
                echo "Waiting for 5m due to high job count"
                sleep 5m
            fi
        fi
    done
}




export K=50

# Evaluation
export METHOD_NAME="evaluation"
export METHOD_ARGS="--selection_method evaluation --evaluation_csv_path new_results.csv --job_summary_dir outputs/mass_eval/"
generate_list

# Random
export METHOD_NAME="random"
export METHOD_ARGS="--selection_method random"
generate_list

# Random seed 1
export METHOD_NAME="randomseed1"
export METHOD_ARGS="--selection_method random --random_seed 1"
generate_list

# Random seed 2
export METHOD_NAME="randomseed2"
export METHOD_ARGS="--selection_method random --random_seed 2"
generate_list

# Random seed 3-0
for seed in 3 4 5 6 7 8 9 0; do
    export METHOD_NAME="randomseed${seed}"
    export METHOD_ARGS="--selection_method random --random_seed ${seed}"
    generate_list
done

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

# Cosine
export METHOD_NAME="cosine"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant cosine --cosine_aggregate macro"
generate_list

# Clamp per module
export METHOD_NAME="clamp_pm"
export METHOD_ARGS="--selection_method closeness --closeness_metric cosine --cosine_variant clamp_pm --cosine_aggregate macro"
generate_list




export K_VALUES=(5 10 20 30)
export K_SOURCE=50
function shorten_list() {
    # Input args: 
    # METHOD_NAME, K_VALUES, K_SOURCE
    SOURCE_FOLDER="outputs/hub_selection/${METHOD_NAME}_top${K_SOURCE}"


    for K in "${K_VALUES[@]}"; do
        OUTPUT_FOLDER="outputs/hub_selection/${METHOD_NAME}_top${K}"
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

# Shorten lists
for METHOD_NAME in "abs" "clamp" "quasi_fim"; do
    shorten_list
done
