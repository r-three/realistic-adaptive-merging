source compute_clusters/submit.sh
source experiments/task_info.sh
. $HOME/environments/moose_env.env

#tasks=("anli")
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
NUM_LAYERS=36

# Run merging
rm -f exps_to_check.txt
function run_exp() {
    # Input args: 
    # TASKS, MERGING_METHOD, EXTRA_ARGS; VARIANT
    for task in ${tasks[@]}; do
        echo "Running for task: $task"
        OUTPUT_DIR="${SCRATCH}/code/moose/outputs/baseline_qwen" # TODO: changed for qwen
		if [ -n "$VARIANT" ]; then
        	JOB_NAME="${MERGING_METHOD}_merging_${task}_${VARIANT}"
        else
			JOB_NAME="${MERGING_METHOD}_merging_${task}"
		fi

		EXPERT_FILE_PATH="${SCRATCH}/code/moose/outputs/hub_selection_qwen/${selection}_top30_wo_target/${task}.txt" # TODO: wo_target

		COMMAND="python jobs/run_merging.py \
					--run_name=${JOB_NAME} \
                    --output_dir ${OUTPUT_DIR} \
					--task $task \
					--merging_method ${MERGING_METHOD} \
					--expert_file_path ${EXPERT_FILE_PATH} \
					--model_name ${MODEL_NAME} \
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
			TIME="01:30:00"
            submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
            echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
            if [ $(squeue -u $USER | wc -l) -gt 40 ]; then
                echo "Waiting for 20m due to high job count"
                sleep 20m
            fi
        fi
    done
}

# Simple averaging
export MERGING_METHOD="aggregate"
export selection="random"
export VARIANT="${selection}_top30_wo_target" # wo_target
export EXTRA_ARGS="--aggregate_method avg"

# if variant "wo_target", reuse computed vector
if [[ "${VARIANT}" == *"wo_target"* ]]; then
	EXTRA_ARGS="${EXTRA_ARGS} --save_expert_task_vector --expert_${MERGING_METHOD}_state_dict_path outputs/data_free_task_vector/${MERGING_METHOD}_task_vector_${VARIANT}.pt"
fi
run_exp

# TIES
export MERGING_METHOD="ties"
export selection="random"
export VARIANT="${selection}_top30_wo_target_weight_frac_0.2" # wo_target
export EXTRA_ARGS="--weight_init equal_frac \
                  --majority_sign_method total \
                  --density 0.2"

# if variant "wo_target", reuse computed vector
if [[ "${VARIANT}" == *"wo_target"* ]]; then
	EXTRA_ARGS="${EXTRA_ARGS} --save_expert_task_vector --expert_${MERGING_METHOD}_state_dict_path outputs/data_free_task_vector/${MERGING_METHOD}_task_vector_${VARIANT}.pt"
fi
run_exp

# TSV
export MERGING_METHOD="tsv"
export selection="random"
export VARIANT="${selection}_top30_wo_target_weight_frac_reduced_8" # wo_target
export EXTRA_ARGS="--weight_init equal_frac \
				   		--tsv_reduced_size 8 \
				   		--num_chunks 2 \
				   		--num_base_model_layers ${NUM_LAYERS}"

# if variant "wo_target", reuse computed vector
if [[ "${VARIANT}" == *"wo_target"* ]]; then
	EXTRA_ARGS="${EXTRA_ARGS} --save_expert_task_vector --expert_${MERGING_METHOD}_state_dict_path outputs/data_free_task_vector/${MERGING_METHOD}_task_vector_${VARIANT}.pt"
fi
run_exp


# TSV
#export MERGING_METHOD="tsv"
#export selection="evaluation"
#export VARIANT="${selection}_top30_weight_frac_reduced_8" # wo_target
#export EXTRA_ARGS="--weight_init equal_frac \
#				   		--tsv_reduced_size 8 \
#				   		--num_chunks 2 \
#				   		--num_base_model_layers 36"
#run_exp

