# source experiments/diagnostic_task_info.sh
source experiments/task_info.sh
source compute_clusters/submit.sh
. $HOME/environments/moose_env.env

export STEP=100
#export DATA_SIZE=100
export LR_OPTIONS=(5e-2)
#export LR_OPTIONS=(1e-4 5e-5)
export h100_tasks=("gsm8k" "unit_conversion_si_conversion" "masakhanews_swa" "masakhanews_yor" "openmathinstruct_2" "intersect_geometry" "elementary_math_qa_question_only")

# export tasks=("anli")

# rm exps_to_check.txt
function run_exp() {
    # Input args: 
    # SELECTION_METHOD, TASKS, EXTRA_ARGS
    for task in ${tasks[@]}; do #tasks_ablation
        echo "Running for task: $task"
		# TODO : jje expert list is different on vulcan for evaluation; excludes krisboro..
        INPUT_EXPERT_LIST="${SCRATCH}/code/moose/outputs/hub_selection_10/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="${SCRATCH}/code/moose/outputs/10_sample_ablation/${SELECTION_METHOD}_${VARIANT_NAME}"
        for lr in "${LR_OPTIONS[@]}"; do
            JOB_NAME="lorahub_${SELECTION_METHOD}_${VARIANT_NAME}_lr${lr}_${task}"
            COMMAND="python jobs/run_tuning.py \
                    --peft_type lorahub \
                    --expert_list_path $INPUT_EXPERT_LIST \
                    --task $task \
                    --batch_size 1 \
                    --lr $lr \
                    --data_size $DATA_SIZE \
                    --combine_train_valid \
                    --step $STEP \
                    --use_flash_attention \
                    --log_and_save_step 10 \
                    --save_model \
                    --eval_split train+valid,test \
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

				# Is H100 required flag passed?
				submitted=false
				if [ "${is_h100_job}" == true ]; then
					# TODO: changed for vulcan
					submit_h100 "${JOB_NAME}" "${COMMAND}"
					submitted=true
				else
					# Does the task require H100?
					for x in "${h100_tasks[@]}"; do
						if [ "$task" == "$x" ]; then
							echo "h100: ${JOB_NAME}"
							# TODO: changed for vulcan
							submit_h100 "${JOB_NAME}" "${COMMAND}"
							submitted=true
							break
						fi
					done
				fi

				if [ "$submitted" == false ]; then
					echo "l40: ${JOB_NAME}"
					submit "${JOB_NAME}" "${COMMAND}"
				fi

                echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
                if [ $(squeue -u $USER | wc -l) -gt 60 ]; then
                    echo "Waiting for 15m due to high job count"
                    sleep 15m
                fi
            fi
        done
    done
}

# data 10 ablation
function run_all() {
	export DATA_SIZE=10
	for level in "per_module"; do #"per_expert" "per_layer" "per_sublayer" "per_dimension"; do
		# export is_h100_job=false
		# export VARIANT_NAME="init0_leakyrelu_${level}_10sample"
		# export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0 --router_activation=leaky_relu --weight_granularity=${level}"
		for SELECTION in "evaluation"; do #wait until evaluation done, and also do for other random seeds # NOTE: for reinit, do it for eval
    		for K in 30 20 10 5; do # 10 20 30; do
				for include_target in true false; do
					if [ "${include_target}" == true ]; then
						export SELECTION_METHOD="${SELECTION}_top${K}"
					else
						export SELECTION_METHOD="${SELECTION}_top${K}_wo_target"
					fi

					for reinit in true false; do
						export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0 --router_activation=leaky_relu --weight_granularity=${level}"

						if [[ "${reinit}" == true && "${include_target}" == true ]]; then
							export VARIANT_NAME="init0_leakyrelu_${level}_10sample_reinit"
							EXTRA_ARGS="$EXTRA_ARGS --reinit_case=non_target_experts"
						elif [[ "${reinit}" == true && "${include_target}" == false ]]; then
							export VARIANT_NAME="init0_leakyrelu_${level}_10sample_reinit"
							EXTRA_ARGS="$EXTRA_ARGS --reinit_case=all_experts"
						elif [ "${reinit}" == false ]; then
							export VARIANT_NAME="init0_leakyrelu_${level}_10sample"
						fi
						run_exp
    				
					done
				done
			done
		done
	done
}
#run_all


function run_adamerge() {
	export DATA_SIZE=10
	for level in "per_module"; do
		export VARIANT_NAME="adamerge_real_linear_10sample" #"init0_leakyrelu_${level}"
		export EXTRA_ARGS="--router_weight_init_method=equal_weight --router_activation=real_linear --weight_granularity=${level}"
		for SELECTION in "random"; do # "randomseed0" "randomseed1"; do #"evaluation"; do
    		for K in 30; do
				for include_target in true false; do
					if [ "${include_target}" == true ]; then
						export SELECTION_METHOD="${SELECTION}_top${K}"
					else
						export SELECTION_METHOD="${SELECTION}_top${K}_wo_target"
					fi
        			run_exp
				done
    		done
		done
	done
}
#run_adamerge


function run_pi_tuning() {
	export DATA_SIZE=10
	export LR_OPTIONS=(1e-4)
	for K in 20; do # 30
        export is_h100_job=true
		export SELECTION_METHOD="pi_tuning_top${K}"
		
		for include_target in true false; do
			export EXTRA_ARGS="--pi_tuning --router_activation=softmax --weight_granularity=per_module"
			
			if [ "${include_target}" == true ]; then
				EXTRA_ARGS="$EXTRA_ARGS --reinit_case=none"
				export VARIANT_NAME="w_target_10sample"
			else
				export VARIANT_NAME="wo_target_10sample"
			fi
			
			run_exp
		
		done
	done
}
run_pi_tuning
