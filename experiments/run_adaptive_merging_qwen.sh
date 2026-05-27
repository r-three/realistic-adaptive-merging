source experiments/task_info.sh
source compute_clusters/submit.sh
. $HOME/environments/moose_env.env

export STEP=100
export DATA_SIZE=100
export LR_OPTIONS=(5e-2)
export h100_tasks=("gsm8k" "unit_conversion_si_conversion" "masakhanews_swa" "masakhanews_yor" "openmathinstruct_2" "intersect_geometry" "elementary_math_qa_question_only")

#export tasks=("anli")
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

# rm exps_to_check.txt
function run_exp() {
    # Input args: 
    # SELECTION_METHOD, TASKS, EXTRA_ARGS
    for task in ${tasks[@]}; do #tasks_ablation
        echo "Running for task: $task"
		# TODO : jje expert list is different on vulcan for evaluation; excludes krisboro..
        INPUT_EXPERT_LIST="${SCRATCH}/code/moose/outputs/hub_selection_qwen/${SELECTION_METHOD}/${task}.txt"
        OUTPUT_DIR="${SCRATCH}/code/moose/outputs/qwen_ablation/${SELECTION_METHOD}_${VARIANT_NAME}"
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
                    --model_name $MODEL_NAME \
					--eval_split valid,test \
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
					# TODO: changed for vulcan/trillium
					TIME="02:30:00"
					submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
					submitted=true
				else
					# Does the task require H100?
					for x in "${h100_tasks[@]}"; do
						if [ "$task" == "$x" ]; then
							echo "h100: ${JOB_NAME}"
							# TODO: changed for vulcan/trillium
							TIME="03:00:00"
							submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
							submitted=true
							break
						fi
					done
				fi

				if [ "$submitted" == false ]; then
					echo "l40: ${JOB_NAME}"
					TIME="02:00:00"
					submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
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

# qwen random ablation
function run_all() {
	for level in "per_module"; do #"per_expert" "per_layer" "per_sublayer" "per_dimension"; do
		export is_h100_job=false
		#export VARIANT_NAME="init0_leakyrelu_${level}"
		#export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0 --router_activation=leaky_relu --weight_granularity=${level}"
		for SELECTION in "evaluation"; do #wait until evaluation done, and also do for other random seeds # NOTE: for reinit, do it for eval
    		for K in 5 10 20 30; do # 30 20 10 5; do
				for include_target in true false; do
					
					if [ "${include_target}" == true ]; then
						export SELECTION_METHOD="${SELECTION}_top${K}"
					else
						export SELECTION_METHOD="${SELECTION}_top${K}_wo_target"
					fi
					
					if [ "${SELECTION}" == "evaluation" ]; then
						# for evaluation-based selection, run both reinit and non-reinit
						for reinit in true false; do
							export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0 --router_activation=leaky_relu --weight_granularity=${level}"
							# CASE 1: Reinit.
							if [[ "${reinit}" == true && "${include_target}" == true ]]; then
								export VARIANT_NAME="init0_leakyrelu_${level}_reinit"
								EXTRA_ARGS="$EXTRA_ARGS --reinit_case=non_target_experts"
							elif [[ "${reinit}" == true && "${include_target}" == false ]]; then
								export VARIANT_NAME="init0_leakyrelu_${level}_reinit"
								EXTRA_ARGS="$EXTRA_ARGS --reinit_case=all_experts"
							# CASE 2: Eval.
							else
								export VARIANT_NAME="init0_leakyrelu_${level}"
							fi
							run_exp
						done
					else
						# CASE 3: Random.
						# non-evaluation selection (e.g. random): no reinit, just run w/ and w/o target
						export EXTRA_ARGS="--router_weight_init_method=target --force_routing=0.0 --router_activation=leaky_relu --weight_granularity=${level}"
						export VARIANT_NAME="init0_leakyrelu_${level}"
						run_exp
					fi
    			done
			done
		done
	done
}
run_all

function run_adamerge() {
	for level in "per_module"; do
		export VARIANT_NAME="adamerge_real_linear" #"init0_leakyrelu_${level}"
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
	for K in 20; do # 30
        export is_h100_job=true
		export SELECTION_METHOD="quasi_fim_top${K}"
		
		for include_target in true false; do
			export EXTRA_ARGS="--pi_tuning --router_activation=softmax --weight_granularity=per_module"
			
			if [ "${include_target}" == true ]; then
				EXTRA_ARGS="$EXTRA_ARGS --reinit_case=none"
				export VARIANT_NAME="pi_tuning_w_target"
			else
				export VARIANT_NAME="pi_tuning_wo_target"
			fi
			
			run_exp
		
		done
	done
}
run_pi_tuning
