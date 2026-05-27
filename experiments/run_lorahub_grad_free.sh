
######################### RUN TRAINING ##############################

source experiments/task_info.sh
source compute_clusters/submit.sh
. $HOME/environments/moose_env.env

export step=100
export data_size=100
export include_best_lora="True"
export num_experts_options=(5 10 20 30)
export sample_method_options=("random" "randomseed1" "randomseed0") 
export exclude_tasks=("")
export h100_tasks=("gsm8k" "unit_conversion_si_conversion" "masakhanews_swa" "masakhanews_yor" "openmathinstruct_2" "intersect_geometry")

for task in "${tasks[@]}"; do

	skip=false
    for x in "${exclude_tasks[@]}"; do
		if [ "$task" == "$x" ]; then
	    	skip=true
	    	echo "Excluding task $task"
	    	break
		fi
    done
    if [ "$skip" == false ]; then
		echo "Running for task: $task"
    	for num_experts in ${num_experts_options[@]}; do	    
			for sample_method in ${sample_method_options[@]}; do
	   
				# check if the hub selection list exists
				expert_file_path="${PROJECT_PATH}/moose/hub_selection/${sample_method}_top${num_experts}/${task}.txt"
				if [ ! -f "${expert_file_path}" ]; then
					mkdir -p ${PROJECT_PATH}/moose/hub_selection/${sample_method}_top${num_experts}
					echo "r-three/lora_baseline_lr3e-4_step400_rank64_${task}" > ${expert_file_path} && \
					head -n -1 ${PROJECT_PATH}/moose/hub_selection/${sample_method}_top${num_experts}_wo_target/${task}.txt >> ${expert_file_path}
				fi


				# if include_best_experts is false, drop the first line
				if [ "$include_best_lora" == "False" ]; then
					expert_file_path="${PROJECT_PATH}/moose/hub_selection/${sample_method}_top${num_experts}_wo_target/${task}.txt"
				fi
				
				JOB_NAME="grad_free_${task}_${num_experts}exp_${step}iter_${data_size}data_${include_best_lora}bestexp_${sample_method}"
	        	JOB_COMPLETED_ARTIFACT="results/lorahub/lorahub_grad_free_${task}_${num_experts}expert_${step}iter_${data_size}datasize_${include_best_lora}bestexp_${sample_method}_result.json"
	        	COMMAND="python jobs/run_gradfree_lorahub.py \
				--num_experts $num_experts \
				--budget $step \
				--num_train_example $data_size \
				--sample_method $sample_method \
				--task $task \
				--expert_file_path ${expert_file_path}
				"

	        	if [ -f "${JOB_COMPLETED_ARTIFACT}" ]; then
		    		echo "Run already completed w/ ${JOB_COMPLETED_ARTIFACT}; skipping"
				elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
					echo "Skipping ${JOB_NAME} - already running"
				else
		    		echo "Running ${JOB_COMPLETED_ARTIFACT}"
		    		echo "${COMMAND}"

					# check if h100 needed
					is_h100_job=false
					for x in "${h100_tasks[@]}"; do
						if [ "$task" == "$x" ]; then
							is_h100_job=true
							echo "h100: ${JOB_NAME}"
							submit_h100 "${JOB_NAME}" "${COMMAND}"
							break
						fi
					done

					if [ "$is_h100_job" == false ]; then
						echo "l40: ${JOB_NAME}"
						submit "${JOB_NAME}" "${COMMAND}"
					fi

					if [ $(squeue -u $USER | wc -l) -gt 40 ]; then
						echo "Waiting for 15m due to high job count"
						sleep 15m
					fi
	        	fi
	    	done
        done
    fi
done
