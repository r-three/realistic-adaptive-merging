
######################### RUN TRAINING ##############################

source experiments/task_info.sh
source compute_clusters/submit.sh
. $HOME/environments/moose_env.env

export step=100
export data_size=100
export include_best_lora="False"
export num_experts_options=(30) #(5 10 20 30)
export sample_method_options=("random") #"randomseed1" "randomseed0") 
export exclude_tasks=("")
export h100_tasks=("gsm8k" "unit_conversion_si_conversion" "masakhanews_swa" "masakhanews_yor" "openmathinstruct_2" "intersect_geometry")

#export tasks=("anli")
export FILE_SAVE_DIR="results/lorahub_qwen"
export BASE_MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

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
				expert_file_path="${SCRATCH}/code/moose/outputs/hub_selection_qwen/${sample_method}_top${num_experts}/${task}.txt"

				# if include_best_experts is false, drop the first line
				if [ "$include_best_lora" == "False" ]; then
					expert_file_path="${SCRATCH}/code/moose/outputs/hub_selection_qwen/${sample_method}_top${num_experts}_wo_target/${task}.txt"
				fi
				
				JOB_NAME="grad_free_${task}_${num_experts}exp_${step}iter_${data_size}data_${include_best_lora}bestexp_${sample_method}"
	        	JOB_COMPLETED_ARTIFACT="results/lorahub_qwen/lorahub_grad_free_${task}_${num_experts}expert_${step}iter_${data_size}datasize_${include_best_lora}bestexp_${sample_method}_result.json"
	        	COMMAND="python jobs/run_gradfree_lorahub.py \
				--num_experts $num_experts \
				--budget $step \
				--num_train_example $data_size \
				--sample_method $sample_method \
				--task $task \
				--expert_file_path ${expert_file_path} \
				--file_save_dir ${FILE_SAVE_DIR} \
				--model_name ${BASE_MODEL_NAME} \
				--eval_split train+valid \
				--test_split test \
				--combine_train_valid
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
							# TODO: changed for trillium
							# echo "h100: ${JOB_NAME}"
							TIME="02:00:00"
							submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
							break
						fi
					done

					if [ "$is_h100_job" == false ]; then
						#echo "l40: ${JOB_NAME}"
						TIME="01:00:00"
						submit "${JOB_NAME}" "${COMMAND}" "${TIME}"
					fi

					if [ $(squeue -u $USER | wc -l) -gt 70 ]; then
						echo "Waiting for 15m due to high job count"
						sleep 15m
					fi
	        	fi
	    	done
        done
    fi
done
