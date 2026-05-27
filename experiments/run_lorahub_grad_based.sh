
############################ Training code ############################

cd ~/moose/
source experiments/task_info.sh

export step=100
export data_size=100
export num_expert_options=(2 5 10) # jje: offload 2 to killarney
export lr_options=(1e-1 5e-2 1e-2)
export w_init_options=('uniform')
export activation_options=('softmax' 'linear')

# debug run
#export step=100
#export data_size=100
#export num_expert_options=(5)
#export lr_options=(1e-1) #(5e-2 1e-2)
#export w_init_options=('normal')
#export activation_options=('softmax')
#export per_module_options=('true')
#export include_coeff_options=('true')
#sample_method='oracle'

export exclude_tasks=(9 12 16 47 26 36 57 50 40 25 0 13 60 54 55 35 49 41 37 7 17 33 32) # starting 47 are the tasks for sweep
export sample_methods=('oracle' 'random')

## launch multiple jobs within this script
num_method=${#sample_methods[@]}
num_task_jobs=$(( SLURM_ARRAY_TASK_COUNT / num_method )) # num jobs to split the task ids into per method

# define sample method: random vs. oracle
method_idx=$(( SLURM_ARRAY_TASK_ID / num_task_jobs ))
sample_method=${sample_methods[$method_idx]}

# split the task ids
split_index=$(( SLURM_ARRAY_TASK_ID % num_task_jobs ))

chunk_size=$(( (${#task_ids[@]} + num_task_jobs - 1) / num_task_jobs ))
start=$(( split_index * chunk_size ))
end=$(( start + chunk_size - 1 ))

if [ "$end" -ge ${#task_ids[@]} ]; then
    end=$((${#task_ids[@]} - 1))
fi

new_task_ids=("${task_ids[@]:$start:$((end - start + 1))}") # slice

echo $sample_method
echo "${new_task_ids[@]}"

for i in ${new_task_ids[@]}; do
    dataset=${datasets[$i]}
    subset=${subsets[$i]}
    task_key=${task_keys[$i]}
    echo "Running for dataset: $dataset, subset: $subset"

    skip=false
    for x in "${exclude_tasks[@]}"; do
	if [ "$i" == "$x" ]; then
	    skip=true
	    echo "Excluding task num $i"
	    break
	fi
    done
    if [ "$skip" == true ]; then
	continue
    fi

    for num_experts in "${num_expert_options[@]}"; do
        export WANDB_PROJECT="lorahub_expert${num_experts}_${sample_method}_run_all"
    	OUTPUT_DIR="/home/jayje/projects/aip-craffel/moose/lorahub_weights/"
	EXPERT_MERGED_PATH="/home/jayje/projects/aip-craffel/moose/lorahub_input/${sample_method}/${task_key}/expert_num${num_experts}"
    	# if output artifact does not exist, then do:
    	echo $EXPERT_MERGED_PATH
    	if [ ! -d "${EXPERT_MERGED_PATH}" ]; then
	    echo "doesn't exist!"
	    python jobs/lorahub_load_experts.py \
    		--base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    		--sample_method $sample_method \
    		--num_experts $num_experts \
    		--save_dir "/home/jayje/projects/aip-craffel/moose/lorahub_input" \
		--task $subset \
		--task_key $task_key \
		--include_best_lora
    	fi

    	for lr in "${lr_options[@]}"; do
	    JOB_NAME="lorahub_${task_key}_${num_experts}expert_step${step}_lr${lr}_${sample_method}_datasize${data_size}"
	    cmd="python moerging_methods/lorahub/step2_run_tuning.py \
            	--model_type lorahub \
            	--sample_method $sample_method \
	   	--expert_info_dir $EXPERT_MERGED_PATH \
            	--num_experts $num_experts \
            	--task $subset \
            	--batch_size 1 \
            	--effective_batch_size 8 \
            	--lr $lr \
            	--data_size $data_size \
            	--step $step \
            	--seed 123 \
	    	--use_flash_attention \
            	--log_and_save_step 1 \
		--save_model \
		--output_dir $OUTPUT_DIR"

	    for w_init in "${w_init_options[@]}"; do
		cmd1="${cmd} --router_weight_init_method=${w_init}"
		JOB_NAME1="${JOB_NAME}_${w_init}"
		for act in "${activation_options[@]}"; do
		    cmd2="${cmd1} --router_activation=${act}"
		    JOB_NAME2="${JOB_NAME1}_${act}"
		    
		    # softmax: per-expert, a,b term
		    # e.g. lorahub_wic_2expert_step100_lr5e-2_oracle_datasize100_uniform_softmax_falsepermodule_truecoeff
		    if [ "$act" = "softmax" ]; then
		    	JOB_NAME_FIN="${JOB_NAME2}_falsepermodule_truecoeff"
		        cmd_fin="${cmd2} --include_activation_coeff"
		    # linear: per-module, no term
		    elif [ "$act" = "linear" ]; then
			JOB_NAME_FIN="${JOB_NAME2}_truepermodule_falsecoeff"
		        cmd_fin="${cmd2} --expert_weights_per_module"
		    fi
		
		    # put job name
		    cmd_fin="${cmd_fin} --run_name=${JOB_NAME_FIN}"

		    OUTPUT_ARTIFACT_FILE="${OUTPUT_DIR}/${JOB_NAME_FIN}/job_summary.json"
		    if [ -f "${OUTPUT_ARTIFACT_FILE}" ]; then
		        echo "Skipping ${JOB_NAME_FIN} - already done"
		    else
			echo $cmd_fin
			echo $JOB_NAME_FIN 
			$cmd_fin
		    fi
		done
	    done
        done
        #if [ -d "${EXPERT_MERGED_PATH}" ]; then
	#    rm -rf $EXPERT_MERGED_PATH
        #fi
    done
done
