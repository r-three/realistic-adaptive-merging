source experiments/task_info.sh

# full HP search
# lr_options=(5e-5 1e-4 3e-4)
# step_options=(100 400)
# rank_options=(16 32 64 128)

# debug run
lr_options=(5e-5)
step_options=(100)
rank_options=(32)
task_ids=(0)

# Submitting jobs and saving job names to a file for later checking
rm exps_to_check.txt
for idx in "${task_ids[@]}"; do
# for idx in "${!datasets[@]}"; do
    for lr in "${lr_options[@]}"; do
        for step in "${step_options[@]}"; do
            for rank in "${rank_options[@]}"; do
                task=${tasks[$idx]}
                echo "Running for task $idx, lr $lr, step $step, rank $rank"
                JOB_NAME="lora_baseline_lr${lr}_step${step}_rank${rank}_task${task}"
                OUTPUT_ARTIFACT_PATH="outputs/${JOB_NAME}/job_summary.json"
                COMMAND="python jobs/run_tuning.py \
                    --run_name $JOB_NAME \
                    --task $task \
                    --batch_size 1 \
                    --lr $lr \
                    --data_size 100 \
                    --step $step \
                    --rank $rank \
                    --use_flash_attention \
                    --router_activation linear \
                    --log_and_save_step 10 \
                    --peft_type lora \
                    --save_model \
                    --ds_config 'ds_configs/zero2_config.json'"
                
                if [ -f "${OUTPUT_ARTIFACT_PATH}" ]; then
                    echo "Skipping ${JOB_NAME} - already done"
                elif [ $(squeue -u $USER -n "${JOB_NAME}" | wc -l) -gt 1 ]; then
                    echo "Skipping ${JOB_NAME} - already running"
                else
                    echo $COMMAND
                    submit "${JOB_NAME}" "${COMMAND}"
                    echo -e "${JOB_NAME} :::: ${COMMAND}" >> exps_to_check.txt
                fi
            done
        done
    done
done