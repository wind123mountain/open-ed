#! /bin/bash

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

# model
BASE_PATH=.
CKPT_NAME="qwen3-4B"
CKPT="Qwen/Qwen3-4B-Instruct-2507"
# hp
BATCH_SIZE=1
LR=0.0001
GRAD_ACC=16
EVAL_BATCH_SIZE=32
EPOCHS=3
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/sft_4B_cl_v2"
# seed
SEED=42

# Initial peft path: empty = task 0 starts with a fresh LoRA on the base model.
# Set to an existing checkpoint path to resume from a prior SFT instead.
CURRENT_PEFT_PATH=""

# CL distillation settings
CL_DISTILL_COEF=0.5  # >0 enables CL distillation from frozen old-model snapshot

NUM_TASKS=5
START_TASK=0  # Change to resume from a later task (e.g. 1 if task 0 is done)

# Clean up previous run
rm -rf "${SAVE_PATH}"

# Log all output (stdout + stderr) to file while still printing to terminal
LOG_FILE_PATH="${SAVE_PATH}/full_run.log"
mkdir -p "$(dirname "${LOG_FILE_PATH}")"
exec > >(tee -a "${LOG_FILE_PATH}") 2>&1

for TASK_ID in $(seq ${START_TASK} $((NUM_TASKS - 1))); do

    MASTER_PORT=66$(($RANDOM%90+10))

    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                      --nnodes $NNODES \
                      --node_rank $NODE_RANK \
                      --master_addr $MASTER_ADDR \
                      --master_port $MASTER_PORT"

    DATA_DIR="${BASE_PATH}/processed_data/ace_v2/${TASK_ID}/qwen/"

    OPTS=""
    # model
    OPTS+=" --base-path ${BASE_PATH}"
    OPTS+=" --model-path ${CKPT}"
    OPTS+=" --ckpt-name ${CKPT_NAME}"
    OPTS+=" --model-type qwen"
    OPTS+=" --n-gpu ${GPUS_PER_NODE}"
    # OPTS+=" --gradient-checkpointing"
    # data
    OPTS+=" --data-dir ${DATA_DIR}"
    OPTS+=" --num-workers 0"
    OPTS+=" --dev-num -1"
    # hp
    OPTS+=" --lr ${LR}"
    OPTS+=" --batch-size ${BATCH_SIZE}"
    OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
    OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
    OPTS+=" --warmup-iters 0"
    OPTS+=" --warmup-ratio 0.1"
    OPTS+=" --lr-decay-style wrmup_cosine"
    OPTS+=" --weight-decay 1e-2"
    OPTS+=" --clip-grad 1.0"
    OPTS+=" --epochs ${EPOCHS}"
    # length
    OPTS+=" --max-length ${MAX_LENGTH}"
    OPTS+=" --max-prompt-length 460"
    # runtime
    OPTS+=" --do-train"
    OPTS+=" --do-valid"
    OPTS+=" --do-eval"
    OPTS+=" --eval-gen"
    OPTS+=" --save-interval -1"
    OPTS+=" --eval-interval -1"
    OPTS+=" --log-interval 10"
    OPTS+=" --mid-log-num -1"
    OPTS+=" --save ${SAVE_PATH}/${TASK_ID}"
    OPTS+=" --kd-ratio 0.7"
    # seed
    OPTS+=" --seed ${SEED}"
    # lora
    OPTS+=" --peft lora"
    OPTS+=" --peft-lora-r 16"
    OPTS+=" --peft-lora-alpha 64"
    OPTS+=" --peft-lora-dropout 0.1"
    if [ -n "${CURRENT_PEFT_PATH}" ]; then
        OPTS+=" --peft-path ${CURRENT_PEFT_PATH}"
    fi
    # continual-learning settings
    OPTS+=" --cl-task-id ${TASK_ID}"
    OPTS+=" --cl-distill-coef ${CL_DISTILL_COEF}"
    # deepspeed
    OPTS+=" --deepspeed"
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
    # type
    OPTS+=" --type fkl"
    # gen
    OPTS+=" --do-sample"
    OPTS+=" --top-k 0"
    OPTS+=" --top-p 0.95"
    OPTS+=" --temperature 0.5"

    export NCCL_DEBUG=""
    export WANDB_DISABLED=True
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONPATH=${BASE_PATH}
    CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_v2.py ${OPTS}"

    echo "=============================="
    echo "Task ${TASK_ID} / $((NUM_TASKS - 1))"
    echo "Data dir : ${DATA_DIR}"
    echo "PEFT init: ${CURRENT_PEFT_PATH}"
    echo ${CMD}
    echo "=============================="
    mkdir -p ${SAVE_PATH}

    ${CMD}

    if [ $? -ne 0 ]; then
        echo "ERROR: Task ${TASK_ID} failed. Aborting."
        exit 1
    fi

    # Locate the latest checkpoint saved for this task (highest step number directory)
    NEXT_PEFT_PATH=$(ls -d ${SAVE_PATH}/${TASK_ID}/[0-9]* 2>/dev/null | sort -V | tail -1)

    if [ -z "${NEXT_PEFT_PATH}" ]; then
        echo "ERROR: Could not find checkpoint for task ${TASK_ID} under ${SAVE_PATH}/${TASK_ID}/. Aborting."
        exit 1
    fi

    echo "Task ${TASK_ID} done. Checkpoint: ${NEXT_PEFT_PATH}"
    CURRENT_PEFT_PATH="${NEXT_PEFT_PATH}"

done

echo "=============================="
echo "All ${NUM_TASKS} tasks completed."
echo "Final checkpoint: ${CURRENT_PEFT_PATH}"
echo "=============================="

# Export all eval results to CSV
LOG_FILE="${SAVE_PATH}/${TASK_ID}/log.txt"
CSV_FILE="${SAVE_PATH}/${TASK_ID}/eval_results.csv"
python ${BASE_PATH}/tools/parse_log_to_csv.py --log "${LOG_FILE}" --out "${CSV_FILE}"