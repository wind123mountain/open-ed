#! /bin/bash
# Continual Relation Extraction training script
# Usage: bash scripts/qwen/distillm_cl/v3/re_cl_qwen3_0.6B.sh <data_prefix> <perm> <run_name> [extra_opts]
# Example: bash scripts/qwen/distillm_cl/v3/re_cl_qwen3_0.6B.sh fewrel_wm_re_same_prefix 0 fewrel_kd05 "--kd-ratio 0.5"
#          bash scripts/qwen/distillm_cl/v3/re_cl_qwen3_0.6B.sh fewrel_wm_re_same_prefix 0 fewrel_augd "--augd"
#          bash scripts/qwen/distillm_cl/v3/re_cl_qwen3_0.6B.sh tacred_wm_re_random 0 tacred_random
#          bash scripts/qwen/distillm_cl/v3/re_cl_qwen3_0.6B.sh tacred_wm_re_random 0 tacred_random_adaptive "--kd-ratio 0.5 --kd-ratio-new 0.3 --epochs 5 --patience 2"

data_prefix=$1
perm=$2
run_name=${3:?Usage: $0 <data_prefix> <perm> <run_name> [extra_opts]}
EXTRA_OPTS="${4:-}"


export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python3 -c "import torch; print(f'Visible GPUs: {torch.cuda.device_count()}')"

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

# model
BASE_PATH=.
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
# hp
BATCH_SIZE=2
LR=0.0001
GRAD_ACC=8
EVAL_BATCH_SIZE=32
EPOCHS=3
# length
MAX_LENGTH=640
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/re/${run_name}_${perm}_0.6B_cl"
# seed
SEED=42

# CL distillation settings
NUM_TASKS=10
START_TASK=0

# Initial peft path
CURRENT_PEFT_PATH=""
if [ "${START_TASK}" -gt 0 ]; then
    PREV_TASK=$((START_TASK - 1))
    CURRENT_PEFT_PATH=$(ls -d ${SAVE_PATH}/${PREV_TASK}/[0-9]* 2>/dev/null | sort -V | tail -1)
    echo "Resuming from task ${START_TASK}, previous checkpoint: ${CURRENT_PEFT_PATH}"
fi

# Clean up previous run (skip if resuming)
if [ "${START_TASK}" -eq 0 ]; then
    rm -rf "${SAVE_PATH}"
fi

# Log all output
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

    DATA_DIR="${BASE_PATH}/processed_data/${data_prefix}_${perm}/${TASK_ID}/qwen/"

    OPTS=""
    # model
    OPTS+=" --base-path ${BASE_PATH}"
    OPTS+=" --model-path ${CKPT}"
    OPTS+=" --ckpt-name ${CKPT_NAME}"
    OPTS+=" --model-type qwen"
    OPTS+=" --n-gpu ${GPUS_PER_NODE}"
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
    OPTS+=" --max-prompt-length 512"
    OPTS+=" --t-max-prompt-length 768"
    OPTS+=" --t-max-length 768"
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
    OPTS+=" --kd-ratio 0.8"  # default; can be overridden via EXTRA_OPTS (4th arg)
    # Adaptive KD is opt-in via EXTRA_OPTS, e.g. --kd-ratio-new 0.3
    # seed
    OPTS+=" --seed ${SEED}"
    # lora
    OPTS+=" --peft lora"
    OPTS+=" --peft-lora-r 64"
    OPTS+=" --peft-lora-alpha 128"
    OPTS+=" --peft-lora-dropout 0.1"
    if [ -n "${CURRENT_PEFT_PATH}" ]; then
        OPTS+=" --peft-path ${CURRENT_PEFT_PATH}"
    fi
    # continual-learning settings
    OPTS+=" --cl-task-id ${TASK_ID}"
    OPTS+=" --cl-distill"
    # deepspeed
    OPTS+=" --deepspeed"
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
    # type
    OPTS+=" --type sfkl"
    # Note: eval uses greedy decoding (do_sample=False hardcoded in finetune_v2.py)
    # Sampling flags are NOT passed — they have no effect in CL mode

    export NCCL_DEBUG=""
    export WANDB_DISABLED=True
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONPATH=${BASE_PATH}
    CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_v2.py ${OPTS} ${EXTRA_OPTS}"

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
CSV_FILE="${SAVE_PATH}/eval_results.csv"
python ${BASE_PATH}/tools/parse_log_to_csv.py --log-dir "${SAVE_PATH}" --out "${CSV_FILE}"
