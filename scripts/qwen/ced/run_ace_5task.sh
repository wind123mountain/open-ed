#! /bin/bash
# Sequential CED baseline: LoRA + merge per task (SeqLoRA-with-merge).
# Task 0 starts from the base model; task t>0 starts from the merged model of task t-1.
# Usage: bash scripts/qwen/ced/run_ace_5task.sh [PERM]   (default PERM=0)

set -e

PERM=${1:-0}

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

ENV_BIN=$HOME/miniconda3/envs/mta/bin
export PATH=${ENV_BIN}:$PATH

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

BASE_PATH=.
BASE_MODEL="Qwen/Qwen3-0.6B"
CKPT_NAME="qwen3-0.6B"

# hp (LoRA config matches EventKD span_distillm scripts)
BATCH_SIZE=8
LR=0.0002
GRAD_ACC=2
EVAL_BATCH_SIZE=32
EPOCHS=5
MAX_LENGTH=768
SEED=42

RUN_ROOT="${BASE_PATH}/results/qwen3/ced/perm${PERM}"
INIT_MODEL=${BASE_MODEL}

for T in 0 1 2 3 4
do
    MASTER_PORT=66$(($RANDOM%90+10))
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                      --nnodes $NNODES \
                      --node_rank $NODE_RANK \
                      --master_addr $MASTER_ADDR \
                      --master_port $MASTER_PORT"

    DATA_DIR="${BASE_PATH}/processed_data/ace_perm${PERM}/${T}/qwen/"
    SAVE_PATH="${RUN_ROOT}/task${T}"

    OPTS=""
    OPTS+=" --base-path ${BASE_PATH}"
    OPTS+=" --model-path ${INIT_MODEL}"
    OPTS+=" --ckpt-name ${CKPT_NAME}"
    OPTS+=" --model-type qwen"
    OPTS+=" --n-gpu ${GPUS_PER_NODE}"
    OPTS+=" --data-dir ${DATA_DIR}"
    OPTS+=" --num-workers 0"
    OPTS+=" --dev-num -1"
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
    OPTS+=" --max-length ${MAX_LENGTH}"
    OPTS+=" --max-prompt-length 460"
    OPTS+=" --do-train"
    OPTS+=" --do-valid"
    OPTS+=" --eval-gen"
    OPTS+=" --save-interval -1"
    OPTS+=" --eval-interval -1"
    OPTS+=" --log-interval 20"
    OPTS+=" --mid-log-num -1"
    OPTS+=" --save ${SAVE_PATH}"
    OPTS+=" --seed ${SEED}"
    OPTS+=" --deepspeed"
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
    OPTS+=" --type lm"
    OPTS+=" --do-sample"
    OPTS+=" --top-k 0"
    OPTS+=" --top-p 0.95"
    OPTS+=" --temperature 0.5"
    OPTS+=" --peft lora"
    OPTS+=" --peft-lora-r 8"
    OPTS+=" --peft-lora-alpha 64"
    OPTS+=" --peft-lora-dropout 0.1"

    export NCCL_DEBUG=""
    export WANDB_DISABLED=True
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONPATH=${BASE_PATH}

    mkdir -p ${SAVE_PATH}
    echo "===== perm${PERM} task${T}: init=${INIT_MODEL} data=${DATA_DIR} ====="
    ${ENV_BIN}/torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} \
        > ${SAVE_PATH}/train.log 2>&1

    # the trainer appends a hyperparameter suffix to --save; adapter checkpoints
    # land at ${SAVE_PATH}/<hp-suffix>/<global_step>/ each epoch; take the last
    LAST_CKPT=$(find ${SAVE_PATH} -maxdepth 2 -type d -regextype posix-extended -regex ".*/[0-9]+" | sort -V | tail -1)
    if [ -z "${LAST_CKPT}" ]; then
        echo "no adapter checkpoint found under ${SAVE_PATH}, aborting"
        exit 1
    fi

    echo "===== perm${PERM} task${T}: merging adapter ${LAST_CKPT} ====="
    ${ENV_BIN}/python ${BASE_PATH}/tools/merge_lora.py \
        --base-model-path ${INIT_MODEL} \
        --peft-path ${LAST_CKPT} \
        --out ${SAVE_PATH}/merged \
        > ${SAVE_PATH}/merge.log 2>&1

    INIT_MODEL="${SAVE_PATH}/merged"
done

echo "CED SEQUENCE DONE perm${PERM}"
