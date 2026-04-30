#! /bin/bash

GPUS=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=.
CKPT_NAME="llama3.2-1B"
CKPT="meta-llama/Llama-3.2-1B"
TEACHER_CKPT_NAME="llama3.2-3B"
TEACHER_CKPT="meta-llama/Llama-3.2-3B-Instruct"
# data
DATA_DIR="${BASE_PATH}/processed_data/ace/llama/"
# hp
BATCH_SIZE=2
LR=0.0002
GRAD_ACC=1
EVAL_BATCH_SIZE=64
EPOCHS=5
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/llama/distillm_1B_3B_ace_csd"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
# Teacher LoRA path. Override via env: TEACHER_PEFT_PATH=<path> bash <script>
DEFAULT_TEACHER_SFT_DIR="${BASE_PATH}/results/llama/sft_3B_ace"
if [ -z "${TEACHER_PEFT_PATH}" ]; then
    TEACHER_PEFT_PATH=$(ls -d ${DEFAULT_TEACHER_SFT_DIR}/e*/[0-9]*/ 2>/dev/null \
        | awk -F/ '{print $0, $(NF-1)}' | sort -k2 -n | tail -1 | awk '{print $1}')
    TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH%/}"
fi
if [ -z "${TEACHER_PEFT_PATH}" ]; then
    echo "ERROR: TEACHER_PEFT_PATH not set and no SFT checkpoint found under ${DEFAULT_TEACHER_SFT_DIR}/e*/." >&2
    exit 1
fi
echo "Using teacher PEFT: ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --model-type llama"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 0.7"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 460"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type csd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 8"
OPTS+=" --peft-lora-alpha 64"
OPTS+=" --peft-lora-dropout 0.1"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
