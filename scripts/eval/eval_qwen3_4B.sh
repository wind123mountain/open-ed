#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="VoCuc/Qwen3-4B-ace-sft"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

# training
OPTS+=" --val_batch_size 32"

# devices
OPTS+=" --student_device cuda:1"

# models
OPTS+=" --model_type qwen"
OPTS+=" --output_dir ${OUTPUT_DIR}"

OPTS+=" --bf16"
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "
OPTS+=" --tokenizer VoCuc/Qwen3-4B-ace-sft"

python run_eval.py ${OPTS}
# python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log
