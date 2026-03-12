#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="Qwen/Qwen3-0.6B"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}-ace-distill"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --data_dir processed_data/ace/qwen/"
OPTS+=" --dataset_name ace"
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --model_type qwen"
OPTS+=" --output_dir ${OUTPUT_DIR}"

OPTS+=" --bf16"
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "
OPTS+=" --tokenizer Qwen/Qwen3-0.6B"

python run_eval.py ${OPTS}
# python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log
