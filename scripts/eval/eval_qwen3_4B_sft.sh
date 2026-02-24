#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

# OPTS+=" --data_dir processed_data/qwen/"
# OPTS+=" --dataset_name in_domain"
OPTS+=" --val_batch_size 32"

# devices
OPTS+=" --student_device cuda:1"

# models
OPTS+=" --model_type qwen"
OPTS+=" --output_dir ${OUTPUT_DIR}"

OPTS+=" --bf16"
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
OPTS+=" --lora_path results/qwen3/sft_4B/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/490"
OPTS+=" --tokenizer VoCuc/Qwen3-4B-ace-sft"

python run_eval.py ${OPTS}
# python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log
