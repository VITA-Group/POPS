#!/bin/bash



MODEL_ID="llava-hf/llava-1.5-7b-hf"
UNLEARNED_MODEL_PATH="path/to/unlearned/model"
VANILLA_MODEL_PATH="path/to/vanilla/model"
DATA_DIR="data/MLLMU-Bench"
CONFIG_PATH="configs/attack_config.yaml"
OUTPUT_DIR="attack_results"
FORGET_RATIO=10


mkdir -p ${OUTPUT_DIR}


python attack_eval.py \
    --model_id ${MODEL_ID} \
    --unlearned_model_path ${UNLEARNED_MODEL_PATH} \
    --vanilla_model_path ${VANILLA_MODEL_PATH} \
    --data_split_folder ${DATA_DIR} \
    --few_shot_data ${DATA_DIR}/Full_Set/train-00000-of-00001.parquet \
    --test_data ${DATA_DIR}/Test_Set \
    --celebrity_data ${DATA_DIR}/Retain_Set/train-00000-of-00001.parquet \
    --config_path ${CONFIG_PATH} \
    --forget_ratio ${FORGET_RATIO} \
    --run_attack \
    --output_folder ${OUTPUT_DIR} \
    --output_file pops_attack_${FORGET_RATIO}pct \
    --log_level INFO \
    --use_perplexity_selection \
    --evaluate_stages unlearned full_attack

echo "POPS attack completed! Results saved to ${OUTPUT_DIR}"
