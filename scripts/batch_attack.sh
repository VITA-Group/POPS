#!/bin/bash



MODEL_ID="llava-hf/llava-1.5-7b-hf"
VANILLA_MODEL_PATH="path/to/vanilla/model"
DATA_DIR="data/MLLMU-Bench"
CONFIG_PATH="configs/attack_config.yaml"
BASE_OUTPUT_DIR="attack_results"
FORGET_RATIO=10


UNLEARNING_METHODS=("GA" "GA_Difference" "KL_Min" "NPO")


for METHOD in "${UNLEARNING_METHODS[@]}"; do
    echo "========================================"
    echo "Attacking ${METHOD} unlearned model"
    echo "========================================"

    UNLEARNED_MODEL_PATH="baselines/${METHOD}_models/forget_${FORGET_RATIO}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${METHOD}_attack"


    if [ ! -d "${UNLEARNED_MODEL_PATH}" ]; then
        echo "Warning: Model path ${UNLEARNED_MODEL_PATH} does not exist. Skipping..."
        continue
    fi


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
        --output_file ${METHOD}_attack \
        --log_level INFO \
        --evaluate_stages unlearned full_attack

    echo "${METHOD} attack completed!"
    echo ""
done

echo "========================================"
echo "All attacks completed!"
echo "Results saved to ${BASE_OUTPUT_DIR}/"
echo "========================================"
