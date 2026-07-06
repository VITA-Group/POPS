@echo off
cd /d "%~dp0.."
call conda activate pops_attack
python baselines/KL_Min.py --model_id llava-hf/llava-1.5-7b-hf --vanilla_dir models/vanilla/llava-vanilla --data_split_dir data/MLLMU-Bench --forget_split_ratio 10 --save_dir models/unlearned/KL_Min_10pct --batch_size 1 --lr 1e-4 --num_epochs 1 --gradient_accumulation True
