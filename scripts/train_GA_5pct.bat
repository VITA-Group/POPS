@echo off
cd /d "%~dp0.."
call conda activate pops_attack
python baselines/GA.py --model_id llava-hf/llava-1.5-7b-hf --vanilla_dir models/vanilla/llava-vanilla --data_split_dir data/MLLMU-Bench --forget_split_ratio 5 --save_dir models/unlearned/GA_5pct --batch_size 2 --lr 2e-5 --num_epochs 1
