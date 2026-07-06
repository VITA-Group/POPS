@echo off
cd /d "%~dp0.."
call conda activate pops_attack
python finetune.py --model_id llava-hf/llava-1.5-7b-hf --data_dir data/MLLMU-Bench/ft_Data/train-00000-of-00001.parquet --save_dir models/vanilla/llava-vanilla-lite --batch_size 1 --lr 1e-4 --num_epochs 3 --gradient_accumulation False --lora_rank 4
