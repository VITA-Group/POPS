$env:CONDA_DEFAULT_ENV = "pops_attack"
$env:CONDA_PREFIX = "C:\Users\Zohar\miniconda3\envs\pops_attack"
$env:PATH = "C:\Users\Zohar\miniconda3\envs\pops_attack;C:\Users\Zohar\miniconda3\envs\pops_attack\Scripts;C:\Users\Zohar\miniconda3\envs\pops_attack\Library\bin;" + $env:PATH
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = $repoRoot.Path

Set-Location $repoRoot.Path

& "C:\Users\Zohar\miniconda3\envs\pops_attack\python.exe" finetune.py --model_id llava-hf/llava-1.5-7b-hf --data_dir data/MLLMU-Bench/ft_Data/train-00000-of-00001.parquet --save_dir models/vanilla/llava-vanilla-lite --batch_size 1 --lr 1e-4 --num_epochs 3 --gradient_accumulation False --lora_rank 4
