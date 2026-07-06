$env:CONDA_DEFAULT_ENV = "pops_attack"
$env:CONDA_PREFIX = "C:\Users\Zohar\miniconda3\envs\pops_attack"
$env:PATH = "C:\Users\Zohar\miniconda3\envs\pops_attack;C:\Users\Zohar\miniconda3\envs\pops_attack\Scripts;C:\Users\Zohar\miniconda3\envs\pops_attack\Library\bin;" + $env:PATH
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = $repoRoot.Path

Set-Location $repoRoot.Path

& "C:\Users\Zohar\miniconda3\envs\pops_attack\python.exe" baselines/GA.py --model_id llava-hf/llava-1.5-7b-hf --vanilla_dir models/vanilla/llava-vanilla --data_split_dir data/MLLMU-Bench --forget_split_ratio 5 --save_dir models/unlearned/GA_5pct --batch_size 1 --lr 2e-5 --num_epochs 1 --gradient_accumulation True
