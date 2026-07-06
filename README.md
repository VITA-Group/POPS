# POPS

Official code release for **POPS: Recovering Unlearned Multi-Modality Knowledge in MLLMs with Prompt-Optimized Parameter Shaking**.

POPS studies the robustness of multimodal machine unlearning. Machine unlearning is often evaluated by checking whether a model can still directly recall removed examples. Our work asks a stronger question: after unlearning, can residual multimodal knowledge still be reactivated by an adaptive adversary? We show that unlearned MLLMs can retain latent visual-textual associations, and that these associations can be exposed through prompt optimization and amplified through lightweight fine-tuning.

The attack has two main stages:

- **PromptSuffix optimization** searches for suffix prompts that elicit residual knowledge from an unlearned multimodal model using domain-similar OOD samples.
- **Parameter shaking** uses the model's own generated responses to fine-tune lightweight adapters, amplifying weak residual signals into stronger recovery behavior.

Together, these stages provide a stress test for multimodal unlearning methods and help evaluate whether supposedly removed knowledge is actually robustly erased or merely hidden from direct prompting.

## Repository Structure

```text
attack/          POPS attack pipeline, PromptSuffix optimization, S2L fine-tuning
baselines/       Unlearning baselines
configs/         Paper-facing attack configuration
data_process/    Dataset and collator utilities
experiments/     Ablations, defenses, and comparison experiments
scripts/         Dataset/model download helpers and experiment launchers
eval.py          Evaluation entry point
attack_eval.py   End-to-end POPS attack and evaluation entry point
finetune.py      Vanilla model fine-tuning entry point
```

## Setup

```bash
conda create -n pops_attack python=3.10
conda activate pops_attack
pip install -r requirements.txt
```

Prepare the data resources:

```bash
python scripts/download_dataset.py --download_splits
```

Optional model download helpers are provided in:

```bash
python scripts/download_models.py
```

## Running POPS

The main attack entry point is `attack_eval.py`. The paper-facing attack configuration is in:

```text
configs/attack_config.yaml
```

Scripts for preparing data, running unlearning baselines, launching POPS, and reproducing ablations are provided under `scripts/` and `experiments/`.

## Evaluation

`eval.py` contains the standalone evaluation pipeline used by the experiments. Factuality scoring with GPT-based judging is implemented separately in `eval_gpt.py`.

## Experiments

The repository includes scripts for:

- vanilla model fine-tuning,
- unlearning baselines,
- POPS attack evaluation,
- OOD ablations,
- GCG comparison,
- defense analysis,
- multi-seed statistical analysis.

See `scripts/` and `experiments/` for the corresponding launchers.

## Citation

```bibtex
@article{li2026pops,
  title={POPS: Recovering Unlearned Multi-Modality Knowledge in MLLMs with Prompt-Optimized Parameter Shaking},
  author={Li, Zhangheng and Hong, Junyuan and Zhu, Jianing and Eum, Sungmin and Hu, Shuowen and You, Suya and Wang, Zhangyang},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```

## Acknowledgements

This repository uses resources from MLLMU-Bench: https://github.com/franciscoliu/MLLMU-Bench.
