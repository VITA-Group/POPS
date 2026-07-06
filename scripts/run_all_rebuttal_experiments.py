                                                                            

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_METHOD_DIRS = {
    "GA": "GA_10pct",
    "GA_Difference": "GA_Diff_10pct",
    "KL_Min": "KL_Min_10pct",
    "NPO": "NPO_10pct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run POPS attack/evaluation jobs for revised-paper MLLMU-Bench experiments."
    )
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--vanilla_dir", required=True)
    parser.add_argument("--unlearned_root", default="models/unlearned")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--config_path", default="configs/attack_config.yaml")
    parser.add_argument("--output_root", default="attack_results/rebuttal")
    parser.add_argument("--forget_ratio", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHOD_DIRS),
        choices=list(DEFAULT_METHOD_DIRS),
    )
    parser.add_argument(
        "--skip_attack",
        action="store_true",
        help="Evaluate existing attacked models instead of running POPS.",
    )
    return parser.parse_args()


def build_command(args, method, unlearned_path, output_dir):
    command = [
        sys.executable,
        "attack_eval.py",
        "--model_id",
        args.model_id,
        "--unlearned_model_path",
        str(unlearned_path),
        "--vanilla_model_path",
        args.vanilla_dir,
        "--data_split_folder",
        args.data_split_dir,
        "--few_shot_data",
        str(Path(args.data_split_dir) / "Full_Set" / "train-00000-of-00001.parquet"),
        "--test_data",
        str(Path(args.data_split_dir) / "Test_Set"),
        "--celebrity_data",
        str(Path(args.data_split_dir) / "Retain_Set" / "train-00000-of-00001.parquet"),
        "--config_path",
        args.config_path,
        "--forget_ratio",
        str(args.forget_ratio),
        "--output_folder",
        str(output_dir),
        "--output_file",
        f"{method}_{args.forget_ratio}pct_pops",
        "--log_level",
        "INFO",
        "--evaluate_stages",
        "unlearned",
        "full_attack",
    ]

    if not args.skip_attack:
        command.append("--run_attack")

    return command


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        default_dir = DEFAULT_METHOD_DIRS[method].replace("10", str(args.forget_ratio))
        unlearned_path = Path(args.unlearned_root) / default_dir
        output_dir = output_root / f"{method}_{args.forget_ratio}pct"

        if not unlearned_path.exists():
            print(f"[skip] {method}: missing unlearned model at {unlearned_path}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, method, unlearned_path, output_dir)

        print(f"[run] {method}: {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
