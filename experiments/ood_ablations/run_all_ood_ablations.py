\
\
\
   

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
from scipy import stats

                                
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_experiment(script_name, args_dict, seed):
                                                                      
    cmd = ["python", script_name]

    for key, value in args_dict.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    cmd.append("--seed")
    cmd.append(str(seed))

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None

    print(result.stdout)
    return result.returncode == 0


def load_metrics(output_dir):
                                                       
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compute_statistics(results_list):
                                                            
    test_accs = [r['test_acc'] for r in results_list]
    rouge_ls = [r['rouge_l'] for r in results_list]
    recovery_rates = [r['recovery_rate'] for r in results_list]

    stats_dict = {
        'test_acc': {
            'mean': np.mean(test_accs),
            'std': np.std(test_accs, ddof=1),
            'ci95': stats.t.interval(0.95, len(test_accs)-1,
                                    loc=np.mean(test_accs),
                                    scale=stats.sem(test_accs))
        },
        'rouge_l': {
            'mean': np.mean(rouge_ls),
            'std': np.std(rouge_ls, ddof=1),
            'ci95': stats.t.interval(0.95, len(rouge_ls)-1,
                                    loc=np.mean(rouge_ls),
                                    scale=stats.sem(rouge_ls))
        },
        'recovery_rate': {
            'mean': np.mean(recovery_rates),
            'std': np.std(recovery_rates, ddof=1),
            'ci95': stats.t.interval(0.95, len(recovery_rates)-1,
                                    loc=np.mean(recovery_rates),
                                    scale=stats.sem(recovery_rates))
        }
    }

    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="Run All OOD Baseline Ablations")
    parser.add_argument('--vanilla_dir', type=str, required=True,
                       help='Path to vanilla model')
    parser.add_argument('--unlearned_dir', type=str, required=True,
                       help='Path to unlearned (GA) model')
    parser.add_argument('--data_split_dir', type=str, required=True,
                       help='Path to data splits (forget/retain)')
    parser.add_argument('--output_base_dir', type=str,
                       default='results/ood_ablations',
                       help='Base output directory for all experiments')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 2024],
                       help='Random seeds for multi-seed experiments')
    args = parser.parse_args()

                             
    os.makedirs(args.output_base_dir, exist_ok=True)

                        
    experiments = [
        {
            'name': 'direct_ood_ft',
            'script': 'direct_ood_ft.py',
            'description': 'Direct Fine-Tuning on OOD (Retain Set)',
            'expected_recovery': 0.12
        },
        {
            'name': 's2l_ood',
            'script': 's2l_ood.py',
            'description': 'S2L on OOD (Retain Set Synthesis)',
            'expected_recovery': 0.21
        },
        {
            'name': 's2l_forget_no_suffix',
            'script': 's2l_forget_no_suffix.py',
            'description': 'S2L on Forget (No PromptSuffix)',
            'expected_recovery': 0.30
        }
    ]

                                        
    all_results = {}

    for exp in experiments:
        print(f"\n{'#'*80}")
        print(f"# Experiment: {exp['name']}")
        print(f"# Description: {exp['description']}")
        print(f"# Expected Recovery: {exp['expected_recovery']:.1%}")
        print(f"{'#'*80}\n")

        exp_results = []

        for seed in args.seeds:
            output_dir = os.path.join(args.output_base_dir, exp['name'], f'seed_{seed}')
            os.makedirs(output_dir, exist_ok=True)

            exp_args = {
                'vanilla_dir': args.vanilla_dir,
                'unlearned_dir': args.unlearned_dir,
                'data_split_dir': args.data_split_dir,
                'output_dir': output_dir
            }

                            
            success = run_experiment(exp['script'], exp_args, seed)

            if success:
                              
                metrics = load_metrics(output_dir)
                exp_results.append(metrics)
                print(f"✓ Seed {seed}: Recovery Rate = {metrics['recovery_rate']:.2%}")
            else:
                print(f"✗ Seed {seed}: FAILED")

                                         
        if exp_results:
            stats_dict = compute_statistics(exp_results)
            all_results[exp['name']] = {
                'description': exp['description'],
                'expected_recovery': exp['expected_recovery'],
                'results': exp_results,
                'statistics': stats_dict
            }

            print(f"\n{exp['name']} Statistics:")
            print(f"  Recovery Rate: {stats_dict['recovery_rate']['mean']:.2%} ± {stats_dict['recovery_rate']['std']:.2%}")
            print(f"  95% CI: [{stats_dict['recovery_rate']['ci95'][0]:.2%}, {stats_dict['recovery_rate']['ci95'][1]:.2%}]")
            print(f"  Expected: {exp['expected_recovery']:.1%}")

                             
    aggregated_path = os.path.join(args.output_base_dir, 'aggregated_results.json')
    with open(aggregated_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"All OOD Ablation Experiments Complete!")
    print(f"Aggregated results saved to: {aggregated_path}")
    print(f"{'='*80}\n")

                         
    print("\nSummary Table:")
    print(f"{'Method':<30} {'Recovery Rate':<20} {'Expected':<15}")
    print("-" * 65)

    for exp_name, exp_data in all_results.items():
        stats = exp_data['statistics']
        mean = stats['recovery_rate']['mean']
        std = stats['recovery_rate']['std']
        expected = exp_data['expected_recovery']

        print(f"{exp_data['description']:<30} {mean:.2%} ± {std:.2%}      {expected:.1%}")


if __name__ == "__main__":
    main()
