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


class MultiSeedRunner:
                                                                      

    def __init__(self, seeds=[42, 123, 456, 789, 2024]):
        self.seeds = seeds

    def run_experiment(self, script_path, base_args, output_base_dir):
\
\
\
\
\
\
\
\
\
\
           
        results = []

        for seed in self.seeds:
            print(f"\n{'='*80}")
            print(f"Running with seed {seed}")
            print(f"{'='*80}\n")

                                                   
            output_dir = os.path.join(output_base_dir, f'seed_{seed}')
            os.makedirs(output_dir, exist_ok=True)

                           
            cmd = ["python", script_path]
            for key, value in base_args.items():
                cmd.extend([f"--{key}", str(value)])
            cmd.extend(["--output_dir", output_dir])
            cmd.extend(["--seed", str(seed)])

                            
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(result.stdout)

                              
                metrics_path = os.path.join(output_dir, 'metrics.json')
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                metrics['seed'] = seed
                results.append(metrics)

                print(f"✓ Seed {seed}: Success")
                print(f"  Test Acc: {metrics.get('test_acc', 'N/A')}")
                print(f"  Recovery: {metrics.get('recovery_rate', 'N/A')}")
            else:
                print(f"✗ Seed {seed}: FAILED")
                print(f"Error: {result.stderr}")

        return results

    def compute_statistics(self, results, baseline_results=None):
\
\
\
\
\
\
\
\
\
           
        if not results:
            return {}

                         
        test_accs = [r['test_acc'] for r in results if 'test_acc' in r]
        rouge_ls = [r['rouge_l'] for r in results if 'rouge_l' in r]
        recovery_rates = [r['recovery_rate'] for r in results if 'recovery_rate' in r]

        stats_dict = {
            'num_seeds': len(results),
            'seeds': [r['seed'] for r in results]
        }

                                  
        if test_accs:
            stats_dict['test_acc'] = {
                'mean': np.mean(test_accs),
                'std': np.std(test_accs, ddof=1),
                'min': np.min(test_accs),
                'max': np.max(test_accs),
                'ci95': stats.t.interval(0.95, len(test_accs)-1,
                                        loc=np.mean(test_accs),
                                        scale=stats.sem(test_accs))
            }

                            
        if rouge_ls:
            stats_dict['rouge_l'] = {
                'mean': np.mean(rouge_ls),
                'std': np.std(rouge_ls, ddof=1),
                'min': np.min(rouge_ls),
                'max': np.max(rouge_ls),
                'ci95': stats.t.interval(0.95, len(rouge_ls)-1,
                                        loc=np.mean(rouge_ls),
                                        scale=stats.sem(rouge_ls))
            }

                                  
        if recovery_rates:
            stats_dict['recovery_rate'] = {
                'mean': np.mean(recovery_rates),
                'std': np.std(recovery_rates, ddof=1),
                'min': np.min(recovery_rates),
                'max': np.max(recovery_rates),
                'ci95': stats.t.interval(0.95, len(recovery_rates)-1,
                                        loc=np.mean(recovery_rates),
                                        scale=stats.sem(recovery_rates))
            }

                                              
        if baseline_results and test_accs:
            baseline_accs = [r['test_acc'] for r in baseline_results if 'test_acc' in r]
            if baseline_accs:
                                             
                if len(test_accs) == len(baseline_accs):
                    t_stat, p_value = stats.ttest_rel(test_accs, baseline_accs)
                else:
                                                  
                    t_stat, p_value = stats.ttest_ind(test_accs, baseline_accs)

                stats_dict['vs_baseline'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        return stats_dict

    def save_results(self, results, statistics, output_path):
                                                 
        output = {
            'results': results,
            'statistics': statistics
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, statistics, experiment_name="Experiment"):
                                      
        print(f"\n{'='*80}")
        print(f"{experiment_name} - Statistical Summary")
        print(f"{'='*80}")
        print(f"Number of seeds: {statistics['num_seeds']}")
        print(f"Seeds: {statistics['seeds']}")

        if 'test_acc' in statistics:
            acc = statistics['test_acc']
            print(f"\nTest Accuracy:")
            print(f"  Mean: {acc['mean']:.4f} ({acc['mean']*100:.2f}%)")
            print(f"  Std:  {acc['std']:.4f} ({acc['std']*100:.2f}%)")
            print(f"  95% CI: [{acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
            print(f"  Range: [{acc['min']:.4f}, {acc['max']:.4f}]")

        if 'rouge_l' in statistics:
            rouge = statistics['rouge_l']
            print(f"\nROUGE-L:")
            print(f"  Mean: {rouge['mean']:.4f}")
            print(f"  Std:  {rouge['std']:.4f}")
            print(f"  95% CI: [{rouge['ci95'][0]:.4f}, {rouge['ci95'][1]:.4f}]")
            print(f"  Range: [{rouge['min']:.4f}, {rouge['max']:.4f}]")

        if 'recovery_rate' in statistics:
            rec = statistics['recovery_rate']
            print(f"\nRecovery Rate:")
            print(f"  Mean: {rec['mean']:.4f} ({rec['mean']*100:.2f}%)")
            print(f"  Std:  {rec['std']:.4f} ({rec['std']*100:.2f}%)")
            print(f"  95% CI: [{rec['ci95'][0]:.4f}, {rec['ci95'][1]:.4f}]")
            print(f"  Range: [{rec['min']:.4f}, {rec['max']:.4f}]")

        if 'vs_baseline' in statistics:
            vs = statistics['vs_baseline']
            print(f"\nComparison vs Baseline:")
            print(f"  t-statistic: {vs['t_statistic']:.4f}")
            print(f"  p-value: {vs['p_value']:.6f}")
            print(f"  Significant (p < 0.05): {'YES' if vs['significant'] else 'NO'}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner")
    parser.add_argument('--script', type=str, required=True,
                       help='Path to experiment script to run')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name of experiment (for output directory)')
    parser.add_argument('--output_base_dir', type=str,
                       default='results/statistical_analysis',
                       help='Base output directory')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 2024],
                       help='Random seeds')
    parser.add_argument('--baseline_results', type=str,
                       help='Path to baseline results JSON for comparison')

                                                   
    args, remaining_args = parser.parse_known_args()

                                          
    base_args = {}
    i = 0
    while i < len(remaining_args):
        if remaining_args[i].startswith('--'):
            key = remaining_args[i][2:]
            if i + 1 < len(remaining_args) and not remaining_args[i + 1].startswith('--'):
                value = remaining_args[i + 1]
                i += 2
            else:
                value = True
                i += 1
            base_args[key] = value
        else:
            i += 1

                             
    output_dir = os.path.join(args.output_base_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

                   
    runner = MultiSeedRunner(seeds=args.seeds)

                    
    print(f"\nRunning {args.experiment_name} with {len(args.seeds)} seeds...")
    results = runner.run_experiment(args.script, base_args, output_dir)

                               
    baseline_results = None
    if args.baseline_results:
        with open(args.baseline_results, 'r') as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data.get('results', [])

                        
    statistics = runner.compute_statistics(results, baseline_results)

                  
    output_path = os.path.join(output_dir, 'results_statistics.json')
    runner.save_results(results, statistics, output_path)

                   
    runner.print_summary(statistics, args.experiment_name)


if __name__ == "__main__":
    main()
