\
\
\
   

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import torch
from transformers import LlavaForConditionalGeneration, Idefics2ForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer

                                                   
from eval import (
    evaluate_classification,
    evaluate_fill_in_the_blank,
    evaluate_generation
)

                          
from attack.pops_attack import POPSAttack, setup_logging
from attack.perplexity_selector import PerplexitySelector

logger = logging.getLogger(__name__)


def parse_args():
                                                             
    parser = argparse.ArgumentParser(
        description="Evaluate POPS attack on unlearned MLLM models"
    )

                 
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='Base model ID (e.g., llava-hf/llava-1.5-7b-hf)'
    )
    parser.add_argument(
        '--unlearned_model_path',
        type=str,
        required=True,
        help='Path to unlearned model to attack'
    )
    parser.add_argument(
        '--vanilla_model_path',
        type=str,
        default=None,
        help='Path to vanilla model for comparison'
    )
    parser.add_argument(
        '--attacked_model_path',
        type=str,
        default=None,
        help='Path to already attacked model (skip attack if provided)'
    )

                
    parser.add_argument(
        '--data_split_folder',
        type=str,
        required=True,
        help='Path to MLLMU-Bench data splits'
    )
    parser.add_argument(
        '--few_shot_data',
        type=str,
        required=True,
        help='Path to few-shot data parquet file'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test set directory'
    )
    parser.add_argument(
        '--celebrity_data',
        type=str,
        required=True,
        help='Path to celebrity/retain data'
    )

                          
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/attack_config.yaml',
        help='Path to attack configuration file'
    )
    parser.add_argument(
        '--forget_ratio',
        type=int,
        default=10,
        help='Forget set percentage (paper main setting: 10; MLLMU-Bench also provides 5, 15, 20)'
    )
    parser.add_argument(
        '--run_attack',
        action='store_true',
        help='Run POPS attack before evaluation'
    )

                          
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Folder to save evaluation results'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output file name prefix'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

                        
    parser.add_argument(
        '--use_perplexity_selection',
        action='store_true',
        help='Use perplexity-based response selection during evaluation'
    )
    parser.add_argument(
        '--evaluate_stages',
        nargs='+',
        default=['unlearned', 'full_attack'],
        choices=['unlearned', 'prompt_only', 'full_attack'],
        help='Which attack stages to evaluate'
    )

    return parser.parse_args()


class AttackArgs:
                                                                       
    def __init__(self, args):
        self.model_id = args.model_id
        self.output_folder = args.output_folder
        self.forget_ratio = args.forget_ratio


def load_model(model_path: str, model_id: str):
                                               
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model_id.startswith("llava"):
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    elif model_id.startswith("HuggingFaceM4"):
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    return model, processor, tokenizer


def evaluate_model(
    model,
    processor,
    tokenizer,
    args,
    eval_args,
    stage_name: str
) -> dict:
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
\
\
\
       
    logger.info(f"=" * 80)
    logger.info(f"EVALUATING: {stage_name}")
    logger.info(f"=" * 80)

                             
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    forget_parquet = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet = os.path.join(retain_folder, "train-00000-of-00001.parquet")

    results = {}

                         
    logger.info("### Evaluating Forget Set ###")
    try:
        forget_classification = evaluate_classification(
            parquet_file=forget_parquet,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="forget"
        )
        results['forget_classification'] = forget_classification

        forget_cloze = evaluate_fill_in_the_blank(
            parquet_file=forget_parquet,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="forget"
        )
        results['forget_cloze'] = forget_cloze

        forget_generation = evaluate_generation(
            parquet_file=forget_parquet,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="forget"
        )
        results['forget_generation'] = forget_generation

    except Exception as e:
        logger.error(f"Error evaluating forget set: {e}")
        results['forget_error'] = str(e)

                       
    logger.info("### Evaluating Test Set ###")
    try:
        test_classification = evaluate_classification(
            parquet_file=args.test_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="test",
            forget_parquet_file=forget_parquet
        )
        results['test_classification'] = test_classification

        test_cloze = evaluate_fill_in_the_blank(
            parquet_file=args.test_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="test",
            forget_parquet_file=forget_parquet
        )
        results['test_cloze'] = test_cloze

        test_generation = evaluate_generation(
            parquet_file=args.test_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="test",
            forget_parquet_file=forget_parquet
        )
        results['test_generation'] = test_generation

    except Exception as e:
        logger.error(f"Error evaluating test set: {e}")
        results['test_error'] = str(e)

                         
    logger.info("### Evaluating Retain Set ###")
    try:
        retain_classification = evaluate_classification(
            parquet_file=retain_parquet,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="retain_shared"
        )
        results['retain_classification'] = retain_classification

        retain_cloze = evaluate_fill_in_the_blank(
            parquet_file=retain_parquet,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="retain_shared"
        )
        results['retain_cloze'] = retain_cloze

        retain_generation = evaluate_generation(
            parquet_file=retain_parquet,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=eval_args,
            mode="retain_shared"
        )
        results['retain_generation'] = retain_generation

    except Exception as e:
        logger.error(f"Error evaluating retain set: {e}")
        results['retain_error'] = str(e)

    return results


def compute_attack_metrics(
    vanilla_results: dict,
    unlearned_results: dict,
    attacked_results: dict
) -> dict:
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
       
    metrics = {}

                                                       
    if all(k in r.get('test_classification', {}) for r in [vanilla_results, unlearned_results, attacked_results] for k in ['Image-Textual Question Accuracy']):
        vanilla_acc = vanilla_results['test_classification']['Image-Textual Question Accuracy']
        unlearned_acc = unlearned_results['test_classification']['Image-Textual Question Accuracy']
        attacked_acc = attacked_results['test_classification']['Image-Textual Question Accuracy']

        if vanilla_acc - unlearned_acc != 0:
            recovery_rate = (attacked_acc - unlearned_acc) / (vanilla_acc - unlearned_acc) * 100
            metrics['recovery_rate'] = recovery_rate
            metrics['accuracy_gain'] = attacked_acc - unlearned_acc

    return metrics


def main():
                                  
    args = parse_args()

                   
    setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("POPS ATTACK EVALUATION")
    logger.info("=" * 80)
    if args.use_perplexity_selection:
        logger.warning(
            "--use_perplexity_selection is set, but attack_eval.py currently uses the standard eval.py metrics path. "
            "Perplexity-based response selection is not applied to full_attack metrics."
        )
    if 'prompt_only' in args.evaluate_stages:
        logger.warning(
            "prompt_only was requested, but attack_eval.py does not currently implement PromptSuffix-only metrics."
        )

                             
    os.makedirs(args.output_folder, exist_ok=True)

                              
    eval_args = AttackArgs(args)

    all_results = {}

                                                  
    if 'unlearned' in args.evaluate_stages:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: BASELINE UNLEARNED MODEL")
        logger.info("=" * 80)

        model, processor, tokenizer = load_model(args.unlearned_model_path, args.model_id)
        unlearned_results = evaluate_model(
            model, processor, tokenizer, args, eval_args, "Unlearned Model"
        )
        all_results['unlearned'] = unlearned_results

                      
        with open(os.path.join(args.output_folder, f"{args.output_file}_unlearned_results.json"), 'w') as f:
            json.dump(unlearned_results, f, indent=2)

                      
        del model
        torch.cuda.empty_cache()

                                           
    if args.run_attack and args.attacked_model_path is None:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: RUNNING POPS ATTACK")
        logger.info("=" * 80)

                           
        pops = POPSAttack(
            config_path=args.config_path,
            unlearned_model_path=args.unlearned_model_path,
            vanilla_model_path=args.vanilla_model_path,
            forget_ratio=args.forget_ratio,
            device="cuda"
        )

                         
        attack_output_dir = os.path.join(args.output_folder, "pops_attack")
        attacked_model_path = pops.run_full_attack(
            output_dir=attack_output_dir,
            save_artifacts=True
        )

        args.attacked_model_path = attacked_model_path

                      
        del pops
        torch.cuda.empty_cache()

                                      
    if 'full_attack' in args.evaluate_stages and args.attacked_model_path:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: EVALUATING ATTACKED MODEL")
        logger.info("=" * 80)

        model, processor, tokenizer = load_model(args.attacked_model_path, args.model_id)
        attacked_results = evaluate_model(
            model, processor, tokenizer, args, eval_args, "Attacked Model (POPS)"
        )
        all_results['attacked'] = attacked_results

                      
        with open(os.path.join(args.output_folder, f"{args.output_file}_attacked_results.json"), 'w') as f:
            json.dump(attacked_results, f, indent=2)

                      
        del model
        torch.cuda.empty_cache()

                                     
    if 'unlearned' in all_results and 'attacked' in all_results:
                                           
        vanilla_results = {}
        if args.vanilla_model_path:
            logger.info("Loading vanilla model for comparison...")
            model, processor, tokenizer = load_model(args.vanilla_model_path, args.model_id)
            vanilla_results = evaluate_model(
                model, processor, tokenizer, args, eval_args, "Vanilla Model"
            )
            all_results['vanilla'] = vanilla_results

                                
        if vanilla_results:
            attack_metrics = compute_attack_metrics(
                vanilla_results,
                all_results['unlearned'],
                all_results['attacked']
            )
            all_results['attack_metrics'] = attack_metrics

            logger.info("\n" + "=" * 80)
            logger.info("ATTACK METRICS")
            logger.info("=" * 80)
            for metric, value in attack_metrics.items():
                logger.info(f"{metric}: {value:.2f}")

                        
    final_output = os.path.join(args.output_folder, f"{args.output_file}_final_results.json")
    with open(final_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {final_output}")


if __name__ == "__main__":
    main()
