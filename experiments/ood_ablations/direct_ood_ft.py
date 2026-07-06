\
\
\
   

import os
import sys
import torch
import argparse
import json
from pathlib import Path

                                
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from data_process.dataset import MLLMU_Dataset


def set_seed(seed):
                                             
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_dir):
                                            
    from transformers import LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    return model, processor


def apply_lora(model):
                                                          
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model


def create_trainer(model, dataset, processor, args):
                                                      
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer
    )

    return trainer


def evaluate_on_forget_set(model, processor, data_split_dir):
                                                 
    from attack_eval import compute_accuracy, compute_rouge_l

                     
    forget_dataset = MLLMU_Dataset(data_split_dir, split='forget')

                     
    test_acc = compute_accuracy(model, processor, forget_dataset)
    rouge_l = compute_rouge_l(model, processor, forget_dataset)

                                                            
    vanilla_metrics = json.load(open(f"{data_split_dir}/../vanilla_metrics.json"))
    unlearned_metrics = json.load(open(f"{data_split_dir}/../unlearned_metrics.json"))

                                                                                  
    recovery_rate = (test_acc - unlearned_metrics['test_acc']) / \
                   (vanilla_metrics['test_acc'] - unlearned_metrics['test_acc'])

    metrics = {
        'test_acc': test_acc,
        'rouge_l': rouge_l,
        'recovery_rate': recovery_rate
    }

    return metrics


def save_metrics(metrics, output_path):
                                         
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Direct OOD Fine-Tuning Baseline")
    parser.add_argument('--vanilla_dir', type=str, required=True,
                       help='Path to vanilla model')
    parser.add_argument('--unlearned_dir', type=str, required=True,
                       help='Path to unlearned (GA) model')
    parser.add_argument('--data_split_dir', type=str, required=True,
                       help='Path to data splits (forget/retain)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fine-tuned model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    print(f"[Direct OOD FT] Starting experiment with seed {args.seed}")

              
    set_seed(args.seed)

                          
    print("[Direct OOD FT] Loading unlearned model...")
    model, processor = load_model(args.unlearned_dir)

                
    print("[Direct OOD FT] Applying LoRA...")
    model = apply_lora(model)

                               
    print("[Direct OOD FT] Loading retain set (OOD data)...")
    retain_dataset = MLLMU_Dataset(args.data_split_dir, split='retain')

                                  
    print(f"[Direct OOD FT] Fine-tuning on {len(retain_dataset)} retain samples...")
    trainer = create_trainer(model, retain_dataset, processor, args)
    trainer.train()

                
    print(f"[Direct OOD FT] Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

                            
    print("[Direct OOD FT] Evaluating on forget set...")
    eval_metrics = evaluate_on_forget_set(model, processor, args.data_split_dir)

                  
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics(eval_metrics, metrics_path)

    print(f"[Direct OOD FT] Results:")
    print(f"  Test Acc: {eval_metrics['test_acc']:.2%}")
    print(f"  ROUGE-L: {eval_metrics['rouge_l']:.3f}")
    print(f"  Recovery Rate: {eval_metrics['recovery_rate']:.2%}")
    print(f"  Expected: ~12% recovery (shows OOD alone insufficient)")

    print(f"[Direct OOD FT] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
