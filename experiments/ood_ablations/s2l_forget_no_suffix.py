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
from data_process.mllmu_dataset import MLLMU_Dataset
from attack.s2l_finetune import S2LFineTuner


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
    return get_peft_model(model, lora_config)


def synthesize_data_from_forget(model, processor, tokenizer, forget_dataset, config, num_samples=500):
    s2l_finetuner = S2LFineTuner(model, processor, tokenizer, config)
    print(f"[S2L Forget] Synthesizing {num_samples} samples...")

    forget_samples = []
    for i in range(len(forget_dataset)):
        sample = forget_dataset[i]
        forget_samples.append({
            'image': sample['image'],
            'question': sample.get('question', ''),
            'answer': sample.get('answer', '')
        })

    synthetic_samples = s2l_finetuner.generate_synthetic_data(
        model=model,
        forget_samples=forget_samples,
        optimized_suffixes=[''],
        num_synthetic_samples=num_samples
    )

    print(f"[S2L Forget] Synthesis complete")
    return synthetic_samples


def create_synthetic_dataset(synthetic_samples):
    from torch.utils.data import Dataset
    class SyntheticDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
    return SyntheticDataset(synthetic_samples)


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
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer
    )


def evaluate_on_forget_set(model, processor, data_split_dir):
    from attack_eval import compute_accuracy, compute_rouge_l
    forget_dataset = MLLMU_Dataset(data_split_dir, split='forget')
    test_acc = compute_accuracy(model, processor, forget_dataset)
    rouge_l = compute_rouge_l(model, processor, forget_dataset)

    vanilla_metrics = json.load(open(f"{data_split_dir}/../vanilla_metrics.json"))
    unlearned_metrics = json.load(open(f"{data_split_dir}/../unlearned_metrics.json"))

    recovery_rate = (test_acc - unlearned_metrics['test_acc']) / \
                   (vanilla_metrics['test_acc'] - unlearned_metrics['test_acc'])

    return {
        'test_acc': test_acc,
        'rouge_l': rouge_l,
        'recovery_rate': recovery_rate
    }


def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="S2L on Forget (No PromptSuffix)")
    parser.add_argument('--vanilla_dir', type=str, required=True,
                       help='Path to vanilla model')
    parser.add_argument('--unlearned_dir', type=str, required=True,
                       help='Path to unlearned (GA) model')
    parser.add_argument('--data_split_dir', type=str, required=True,
                       help='Path to data splits (forget/retain)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fine-tuned model')
    parser.add_argument('--num_synthetic', type=int, default=500,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    model, processor = load_model(args.unlearned_dir)
    forget_dataset = MLLMU_Dataset(args.data_split_dir, split='forget')

    s2l_config = {
        'lora_rank': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'batch_size': 4,
        'gradient_accumulation_steps': 4
    }

    synthetic_samples = synthesize_data_from_forget(
        model, processor, processor.tokenizer, forget_dataset, s2l_config, args.num_synthetic
    )
    synthetic_dataset = create_synthetic_dataset(synthetic_samples)

    model = apply_lora(model)
    trainer = create_trainer(model, synthetic_dataset, processor, args)
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    eval_metrics = evaluate_on_forget_set(model, processor, args.data_split_dir)
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics(eval_metrics, metrics_path)

    print(f"\nRecovery Rate: {eval_metrics['recovery_rate']:.2%}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
