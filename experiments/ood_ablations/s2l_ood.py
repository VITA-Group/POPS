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
from attack.s2l_synthesis import S2LSynthesizer


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


def synthesize_data_from_ood(model, processor, retain_dataset, num_samples=500):
\
\
\
       
    synthesizer = S2LSynthesizer(model, processor)

    print(f"[S2L OOD] Synthesizing {num_samples} samples from retain set...")
    synthetic_samples = []

    for i in range(min(num_samples, len(retain_dataset))):
        sample = retain_dataset[i]

                                                   
        synthetic_qa = synthesizer.generate_qa_pair(
            image=sample['image'],
            context=sample.get('context', ''),
            temperature=0.7
        )

        synthetic_samples.append({
            'image': sample['image'],
            'question': synthetic_qa['question'],
            'answer': synthetic_qa['answer']
        })

        if (i + 1) % 100 == 0:
            print(f"[S2L OOD] Synthesized {i + 1}/{num_samples} samples")

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
    parser = argparse.ArgumentParser(description="S2L on OOD (Retain Set)")
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

    print(f"[S2L OOD] Starting experiment with seed {args.seed}")

              
    set_seed(args.seed)

                          
    print("[S2L OOD] Loading unlearned model...")
    model, processor = load_model(args.unlearned_dir)

                               
    print("[S2L OOD] Loading retain set (OOD data)...")
    retain_dataset = MLLMU_Dataset(args.data_split_dir, split='retain')

                                           
    synthetic_samples = synthesize_data_from_ood(
        model, processor, retain_dataset, args.num_synthetic
    )

                              
    synthetic_dataset = create_synthetic_dataset(synthetic_samples)

                
    print("[S2L OOD] Applying LoRA...")
    model = apply_lora(model)

                                                    
    print(f"[S2L OOD] Fine-tuning on {len(synthetic_dataset)} synthetic samples...")
    trainer = create_trainer(model, synthetic_dataset, processor, args)
    trainer.train()

                
    print(f"[S2L OOD] Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

                            
    print("[S2L OOD] Evaluating on forget set...")
    eval_metrics = evaluate_on_forget_set(model, processor, args.data_split_dir)

                  
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics(eval_metrics, metrics_path)

    print(f"[S2L OOD] Results:")
    print(f"  Test Acc: {eval_metrics['test_acc']:.2%}")
    print(f"  ROUGE-L: {eval_metrics['rouge_l']:.3f}")
    print(f"  Recovery Rate: {eval_metrics['recovery_rate']:.2%}")
    print(f"  Expected: ~21% recovery (synthesis from OOD has limited targeting)")

    print(f"[S2L OOD] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
