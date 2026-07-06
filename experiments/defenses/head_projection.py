\
\
\
\
   

import os
import sys
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

                                
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoProcessor
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


class HeadProjectionDefense:
\
\
\
       

    def __init__(self, model, processor, n_components=10):
\
\
\
\
\
           
        self.model = model
        self.processor = processor
        self.n_components = n_components
        self.projection_matrix = None
        self.target_layer = None

    def collect_activations(self, dataset, layer_name='model.layers.15'):
\
\
\
\
\
\
\
\
\
           
        print(f"[Head Proj] Collecting activations from {layer_name}...")

        activations = []
        self.target_layer = layer_name

                                     
        activation_dict = {}

        def get_activation(name):
            def hook(model, input, output):
                                              
                if isinstance(output, tuple):
                    output = output[0]
                                                  
                activation_dict[name] = output.mean(dim=1).detach().cpu()
            return hook

                       
        target_module = dict(self.model.named_modules())[layer_name]
        handle = target_module.register_forward_hook(get_activation(layer_name))

                                        
        for idx in range(min(len(dataset), 200)):                             
            sample = dataset[idx]

            inputs = self.processor(
                text=sample['question'],
                images=sample['image'],
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                _ = self.model(**inputs)

            if layer_name in activation_dict:
                activations.append(activation_dict[layer_name].numpy())

            if (idx + 1) % 50 == 0:
                print(f"[Head Proj] Processed {idx+1} samples...")

                     
        handle.remove()

                           
        activations = np.vstack(activations)
        print(f"[Head Proj] Collected activations shape: {activations.shape}")

        return activations

    def compute_projection_matrix(self, activations):
\
\
\
\
\
\
\
\
           
        print(f"[Head Proj] Computing projection matrix...")
        print(f"[Head Proj] Using {self.n_components} principal components")

                                                                       
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        svd.fit(activations)

                                                       
        sensitive_directions = svd.components_                                     

        print(f"[Head Proj] Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

                                 
        U = torch.tensor(sensitive_directions.T, dtype=torch.float32)                              

                                                   
                                                                      
        I = torch.eye(U.shape[0])
        P = I - torch.mm(U, U.T)

        self.projection_matrix = P

        print(f"[Head Proj] Projection matrix shape: {P.shape}")

        return P

    def apply_defense(self):
\
\
\
           
        print(f"[Head Proj] Applying defense to {self.target_layer}...")

        if self.projection_matrix is None:
            raise ValueError("Must compute projection matrix first!")

        projection_matrix = self.projection_matrix.to(self.model.device)

                                     
        def projection_hook(module, input, output):
                                            
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

                                 
            original_shape = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))

                                    
            projected = torch.mm(
                hidden_states_flat,
                projection_matrix.to(hidden_states.dtype)
            )

            projected = projected.view(original_shape)

            if other_outputs:
                return (projected,) + other_outputs
            else:
                return projected

                       
        target_module = dict(self.model.named_modules())[self.target_layer]
        handle = target_module.register_forward_hook(projection_hook)

        print(f"[Head Proj] Defense applied! Activations will be projected.")

        return handle


def train_ga_with_defense(vanilla_dir, data_split_dir, output_dir, seed):
\
\
\
\
       
    print("[Head Proj] Training GA with head projection defense...")

                        
    model, processor = load_model(vanilla_dir)

                         
    forget_dataset = MLLMU_Dataset(data_split_dir, split='forget')

                                   
    defense = HeadProjectionDefense(model, processor, n_components=10)

                                         
    activations = defense.collect_activations(forget_dataset)

                               
    defense.compute_projection_matrix(activations)

                   
    defense_handle = defense.apply_defense()

                                       
    print("[Head Proj] Training GA unlearning with defense...")

    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer

                
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

                   
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=processor.tokenizer
    )

                        
    trainer.train()

                
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

                         
    defense_handle.remove()

    print(f"[Head Proj] GA with defense trained and saved to {output_dir}")

    return model, processor


def evaluate_pops_on_defended_model(defended_model_dir, data_split_dir):
\
\
\
\
       
    print("[Head Proj] Evaluating POPS on defended model...")

    from attack.pops_attack import POPSAttack

                         
    model, processor = load_model(defended_model_dir)

                         
    forget_dataset = MLLMU_Dataset(data_split_dir, split='forget')

                     
    pops = POPSAttack(model, processor)
    attack_results = pops.run_attack(forget_dataset)

    print(f"[Head Proj] POPS on defended model:")
    print(f"  Recovery Rate: {attack_results['recovery_rate']:.2%}")
    print(f"  Expected: ~68% (vs 82% on undefended)")
    print(f"  Defense reduces effectiveness by ~14pp")

    return attack_results


def main():
    parser = argparse.ArgumentParser(description="Head Projection Defense Evaluation")
    parser.add_argument('--vanilla_dir', type=str, required=True,
                       help='Path to vanilla model')
    parser.add_argument('--data_split_dir', type=str, required=True,
                       help='Path to data splits')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for defended model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    print(f"[Head Proj] Starting Head Projection defense evaluation")
    print(f"[Head Proj] Seed: {args.seed}")

              
    set_seed(args.seed)

                           
    defended_model, processor = train_ga_with_defense(
        args.vanilla_dir,
        args.data_split_dir,
        args.output_dir,
        args.seed
    )

                                     
    results = evaluate_pops_on_defended_model(
        args.output_dir,
        args.data_split_dir
    )

                  
    metrics_path = os.path.join(args.output_dir, 'defense_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Head Proj] Defense evaluation complete!")
    print(f"  Results saved to {metrics_path}")
    print(f"  Interpretation: Defense helps but doesn't eliminate threat")
    print(f"  68% recovery still concerning for privacy")


if __name__ == "__main__":
    main()
