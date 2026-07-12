\
\
\
   

import logging
import os
import sys
from io import BytesIO
from typing import List, Dict, Optional, Tuple

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.data_preprocess import train_collate_fn_llava, train_collate_fn_idefics

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
                                                                          

    def __init__(self, synthetic_samples: List[Dict]):
                                           
        self.samples = synthetic_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class S2LFineTuner:
                                                                     

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        config: Dict,
        device: str = "cuda"
    ):
                                        
        self.base_model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

                               
        self.lora_rank = config.get('lora_rank', 8)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.05)
        self.learning_rate = config.get('learning_rate', 1e-6)
        self.num_epochs = config.get('num_epochs', 3)
        self.batch_size = config.get('batch_size', 4)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.kl_penalty = config.get('kl_penalty', 0.2)
        self.max_length = config.get('max_length', 384)
        self.target_modules = config.get('target_modules', ["q_proj", "v_proj"])

        if self.kl_penalty:
            logger.warning(
                "S2L kl_penalty=%s is configured, but retain-set KL regularization is not implemented in this training loop.",
                self.kl_penalty,
            )
        if self.max_length:
            logger.warning(
                "S2L max_length=%s is configured, but the current collate functions rely on processor defaults and do not pass max_length.",
                self.max_length,
            )

        logger.info(f"Initialized S2LFineTuner with LoRA rank {self.lora_rank}")

    def find_all_linear_names(self, model) -> List[str]:
                                                                       
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']

        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')

        return list(lora_module_names)

    def resolve_target_modules(self, model) -> List[str]:
                                                                                
        available = set(self.find_all_linear_names(model))
        configured = self.target_modules or ["q_proj", "v_proj"]
        matched = [name for name in configured if name in available]

        if not matched:
            logger.warning(
                "None of the configured LoRA targets %s were found; falling back to all language linear layers",
                configured,
            )
            return sorted(available)

        missing = sorted(set(configured) - set(matched))
        if missing:
            logger.warning("Configured LoRA targets not found and skipped: %s", missing)

        return matched

    def prepare_model_for_lora(self, model):
                                                                         
        target_modules = self.resolve_target_modules(model)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        logger.info("Model prepared with LoRA adapters on target modules: %s", target_modules)
        model.print_trainable_parameters()

        return model

    def generate_synthetic_data(
        self,
        model,
        source_samples: List[Dict],
        optimized_suffixes: List[str],
        num_synthetic_samples: int = 100
    ) -> List[Dict]:
                                                                                       
        logger.info(f"Generating {num_synthetic_samples} synthetic samples...")

        if not source_samples:
            raise ValueError("Cannot generate synthetic data: source_samples is empty")
        if not optimized_suffixes:
            raise ValueError("Cannot generate synthetic data: optimized_suffixes is empty")

        synthetic_data = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(num_synthetic_samples), desc="Generating Synthetic Data"):
                sample = np.random.choice(source_samples)
                suffix = np.random.choice(optimized_suffixes)

                image = sample.get('image', None)
                question = sample.get('question', '')
                full_prompt = f"{question} {suffix}"

                if image is not None:
                    prompt_text = f"USER: <image>\n{full_prompt}\nASSISTANT:"
                    inputs = self.processor(
                        images=image,
                        text=prompt_text,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    prompt_text = f"USER: {full_prompt}\nASSISTANT:"
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt"
                    ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                generated_text = self.processor.decode(outputs[0][2:], skip_special_tokens=True)

                if "ASSISTANT:" in generated_text:
                    answer = generated_text.split("ASSISTANT:")[1].strip()
                else:
                    answer = generated_text.strip()

                synthetic_data.append({
                    'image': image,
                    'question': question,
                    'answer': answer,
                    'suffix': suffix
                })

        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def decompose_into_facets(
        self,
        samples: List[Dict],
        num_facets: int = 3
    ) -> List[Dict]:
                                                                                     
        logger.info(f"Decomposing samples into {num_facets} facets each...")

        decomposed_samples = []

        for sample in samples:
            image = sample.get('image', None)
            question = sample.get('question', '')
            answer = sample.get('answer', '')

            facet_templates = [
                f"Can you tell me about {question}",
                f"What do you know about {question}",
                f"Describe {question}",
                f"Provide information about {question}",
                f"Tell me what you remember about {question}",
            ]

            selected_templates = np.random.choice(
                facet_templates,
                size=min(num_facets, len(facet_templates)),
                replace=False
            )

            for template in selected_templates:
                decomposed_samples.append({
                    'image': image,
                    'question': template,
                    'answer': answer
                })

        logger.info(f"Created {len(decomposed_samples)} faceted samples")
        return decomposed_samples

    def create_multi_concept_batch(
        self,
        samples: List[Dict],
        batch_size: int = 4
    ) -> List[List[Dict]]:
                                                                                           
        shuffled = copy.deepcopy(samples)
        np.random.shuffle(shuffled)

        batches = []
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            batches.append(batch)

        return batches

    def compute_kl_divergence(
        self,
        model_outputs,
        reference_outputs
    ) -> torch.Tensor:
                                                       
        log_probs = F.log_softmax(model_outputs, dim=-1)
        ref_probs = F.softmax(reference_outputs, dim=-1)
        kl_div = F.kl_div(log_probs, ref_probs, reduction='batchmean')
        return kl_div

    def fine_tune(
        self,
        synthetic_samples: List[Dict],
        model_id: str,
        output_dir: str,
        use_faceted_decomposition: bool = True,
        num_facets: int = 3
    ):
                                                                  
        logger.info("Starting S2L fine-tuning...")

        if use_faceted_decomposition:
            training_samples = self.decompose_into_facets(synthetic_samples, num_facets)
        else:
            training_samples = synthetic_samples

        dataset = SyntheticDataset(training_samples)

        if model_id.startswith("llava"):
            collate_fn = lambda x: train_collate_fn_llava(x, self.processor, self.config)
        elif model_id.startswith("HuggingFaceM4"):
            collate_fn = lambda x: train_collate_fn_idefics(x, self.processor, self.config)
        else:
            raise ValueError(f"Unknown model type: {model_id}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        model = self.prepare_model_for_lora(self.base_model)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        num_training_steps = len(dataloader) * self.num_epochs
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )

        model.gradient_checkpointing_enable()
        model.train()

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask, pixel_values, labels = batch

                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels
                    )

                    loss = outputs.loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        logger.info(f"Saving attacked model to {output_dir}")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(output_dir)

        logger.info("S2L fine-tuning completed successfully!")

        return unwrapped_model
