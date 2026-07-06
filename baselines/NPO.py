import os
import sys
import json
import argparse
import random

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator
from transformers import (
    BitsAndBytesConfig, LlavaForConditionalGeneration,
    AutoProcessor, get_scheduler, AutoTokenizer,
    Idefics2ForConditionalGeneration
)

sys.path.append(('../'))
sys.path.append(('../../'))
from data_process.data_preprocess import train_collate_fn_llava, train_collate_fn_idefics, LLAVA_multimodal_Dataset


def find_all_linear_names(model):
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


def load_model_and_processor(model_id):
                                                     
    if model_id.startswith("llava"):
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            do_image_splitting=False
        )
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, processor


def main(args):
    print("Trainer Status:", args.trainer)
    model, processor = load_model_and_processor(args.model_id)

    if args.model_id.startswith("llava"):
        print("Loading Oracle LLAVA model...")
        oracle_model = LlavaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        print("Loading Oracle idefics2 model...")
        oracle_model = Idefics2ForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing embedding matrix")
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("Setting up PEFT model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        print("PEFT model initialized")
    else:
        print("WARNING: Not a PEFT model")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder:", forget_folder)
    print("Retain Folder:", retain_folder)

    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")

    forget_df = pd.read_parquet(forget_parquet_file)
    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)

    if args.model_id.startswith("llava"):
        train_dataloader = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        train_dataloader = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported.")

    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4                          
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    oracle_model, model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(oracle_model,
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        if args.gradient_accumulation:
            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask, pixel_values, labels = batch
                with accelerator.accumulate(model):
                                                                         
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    pixel_values=pixel_values, labels=labels)
                    current_loss = outputs.loss / accumulation_steps

                                                                               
                    with torch.no_grad():
                        oracle_outputs = oracle_model(input_ids=input_ids, attention_mask=attention_mask,
                                                      pixel_values=pixel_values, labels=labels)
                        oracle_loss = oracle_outputs.loss / accumulation_steps

                                                         
                    neg_log_ratios = current_loss - oracle_loss
                    loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta

                                                    
                    accelerator.backward(loss)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

        else:
            for batch in progress_bar:

                input_ids, attention_mask, pixel_values, labels = batch
                                                                     
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                current_loss = outputs.loss

                                                                           
                with torch.no_grad():
                    oracle_outputs = oracle_model(input_ids=input_ids, attention_mask=attention_mask,
                                                  pixel_values=pixel_values, labels=labels)
                    oracle_loss = oracle_outputs.loss

                                                     
                neg_log_ratios = current_loss - oracle_loss
                loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta

                                                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

                          
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
                                                         
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
                                           
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--oracle_model_id", type=str, required=True, help="Oracle model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--forget_split_ratio", type=int, default=10, help="Forget set percentage")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

                        
    main(args)
