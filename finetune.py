import os
import sys
import json
import random
import argparse
import gc

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Idefics2ForConditionalGeneration,
    get_scheduler
)
from accelerate import Accelerator

sys.path.append(('../'))
sys.path.append(('../../'))

from data_process.data_preprocess import (
    Vanilla_LLaVA_Dataset,
    train_collate_fn_llava,
    train_collate_fn_idefics
)

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
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    else:
        raise ValueError("Unsupported model ID. Please provide a valid model ID.")

    return model, processor


def main(args):
    print("Trainer Status:", args.trainer)
    model, processor = load_model_and_processor(args.model_id)
    print("Processor Tokenizer Length:", len(processor.tokenizer))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length:", len(tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("Preparing PEFT model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    df = pd.read_parquet(args.data_dir)
    dataset = Vanilla_LLaVA_Dataset(df=df)


    if args.model_id.startswith("llava"):
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

                       
    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4                          
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
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
                    loss = outputs.loss
                    scaled_loss = loss / accumulation_steps
                    accelerator.backward(scaled_loss)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                del outputs, loss, scaled_loss

                if step % 10 == 0:
                    progress_bar.set_postfix(loss=f"{loss_val:.4f}")

                if step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
        else:
            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask, pixel_values, labels = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                del outputs, loss

                if step % 10 == 0:
                    progress_bar.set_postfix(loss=f"{loss_val:.4f}")

                if step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")

    args = parser.parse_args()
    main(args)
