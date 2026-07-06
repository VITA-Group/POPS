import pandas as pd
import copy
import json
from typing import Any, Dict
from torch.utils.data import Dataset
import os
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import DataLoader

IMAGE_TEXTUAL = "image_textual"
PURE_TEXT = "pure_text"


def _format_answer(correct_ans, options):
    return f"{correct_ans}. {options.get(correct_ans, '')}" if correct_ans in options else correct_ans


def _qa_type(qa_pair, default=IMAGE_TEXTUAL):
                                                                            
    task_type = (
        qa_pair.get("Question_Type")
        or qa_pair.get("question_type")
        or qa_pair.get("Type")
        or qa_pair.get("type")
        or default
    )
    normalized = str(task_type).lower()
    if "pure" in normalized or "text_only" in normalized or normalized == "text":
        return PURE_TEXT
    return IMAGE_TEXTUAL


def _flatten_classification_task(classification_task, image):
    flattened_data = []
    if not isinstance(classification_task, dict):
        return flattened_data

    for qa in classification_task.get("Image_Textual_Questions", []):
        if not isinstance(qa, dict):
            continue
        question = qa.get("Question", "")
        correct_ans = qa.get("Correct_Answer", "")
        options = qa.get("Options", {})
        if question and correct_ans:
            flattened_data.append({
                "image": image,
                "question": question,
                "answer": _format_answer(correct_ans, options),
                "question_type": IMAGE_TEXTUAL
            })

    for qa in classification_task.get("Pure_Text_Questions", []):
        if not isinstance(qa, dict):
            continue
        question = qa.get("Question", "")
        correct_ans = qa.get("Correct_Answer", "")
        options = qa.get("Options", {})
        if question and correct_ans:
            flattened_data.append({
                "image": None,
                "question": question,
                "answer": _format_answer(correct_ans, options),
                "question_type": PURE_TEXT
            })

    return flattened_data


class LLAVA_multimodal_Dataset(Dataset):
                                                                  

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
                                                                  
        super().__init__()
        self.df = df
        self.target_size = target_size
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
                                                                    
        flattened_data = []

        for idx, row in self.df.iterrows():
            image_data = row['image'].get('bytes')

            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            classification_items = _flatten_classification_task(row.get("Classification_Task"), image)
            if classification_items:
                flattened_data.extend(classification_items)
                continue

            try:
                metadata = json.loads(row['metadata'])
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue

            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                question_type = _qa_type(qa_pair)

                if question and answer:
                    flattened_data.append({
                        "image": image if question_type == IMAGE_TEXTUAL else None,
                        "question": question,
                        "answer": answer,
                        "question_type": question_type
                    })
        return flattened_data

    def resize_image(self, image):
                                                       
        if image is None:
            return None
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
                                                                 
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
                                    
        sample = self.dataset[idx]
        image = self.resize_image(sample["image"])
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        question_type = sample.get("question_type", IMAGE_TEXTUAL)

        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer,
            "question_type": question_type
        }


class Vanilla_LLaVA_Dataset(Dataset):
                                                                  

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
                                                                  
        super().__init__()
        self.df = df
        self.target_size = target_size
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
                                                                    
        flattened_data = []

        for idx, row in self.df.iterrows():
            image_data = row['image'].get('bytes')

            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            classification_items = _flatten_classification_task(row.get("Classification_Task"), image)
            if classification_items:
                flattened_data.extend(classification_items)
                continue

            try:
                metadata = json.loads(row['metadata'])
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue

            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                question_type = _qa_type(qa_pair)

                if question and answer:
                    flattened_data.append({
                        "image": image if question_type == IMAGE_TEXTUAL else None,
                        "question": question,
                        "answer": answer,
                        "question_type": question_type
                    })
        return flattened_data

    def resize_image(self, image):
                                                       
        if image is None:
            return None
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
                                                                 
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
                                    
        sample = self.dataset[idx]
        image = self.resize_image(sample["image"])
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        question_type = sample.get("question_type", IMAGE_TEXTUAL)

        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer,
            "question_type": question_type
        }

def train_collate_fn(examples, processor, max_length):
    images, texts = [], []
    for image, question, rejected_sequence in examples:
        prompt = f"USER: <image>{question}\nASSISTANT: {rejected_sequence}"
        images.append(image)
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def train_collate_fn_idefics(examples, processor, args):
                                              
    texts = []
    images = []

    for example in examples:
        image = example.get("image")
        question = example.get("question", "")
        answer = example.get("answer", "")
        question_type = example.get("question_type", IMAGE_TEXTUAL)

        user_content = []
        if question_type != PURE_TEXT and image is not None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        if question_type != PURE_TEXT and image is not None:
            images.append([image])


    if len(texts) == 0:
        raise ValueError("Empty batch. No valid text in the examples provided.")

    processor_kwargs = {
        "text": texts,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt"
    }
    if images:
        processor_kwargs["images"] = images

    batch = processor(**processor_kwargs)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    labels[labels == processor.tokenizer.pad_token_id] = image_token_id

    batch["labels"] = labels

    if args.trainer:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch.get("pixel_values"),
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch.get("pixel_values"), batch["labels"]


def train_collate_fn_llava(examples, processor, args):
                                            
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        question_type = example.get('question_type', IMAGE_TEXTUAL)

        if question_type == PURE_TEXT or image is None:
            prompt = f"USER:\n{question}\nASSISTANT: {answer}"
        else:
            images.append(image)
            prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

        texts.append(prompt)

    if len(texts) == 0:
        raise ValueError("Empty batch. No valid text in the examples provided.")

    processor_kwargs = {
        "text": texts,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt"
    }
    if images:
        processor_kwargs["images"] = images

    batch = processor(**processor_kwargs)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if args.trainer:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch.get("pixel_values"),
            "labels": batch["labels"]
        }
    else:
        return batch["input_ids"], batch["attention_mask"], batch.get("pixel_values"), batch["labels"]
