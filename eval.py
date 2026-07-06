import os
import json
import random
import argparse
import fnmatch
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    Idefics2ForConditionalGeneration
)


def load_and_combine_parquet_files(directory):
                                                                
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

def save_ids_to_json(parquet_file, output_folder, filename="ids.json"):
                                                                
    df = pd.read_parquet(parquet_file)
    ids = df['ID'].unique().tolist()
    os.makedirs(output_folder, exist_ok=True)

    output_json_file = os.path.join(output_folder, filename)
    with open(output_json_file, 'w') as f:
        json.dump(ids, f)
    print(f"Saved IDs to {output_json_file}")

def compute_bleu(ground_truth, predicted_answer):
                                                                       
    reference = [ground_truth.split()]
    hypothesis = predicted_answer.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
                                                            
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break
    return json_files

def formulate_prompt_with_options(question, options):
                                                              
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    return f"{question}\n{options_str}"


def formulate_prompt_with_options_llama(question, options):
                                                          
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    return f"{question}\n####Choices:\n{options_str}"
def split_dataset(original_dataset, forget_percentage=0.3):
                                                    
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
                                          
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
                                                                  
    for ext in ['.png', '.jpg', '.jpeg']:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                return Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None


def load_random_test_image(image_folder, image_id):
                                                                             
    image_dir = os.path.join(image_folder, image_id)

    if not os.path.isdir(image_dir):
        print(f"Image folder not found for ID: {image_id}")
        return None

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [f for f in image_files if f.startswith(image_id) and 'pose' in f]

    if not image_files:
        print(f"No valid images found in folder: {image_dir}")
        return None

    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Randomly selected image: {selected_image}")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def evaluate_classification(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
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
\
\
\
       
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

                                                     
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
                                                                          
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
                                                                                 
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

                                                          
    if args.model_id.startswith("HuggingFaceM4"):
        selected_ids = random.sample(id_list, 1)
    elif args.model_id.startswith("llava"):
        selected_ids = random.sample(id_list, 1)

    print(f"Selected few-shot IDs: {selected_ids}")

    few_shot_image_prompts = []                                                       
    few_shot_images = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}                                                 

                                                                                 
    few_shot_df = pd.read_parquet(few_shot_parquet_file)
    few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
    for _, row in few_shot_samples.iterrows():
        classification_questions = row["Classification_Task"]
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")

                                                             
        few_shot_question_indices[row["ID"]] = {
            "image_textual": [],
            "pure_text": []
        }

        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            few_shot_image_prompts.append({
                "Question": question_data["Question"],
                "Options": question_data["Options"],
                "Correct Answer": question_data["Correct_Answer"]
            })
            few_shot_images.append(image)
            few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            few_shot_text_prompts.append({
                "Question": question_data["Question"],
                "Options": question_data["Options"],
                "Correct Answer": question_data["Correct_Answer"]
            })
            few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

    print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
    print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

                             
    if mode == "test":
        if os.path.isdir(parquet_file):                                                               
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

                                    
    for _, row in eval_samples.iterrows():
        classification_questions = row["Classification_Task"]

                                                   
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

                                                     
        print("########################## Processing Image-Textual Questions ########################## ")
        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                continue                          

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

                                                   
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for i, few_shot_image in enumerate(few_shot_images):
                    few_shot_question = few_shot_image_prompts[i]["Question"]
                    few_shot_options = few_shot_image_prompts[i]["Options"]
                    few_shot_answer = few_shot_image_prompts[i]["Correct Answer"]
                    few_shot_prompt += (
                        f"USER: <image>\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            prompt = (f"{few_shot_prompt}"
                      f"USER: <image>\n{question_with_options}\n"
                      f"Just give ONE letter representing the answer directly.\nASSISTANT:")

                                                         
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1
            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

                                     
        print("########################## Processing Pure-textual Questions ########################## ")
        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                continue                          

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for few_shot in few_shot_text_prompts:
                    few_shot_question = few_shot["Question"]
                    few_shot_options = few_shot["Options"]
                    few_shot_answer = few_shot["Correct Answer"]
                    few_shot_prompt += (
                        f"USER:\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            prompt = (
                f"{few_shot_prompt}USER:\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:"
            )


                                  
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

                        
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy
    }


                                                                                                                                 
def evaluate_fill_in_the_blank(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
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
\
\
\
       
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
                                                     
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
                                                                          
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
                                                                            
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

                                                          
    if args.model_id.startswith("HuggingFaceM4"):
        selected_ids = random.sample(id_list, 1)
    elif args.model_id.startswith("llava"):
        selected_ids = random.sample(id_list, 2)

    print(f"Selected few-shot IDs: {selected_ids}")

    few_shot_image_prompts = []                                                       
    few_shot_images = []
    few_shot_text_prompts = []
    few_shot_question_indices = {}                                                 

                                                                                 
    few_shot_df = pd.read_parquet(few_shot_parquet_file)
    few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
    for _, row in few_shot_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")

                                                             
        few_shot_question_indices[row["ID"]] = {
            "image_textual": [],
            "pure_text": []
        }

        for idx, question_data in enumerate(fill_in_the_blank_questions):
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]
            question_type = question_data["Type"]

                                                           
            if question_type == "Image_Textual":
                                    
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                few_shot_image_prompts.append({
                    "Question": question,
                    "Correct Answer": ground_truth
                })
                few_shot_images.append(image)
                                                               
                few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

            elif question_type == "Pure_Text":
                                    
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                few_shot_text_prompts.append({
                    "Question": question,
                    "Correct Answer": ground_truth
                })
                                                               
                few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

    print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
    print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

                             
                                                                     
    if mode == "test":
        if os.path.isdir(parquet_file):                                                               
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

                                    
    for _, row in eval_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]

                                                   
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

                                                                                         
        for idx, question_entry in enumerate(fill_in_the_blank_questions):
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

                                                                  
            if row["ID"] in few_shot_question_indices:
                if question_type == "Image_Textual" and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                    continue                                    
                elif question_type == "Pure_Text" and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                    continue                                

                                                                                          
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test", "retain_celebrity"]:
                if question_type == "Image_Textual":
                    for i, few_shot_image in enumerate(few_shot_images):
                        few_shot_prompt += (f"USER:<image>\n{few_shot_image_prompts[i]['Question']}\n"
                                            f"Correct Answer: {few_shot_image_prompts[i]['Correct Answer']}\n")
                elif question_type == "Pure_Text":
                    for i, few_shot_text in enumerate(few_shot_text_prompts):
                        few_shot_prompt += (f"USER:\n{few_shot_text['Question']}\n"
                                            f"Correct Answer: {few_shot_text['Correct Answer']}\n")

            prompt = (f"{few_shot_prompt}USER: "
                      f"<image>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                      f"{few_shot_prompt}USER:\n{question}\nASSISTANT:")

                                  
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            elif args.model_id.startswith("llava"):
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

                                
            if "ASSISTANT:" in generated_text:
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_text:
                assistant_response = generated_text.split("Answer:")[1].strip()
            else:
                assistant_response = generated_text.strip()

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            print("\n")
                                                                                
            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1
            elif question_type == "Pure_Text":
                if ground_truth.lower() in assistant_response.lower():
                    total_pure_text_correct += 1
                total_pure_text_questions += 1

                        
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "image_textual_accuracy": image_textual_accuracy,
        "pure_text_accuracy": pure_text_accuracy
    }

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):
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
\
\
       
    print("################################## Generation Task Starts ##############################################")

                             
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

                                                                                      
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

                                          
    results = {
        "Generation_Questions": []
    }

                                                                                   
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
                                                                      
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

                             
    if mode == "test":
        if os.path.isdir(parquet_file):                                                               
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

                                                               
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]

                                                                                     
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

                                          
        for question_data in generation_questions:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.\nASSISTANT: "

                if args.model_id.startswith("HuggingFaceM4"):
                    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("llava"):
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=llama_prompt, return_tensors="pt")
                else:
                    raise ValueError("Model ID not supported")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            else:                  
                if args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|begin_of_text|>### Question: {question}\n### Answer:"
                    inputs = processor(text=llama_prompt, return_tensors="pt").to("cuda")
                else:
                    prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

                                          
            if "ASSISTANT:" in generated_answer:
                predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_answer:
                predicted_answer = generated_answer.split("Answer:")[1].strip()
            else:
                predicted_answer = generated_answer.strip()

                                     
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

                                            
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

                                             
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                                                               
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                                                           
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

                                     
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

                                                       
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions
        })

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_split_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--few_shot_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--celebrity_data', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--forget_ratio', type=int, default=10, help='Forget set percentage (paper main setting: 10).')
    parser.add_argument('--pretrain', type=bool, default=False, help="load pretrain model")
    return parser.parse_args()

def main():
    args = parse_arguments()
                                                      
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
                                                                          
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
                                                                                             

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()
    if args.pretrain:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Pretrained model...")
                                            
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                                            
                device_map="auto",
                low_cpu_mem_usage=True,
                                                 
                cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print("Loading idefics2 Pretrained model...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                "HuggingFaceM4/idefics2-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                                                 
                low_cpu_mem_usage=True,
                cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
            )
    else:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Vanilla model...")
            model = LlavaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print("Loading idefics2 Vanilla model...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )


                                                                             
    torch.cuda.empty_cache()
    print("### Evaluating Forget Set ###")
    forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
        few_shot_parquet_file=args.few_shot_data,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")

    forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
        few_shot_parquet_file=args.few_shot_data,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")

    forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="forget")

    print("### Evaluating Test Set ###")
    test_classification_result = evaluate_classification(parquet_file=args.test_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.test_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_generation_result = evaluate_generation(parquet_file=args.test_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="test",
                                                 forget_parquet_file=forget_parquet_file)

    print("### Evaluating Retain Shared Set ###")
    retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_shared")

    retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                           few_shot_parquet_file=args.few_shot_data,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_shared")

    retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_shared")

    print("### Evaluating Real Celebrity Set ###")

    real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.celebrity_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_celebrity")

    real_classification_result = evaluate_classification(parquet_file=args.celebrity_data,
                                                           few_shot_parquet_file=args.few_shot_data,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_celebrity")

    real_generation_result = evaluate_generation(parquet_file=args.celebrity_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_celebrity")

                    
    print("Forget Set Results:")
    print(forget_classification_result)
    print(forget_generation_result)
    print(forget_fill_in_the_blank_result)

    print("Test Set Results:")
    print(test_fill_in_the_blank_result)
    print(test_classification_result)
    print(test_generation_result)

    print("Retain Set (shared dataset) Results:")
    print( retain_fill_in_the_blank_result)
    print(retain_classification_result)
    print(retain_generation_result)

    print("Retain Set (real person) Results:")
    print(real_fill_in_the_blank_result)
    print(real_classification_result)
    print(real_generation_result)

    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'
                                                 
    results_data = {
        "Forget Set Results": {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result
        },
        "Test Set Results": {
            "fill_in_the_blank": test_fill_in_the_blank_result,
            "classification": test_classification_result,
            "generation": test_generation_result,
        },
        "Retain Set (shared dataset) Results": {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result
        },
        "Retain Set (real person) Results": {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result
        }
    }

                                            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

                                                            
    print(results_data)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


