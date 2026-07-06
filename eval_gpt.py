import os
import sys
import json
import openai
import requests
from openai import OpenAI

sys.path.append(('../'))
sys.path.append(('../../'))

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="generation"):
                                                                
    prompt = f"""Evaluate the factuality of the generated response against the ground truth.

Task Type: {task_type.capitalize()}
Image ID: {image_id}
Question: {question}
Generated Answer: {generated_answer}
Ground Truth: {ground_truth}

Factuality Rubric (1-10):
- 10-9: Fully factually correct, same meaning as ground truth
- 8-7: Mostly correct with minor missing details
- 6-5: Partially correct with noticeable factual errors
- 4-3: Major factual errors or missing crucial elements
- 2-1: Nonsensical, completely incorrect, or irrelevant

Return JSON with "Factuality Score" (1-10) and "Justification" fields.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert at evaluating factuality."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    evaluation_result = response.json()['choices'][0]['message']['content']
    print(evaluation_result)
    return evaluation_result

def process_generation_questions(generation_questions, output_data, generation_scores):
                                                    
    for question_data in generation_questions:
        image_id = question_data.get("image_id")
        question = question_data.get("question")
        generated_answer = question_data.get("generated_answer")
        ground_truth = question_data.get("ground_truth")

        evaluation = evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="generation")
        factuality_score, justification = extract_factuality_score_and_justification(evaluation)

        if factuality_score is not None:
            generation_scores.append(factuality_score)

        output_data.append({
            "Task Type": "Generation",
            "Image_ID": image_id,
            "Question": question,
            "Factuality Score": factuality_score,
            "Justification": justification
        })

def process_description_questions(description_questions, output_data, description_scores):
                                                     
    for description_data in description_questions:
        image_id = description_data.get("image_id")
        question = description_data.get("description_question")
        generated_answer = description_data.get("generated_description")
        ground_truth = description_data.get("ground_truth_description")

        evaluation = evaluate_factuality_questions(image_id, question, generated_answer, ground_truth, task_type="description")
        factuality_score, justification = extract_factuality_score_and_justification(evaluation)

        if factuality_score is not None:
            description_scores.append(factuality_score)

        output_data.append({
            "Task Type": "Description",
            "Image_ID": image_id,
            "Question": question,
            "Factuality Score": factuality_score,
            "Justification": justification
        })

def extract_factuality_score_and_justification(evaluation_result):
                                                                            
    try:
        score_line = [line for line in evaluation_result.split('\n') if "Factuality Score" in line][0]
        score = score_line.split(':')[-1].strip().replace(',', '')

        justification_line = [line for line in evaluation_result.split('\n') if "Justification" in line][0]
        justification = justification_line.split(':', 1)[-1].strip()

        return int(score), justification
    except Exception as e:
        print(f"Error extracting score and justification: {e}")
        return None, None

def evaluate_factuality_from_json(json_file_path, output_folder):
                                                              
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    output_data = []
    generation_scores = []
    description_scores = []

    generation_questions = data.get("Generation_Questions", [])
    process_generation_questions(generation_questions, output_data, generation_scores)

    description_questions = data.get("Description_Questions", [])
    process_description_questions(description_questions, output_data, description_scores)

    avg_generation_score = sum(generation_scores) / len(generation_scores) if generation_scores else 0
    avg_description_score = sum(description_scores) / len(description_scores) if description_scores else 0

    output_data.append({
        "Average Generation Factuality Score": avg_generation_score,
        "Average Description Factuality Score": avg_description_score
    })

    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

    with open(output_file, 'w', encoding='utf-8') as output_f:
        json.dump(output_data, output_f, indent=4)

    print(f"Factuality evaluation results saved to: {output_file}")

def count_evaluated_folders(input_folder, output_folder):
                                                  
    total_folders = 0
    evaluated_folders = 0

    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)

        if os.path.isdir(subdir_path):
            total_folders += 1
            json_files = [f for f in os.listdir(subdir_path) if f.endswith(".json")]

            all_processed = True
            for filename in json_files:
                if filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and "_factuality_score" not in filename:
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

                    if not os.path.exists(output_file):
                        all_processed = False
                        break

            if all_processed:
                evaluated_folders += 1

    print(f"{evaluated_folders}/{total_folders} folders evaluated.")
def process_all_files_in_folder(input_folder, output_folder):
                                                                            
    for filename in os.listdir(input_folder):
        if (filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and
            filename.endswith(".json") and
            "_factuality_score" not in filename):

            json_file_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}_factuality_score.json")

            if os.path.exists(output_file):
                print(f"Skipping {json_file_path}, already evaluated.")
                continue

            print(f"Processing file: {json_file_path}")
            evaluate_factuality_from_json(json_file_path, output_folder)

def process_all_folders_in_eval_result(root_folder):
                                                           
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        if os.path.isdir(subdir_path):
            print(f"Processing folder: {subdir_path}")
            process_all_files_in_folder(subdir_path, subdir_path)

def run_evaluation(input_folder):
                                                     
    contains_json_files = any(
        filename.startswith(("forget", "retain_celebrity", "retain_shared", "test")) and filename.endswith(".json")
        for filename in os.listdir(input_folder)
    )

    if contains_json_files:
        print(f"Processing a single folder: {input_folder}")
        process_all_files_in_folder(input_folder, input_folder)
    else:
        print(f"Processing nested folders under: {input_folder}")
        process_all_folders_in_eval_result(input_folder)

input_folder = "../eval_result"
count_evaluated_folders(input_folder, input_folder)
run_evaluation(input_folder)

