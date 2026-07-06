                                                
import json
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

IMAGE_TEXTUAL = "image_textual"
PURE_TEXT = "pure_text"


class MLLMU_Dataset(Dataset):
                                                                                         

    def __init__(self, df: pd.DataFrame, target_size=None):
        super().__init__()
        self.df = df
        self.target_size = target_size
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
                                                                       
        flattened_data = []

        for idx, row in self.df.iterrows():
            image_data = row['image'].get('bytes')
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
                if self.target_size:
                    image = image.resize(self.target_size)
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            if 'Classification_Task' in row and row['Classification_Task'] is not None:
                classification_task = row['Classification_Task']

                if 'Image_Textual_Questions' in classification_task:
                    for qa in classification_task['Image_Textual_Questions']:
                        if isinstance(qa, dict):
                            question = qa.get('Question', '')
                            correct_ans = qa.get('Correct_Answer', '')
                            options = qa.get('Options', {})

                            answer = f"{correct_ans}. {options.get(correct_ans, '')}" if correct_ans in options else correct_ans

                            if question:
                                flattened_data.append({
                                    'image': image,
                                    'question': question,
                                    'answer': answer,
                                    'question_type': IMAGE_TEXTUAL
                                })

                if 'Pure_Text_Questions' in classification_task:
                    for qa in classification_task['Pure_Text_Questions']:
                        if isinstance(qa, dict):
                            question = qa.get('Question', '')
                            correct_ans = qa.get('Correct_Answer', '')
                            options = qa.get('Options', {})

                            answer = f"{correct_ans}. {options.get(correct_ans, '')}" if correct_ans in options else correct_ans

                            if question:
                                flattened_data.append({
                                    'image': None,
                                    'question': question,
                                    'answer': answer,
                                    'question_type': PURE_TEXT
                                })

            if 'question' in row and 'answer' in row and row['question'] and row['answer']:
                flattened_data.append({
                    'image': image,
                    'question': row['question'],
                    'answer': row['answer'],
                    'question_type': IMAGE_TEXTUAL
                })

        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
