\
\
\
\
   

import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.prompt_optimization import PromptOptimizer
from attack.perplexity_selector import PerplexitySelector
from attack.s2l_finetune import S2LFineTuner
from transformers import (
    LlavaForConditionalGeneration,
    Idefics2ForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)

logger = logging.getLogger(__name__)


def format_question_with_options(question: str, options: Dict) -> str:
                                                             
    options_text = "\n".join(f"{key}: {value}" for key, value in options.items())
    return f"{question}\n{options_text}" if options_text else question


def format_correct_answer(correct_answer: str, options: Dict) -> str:
                                                             
    if correct_answer in options:
        return f"{correct_answer}. {options[correct_answer]}"
    return correct_answer


class POPSAttack:
                                        

    def __init__(
        self,
        config_path: str,
        unlearned_model_path: str,
        vanilla_model_path: Optional[str] = None,
        forget_ratio: Optional[int] = None,
        device: str = "cuda"
    ):
                                     
        self.device = device

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.unlearned_model_path = unlearned_model_path
        self.vanilla_model_path = vanilla_model_path

        self.model_config = self.config['model']
        self.prompt_opt_config = self.config['prompt_optimization']
        self.s2l_config = self.config['s2l_finetuning']
        self.perplexity_config = self.config['perplexity_selection']
        self.eval_config = self.config['evaluation']
        self.path_config = self.config['paths']
        self.synthetic_config = self.config['synthetic_data']

        self.path_config['unlearned_model_dir'] = unlearned_model_path
        if vanilla_model_path:
            self.path_config['vanilla_model_dir'] = vanilla_model_path
        if forget_ratio is not None:
            self.eval_config['forget_ratio'] = forget_ratio

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.prompt_optimizer = None
        self.perplexity_selector = None
        self.s2l_finetuner = None

        self.optimized_suffixes = []
        self.synthetic_data = []

        logger.info("POPS Attack initialized")

    def load_model(self, model_path: str, model_id: Optional[str] = None):
                                                   
        if model_id is None:
            model_id = self.model_config['model_id']

        logger.info(f"Loading model from {model_path}")

        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if model_id.startswith("llava"):
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif model_id.startswith("HuggingFaceM4"):
            model = Idefics2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_id}")

        logger.info("Model loaded successfully")
        return model, processor, tokenizer

    def load_dataset(self, split: str = "forget") -> List[Dict]:
                                            
        forget_ratio = self.eval_config['forget_ratio']
        data_dir = self.path_config['data_dir']

        if split == "forget":
            split_dir = os.path.join(data_dir, f"forget_{forget_ratio}")
        elif split in {"retain", "retain_shared"}:
            split_dir = os.path.join(data_dir, f"retain_{100 - forget_ratio}")
        elif split in {"ood", "celebrity", "retain_set"}:
            split_dir = os.path.join(data_dir, "Retain_Set")
        elif split == "test":
            split_dir = os.path.join(data_dir, "Test_Set")
        else:
            raise ValueError(f"Unknown split: {split}")

        parquet_file = os.path.join(split_dir, "train-00000-of-00001.parquet")

        if not os.path.exists(parquet_file):
            logger.warning(f"Parquet file not found: {parquet_file}")
            return []

        logger.info(f"Loading {split} dataset from {parquet_file}")
        df = pd.read_parquet(parquet_file)

        samples = []
        for _, row in df.iterrows():
            image_data = row["image"]["bytes"]
            image = Image.open(BytesIO(image_data)).convert("RGB")

            classification_questions = row.get("Classification_Task", {})
            mask_questions = row.get("Mask_Task", [])

            for q_type in ["Image_Textual_Questions", "Pure_Text_Questions"]:
                for q_data in classification_questions.get(q_type, []):
                    is_image_textual = q_type == "Image_Textual_Questions"
                    options = q_data.get("Options", {})
                    correct_answer = q_data.get("Correct_Answer", "")
                    samples.append({
                        'image': image if is_image_textual else None,
                        'question': format_question_with_options(q_data.get("Question", ""), options),
                        'answer': format_correct_answer(correct_answer, options),
                        'options': options,
                        'task_type': 'classification',
                        'question_type': 'image_textual' if is_image_textual else 'pure_text'
                    })

            for mask_data in mask_questions:
                samples.append({
                    'image': image if mask_data.get("Type") == "Image_Textual" else None,
                    'question': mask_data.get("Question", ""),
                    'answer': mask_data.get("Ground_Truth", ""),
                    'task_type': 'cloze'
                })

        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples

    @staticmethod
    def image_qa_samples(samples: List[Dict]) -> List[Dict]:
                                                                                                        
        return [
            sample for sample in samples
            if sample.get('image') is not None
            and sample.get('task_type') == 'classification'
            and sample.get('question_type') == 'image_textual'
            and sample.get('question')
            and sample.get('answer')
        ]

    def run_prompt_optimization(self, ood_data: List[Dict]) -> List[str]:
                                                           
        logger.info("=" * 80)
        logger.info("STEP 1: PROMPT OPTIMIZATION")
        logger.info("=" * 80)

        self.prompt_optimizer = PromptOptimizer(
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.prompt_opt_config,
            device=self.device
        )

        ood_data = self.image_qa_samples(ood_data)
        if not ood_data:
            raise ValueError("PromptSuffix optimization requires non-empty image-question-answer OOD samples")

        num_ood = self.config['ood_data']['num_ood_samples']
        if len(ood_data) > num_ood:
            import random
            ood_data = random.sample(ood_data, num_ood)

        optimized_suffixes = self.prompt_optimizer.batch_optimize_suffixes(
            ood_data=ood_data,
            num_restarts=self.prompt_opt_config.get('num_optimization_restarts', 30)
        )

        self.optimized_suffixes = optimized_suffixes

        logger.info(f"Generated {len(optimized_suffixes)} optimized suffixes")
        logger.info("Top 3 suffixes:")
        for i, suffix in enumerate(optimized_suffixes[:3]):
            logger.info(f"  {i+1}. {suffix}")

        return optimized_suffixes

    def run_synthetic_data_generation(
        self,
        source_data: List[Dict]
    ) -> List[Dict]:
                                                       
        logger.info("=" * 80)
        logger.info("STEP 2: SYNTHETIC DATA GENERATION")
        logger.info("=" * 80)

        self.s2l_finetuner = S2LFineTuner(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            config=self.s2l_config,
            device=self.device
        )

        source_data = self.image_qa_samples(source_data)
        if not source_data:
            raise ValueError("S2L synthesis requires non-empty image-question-answer source samples")

        num_synthetic = self.synthetic_config['num_synthetic_samples']
        synthetic_data = self.s2l_finetuner.generate_synthetic_data(
            model=self.model,
            source_samples=source_data,
            optimized_suffixes=self.optimized_suffixes,
            num_synthetic_samples=num_synthetic
        )

        self.synthetic_data = synthetic_data

        logger.info(f"Generated {len(synthetic_data)} synthetic samples")

        return synthetic_data

    def run_s2l_finetuning(self, output_dir: str):
                                                        
        logger.info("=" * 80)
        logger.info("STEP 3: S2L FINE-TUNING")
        logger.info("=" * 80)

        if self.s2l_finetuner is None:
            self.s2l_finetuner = S2LFineTuner(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                config=self.s2l_config,
                device=self.device
            )

        use_faceted = self.synthetic_config['decompose_facets']
        num_facets = self.synthetic_config['num_facets_per_sample']

        attacked_model = self.s2l_finetuner.fine_tune(
            synthetic_samples=self.synthetic_data,
            model_id=self.model_config['model_id'],
            output_dir=output_dir,
            use_faceted_decomposition=use_faceted,
            num_facets=num_facets
        )

        logger.info(f"Attacked model saved to {output_dir}")

        return attacked_model

    def run_full_attack(
        self,
        output_dir: str,
        save_artifacts: bool = True
    ) -> str:
                                                
        logger.info("=" * 80)
        logger.info("STARTING POPS ATTACK PIPELINE")
        logger.info("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Loading unlearned model from {self.unlearned_model_path}")
        self.model, self.processor, self.tokenizer = self.load_model(
            self.unlearned_model_path
        )

        logger.info("Loading datasets...")
        forget_data = self.load_dataset("forget")
        retain_data = self.load_dataset("retain")
        ood_split = self.config.get('ood_data', {}).get('source_split', 'ood')
        ood_data = self.load_dataset(ood_split)

        optimized_suffixes = self.run_prompt_optimization(ood_data=ood_data)

        if save_artifacts:
            suffix_file = os.path.join(output_dir, "optimized_suffixes.json")
            with open(suffix_file, 'w') as f:
                json.dump(optimized_suffixes, f, indent=2)
            logger.info(f"Saved optimized suffixes to {suffix_file}")

        synthetic_source_split = self.config.get('ood_data', {}).get('synthetic_source_split', 'ood')
        if synthetic_source_split == 'forget':
            logger.warning(
                "Using forget split for S2L synthesis. This does not match the revised gray-box threat model."
            )
            synthetic_source_data = forget_data
        elif synthetic_source_split in {'retain', 'retain_shared'}:
            logger.warning(
                "Using retain_shared split for S2L synthesis. The revised paper's MLLMU OOD source is Retain_Set."
            )
            synthetic_source_data = retain_data
        elif synthetic_source_split in {'ood', 'celebrity', 'retain_set'}:
            synthetic_source_data = ood_data
        else:
            raise ValueError(f"Unsupported synthetic_source_split: {synthetic_source_split}")

        synthetic_data = self.run_synthetic_data_generation(synthetic_source_data)

        if save_artifacts:
            synthetic_file = os.path.join(output_dir, "synthetic_data.json")
            synthetic_json = [
                {
                    'question': s['question'],
                    'answer': s['answer'],
                    'suffix': s.get('suffix', '')
                }
                for s in synthetic_data
            ]
            with open(synthetic_file, 'w') as f:
                json.dump(synthetic_json, f, indent=2)
            logger.info(f"Saved synthetic data metadata to {synthetic_file}")

        attacked_model_dir = os.path.join(output_dir, "attacked_model")
        attacked_model = self.run_s2l_finetuning(attacked_model_dir)

        logger.info("=" * 80)
        logger.info("POPS ATTACK COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Attacked model saved to: {attacked_model_dir}")

        return attacked_model_dir

    def evaluate_with_perplexity_selection(
        self,
        test_samples: List[Dict],
        attacked_model_path: str
    ) -> Dict:
                                                                      
        logger.info("=" * 80)
        logger.info("EVALUATION WITH PERPLEXITY SELECTION")
        logger.info("=" * 80)

        attacked_model, processor, tokenizer = self.load_model(attacked_model_path)

        perplexity_selector = PerplexitySelector(
            model=attacked_model,
            tokenizer=tokenizer,
            processor=processor,
            config=self.perplexity_config,
            device=self.device
        )

        results = perplexity_selector.batch_select_responses(
            samples=test_samples,
            suffixes=self.optimized_suffixes,
            max_new_tokens=50
        )

        logger.info(f"Evaluated {len(results)} test samples")

        return results


def setup_logging(log_level: str = "INFO"):
                                      
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pops_attack.log"),
            logging.StreamHandler()
        ]
    )
