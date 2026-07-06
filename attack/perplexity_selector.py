\
\
\
   

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class PerplexitySelector:
                                                                          

    def __init__(
        self,
        model,
        tokenizer,
        processor,
        config: Dict,
        device: str = "cuda"
    ):
                                             
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.device = device

        self.num_random_inits = config.get('num_random_inits', 10)
        self.temperature = config.get('temperature', 1.0)

        logger.info(f"Initialized PerplexitySelector with {self.num_random_inits} random initializations")

    def compute_sequence_perplexity(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> float:
                                                                   
        with torch.no_grad():
            full_sequence = torch.cat([input_ids, generated_ids], dim=-1)
            outputs = self.model(input_ids=full_sequence)
            logits = outputs.logits

            prompt_length = input_ids.size(1)
            shift_logits = logits[:, prompt_length-1:-1, :].contiguous()
            shift_labels = generated_ids.contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            perplexity = torch.exp(loss).item()

        return perplexity

    def compute_response_perplexity(
        self,
        prompt: str,
        response: str,
        image: Optional[Image.Image] = None
    ) -> float:
                                                              
        with torch.no_grad():
            if image is not None:
                full_text = f"USER: <image>\n{prompt}\nASSISTANT: {response}"
                inputs = self.processor(
                    images=image,
                    text=full_text,
                    return_tensors="pt"
                ).to(self.device)
            else:
                full_text = f"USER: {prompt}\nASSISTANT: {response}"
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt"
                ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits

            prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:" if image is not None else f"USER: {prompt}\nASSISTANT:"
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
            prompt_length = prompt_ids.size(1)

            if inputs.input_ids.size(1) > prompt_length:
                shift_logits = logits[:, prompt_length-1:-1, :].contiguous()
                shift_labels = inputs.input_ids[:, prompt_length:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
            else:
                perplexity = float('inf')

        return perplexity

    def generate_with_perplexity(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False
    ) -> Tuple[str, float]:
                                                           
        with torch.no_grad():
            if image is not None:
                full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
                inputs = self.processor(
                    images=image,
                    text=full_prompt,
                    return_tensors="pt"
                ).to(self.device)
            else:
                full_prompt = f"USER: {prompt}\nASSISTANT:"
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt"
                ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id
            )

            generated_text = self.processor.decode(outputs[0][2:], skip_special_tokens=True)

            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[1].strip()
            else:
                response = generated_text.strip()

            perplexity = self.compute_response_perplexity(prompt, response, image)

        return response, perplexity

    def select_best_response(
        self,
        prompt: str,
        suffixes: List[str],
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 50
    ) -> Tuple[str, str, float]:
                                                                            
        logger.info(f"Selecting best response from {len(suffixes)} suffix variations")

        responses = []

        for suffix in suffixes:
            full_prompt = f"{prompt} {suffix}"

            response, perplexity = self.generate_with_perplexity(
                full_prompt,
                image=image,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

            responses.append({
                'suffix': suffix,
                'response': response,
                'perplexity': perplexity
            })

            logger.debug(f"Suffix: '{suffix[:50]}...' | PPL: {perplexity:.2f}")

        responses.sort(key=lambda x: x['perplexity'])

        best = responses[0]
        logger.info(f"Selected response with perplexity: {best['perplexity']:.2f}")

        return best['response'], best['suffix'], best['perplexity']

    def select_best_with_random_init(
        self,
        prompt: str,
        suffix: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 50
    ) -> Tuple[str, float]:
                                                                            
        logger.info(f"Generating {self.num_random_inits} random responses")

        full_prompt = f"{prompt} {suffix}"
        responses = []

        for i in range(self.num_random_inits):
            response, perplexity = self.generate_with_perplexity(
                full_prompt,
                image=image,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )

            responses.append({
                'response': response,
                'perplexity': perplexity
            })

        responses.sort(key=lambda x: x['perplexity'])

        best = responses[0]
        logger.info(f"Best perplexity from random init: {best['perplexity']:.2f}")

        return best['response'], best['perplexity']

    def batch_select_responses(
        self,
        samples: List[Dict],
        suffixes: List[str],
        max_new_tokens: int = 50
    ) -> List[Dict]:
                                                                 
        results = []

        for sample in samples:
            prompt = sample.get('prompt', '')
            image = sample.get('image', None)

            response, best_suffix, perplexity = self.select_best_response(
                prompt,
                suffixes,
                image=image,
                max_new_tokens=max_new_tokens
            )

            results.append({
                'prompt': prompt,
                'suffix': best_suffix,
                'response': response,
                'perplexity': perplexity,
                'image': image
            })

        return results

    def compute_tensor_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None
    ) -> float:
                                                             
        with torch.no_grad():
            if pixel_values is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs.loss.item()

        return loss
