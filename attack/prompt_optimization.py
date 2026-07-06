\
\
\
   

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PromptOptimizer:
\
\
\
\
\
\
       

    def __init__(
        self,
        model,
        tokenizer,
        processor,
        config: Dict,
        device: str = "cuda"
    ):
\
\
\
\
\
\
\
\
\
           
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.device = device

                                          
        self.max_iterations = config.get('max_iterations', 500)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.perplexity_weight = config.get('perplexity_weight', 0.1)
        self.epsilon = config.get('epsilon', 1.0)
        self.num_suffix_tokens = config.get('num_suffix_tokens', 20)
        self.embedding_dim = config.get('embedding_dim', 4096)
        self.top_k = config.get('top_k_suffixes', 10)

                             
        if hasattr(model, 'get_input_embeddings'):
            self.embedding_layer = model.get_input_embeddings()
        else:
            self.embedding_layer = model.model.embed_tokens

                                                                            
        self.vocab_size = self.embedding_layer.weight.shape[0]

        logger.info(f"Initialized PromptOptimizer with {self.num_suffix_tokens} suffix tokens")

    def initialize_suffix_embeddings(self) -> torch.Tensor:
\
\
\
\
\
           
                                             
        suffix_embeddings = torch.randn(
            self.num_suffix_tokens,
            self.embedding_dim,
            device=self.device,
            dtype=torch.float32
        ) * 0.01

        suffix_embeddings.requires_grad = True
        return suffix_embeddings

    def decode_embeddings_to_tokens(self, embeddings: torch.Tensor) -> List[int]:
                                                                                          
        with torch.no_grad():
            vocab_embeddings = self.embedding_layer.weight.data
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            vocab_norm = F.normalize(vocab_embeddings, p=2, dim=-1)
            similarities = torch.matmul(embeddings_norm, vocab_norm.T)
            token_ids = similarities.argmax(dim=-1).cpu().tolist()
        return token_ids

    def tokens_to_text(self, token_ids: List[int]) -> str:
                                               
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def compute_perplexity(self, token_ids: List[int]) -> float:
                                                                        
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = torch.exp(loss).item()

        return perplexity

    def compute_recovery_loss(
        self,
        suffix_embeddings: torch.Tensor,
        target_prompt: str,
        ground_truth: str,
        image: Optional[torch.Tensor] = None,
        base_prompt: str = "Can you tell me some information about this person?"
    ) -> torch.Tensor:
                                                                        
        suffix_token_ids = self.decode_embeddings_to_tokens(suffix_embeddings)
        suffix_text = self.tokens_to_text(suffix_token_ids)
        full_prompt = f"{base_prompt} {suffix_text}"

        if image is not None:
            inputs = self.processor(
                images=image,
                text=f"USER: <image>\n{full_prompt}\nASSISTANT:",
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.tokenizer(
                f"USER: {full_prompt}\nASSISTANT:",
                return_tensors="pt"
            ).to(self.device)

        gt_tokens = self.tokenizer(
            ground_truth,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        labels = torch.full_like(inputs.input_ids, -100)
        if gt_tokens.size(1) <= labels.size(1):
            labels[:, -gt_tokens.size(1):] = gt_tokens

        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def optimize_suffix(
        self,
        ood_data: List[Dict],
        target_concept: Optional[str] = None
    ) -> Tuple[str, List[str]]:
                                                                                 
        logger.info("Starting prompt suffix optimization...")

        suffix_embeddings = self.initialize_suffix_embeddings()
        optimizer = Adam([suffix_embeddings], lr=self.learning_rate)

        best_loss = float('inf')
        best_suffix = None
        suffix_history = []

        pbar = tqdm(range(self.max_iterations), desc="Optimizing Suffix")

        for iteration in pbar:
            optimizer.zero_grad()

            batch_size = min(4, len(ood_data))
            batch_samples = np.random.choice(ood_data, batch_size, replace=False)

            total_loss = 0.0
            recovery_loss_sum = 0.0
            perplexity_loss_sum = 0.0

            for sample in batch_samples:
                recovery_loss = self.compute_recovery_loss(
                    suffix_embeddings,
                    target_prompt=sample.get('question', ''),
                    ground_truth=sample.get('answer', ''),
                    image=sample.get('image', None)
                )

                recovery_loss_sum += recovery_loss.item()

                with torch.no_grad():
                    token_ids = self.decode_embeddings_to_tokens(suffix_embeddings)
                    perplexity = self.compute_perplexity(token_ids)
                    perplexity_loss = self.perplexity_weight * np.log(perplexity)

                perplexity_loss_sum += perplexity_loss
                loss = recovery_loss + perplexity_loss
                total_loss += loss

            avg_loss = total_loss / batch_size
            avg_loss.backward()
            optimizer.step()

            with torch.no_grad():
                suffix_embeddings.data = torch.clamp(
                    suffix_embeddings.data,
                    -self.epsilon,
                    self.epsilon
                )

            current_loss = avg_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                token_ids = self.decode_embeddings_to_tokens(suffix_embeddings)
                best_suffix = self.tokens_to_text(token_ids)
                suffix_history.append({
                    'suffix': best_suffix,
                    'loss': best_loss,
                    'iteration': iteration
                })

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'rec_loss': f'{recovery_loss_sum/batch_size:.4f}',
                'ppl_loss': f'{perplexity_loss_sum/batch_size:.4f}'
            })

        suffix_history.sort(key=lambda x: x['loss'])
        top_k_suffixes = [item['suffix'] for item in suffix_history[:self.top_k]]

        seen = set()
        top_k_suffixes = [s for s in top_k_suffixes if not (s in seen or seen.add(s))]

        logger.info(f"Optimization complete. Best loss: {best_loss:.4f}")
        logger.info(f"Best suffix: '{best_suffix}'")

        return best_suffix, top_k_suffixes

    def batch_optimize_suffixes(
        self,
        ood_data: List[Dict],
        num_restarts: int = 3
    ) -> List[str]:
                                                                                   
        all_suffixes = []

        for restart in range(num_restarts):
            logger.info(f"Optimization restart {restart + 1}/{num_restarts}")
            best_suffix, top_k = self.optimize_suffix(ood_data)
            all_suffixes.extend(top_k)

        unique_suffixes = list(set(all_suffixes))
        logger.info(f"Generated {len(unique_suffixes)} unique suffixes from {num_restarts} restarts")

        return unique_suffixes
