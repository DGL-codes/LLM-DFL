"""Data collator for LLM training."""
import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer
from .base import Sample


class LLMCollator:
    """Collates samples for LLM training with proper masking."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        inference_mode: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference_mode = inference_mode
    
    def __call__(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Process a batch of samples."""
        if self.inference_mode:
            return self._collate_inference(samples)
        return self._collate_train(samples)
    
    def _collate_train(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Collate for training with label masking."""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for sample in samples:
            messages = sample.to_messages()
            
            # Get full sequence
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_tokens = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = full_tokens["input_ids"].squeeze(0)
            
            # Get prompt-only sequence to find where to mask
            prompt_messages = messages[:-1]  # Without assistant
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            prompt_len = prompt_tokens["input_ids"].shape[1]
            
            # Create labels: mask prompt tokens with -100
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(torch.ones_like(input_ids))
        
        # Pad to same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        for input_ids, labels, attn_mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                labels = torch.cat([labels, torch.full((pad_len,), -100)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len)])
            
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_mask.append(attn_mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask).long()
        }
    
    def _collate_inference(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Collate for inference (no labels)."""
        input_ids_list = []
        attention_mask_list = []
        
        for sample in samples:
            messages = sample.to_inference_messages()
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
            attention_mask_list.append(tokens["attention_mask"].squeeze(0))
        
        max_len = max(ids.shape[0] for ids in input_ids_list)
        
        # Left-pad for generation
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id), input_ids])
                attn_mask = torch.cat([torch.zeros(pad_len), attn_mask])
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attn_mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask).long()
        }

