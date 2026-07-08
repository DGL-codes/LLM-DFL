"""Base dataset class for LLM-DFL."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Sample:
    """A processed sample ready for LLM training."""
    instruction: str
    input_text: str
    output_text: str
    label: Optional[int] = None  # For classification tasks
    paraphrased_answer: Optional[str] = None  # TOFU: 改写答案
    perturbed_answers: Optional[List[str]] = None  # TOFU: 扰动答案列表
    
    def to_messages(self, system_prompt: str = "You are a helpful assistant.") -> List[Dict[str, str]]:
        """Convert to TinyLlama chat format."""
        user_content = f"{self.instruction}\n\nInput:\n{self.input_text}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": self.output_text}
        ]
    
    def to_inference_messages(self, system_prompt: str = "You are a helpful assistant.") -> List[Dict[str, str]]:
        """Convert to messages for inference (no assistant response)."""
        user_content = f"{self.instruction}\n\nInput:\n{self.input_text}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    TASK_TYPE: str = "classification"  # or "generation"
    DATASET_NAME: str = ""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        self.split = split
        self.max_samples = max_samples
        self.samples: List[Sample] = []
        self.label_names: List[str] = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load and preprocess data from source."""
        pass
    
    @abstractmethod
    def _process_sample(self, raw_sample: Any) -> Sample:
        """Convert a raw sample to a Sample object."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]
    
    def get_labels(self) -> List[int]:
        """Return list of labels for all samples."""
        return [s.label for s in self.samples if s.label is not None]
    
    def get_subset(self, indices: List[int]) -> List[Sample]:
        """Get a subset of samples by indices."""
        return [self.samples[i] for i in indices]
    
    @property
    def num_classes(self) -> int:
        return len(self.label_names)

