"""LoRA model wrapper for TinyLlama."""
import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class LoRAModelWrapper:
    """Wrapper for TinyLlama with LoRA."""
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        self.tokenizer = None
        self.base_model = None
        self.model = None
    
    def load_base_model(self):
        """Load the base TinyLlama model."""
        def _safe_exists(path: Path) -> bool:
            try:
                return path.exists()
            except PermissionError:
                # Broken/unauthorized symlink (common when folders are copied across machines).
                return False

        # Prefer a local snapshot if present (avoids network and makes runs reproducible).
        model_id_or_path = os.environ.get("LLMDFL_BASE_MODEL") or os.environ.get("LLMDFL_BASE_MODEL_DIR")
        if not model_id_or_path and _safe_exists(self.DEFAULT_LOCAL_DIR):
            model_id_or_path = str(self.DEFAULT_LOCAL_DIR)
        if not model_id_or_path:
            model_id_or_path = self.MODEL_NAME

        # Default to local-files-only to avoid slow network retries in restricted environments.
        local_only_env = os.environ.get("LLMDFL_LOCAL_FILES_ONLY", "1")
        local_files_only = str(local_only_env).strip() not in {"0", "false", "False", "no", "NO"}
        cache_dir = os.environ.get("LLMDFL_HF_CACHE_DIR") or None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except OSError as e:
            if local_files_only:
                raise RuntimeError(
                    "Base model files not found locally. "
                    "Run `python scripts/cache_base_model.py` to cache the model, "
                    "or set `LLMDFL_LOCAL_FILES_ONLY=0` to allow downloading."
                ) from e
            raise
        # For decoder-only generation (our eval path uses `generate`), left-padding avoids
        # incorrect batched generation when sequences have different lengths.
        if getattr(self.tokenizer, "padding_side", None) is not None:
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect CUDA even when an explicit device index is provided (e.g., "cuda:0")
        is_cuda = str(self.device).startswith("cuda")
        torch_dtype = torch.float16 if is_cuda else torch.float32

        device_map = None
        if is_cuda:
            # Respect explicit GPU index if provided, otherwise let HF place on CUDA
            if ":" in str(self.device):
                try:
                    device_idx = int(str(self.device).split(":")[1])
                    device_map = {"": device_idx}
                except (ValueError, IndexError):
                    device_map = "cuda"
            else:
                device_map = "cuda"

        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except OSError as e:
            if local_files_only:
                raise RuntimeError(
                    "Base model files not found locally. "
                    "Run `python scripts/cache_base_model.py` to cache the model, "
                    "or set `LLMDFL_LOCAL_FILES_ONLY=0` to allow downloading."
                ) from e
            raise
        # from_pretrained may already place the model; ensure it sits on the requested device
        model_device = next(self.base_model.parameters()).device
        if self.device and model_device.type != str(self.device).split(":")[0]:
            self.base_model = self.base_model.to(self.device)
    
    def init_lora(self):
        """Initialize LoRA on top of base model."""
        if self.base_model is None:
            self.load_base_model()
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
    
    def save_lora(self, path: str):
        """Save LoRA weights only."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
    
    def load_lora(self, path: str):
        """Load LoRA weights from path."""
        if self.base_model is None:
            self.load_base_model()
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.model = self.model.to(self.device)
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters as state dict."""
        return {
            k: v.clone() for k, v in self.model.named_parameters()
            if "lora" in k.lower()
        }
    
    def set_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Set LoRA parameters from state dict."""
        model_state = self.model.state_dict()
        for k, v in state_dict.items():
            if k in model_state:
                model_state[k] = v
        self.model.load_state_dict(model_state)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask, max_new_tokens=128, **kwargs):
        """Generate text."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    @property
    def trainable_params(self):
        """Get trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
