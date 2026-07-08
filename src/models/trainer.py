"""Trainer for LLM with LoRA."""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Optional
from contextlib import nullcontext
from tqdm import tqdm
import json
from pathlib import Path

from .lora_model import LoRAModelWrapper
from .ada_hessian import AdaHessian
from ..data.base import Sample
from ..data.collator import LLMCollator


class LLMTrainer:
    """Trainer for LLM fine-tuning with LoRA."""
    
    def __init__(
        self,
        model: LoRAModelWrapper,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        optimizer_name: str = "adamw"
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer_name = optimizer_name.lower()
        
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.history: Dict[str, List[float]] = {"train_loss": [], "eval_loss": []}
    
    def setup_optimizer(self, total_steps: int):
        """Setup optimizer and scheduler."""
        if self.optimizer_name in {"adahessian", "ada_hessian"}:
            self.optimizer = AdaHessian(
                self.model.trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = SGD(
                self.model.trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = AdamW(
                self.model.trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.lr * 0.1
        )
    
    def train_epoch(
        self,
        train_samples: List[Sample],
        collator: LLMCollator,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        max_steps: Optional[int] = None,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        progress_leave: bool = False,
        seed: int = None
    ) -> float:
        """Train for one epoch or max_steps (optimizer steps, not batches)."""
        self.model.train()

        use_hessian = getattr(self.optimizer, "requires_hessian", False)

        # 设置随机种子以保证可复现性
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        dataloader = DataLoader(
            train_samples, batch_size=batch_size, shuffle=True,
            collate_fn=collator, generator=generator
        )

        total_loss = 0.0
        num_batches = 0
        optimizer_steps = 0

        # Calculate total batches needed for max_steps optimizer steps
        if max_steps:
            total_batches = max_steps * grad_accum_steps
        else:
            total_batches = len(dataloader)

        pbar_kwargs = {
            "total": total_batches,
            "desc": progress_desc or "Training",
            "dynamic_ncols": True,
            "leave": progress_leave,
            "disable": not show_progress
        }
        if progress_position is not None:
            pbar_kwargs["position"] = progress_position
        pbar = tqdm(**pbar_kwargs)

        # May need multiple passes through data if max_steps > len(dataloader)
        while True:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                sdp_ctx = self._get_hessian_sdp_context() if use_hessian else nullcontext()
                with sdp_ctx:
                    outputs = self.model.forward(input_ids, attention_mask, labels)
                    loss = outputs.loss / grad_accum_steps
                    loss.backward(create_graph=use_hessian)

                total_loss += outputs.loss.item()
                num_batches += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

                if num_batches % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_params, self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    optimizer_steps += 1

                if max_steps and optimizer_steps >= max_steps:
                    break

            if max_steps is None or optimizer_steps >= max_steps:
                break

        if pbar:
            pbar.close()

        # Handle remaining gradients
        if num_batches % grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, num_batches)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def train_epoch_with_custom_loss(
        self,
        samples: List[Sample],
        collator: LLMCollator,
        loss_fn,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        max_steps: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        progress_leave: bool = False,
        seed: int = None
    ) -> float:
        """使用自定义损失函数训练一个 epoch。
        
        Args:
            samples: 训练样本
            collator: 数据整理器
            loss_fn: 自定义损失函数，接收 (logits, labels) 返回 loss
            batch_size: 批次大小
            grad_accum_steps: 梯度累积步数
            max_steps: 最大优化器步数
            show_progress: 是否显示进度条
            seed: 随机种子（用于 DataLoader shuffle）
            
        Returns:
            平均损失
        """
        self.model.train()
        
        # 设置随机种子以保证可复现性
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        
        dataloader = DataLoader(
            samples, batch_size=batch_size, shuffle=True,
            collate_fn=collator, generator=generator
        )
        
        total_loss = 0.0
        num_batches = 0
        optimizer_steps = 0
        
        if max_steps:
            total_batches = max_steps * grad_accum_steps
        else:
            total_batches = len(dataloader)
        
        pbar_kwargs = {
            "total": total_batches,
            "desc": progress_desc or "Training",
            "dynamic_ncols": True,
            "leave": progress_leave,
            "disable": not show_progress
        }
        if progress_position is not None:
            pbar_kwargs["position"] = progress_position
        pbar = tqdm(**pbar_kwargs)
        
        while True:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播获取 logits
                outputs = self.model.forward(input_ids, attention_mask, labels)
                logits = outputs.logits
                
                # 使用自定义损失函数
                loss = loss_fn(logits, labels) / grad_accum_steps
                
                # 检查 loss 是否为 nan 或 inf
                if torch.isnan(loss) or torch.isinf(loss):
                    # 跳过这个 batch，避免污染梯度
                    continue
                
                loss.backward()
                
                # 记录原始损失（未除以 grad_accum_steps）
                raw_loss = loss.item() * grad_accum_steps
                
                # 如果 raw_loss 是 nan，用 0 替代（用于显示）
                if not (raw_loss == raw_loss):  # nan check
                    raw_loss = 0.0
                
                total_loss += raw_loss
                num_batches += 1
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{raw_loss:.4f}"})
                
                if num_batches % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_params, self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    optimizer_steps += 1
                
                if max_steps and optimizer_steps >= max_steps:
                    break
            
            if max_steps is None or optimizer_steps >= max_steps:
                break
        
        if pbar:
            pbar.close()
        
        if num_batches % grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def train_local(
        self,
        samples: List[Sample],
        collator: LLMCollator,
        *,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        local_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        loss_fn=None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        progress_leave: bool = False,
        seed: Optional[int] = None,
        reset_optimizer: bool = False,
    ) -> float:
        """Unified local training entry for DFL/DFU algorithms.

        - If `local_steps` is provided (>0), trains for that many optimizer steps (DFL-style).
        - Otherwise, trains for `epochs` full passes over the local dataset (legacy).
        - If `loss_fn` is provided, uses `train_epoch_with_custom_loss` (e.g., GradAscent).
        - `seed` controls DataLoader shuffling deterministically (and derives per-epoch seeds).
        """
        if reset_optimizer:
            self.optimizer = None
            self.scheduler = None

        effective_steps = int(local_steps) if (local_steps is not None and int(local_steps) > 0) else None

        # Steps-based training (preferred for fair comparisons)
        if effective_steps is not None:
            if self.optimizer is None:
                self.setup_optimizer(effective_steps * 10)
            if loss_fn is None:
                return self.train_epoch(
                    train_samples=samples,
                    collator=collator,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    max_steps=effective_steps,
                    show_progress=show_progress,
                    progress_desc=progress_desc,
                    progress_position=progress_position,
                    progress_leave=progress_leave,
                    seed=seed,
                )
            return self.train_epoch_with_custom_loss(
                samples=samples,
                collator=collator,
                loss_fn=loss_fn,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                max_steps=effective_steps,
                show_progress=show_progress,
                progress_desc=progress_desc,
                progress_position=progress_position,
                progress_leave=progress_leave,
                seed=seed,
            )

        # Epochs-based training (legacy)
        effective_epochs = 1 if epochs is None else int(epochs)
        if effective_epochs <= 0:
            return 0.0

        if self.optimizer is None:
            approx_steps = max(1, effective_epochs * max(1, len(samples) // max(1, batch_size)))
            self.setup_optimizer(approx_steps)

        total_loss = 0.0
        for epoch_idx in range(effective_epochs):
            epoch_seed = None if seed is None else int(seed) + epoch_idx * 1000
            if loss_fn is None:
                total_loss += self.train_epoch(
                    train_samples=samples,
                    collator=collator,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    max_steps=None,
                    show_progress=show_progress,
                    progress_desc=progress_desc,
                    progress_position=progress_position,
                    progress_leave=progress_leave,
                    seed=epoch_seed,
                )
            else:
                total_loss += self.train_epoch_with_custom_loss(
                    samples=samples,
                    collator=collator,
                    loss_fn=loss_fn,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    max_steps=None,
                    show_progress=show_progress,
                    progress_desc=progress_desc,
                    progress_position=progress_position,
                    progress_leave=progress_leave,
                    seed=epoch_seed,
                )

        return total_loss / float(effective_epochs)

    def _get_hessian_sdp_context(self):
        """Force math SDPA for second-order gradients (AdaHessian)."""
        if not torch.cuda.is_available():
            return nullcontext()
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True
            )
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        return nullcontext()

    @torch.no_grad()
    def evaluate(
        self,
        eval_samples: List[Sample],
        collator: LLMCollator,
        batch_size: int = 8
    ) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        
        dataloader = DataLoader(
            eval_samples, batch_size=batch_size, shuffle=False,
            collate_fn=collator
        )
        
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model.forward(input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / max(1, len(dataloader))
        self.history["eval_loss"].append(avg_loss)
        return avg_loss
    
    def save_history(self, path: str):
        """Save training history."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
