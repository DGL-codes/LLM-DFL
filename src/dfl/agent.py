"""DFL Agent implementation."""
import torch
from typing import Dict, List, Optional
from copy import deepcopy

from ..models.lora_model import LoRAModelWrapper
from ..models.trainer import LLMTrainer
from ..data.base import Sample
from ..data.collator import LLMCollator


class DFLAgent:
    """A single agent in DFL with its own LoRA parameters and local data."""
    
    def __init__(
        self,
        agent_id: int,
        local_samples: List[Sample],
        model: LoRAModelWrapper,
        collator: LLMCollator,
        lr: float = 1e-4,
        device: str = "cuda",
        optimizer_name: str = "adamw"
    ):
        self.agent_id = agent_id
        self.local_samples = local_samples
        self.model = model
        self.collator = collator
        self.device = device
        
        # Initialize trainer
        self.trainer = LLMTrainer(
            model=model,
            lr=lr,
            device=device,
            optimizer_name=optimizer_name
        )
        
        # Store LoRA state
        self.lora_state: Optional[Dict[str, torch.Tensor]] = None
    
    def local_train(
        self,
        steps: int,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        progress_leave: bool = False,
        seed: Optional[int] = None
    ) -> float:
        """Perform local training for specified steps."""
        if self.lora_state is not None:
            self.model.set_lora_state_dict(self.lora_state)

        loss = self.trainer.train_local(
            self.local_samples,
            self.collator,
            local_steps=steps,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=progress_leave,
            seed=seed,
        )
        
        # Save current LoRA state
        self.lora_state = self.model.get_lora_state_dict()
        return loss

    def set_optimizer_name(self, optimizer_name: str):
        """Switch optimizer for next local training round."""
        optimizer_name = optimizer_name.lower()
        if self.trainer.optimizer_name != optimizer_name:
            self.trainer.optimizer_name = optimizer_name
            self.trainer.optimizer = None
            self.trainer.scheduler = None
    
    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """Get current LoRA parameters."""
        if self.lora_state is None:
            self.lora_state = self.model.get_lora_state_dict()
        return {k: v.clone() for k, v in self.lora_state.items()}
    
    def set_lora_params(self, params: Dict[str, torch.Tensor]):
        """Set LoRA parameters."""
        self.lora_state = {k: v.clone() for k, v in params.items()}
    
    def evaluate(self, samples: List[Sample], batch_size: int = 8) -> float:
        """Evaluate on given samples."""
        if self.lora_state is not None:
            self.model.set_lora_state_dict(self.lora_state)
        return self.trainer.evaluate(samples, self.collator, batch_size)


class RingTopology:
    """Ring topology for DFL agents."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
    
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get left and right neighbors in ring."""
        left = (agent_id - 1) % self.num_agents
        right = (agent_id + 1) % self.num_agents
        return [left, right]
    
    def aggregate(
        self,
        agent_id: int,
        agent_params: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate parameters with neighbors using equal weights (1/3)."""
        neighbors = self.get_neighbors(agent_id)
        all_ids = [agent_id] + neighbors  # Self + neighbors
        
        result = {}
        for key in agent_params[agent_id].keys():
            stacked = torch.stack([agent_params[i][key] for i in all_ids])
            result[key] = stacked.mean(dim=0)
        
        return result
