"""Data partitioners for non-IID distribution."""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class PartitionInfo:
    """Stores partition information for reproducibility."""
    dataset_name: str
    num_agents: int
    alpha: Optional[float]  # For Dirichlet
    seed: int
    partition_type: str  # "dirichlet" or "task_skew"
    agent_indices: Dict[int, List[int]]  # agent_id -> sample indices
    agent_weights: Optional[Dict[int, Dict[str, float]]] = None  # For task skew
    
    def save(self, path: str):
        """Save partition info to JSON."""
        data = asdict(self)
        # Convert int keys to strings for JSON
        data["agent_indices"] = {str(k): v for k, v in data["agent_indices"].items()}
        if data["agent_weights"]:
            data["agent_weights"] = {str(k): v for k, v in data["agent_weights"].items()}
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PartitionInfo":
        """Load partition info from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        data["agent_indices"] = {int(k): v for k, v in data["agent_indices"].items()}
        if data["agent_weights"]:
            data["agent_weights"] = {int(k): v for k, v in data["agent_weights"].items()}
        return cls(**data)


class DirichletPartitioner:
    """Partition data using Dirichlet distribution for label skew."""
    
    def __init__(self, num_agents: int, alpha: float = 0.5, seed: int = 42):
        self.num_agents = num_agents
        self.alpha = alpha
        self.seed = seed
    
    def partition(self, labels: List[int], dataset_name: str) -> PartitionInfo:
        """Partition samples by labels using Dirichlet distribution."""
        np.random.seed(self.seed)
        
        labels = np.array(labels)
        num_classes = len(np.unique(labels))
        agent_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_agents)}
        
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            
            # Sample Dirichlet proportions for this class
            proportions = np.random.dirichlet([self.alpha] * self.num_agents)
            proportions = proportions / proportions.sum()
            
            # Split indices according to proportions
            splits = (proportions * len(class_indices)).astype(int)
            splits[-1] = len(class_indices) - splits[:-1].sum()  # Handle rounding
            
            start = 0
            for agent_id, size in enumerate(splits):
                agent_indices[agent_id].extend(class_indices[start:start + size].tolist())
                start += size
        
        # Shuffle each agent's data
        for agent_id in agent_indices:
            np.random.shuffle(agent_indices[agent_id])
        
        return PartitionInfo(
            dataset_name=dataset_name,
            num_agents=self.num_agents,
            alpha=self.alpha,
            seed=self.seed,
            partition_type="dirichlet",
            agent_indices=agent_indices,
            agent_weights=None
        )


class TaskSkewPartitioner:
    """Partition data for task/domain skew scenarios."""
    
    TASK_NAMES = ["general", "finance", "medical", "code"]
    
    def __init__(self, num_agents: int, mode: str = "hard", seed: int = 42):
        """
        Args:
            num_agents: Number of agents (should be divisible by 4 for hard mode)
            mode: "hard" or "soft"
            seed: Random seed
        """
        self.num_agents = num_agents
        self.mode = mode
        self.seed = seed
    
    def partition(self, task_samples: Dict[str, int]) -> PartitionInfo:
        """
        Partition samples across agents.
        
        Args:
            task_samples: Dict mapping task name to number of samples
        """
        np.random.seed(self.seed)
        agent_weights: Dict[int, Dict[str, float]] = {}
        
        if self.mode == "hard":
            # Each group of agents gets only one task
            agents_per_task = self.num_agents // 4
            for i in range(self.num_agents):
                task_idx = i // agents_per_task if agents_per_task > 0 else i % 4
                task_idx = min(task_idx, 3)
                weights = {t: 0.0 for t in self.TASK_NAMES}
                weights[self.TASK_NAMES[task_idx]] = 1.0
                agent_weights[i] = weights
        else:  # soft
            # Each agent has a primary task (0.8) and shares others (0.2)
            for i in range(self.num_agents):
                primary_task = self.TASK_NAMES[i % 4]
                weights = {t: 0.2 / 3 for t in self.TASK_NAMES}
                weights[primary_task] = 0.8
                agent_weights[i] = weights
        
        # Generate sample indices based on weights
        agent_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_agents)}
        offset = 0
        
        for task, num in task_samples.items():
            task_indices = list(range(offset, offset + num))
            np.random.shuffle(task_indices)
            
            # Distribute based on weights
            for agent_id in range(self.num_agents):
                weight = agent_weights[agent_id].get(task, 0)
                n_samples = int(weight * num / (self.num_agents / 4))
                if n_samples > 0 and task_indices:
                    take = min(n_samples, len(task_indices))
                    agent_indices[agent_id].extend(task_indices[:take])
                    task_indices = task_indices[take:]
            
            offset += num
        
        return PartitionInfo(
            dataset_name="multitask",
            num_agents=self.num_agents,
            alpha=None,
            seed=self.seed,
            partition_type=f"task_skew_{self.mode}",
            agent_indices=agent_indices,
            agent_weights=agent_weights
        )

