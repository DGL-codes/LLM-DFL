"""Snapshot loader for DFL checkpoints - READ ONLY."""
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DFLSnapshot:
    """Represents a single DFL training snapshot (read-only)."""
    snapshot_dir: Path
    config: Dict[str, Any]
    num_agents: int
    global_rounds: int
    available_rounds: List[int]  # List of round indices that have been saved
    
    @property
    def dataset(self) -> str:
        return self.config.get("dataset", "unknown")
    
    @property
    def alpha(self) -> float:
        return self.config.get("alpha", 0.5)
    
    @property
    def seed(self) -> int:
        return self.config.get("seed", 42)


class SnapshotLoader:
    """Loads DFL snapshots for DFU processing - READ ONLY access."""
    
    def __init__(self, snapshot_dir: str):
        """Initialize snapshot loader.
        
        Args:
            snapshot_dir: Path to DFL snapshot directory (e.g., 
                checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251203_230136)
        """
        self.snapshot_dir = Path(snapshot_dir)
        if not self.snapshot_dir.exists():
            raise ValueError(f"Snapshot directory not found: {snapshot_dir}")
        
        # Load config
        config_path = self.snapshot_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Discover available rounds
        self.available_rounds = self._find_available_rounds()
        
        # Create snapshot info
        self.snapshot = DFLSnapshot(
            snapshot_dir=self.snapshot_dir,
            config=self.config,
            num_agents=self.config["num_agents"],
            global_rounds=self.config["global_rounds"],
            available_rounds=self.available_rounds
        )
    
    def _find_available_rounds(self) -> List[int]:
        """Find all round directories in the snapshot."""
        round_dirs = list(self.snapshot_dir.glob("round_*"))
        rounds = []
        for d in round_dirs:
            try:
                round_idx = int(d.name.split("_")[1])
                rounds.append(round_idx)
            except (ValueError, IndexError):
                continue
        return sorted(rounds)
    
    def load_agent_state(self, round_idx: int, agent_id: int, pre_agg: bool = False) -> Dict[str, torch.Tensor]:
        """Load a single agent's LoRA state from a specific round.
        
        Args:
            round_idx: Global round index
            agent_id: Agent ID
            pre_agg: If True, load pre-aggregation state (round_t-c_u_i)
                    If False, load post-aggregation state (round_t-c_i)
            
        Returns:
            LoRA state dict
        """
        if round_idx not in self.available_rounds:
            raise ValueError(f"Round {round_idx} not available. Available: {self.available_rounds}")
        
        agent_dir = self.snapshot_dir / f"round_{round_idx}" / f"agent_{agent_id}"
        
        if pre_agg:
            state_path = agent_dir / "lora_state_pre_agg.pt"
            if not state_path.exists():
                # Fallback to post-aggregation state with warning
                logger.warning(
                    f"Pre-aggregation state not found for round {round_idx} agent {agent_id}, "
                    f"falling back to post-aggregation state"
                )
                state_path = agent_dir / "lora_state.pt"
        else:
            state_path = agent_dir / "lora_state.pt"
        
        if not state_path.exists():
            raise ValueError(f"Agent state not found: {state_path}")
        
        return torch.load(state_path, map_location="cpu")
    
    def load_all_agents_for_round(self, round_idx: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load all agents' LoRA states for a specific round.
        
        Args:
            round_idx: Global round index
            
        Returns:
            Dict mapping agent_id to LoRA state dict
        """
        if round_idx not in self.available_rounds:
            raise ValueError(f"Round {round_idx} not available. Available: {self.available_rounds}")
        
        round_dir = self.snapshot_dir / f"round_{round_idx}"
        agent_dirs = sorted(round_dir.glob("agent_*"), key=lambda x: int(x.name.split("_")[1]))
        
        states = {}
        for agent_dir in agent_dirs:
            agent_id = int(agent_dir.name.split("_")[1])
            state_path = agent_dir / "lora_state.pt"
            if state_path.exists():
                states[agent_id] = torch.load(state_path, map_location="cpu")
        
        return states
    
    def load_partition_info(self) -> Optional[Dict]:
        """Load partition information if available."""
        partition_path = self.snapshot_dir / "partition.json"
        if partition_path.exists():
            with open(partition_path) as f:
                return json.load(f)
        return None
    
    def load_history(self) -> Optional[Dict]:
        """Load training history if available."""
        history_path = self.snapshot_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                return json.load(f)
        return None
    
    def has_pre_agg_states(self, round_idx: int) -> bool:
        """Check if pre-aggregation states are available for a round.
        
        Args:
            round_idx: Global round index
            
        Returns:
            True if at least one agent has pre-aggregation state for this round
        """
        if round_idx not in self.available_rounds:
            return False
        
        round_dir = self.snapshot_dir / f"round_{round_idx}"
        agent_dirs = list(round_dir.glob("agent_*"))
        
        for agent_dir in agent_dirs:
            pre_agg_path = agent_dir / "lora_state_pre_agg.pt"
            if pre_agg_path.exists():
                return True
        
        logger.warning(f"No pre-aggregation states found for round {round_idx}")
        return False
    
    def get_snapshot_name(self) -> str:
        """Get a unique name for this snapshot based on its path."""
        # e.g., "seed42_20251203_230136"
        return self.snapshot_dir.name

