"""LoRA Parameter Selection for DFU (Decentralized Federated Unlearning).

基于文档 md/llm-dfu-lora参数选择.md 实现的 layer-wise LoRA 模块选择。

核心思想：
- 计算每个 LoRA 模块对目标客户端（Agent 0）的遗忘敏感度
- 使用贪心算法选择需要参与 DFU 的模块集合
- 只在选定模块上执行 DFU 校准，降低计算和通信开销

敏感度计算：
- I_g = S_g^(0) = Σ_j ||ΔW_g_{0,j}||_F^2
- ΔW_g ≈ ΔB_g * A_g + B_g * ΔA_g (一阶近似)
"""
import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoRAModuleInfo:
    """Information about a LoRA module."""
    module_name: str  # e.g., "model.layers.0.self_attn.q_proj"
    layer_idx: int    # e.g., 0
    module_type: str  # e.g., "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    lora_A_key: str   # e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    lora_B_key: str   # e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"


@dataclass
class LoRASelectionResult:
    """Result of LoRA module selection."""
    selected_modules: List[str]      # 选中的模块名列表
    all_modules: List[str]           # 所有模块名列表
    sensitivities: Dict[str, float]  # 每个模块的敏感度 {module_name: I_g}
    total_sensitivity: float          # 总敏感度 S_tot
    covered_sensitivity: float        # 选中模块的敏感度之和
    selection_ratio: float            # |M| / G
    epsilon_W: float                   # 使用的阈值


def parse_lora_modules(state_dict: Dict[str, torch.Tensor]) -> List[LoRAModuleInfo]:
    """Parse LoRA module information from state dict.
    
    Args:
        state_dict: LoRA state dict with keys like 
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    
    Returns:
        List of LoRAModuleInfo for each unique LoRA module
    """
    modules = {}
    
    for key in state_dict.keys():
        if "lora_A" in key and "weight" in key:
            # Extract module path
            # key: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
            parts = key.split(".")
            
            # Find layer index and module type
            layer_idx = -1
            module_type = None
            
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                if part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    module_type = part
            
            if layer_idx >= 0 and module_type:
                # Construct module name
                module_name = f"layer{layer_idx}.{module_type}"
                
                # Find corresponding lora_B key
                lora_B_key = key.replace("lora_A", "lora_B")
                
                if module_name not in modules:
                    modules[module_name] = LoRAModuleInfo(
                        module_name=module_name,
                        layer_idx=layer_idx,
                        module_type=module_type,
                        lora_A_key=key,
                        lora_B_key=lora_B_key
                    )
    
    return sorted(modules.values(), key=lambda x: (x.layer_idx, x.module_type))


def compute_effective_weight_change(
    B_t: torch.Tensor,
    A_t: torch.Tensor,
    B_t1: torch.Tensor,
    A_t1: torch.Tensor
) -> torch.Tensor:
    """Compute effective weight change using first-order approximation.
    
    ΔW ≈ ΔB * A_t + B_t * ΔA
    
    where:
    - ΔB = B_{t+1} - B_t
    - ΔA = A_{t+1} - A_t
    
    Args:
        B_t: LoRA B matrix at time t, shape (d_out, r)
        A_t: LoRA A matrix at time t, shape (r, d_in)
        B_t1: LoRA B matrix at time t+1
        A_t1: LoRA A matrix at time t+1
    
    Returns:
        Effective weight change matrix, shape (d_out, d_in)
    """
    delta_B = B_t1 - B_t
    delta_A = A_t1 - A_t
    
    # First-order approximation
    delta_W = delta_B @ A_t + B_t @ delta_A
    
    return delta_W


def compute_module_energy(
    state_t: Dict[str, torch.Tensor],
    state_t1: Dict[str, torch.Tensor],
    module_info: LoRAModuleInfo
) -> float:
    """Compute energy (Frobenius norm squared) for a LoRA module change.
    
    E_g = ||ΔW_g||_F^2
    
    Args:
        state_t: LoRA state at time t
        state_t1: LoRA state at time t+1
        module_info: Module information
    
    Returns:
        Energy value (float)
    """
    A_t = state_t[module_info.lora_A_key].float()
    B_t = state_t[module_info.lora_B_key].float()
    A_t1 = state_t1[module_info.lora_A_key].float()
    B_t1 = state_t1[module_info.lora_B_key].float()

    delta_B = B_t1 - B_t
    delta_A = A_t1 - A_t

    # Equivalent to ||delta_B @ A_t + B_t @ delta_A||_F^2, but avoids
    # materializing the dense LoRA effective weight matrix for every module.
    aa_t = A_t @ A_t.T
    db_t_db = delta_B.T @ delta_B
    b_t_b = B_t.T @ B_t
    da_da_t = delta_A @ delta_A.T
    cross = delta_B.T @ B_t
    da_a_t = delta_A @ A_t.T

    energy = (
        torch.sum(db_t_db * aa_t)
        + torch.sum(b_t_b * da_da_t)
        + 2.0 * torch.sum(cross * da_a_t.T)
    ).item()

    return energy


def compute_module_sensitivities(
    snapshot_loader,
    target_agent: int = 0,
    verbose: bool = False
) -> Dict[str, float]:
    """Compute sensitivity scores for all LoRA modules based on DFL history.

    I_g = S_g^(0) = Σ_j E_g_{0,j} = Σ_j ||ΔW_g_{0,j}||_F^2

    Args:
        snapshot_loader: SnapshotLoader instance
        target_agent: Target agent to unlearn (default: 0)
        verbose: Print progress

    Returns:
        Dict mapping module_name to sensitivity score
    """
    available_rounds = snapshot_loader.snapshot.available_rounds

    if len(available_rounds) < 2:
        raise ValueError("Need at least 2 DFL snapshots for sensitivity computation")

    # Load first state to get module information
    first_state = snapshot_loader.load_agent_state(available_rounds[0], target_agent)
    modules = parse_lora_modules(first_state)

    if verbose:
        print(f"Found {len(modules)} LoRA modules")
        print(f"Computing sensitivities from {len(available_rounds)} snapshots...")

    # Initialize sensitivities
    sensitivities = {m.module_name: 0.0 for m in modules}

    # Compute energy for each time interval
    for j in range(len(available_rounds) - 1):
        t_j = available_rounds[j]
        t_j1 = available_rounds[j + 1]

        state_t = snapshot_loader.load_agent_state(t_j, target_agent)
        state_t1 = snapshot_loader.load_agent_state(t_j1, target_agent)

        for module in modules:
            energy = compute_module_energy(state_t, state_t1, module)
            sensitivities[module.module_name] += energy

        if verbose:
            print(f"  Processed interval {t_j} → {t_j1}")

    if verbose:
        print(f"Sensitivity computation complete")
        top_5 = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 sensitive modules:")
        for name, score in top_5:
            print(f"  {name}: {score:.6f}")

    return sensitivities


def compute_module_sensitivities_relative(
    snapshot_loader,
    target_agent: int = 0,
    alpha: float = 1.0,
    verbose: bool = False
) -> Dict[str, float]:
    """Compute sensitivity scores with target-specific ratio weighting.

    I_g = S_g^(0) * (S_g^(0) / S_g^(all))^alpha

    Args:
        snapshot_loader: SnapshotLoader instance
        target_agent: Target agent to unlearn (default: 0)
        alpha: Ratio exponent (alpha >= 0)
        verbose: Print progress

    Returns:
        Dict mapping module_name to sensitivity score
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    available_rounds = snapshot_loader.snapshot.available_rounds

    if len(available_rounds) < 2:
        raise ValueError("Need at least 2 DFL snapshots for sensitivity computation")

    # Load first state to get module information
    first_state = snapshot_loader.load_agent_state(available_rounds[0], target_agent)
    modules = parse_lora_modules(first_state)

    if verbose:
        print(f"Found {len(modules)} LoRA modules")
        print(f"Computing relative sensitivities from {len(available_rounds)} snapshots...")

    sensitivities_target = {m.module_name: 0.0 for m in modules}
    sensitivities_all = {m.module_name: 0.0 for m in modules}

    num_agents = snapshot_loader.snapshot.num_agents

    for j in range(len(available_rounds) - 1):
        t_j = available_rounds[j]
        t_j1 = available_rounds[j + 1]

        for agent_id in range(num_agents):
            state_t = snapshot_loader.load_agent_state(t_j, agent_id)
            state_t1 = snapshot_loader.load_agent_state(t_j1, agent_id)

            for module in modules:
                energy = compute_module_energy(state_t, state_t1, module)
                sensitivities_all[module.module_name] += energy
                if agent_id == target_agent:
                    sensitivities_target[module.module_name] += energy

        if verbose:
            print(f"  Processed interval {t_j} → {t_j1}")

    sensitivities = {}
    for module in modules:
        name = module.module_name
        s0 = sensitivities_target[name]
        sall = sensitivities_all[name]
        if sall <= 0:
            ratio = 0.0
        else:
            ratio = min(1.0, s0 / sall)
        sensitivities[name] = s0 * (ratio ** alpha)

    if verbose:
        print("Relative sensitivity computation complete")
        top_5 = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 sensitive modules (relative):")
        for name, score in top_5:
            print(f"  {name}: {score:.6f}")

    return sensitivities


def select_lora_modules(
    sensitivities: Dict[str, float],
    epsilon_W: float = 0.1,
    verbose: bool = False,
    max_ratio: Optional[float] = None
) -> LoRASelectionResult:
    """Select LoRA modules using greedy algorithm based on sensitivities.

    按照文档 md/llm-dfu-lora参数选择.md 中的贪心算法：
    1. 按 I_g 从大到小排序模块
    2. 从 M=∅ 开始，依次加入模块
    3. 直到 Σ_{g∈M} I_g >= S_tot - ε_W

    Args:
        sensitivities: Dict mapping module_name to sensitivity score
        epsilon_W: 允许丢弃的敏感度总量（绝对阈值）
        verbose: Print progress
        max_ratio: 选择比例上限（在阈值选择后截断 top-k）

    Returns:
        LoRASelectionResult with selected modules
    """
    all_modules = list(sensitivities.keys())
    G = len(all_modules)

    if G == 0:
        raise ValueError("No modules found")

    # Compute total sensitivity
    S_tot = sum(sensitivities.values())

    if S_tot == 0:
        # All sensitivities are zero, select all modules
        return LoRASelectionResult(
            selected_modules=all_modules,
            all_modules=all_modules,
            sensitivities=sensitivities,
            total_sensitivity=0.0,
            covered_sensitivity=0.0,
            selection_ratio=1.0,
            epsilon_W=epsilon_W
        )

    # Sort modules by sensitivity (descending)
    sorted_modules = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    # Greedy selection
    selected = []
    covered = 0.0
    threshold = max(0.0, S_tot - epsilon_W)

    for module_name, score in sorted_modules:
        selected.append(module_name)
        covered += score

        # Check if threshold is reached
        if covered >= threshold:
            break

    selection_ratio = len(selected) / G

    # Apply ratio cap after sensitivity-based selection (top-k by sensitivity)
    if max_ratio is not None and max_ratio > 0 and max_ratio < 1.0:
        max_k = max(1, int(G * max_ratio))
        if len(selected) > max_k:
            selected = sorted(selected, key=lambda m: sensitivities[m], reverse=True)[:max_k]
            covered = sum(sensitivities[m] for m in selected)
            selection_ratio = len(selected) / G

    if verbose:
        print(f"LoRA Module Selection:")
        print(f"  Total modules: {G}")
        print(f"  Selected modules: {len(selected)} ({selection_ratio:.1%})")
        print(f"  Total sensitivity: {S_tot:.6f}")
        print(f"  Covered sensitivity: {covered:.6f} ({covered/S_tot:.1%})")
        print(f"  Dropped sensitivity: {S_tot - covered:.6f} ({(S_tot-covered)/S_tot:.1%})")
        print(f"  Threshold (ε_W): {epsilon_W}")
        if max_ratio is not None and max_ratio > 0 and max_ratio < 1.0:
            print(f"  Ratio cap applied: {max_ratio:.2f}")

    return LoRASelectionResult(
        selected_modules=selected,
        all_modules=all_modules,
        sensitivities=sensitivities,
        total_sensitivity=S_tot,
        covered_sensitivity=covered,
        selection_ratio=selection_ratio,
        epsilon_W=epsilon_W
    )


def select_lora_modules_top_ratio(
    sensitivities: Dict[str, float],
    ratio: float,
    verbose: bool = False,
) -> LoRASelectionResult:
    """Select top-ratio LoRA modules by sensitivity (deterministic).

    This mode matches the "score first, then take top-k%" requirement:
    1) compute sensitivities I_g
    2) sort by (I_g desc, module_name asc) for deterministic tie-breaking
    3) take top floor(G * ratio) modules (at least 1; ratio=1 -> all)
    """
    all_modules = list(sensitivities.keys())
    G = len(all_modules)
    if G == 0:
        raise ValueError("No modules found")
    if ratio <= 0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    S_tot = float(sum(sensitivities.values()))

    # Determine k deterministically (match existing max_ratio cap semantics: floor)
    if ratio >= 1.0:
        k = G
    else:
        k = max(1, int(math.floor(G * ratio)))

    sorted_modules = sorted(
        sensitivities.items(),
        key=lambda x: (-float(x[1]), x[0]),
    )
    selected = [name for name, _ in sorted_modules[:k]]
    covered = float(sum(sensitivities[m] for m in selected))
    selection_ratio = len(selected) / G

    if verbose:
        print("LoRA Module Selection (top-ratio):")
        print(f"  Total modules: {G}")
        print(f"  Selected modules: {len(selected)} ({selection_ratio:.1%})")
        print(f"  Total sensitivity: {S_tot:.6f}")
        if S_tot > 0:
            print(f"  Covered sensitivity: {covered:.6f} ({covered/S_tot:.1%})")
            print(f"  Dropped sensitivity: {S_tot - covered:.6f} ({(S_tot-covered)/S_tot:.1%})")
        print(f"  Ratio: {ratio:.2f}")

    # epsilon_W is not used in top-ratio mode; keep it as 0.0 for bookkeeping.
    return LoRASelectionResult(
        selected_modules=selected,
        all_modules=all_modules,
        sensitivities=sensitivities,
        total_sensitivity=S_tot,
        covered_sensitivity=covered,
        selection_ratio=selection_ratio,
        epsilon_W=0.0,
    )


def get_lora_module_keys(
    state_dict: Dict[str, torch.Tensor],
    selected_modules: List[str]
) -> Set[str]:
    """Get state dict keys for selected LoRA modules.

    Args:
        state_dict: Full LoRA state dict
        selected_modules: List of selected module names (e.g., "layer0.q_proj")

    Returns:
        Set of state dict keys belonging to selected modules
    """
    modules = parse_lora_modules(state_dict)
    module_map = {m.module_name: m for m in modules}

    selected_keys = set()
    for module_name in selected_modules:
        if module_name in module_map:
            module = module_map[module_name]
            selected_keys.add(module.lora_A_key)
            selected_keys.add(module.lora_B_key)

    return selected_keys
