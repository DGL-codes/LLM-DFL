"""Agent selection strategies for DFU (Decentralized Federated Unlearning).

实现三种节点选择策略:
- full: 所有幸存agent参与DFU
- random: 随机选择指定比例的agent
- ours: 基于LP松弛+整数化的最少客户端选择算法

基于文档 md/llm-dfu-节点选择.md 中描述的完整算法实现。

算法概述（ours策略）:
1. 构建LP松弛问题：最小化客户端数量，约束分布误差 D(S) <= epsilon
2. 求解LP得到实数解 s_i (每个客户端的重要性分数)
3. 整数化：按s_i从大到小排序，逐个加入直到满足误差阈值
"""
import hashlib
import time

import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

try:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AgentSelectionResult:
    """Result of agent selection."""
    selected_agents: List[int]  # 选中的agent ID列表
    strategy: str  # 使用的策略
    scores: Optional[Dict[int, float]] = None  # 每个agent的分数 (仅ours策略)
    distribution_error: Optional[float] = None  # 选中集合的分布误差
    weights: Optional[Dict[int, float]] = None  # optimized aggregation weights for selected agents
    diagnostics: Optional[Dict[str, Any]] = None  # strategy-specific diagnostics


def select_agents_full(surviving_agents: List[int]) -> AgentSelectionResult:
    """Full participation: all surviving agents participate.

    Args:
        surviving_agents: List of surviving agent IDs after removing target

    Returns:
        AgentSelectionResult with all agents selected
    """
    return AgentSelectionResult(
        selected_agents=sorted(surviving_agents),
        strategy="full"
    )


def select_agents_random(
    surviving_agents: List[int],
    ratio: float = None,
    count: int = None,
    seed: int = 42
) -> AgentSelectionResult:
    """Random selection: randomly select a fraction of agents.

    Args:
        surviving_agents: List of surviving agent IDs
        ratio: Selection ratio in (0, 1] (used if count is None)
        count: Exact number of agents to select (takes priority over ratio)
        seed: Random seed for reproducibility

    Returns:
        AgentSelectionResult with randomly selected agents
    """
    np.random.seed(seed)

    # 优先使用 count，否则使用 ratio
    if count is not None:
        num_select = max(1, min(count, len(surviving_agents)))
    elif ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Selection ratio must be in (0, 1], got {ratio}")
        num_select = max(1, math.ceil(len(surviving_agents) * ratio))
        num_select = min(num_select, len(surviving_agents))
    else:
        raise ValueError("Either ratio or count must be provided")

    # 随机选择
    selected = np.random.choice(
        surviving_agents,
        size=num_select,
        replace=False
    ).tolist()

    return AgentSelectionResult(
        selected_agents=sorted(selected),
        strategy="random"
    )


def compute_agent_distributions(
    surviving_agents: List[int],
    agent_indices: Dict[int, List[int]],
    all_labels: List[int],
    num_classes: int,
    target_agent: int = None
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray]:
    """Compute class distributions for each agent.
    
    Args:
        surviving_agents: List of surviving agent IDs
        agent_indices: Mapping from agent_id to sample indices
        all_labels: List of all labels
        num_classes: Number of classes
        target_agent: Target agent to exclude (used for computing original global distribution)
    Returns:
        Tuple of (agent_distributions, agent_sample_counts, global_distribution)
        
    Note:
        global_distribution 是原始全局分布（包含所有客户端，包括目标客户端）。
        这样 LP 才能找到能够"补偿"目标客户端缺失的最优子集。
    """
    labels = np.array(all_labels)

    # 计算每个 surviving agent 的分布
    agent_dists = {}
    agent_counts = {}

    for agent_id in surviving_agents:
        indices = agent_indices[agent_id]
        agent_labels = labels[indices]
        agent_counts[agent_id] = len(indices)

        # 计算类别分布
        dist = np.zeros(num_classes)
        for c in range(num_classes):
            dist[c] = np.sum(agent_labels == c)

        # 归一化为概率分布
        if dist.sum() > 0:
            dist = dist / dist.sum()
        agent_dists[agent_id] = dist

    # 计算原始全局分布（包含所有客户端，包括目标客户端）
    # 这是关键：LP 需要知道原始分布，才能找到能够补偿目标客户端缺失的子集
    all_agents = list(agent_indices.keys())
    total_all = sum(len(agent_indices[aid]) for aid in all_agents)
    global_dist = np.zeros(num_classes)
    
    for agent_id in all_agents:
        indices = agent_indices[agent_id]
        agent_labels = labels[indices]
        weight = len(indices) / total_all
        
        # 计算该 agent 的分布
        dist = np.zeros(num_classes)
        for c in range(num_classes):
            dist[c] = np.sum(agent_labels == c)
        if dist.sum() > 0:
            dist = dist / dist.sum()
        
        global_dist += weight * dist

    return agent_dists, agent_counts, global_dist


def compute_distribution_error(
    selected_agents: List[int],
    agent_dists: Dict[int, np.ndarray],
    agent_counts: Dict[int, int],
    global_dist: np.ndarray
) -> float:
    """Compute L1 distribution error for selected agents.
    
    D(S) = sum_c |p_S(c) - p_global(c)|
    
    where p_S(c) = sum_{i in S} (n_i * p_i(c)) / sum_{i in S} n_i
    """
    if not selected_agents:
        return float('inf')
    
    total = sum(agent_counts[i] for i in selected_agents)
    if total == 0:
        return float('inf')
    
    # 计算选中集合的合成分布
    p_S = np.zeros_like(global_dist)
    for agent_id in selected_agents:
        weight = agent_counts[agent_id] / total
        p_S += weight * agent_dists[agent_id]
    
    # L1距离
    return np.sum(np.abs(p_S - global_dist))


def _compute_remaining_distributions(
    surviving_agents: List[int],
    agent_indices: Dict[int, List[int]],
    all_labels: List[int],
    num_classes: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], np.ndarray]:
    """Compute label sketches using only remaining/surviving agents.

    This is the distribution target used by TDB-AS. It intentionally differs
    from the historical label-only AS helper above, whose global distribution
    includes the removed target client for legacy compatibility.
    """
    labels = np.array(all_labels)
    agent_dists: Dict[int, np.ndarray] = {}
    agent_counts: Dict[int, int] = {}

    total = 0
    global_dist = np.zeros(num_classes, dtype=float)
    for agent_id in surviving_agents:
        indices = list(agent_indices[agent_id])
        agent_counts[agent_id] = len(indices)
        total += len(indices)

        dist = np.zeros(num_classes, dtype=float)
        if indices:
            agent_labels = labels[indices]
            for c in range(num_classes):
                dist[c] = float(np.sum(agent_labels == c))
        if dist.sum() > 0:
            dist /= dist.sum()
        agent_dists[agent_id] = dist

    if total <= 0:
        return agent_dists, agent_counts, global_dist

    for agent_id in surviving_agents:
        global_dist += (agent_counts[agent_id] / total) * agent_dists[agent_id]

    return agent_dists, agent_counts, global_dist


def _stable_seed(text: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _project_module_features(
    module_name: str,
    features: np.ndarray,
    sketch_dim: int,
    seed: int,
) -> np.ndarray:
    """Project a small module-stat vector into a fixed sketch dimension."""
    rng = np.random.default_rng(_stable_seed(module_name, seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(sketch_dim, len(features)))
    return (signs @ features) / math.sqrt(max(1, len(features)))


def _tensor_delta_stats(delta) -> List[float]:
    """Small signed/scale statistics for a LoRA factor delta tensor."""
    import torch

    x = delta.detach().float().cpu()
    n = max(1, x.numel())
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    mean = float(x.mean().item())
    abs_mean = float(x.abs().mean().item())
    rms = float(torch.sqrt((x * x).mean()).item())
    signed_l1 = float(x.sum().item() / math.sqrt(n))
    return [mean, abs_mean, rms, signed_l1]


def _select_intervals(rounds: List[int], max_intervals: int, stride: int) -> List[Tuple[int, int]]:
    intervals = [(rounds[i], rounds[i + 1]) for i in range(len(rounds) - 1)]
    stride = max(1, int(stride))
    intervals = intervals[::stride]
    if max_intervals is not None and int(max_intervals) > 0 and len(intervals) > int(max_intervals):
        # Evenly sample intervals across the trajectory so the sketch is not only early-round.
        idx = np.linspace(0, len(intervals) - 1, int(max_intervals))
        intervals = [intervals[int(round(i))] for i in idx]
    return intervals


def compute_retained_update_sketches(
    snapshot_loader,
    agent_ids: List[int],
    *,
    sketch_dim: int = 64,
    max_intervals: int = 3,
    round_stride: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Compute lightweight retained LoRA update sketches for TDB-AS.

    The paper proposal describes random projections of effective LoRA updates.
    For the current code path we use a faster equivalent sketch over LoRA factor
    update statistics: for every retained interval/module, signed and scale
    statistics of ΔA and ΔB are projected into a shared fixed-dimensional vector.
    This keeps the selector cheap enough for repeated smoke/full sweeps while
    retaining trajectory direction information beyond label histograms.
    """
    if snapshot_loader is None:
        raise ValueError("TDB-AS requires snapshot_loader")

    from .lora_param_selection import parse_lora_modules

    rounds = list(snapshot_loader.snapshot.available_rounds)
    if len(rounds) < 2:
        raise ValueError("TDB-AS requires at least two DFL rounds")

    intervals = _select_intervals(rounds, max_intervals=max_intervals, stride=round_stride)
    if not intervals:
        raise ValueError("No retained intervals available for TDB-AS")

    first_state = snapshot_loader.load_agent_state(intervals[0][0], agent_ids[0])
    modules = parse_lora_modules(first_state)
    module_names = [m.module_name for m in modules]

    sketches: Dict[int, np.ndarray] = {}
    for agent_id in agent_ids:
        pieces: List[np.ndarray] = []
        for t0, t1 in intervals:
            state_t = snapshot_loader.load_agent_state(t0, agent_id, pre_agg=False)
            state_t1 = snapshot_loader.load_agent_state(t1, agent_id, pre_agg=True)
            interval_sketch = np.zeros(int(sketch_dim), dtype=float)

            for module in modules:
                if module.lora_A_key not in state_t or module.lora_A_key not in state_t1:
                    continue
                if module.lora_B_key not in state_t or module.lora_B_key not in state_t1:
                    continue

                delta_a = state_t1[module.lora_A_key] - state_t[module.lora_A_key]
                delta_b = state_t1[module.lora_B_key] - state_t[module.lora_B_key]
                features = np.array(
                    _tensor_delta_stats(delta_a) + _tensor_delta_stats(delta_b),
                    dtype=float,
                )
                interval_sketch += _project_module_features(
                    f"{module.module_name}:{t0}->{t1}",
                    features,
                    int(sketch_dim),
                    int(seed),
                )

            norm = float(np.linalg.norm(interval_sketch))
            if norm > 0:
                interval_sketch /= norm
            pieces.append(interval_sketch)

        sketch = np.concatenate(pieces) if pieces else np.zeros(int(sketch_dim), dtype=float)
        norm = float(np.linalg.norm(sketch))
        if norm > 0:
            sketch = sketch / norm
        sketches[agent_id] = sketch

    meta = {
        "sketch_dim_per_interval": int(sketch_dim),
        "sketch_dim_total": int(len(next(iter(sketches.values())))) if sketches else 0,
        "intervals": [[int(a), int(b)] for a, b in intervals],
        "num_modules": len(module_names),
        "module_names": module_names,
        "sketch_mode": "lora_factor_stat_random_projection",
    }
    if verbose:
        print(f"TDB-AS sketches: agents={agent_ids}, intervals={meta['intervals']}, dim={meta['sketch_dim_total']}")
    return sketches, meta


def compute_ring_target_exposure(
    num_agents: int,
    surviving_agents: List[int],
    target_agent: int,
    *,
    rho: float = 0.8,
    horizon: Optional[int] = None,
) -> np.ndarray:
    """Topology exposure q_i based on powers of the DFL ring mixing matrix."""
    n = int(num_agents)
    horizon = int(horizon or n)
    rho = float(rho)

    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        ids = [i, (i - 1) % n, (i + 1) % n]
        for j in ids:
            M[i, j] += 1.0 / len(ids)

    power = np.eye(n, dtype=float)
    exposure = np.zeros(n, dtype=float)
    for t in range(max(1, horizon)):
        exposure += (rho ** t) * power[:, int(target_agent)]
        power = power @ M

    q = np.array([exposure[i] for i in surviving_agents], dtype=float)
    q = np.maximum(q, 0.0)
    total = float(q.sum())
    if total <= 0:
        q[:] = 1.0 / max(1, len(q))
    else:
        q /= total
    return q


def _solve_tdb_milp(
    U: np.ndarray,
    u_bar: np.ndarray,
    P: np.ndarray,
    p_bar: np.ndarray,
    q: np.ndarray,
    *,
    max_selected: Optional[int],
    epsilon_u: Optional[float],
    epsilon_p: Optional[float],
    tau_q: float,
    alpha_u: float,
    alpha_p: float,
    alpha_q: float,
    time_limit: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for TDB-AS MILP")

    K = int(U.shape[1])
    d_u = int(U.shape[0])
    C = int(P.shape[0])
    z0 = 0
    w0 = K
    r0 = 2 * K
    xi0 = 2 * K + d_u
    num_vars = 2 * K + d_u + C

    use_fixed_budget = max_selected is not None
    c = np.zeros(num_vars, dtype=float)
    if use_fixed_budget:
        c[z0:w0] = 1e-6
        c[w0:r0] = -float(alpha_q) * q
        c[r0:xi0] = float(alpha_u)
        c[xi0:] = float(alpha_p)
    else:
        c[z0:w0] = 1.0
        c[w0:r0] = -float(alpha_q) * q

    rows: List[np.ndarray] = []
    lows: List[float] = []
    highs: List[float] = []

    def add(row: np.ndarray, low: float, high: float) -> None:
        rows.append(row)
        lows.append(low)
        highs.append(high)

    row = np.zeros(num_vars)
    row[w0:r0] = 1.0
    add(row, 1.0, 1.0)

    for i in range(K):
        row = np.zeros(num_vars)
        row[w0 + i] = 1.0
        row[z0 + i] = -1.0
        add(row, -np.inf, 0.0)

    row = np.zeros(num_vars)
    row[z0:w0] = 1.0
    if use_fixed_budget:
        exact_selected = float(max(1, min(int(max_selected), K)))
        add(row, exact_selected, exact_selected)
    else:
        add(row, 1.0, float(K))

    for m in range(d_u):
        row = np.zeros(num_vars)
        row[w0:r0] = U[m, :]
        row[r0 + m] = -1.0
        add(row, -np.inf, float(u_bar[m]))

        row = np.zeros(num_vars)
        row[w0:r0] = -U[m, :]
        row[r0 + m] = -1.0
        add(row, -np.inf, float(-u_bar[m]))

    for cc in range(C):
        row = np.zeros(num_vars)
        row[w0:r0] = P[cc, :]
        row[xi0 + cc] = -1.0
        add(row, -np.inf, float(p_bar[cc]))

        row = np.zeros(num_vars)
        row[w0:r0] = -P[cc, :]
        row[xi0 + cc] = -1.0
        add(row, -np.inf, float(-p_bar[cc]))

    if not use_fixed_budget:
        if epsilon_u is not None:
            row = np.zeros(num_vars)
            row[r0:xi0] = 1.0
            add(row, -np.inf, float(epsilon_u))
        if epsilon_p is not None:
            row = np.zeros(num_vars)
            row[xi0:] = 1.0
            add(row, -np.inf, float(epsilon_p))

    if tau_q is not None and float(tau_q) > 0:
        row = np.zeros(num_vars)
        row[w0:r0] = q
        add(row, float(tau_q), np.inf)

    A = np.vstack(rows)
    constraints = LinearConstraint(A, np.array(lows), np.array(highs))
    lb = np.zeros(num_vars, dtype=float)
    ub = np.full(num_vars, np.inf, dtype=float)
    ub[z0:w0] = 1.0
    ub[w0:r0] = 1.0
    integrality = np.zeros(num_vars, dtype=int)
    integrality[z0:w0] = 1

    options: Dict[str, Any] = {"disp": False}
    if time_limit is not None and float(time_limit) > 0:
        options["time_limit"] = float(time_limit)

    started = time.time()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options=options,
    )
    elapsed = time.time() - started

    diagnostics: Dict[str, Any] = {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_fun": float(result.fun) if result.fun is not None else None,
        "solve_time_sec": float(elapsed),
        "mode": "fixed_budget" if use_fixed_budget else "min_cost_constrained",
        "num_variables": int(num_vars),
        "num_constraints": int(A.shape[0]),
    }
    if not result.success or result.x is None:
        return None, None, diagnostics

    x = np.array(result.x, dtype=float)
    z = x[z0:w0]
    w = x[w0:r0]
    return z, w, diagnostics


def _tdb_fallback_selection(
    surviving_agents: List[int],
    U: np.ndarray,
    u_bar: np.ndarray,
    P: np.ndarray,
    p_bar: np.ndarray,
    q: np.ndarray,
    count: int,
    alpha_u: float,
    alpha_p: float,
    alpha_q: float,
) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
    """Deterministic fallback when MILP is unavailable/infeasible."""
    K = len(surviving_agents)
    count = max(1, min(int(count), K))
    per_agent = []
    for i, agent_id in enumerate(surviving_agents):
        traj = float(np.sum(np.abs(U[:, i] - u_bar)))
        label = float(np.sum(np.abs(P[:, i] - p_bar)))
        score = float(alpha_u) * traj + float(alpha_p) * label - float(alpha_q) * float(q[i])
        per_agent.append((score, agent_id, i))
    per_agent.sort(key=lambda x: (x[0], x[1]))
    selected_idx = [i for _, _, i in per_agent[:count]]
    selected = [surviving_agents[i] for i in selected_idx]
    weights = np.zeros(K, dtype=float)
    weights[selected_idx] = 1.0 / len(selected_idx)
    return selected, weights, {"fallback": "per_agent_proxy_greedy"}


def select_agents_tdb(
    surviving_agents: List[int],
    ratio: float = None,
    count: int = None,
    agent_indices: Dict[int, List[int]] = None,
    all_labels: List[int] = None,
    num_classes: int = None,
    snapshot_loader=None,
    target_agent: int = 0,
    seed: int = 42,
    sketch_dim: int = 64,
    max_intervals: int = 3,
    round_stride: int = 1,
    alpha_u: float = 1.0,
    alpha_p: float = 1.0,
    alpha_q: float = 0.1,
    epsilon_u: Optional[float] = None,
    epsilon_p: Optional[float] = None,
    tau_q: float = 0.0,
    exposure_rho: float = 0.8,
    use_target_similarity: bool = False,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> AgentSelectionResult:
    """Trajectory-aware distribution-balanced mixed-integer AS (TDB-AS)."""
    if agent_indices is None or all_labels is None or num_classes is None:
        raise ValueError("tdb strategy requires agent_indices, all_labels, and num_classes")
    if snapshot_loader is None:
        raise ValueError("tdb strategy requires snapshot_loader")

    K = len(surviving_agents)
    if count is not None:
        max_selected = max(1, min(int(count), K))
    elif ratio is not None:
        max_selected = max(1, min(int(math.ceil(K * float(ratio))), K))
    elif epsilon_u is not None or epsilon_p is not None:
        max_selected = None
    else:
        max_selected = max(1, int(math.ceil(K * 0.5)))

    agent_dists, agent_counts, p_bar = _compute_remaining_distributions(
        surviving_agents, agent_indices, all_labels, num_classes
    )
    counts = np.array([agent_counts[i] for i in surviving_agents], dtype=float)
    if counts.sum() <= 0:
        lambdas = np.ones(K, dtype=float) / max(1, K)
    else:
        lambdas = counts / counts.sum()
    P = np.stack([agent_dists[i] for i in surviving_agents], axis=1)

    sketch_agents = list(surviving_agents)
    if use_target_similarity and target_agent not in sketch_agents:
        sketch_agents = sketch_agents + [target_agent]
    sketches, sketch_meta = compute_retained_update_sketches(
        snapshot_loader,
        sketch_agents,
        sketch_dim=sketch_dim,
        max_intervals=max_intervals,
        round_stride=round_stride,
        seed=seed,
        verbose=verbose,
    )
    U = np.stack([sketches[i] for i in surviving_agents], axis=1)
    u_bar = U @ lambdas

    q = compute_ring_target_exposure(
        snapshot_loader.snapshot.num_agents,
        surviving_agents,
        target_agent,
        rho=exposure_rho,
        horizon=snapshot_loader.snapshot.global_rounds,
    )
    if use_target_similarity and target_agent in sketches:
        target_u = sketches[target_agent]
        sims = []
        for agent_id in surviving_agents:
            ui = sketches[agent_id]
            denom = float(np.linalg.norm(ui) * np.linalg.norm(target_u) + 1e-12)
            sims.append(max(0.0, float(np.dot(ui, target_u) / denom)))
        sims_arr = np.array(sims, dtype=float)
        q = q * sims_arr
        if q.sum() > 0:
            q = q / q.sum()

    z, w, solver_diag = _solve_tdb_milp(
        U,
        u_bar,
        P,
        p_bar,
        q,
        max_selected=max_selected,
        epsilon_u=epsilon_u,
        epsilon_p=epsilon_p,
        tau_q=tau_q,
        alpha_u=alpha_u,
        alpha_p=alpha_p,
        alpha_q=alpha_q,
        time_limit=time_limit,
    )

    fallback_diag: Dict[str, Any] = {}
    if z is None or w is None:
        fallback_count = max_selected or K
        selected, w, fallback_diag = _tdb_fallback_selection(
            surviving_agents,
            U,
            u_bar,
            P,
            p_bar,
            q,
            fallback_count,
            alpha_u,
            alpha_p,
            alpha_q,
        )
    else:
        selected = [surviving_agents[i] for i, val in enumerate(z) if val >= 0.5]
        if not selected:
            best = int(np.argmax(w))
            selected = [surviving_agents[best]]
            z[best] = 1.0
        # Numerical cleanup: keep only selected weights and renormalize.
        mask = np.array([agent_id in selected for agent_id in surviving_agents], dtype=bool)
        w = np.where(mask, np.maximum(w, 0.0), 0.0)
        if w.sum() <= 0:
            w[mask] = 1.0 / max(1, mask.sum())
        else:
            w /= w.sum()

    traj_l1 = float(np.sum(np.abs(U @ w - u_bar)))
    traj_l2 = float(np.linalg.norm(U @ w - u_bar))
    label_l1 = float(np.sum(np.abs(P @ w - p_bar)))
    exposure = float(np.dot(q, w))
    weights = {agent_id: float(w[i]) for i, agent_id in enumerate(surviving_agents) if agent_id in selected}
    scores = {agent_id: float(w[i]) for i, agent_id in enumerate(surviving_agents)}

    diagnostics: Dict[str, Any] = {
        "tdb": True,
        "selected_agents": sorted(int(i) for i in selected),
        "weights": {str(k): float(v) for k, v in weights.items()},
        "trajectory_l1": traj_l1,
        "trajectory_l2": traj_l2,
        "label_l1": label_l1,
        "target_exposure": exposure,
        "target_exposure_by_agent": {str(agent): float(q[i]) for i, agent in enumerate(surviving_agents)},
        "max_selected": int(max_selected) if max_selected is not None else None,
        "exact_selected": int(max_selected) if max_selected is not None else None,
        "alpha_u": float(alpha_u),
        "alpha_p": float(alpha_p),
        "alpha_q": float(alpha_q),
        "epsilon_u": None if epsilon_u is None else float(epsilon_u),
        "epsilon_p": None if epsilon_p is None else float(epsilon_p),
        "tau_q": float(tau_q),
        "exposure_rho": float(exposure_rho),
        "use_target_similarity": bool(use_target_similarity),
        "solver": solver_diag,
        "sketch": sketch_meta,
    }
    diagnostics.update(fallback_diag)

    if verbose:
        print("TDB-AS selection:")
        print(f"  selected={sorted(selected)}")
        print(f"  traj_l1={traj_l1:.4f}, label_l1={label_l1:.4f}, exposure={exposure:.4f}")

    return AgentSelectionResult(
        selected_agents=sorted(selected),
        strategy="tdb",
        scores=scores,
        distribution_error=label_l1,
        weights=weights,
        diagnostics=diagnostics,
    )


def greedy_select_clients(
    surviving_agents: List[int],
    agent_dists: Dict[int, np.ndarray],
    agent_counts: Dict[int, int],
    global_dist: np.ndarray,
    epsilon: float = 0.1,
    max_ratio: float = 1.0,
    verbose: bool = False
) -> Tuple[List[int], Dict[int, float]]:
    """贪心选择算法：每次选择能最大减少分布误差的客户端。

    这是文档 md/llm-dfu-节点选择.md 中算法的贪心实现。

    算法原理：
    - 目标：选择最少的客户端使分布误差 D(S) <= epsilon
    - 贪心策略：每次选择加入后能使误差减少最多的客户端
    - 边际效益：第 k 个客户端的"重要性"由其被选中的顺序决定

    为什么使用贪心算法而不是 LP 松弛：
    - LP 松弛的最优解是 s_i = 1/K（均匀解），无法区分客户端重要性
    - 这是 LP 问题的数学本质：均匀权重完美满足分布约束（误差=0）
    - 贪心算法能够产生有意义的排序和选择

    Args:
        surviving_agents: List of agent IDs
        agent_dists: Agent class distributions
        agent_counts: Agent sample counts
        global_dist: Global class distribution
        epsilon: 分布误差阈值
        max_ratio: 最大选择比例
        verbose: 是否打印详细信息

    Returns:
        Tuple of (selected agents, importance scores)
    """
    K = len(surviving_agents)
    max_select = max(1, math.ceil(K * max_ratio))

    if verbose:
        print(f"Greedy selection: K={K}, max_select={max_select}, epsilon={epsilon}")

    selected = []
    remaining = set(surviving_agents)
    importance_scores = {}
    order = 0

    while remaining and len(selected) < max_select:
        best_agent = None
        best_error = float('inf')

        for agent_id in remaining:
            # 尝试加入该客户端
            temp_selected = selected + [agent_id]
            error = compute_distribution_error(temp_selected, agent_dists, agent_counts, global_dist)

            if error < best_error:
                best_error = error
                best_agent = agent_id

        if best_agent is None:
            break

        # 加入最佳客户端
        selected.append(best_agent)
        remaining.remove(best_agent)

        # 重要性分数：越早被选中越重要
        # 使用递减分数：1.0, 0.9, 0.8, ...
        importance_scores[best_agent] = max(0.1, 1.0 - 0.1 * order)
        order += 1

        if verbose:
            print(f"  Step {order}: Select agent {best_agent}, error={best_error:.4f}")

        # 检查是否达到误差阈值
        if best_error <= epsilon:
            if verbose:
                print(f"  Reached epsilon threshold with {len(selected)} agents")
            break

    # 为未选中的客户端赋予较低分数
    for agent_id in remaining:
        importance_scores[agent_id] = 0.05

    if verbose:
        print(f"Selected {len(selected)} agents: {selected}")
        final_error = compute_distribution_error(selected, agent_dists, agent_counts, global_dist)
        print(f"Final error: {final_error:.4f}")

    return selected, importance_scores


def solve_lp_relaxation(
    surviving_agents: List[int],
    agent_dists: Dict[int, np.ndarray],
    agent_counts: Dict[int, int],
    global_dist: np.ndarray,
    epsilon: float = 0.1,
    verbose: bool = False
) -> Optional[Dict[int, float]]:
    """求解LP松弛问题，获取每个agent的重要性分数 s_i。

    严格按照文档 md/llm-dfu-节点选择.md 中的 LP 松弛公式实现。

    (LP-relax) 问题：
    变量：s_i ∈ [0,1], T ≥ 0, t_c ≥ 0
    目标：min Σ s_i
    约束：
    - -t_c ≤ Σ s_i * n_i * p_i(c) - T * p_global(c) ≤ t_c, ∀c
    - Σ t_c ≤ ε * T
    - T = Σ s_i * n_i
    - Σ s_i ≥ 1 (避免平凡解)

    重要说明：
    - global_dist 必须包含目标客户端的分布（即原始全局分布）
    - 如果 global_dist 只包含 surviving agents，则 LP 会返回均匀解 s_i = 1/K
    - 这是因为均匀权重完美满足分布约束（误差 = 0）

    Args:
        surviving_agents: List of agent IDs (不包含目标客户端)
        agent_dists: Agent class distributions {agent_id: (C,) array}
        agent_counts: Agent sample counts {agent_id: int}
        global_dist: 全局分布，应该包含目标客户端的贡献
        epsilon: 分布误差阈值
        verbose: 是否打印详细信息

    Returns:
        Dict mapping agent_id to importance score s_i in [0, 1], or None if LP fails
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for LP relaxation")

    K = len(surviving_agents)
    C = len(global_dist)
    idx_to_agent = {i: agent_id for i, agent_id in enumerate(surviving_agents)}

    # 构建矩阵形式的分布和样本数
    distributions = np.array([agent_dists[agent_id] for agent_id in surviving_agents])
    client_sizes = np.array([agent_counts[agent_id] for agent_id in surviving_agents])

    if verbose:
        print(f"LP Relaxation: K={K}, C={C}, epsilon={epsilon}")
        print(f"Client sizes: {client_sizes}")

    # LP 变量: [s_0, ..., s_{K-1}, T, t_0, ..., t_{C-1}]
    num_vars = K + 1 + C
    c = np.zeros(num_vars)
    c[:K] = 1.0  # 目标: min Σ s_i

    # 构建不等式约束 A_ub @ x <= b_ub
    A_ub = []
    b_ub = []

    # 约束1: 对每个类别 c，有两个不等式
    # Σ s_i * n_i * p_i(c) - T * p_global(c) - t_c ≤ 0
    # -Σ s_i * n_i * p_i(c) + T * p_global(c) - t_c ≤ 0
    for c_idx in range(C):
        row1 = np.zeros(num_vars)
        row1[:K] = client_sizes * distributions[:, c_idx]
        row1[K] = -global_dist[c_idx]
        row1[K + 1 + c_idx] = -1.0
        A_ub.append(row1)
        b_ub.append(0.0)

        row2 = np.zeros(num_vars)
        row2[:K] = -client_sizes * distributions[:, c_idx]
        row2[K] = global_dist[c_idx]
        row2[K + 1 + c_idx] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)

    # 约束2: Σ t_c - ε * T ≤ 0
    row3 = np.zeros(num_vars)
    row3[K] = -epsilon
    row3[K + 1:] = 1.0
    A_ub.append(row3)
    b_ub.append(0.0)

    # 约束3: -Σ s_i ≤ -1 (即 Σ s_i ≥ 1)
    row4 = np.zeros(num_vars)
    row4[:K] = -1.0
    A_ub.append(row4)
    b_ub.append(-1.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 等式约束: T = Σ s_i * n_i
    A_eq = np.zeros((1, num_vars))
    A_eq[0, :K] = client_sizes
    A_eq[0, K] = -1.0
    b_eq = np.array([0.0])

    # 变量边界: s_i ∈ [0, 1], T ≥ 0, t_c ≥ 0
    bounds = [(0, 1) for _ in range(K)]
    bounds.append((0, None))
    bounds.extend([(0, None) for _ in range(C)])

    try:
        result = linprog(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs',
            options={'disp': False, 'presolve': True}
        )

        if result.success:
            s_hat = result.x[:K]
            T_hat = result.x[K]
            t_hat = result.x[K + 1:]

            if verbose:
                print(f"LP solved successfully")
                print(f"  Objective (Σ s_i): {result.fun:.4f}")
                print(f"  T: {T_hat:.2f}")
                print(f"  s_i range: [{s_hat.min():.4f}, {s_hat.max():.4f}]")
                print(f"  s_i values: {s_hat}")

            return {idx_to_agent[i]: float(s_hat[i]) for i in range(K)}
        else:
            if verbose:
                print(f"LP failed: {result.message}")
            return None
    except Exception as e:
        if verbose:
            print(f"LP error: {e}")
        return None


def integerize_solution(
    surviving_agents: List[int],
    scores: Dict[int, float],
    agent_dists: Dict[int, np.ndarray],
    agent_counts: Dict[int, int],
    global_dist: np.ndarray,
    epsilon: float,
    verbose: bool = False
) -> List[int]:
    """整数化算法：将重要性分数转换为实际选择集合。

    算法步骤（来自文档 md/llm-dfu-节点选择.md）：
    1. 过滤：V' = {i : s_i > 0}
    2. 排序：按 s_i 从大到小排序
    3. 选择：逐个加入直到 D(S) <= epsilon

    Args:
        surviving_agents: List of agent IDs
        scores: Importance scores {agent_id: s_i}
        agent_dists: Agent class distributions
        agent_counts: Agent sample counts
        global_dist: Global class distribution
        epsilon: 分布误差阈值
        verbose: 是否打印详细信息

    Returns:
        List of selected agent IDs
    """
    # 1. 过滤：只保留 s_i > 0 的客户端
    candidates = [(agent_id, scores[agent_id]) for agent_id in surviving_agents if scores[agent_id] > 1e-6]

    if not candidates:
        # 如果所有分数都是0，使用所有客户端
        if verbose:
            print("All scores are 0, selecting all agents")
        return surviving_agents.copy()

    # 2. 按分数从大到小排序（同分按 agent_id 升序，保证确定性）
    candidates.sort(key=lambda x: (-x[1], x[0]))

    if verbose:
        print(f"Candidates sorted by importance:")
        for i, (agent_id, score) in enumerate(candidates[:5]):
            print(f"  {i+1}. Agent {agent_id}: score={score:.4f}")
        if len(candidates) > 5:
            print(f"  ... ({len(candidates) - 5} more)")

    # 3. 选择：逐个加入直到满足误差阈值
    selected = []
    for agent_id, score in candidates:
        selected.append(agent_id)
        error = compute_distribution_error(selected, agent_dists, agent_counts, global_dist)

        if verbose:
            print(f"  Added agent {agent_id} (score={score:.4f}), count={len(selected)}, error={error:.4f}")

        if error <= epsilon:
            if verbose:
                print(f"Reached epsilon threshold ({epsilon}) with {len(selected)} agents")
            break

    return selected


def select_agents_ours(
    surviving_agents: List[int],
    ratio: float = None,
    count: int = None,
    agent_indices: Dict[int, List[int]] = None,
    all_labels: List[int] = None,
    num_classes: int = None,
    epsilon: float = 0.1,
    verbose: bool = False
) -> AgentSelectionResult:
    """LP松弛+整数化的最少客户端选择算法。

    完整算法步骤（来自文档 md/llm-dfu-节点选择.md）：

    1. 计算每个agent的类别分布 p_i(c) 和样本数 n_i
    2. 计算全局分布 p_global(c)
    3. 求解LP松弛问题，得到重要性分数 s_i ∈ [0,1]
    4. 整数化：
       - 如果指定了 count：选择分数最高的前 count 个客户端
       - 否则：按 s_i 从大到小排序，逐个加入直到 D(S) <= epsilon

    Args:
        surviving_agents: List of surviving agent IDs
        ratio: 选择比例（如果 count 未指定）
        count: 选择的客户端数量（优先于 ratio 和 epsilon）
        agent_indices: Mapping from agent_id to sample indices
        all_labels: List of all labels
        num_classes: Number of classes
        epsilon: 分布误差阈值（当 count 未指定时使用）
        verbose: 是否打印详细信息

    Returns:
        AgentSelectionResult with selected agents and scores
    """
    # 1. 计算分布
    agent_dists, agent_counts, global_dist = compute_agent_distributions(
        surviving_agents, agent_indices, all_labels, num_classes
    )

    # 2. 求解LP获取分数
    scores = solve_lp_relaxation(
        surviving_agents, agent_dists, agent_counts, global_dist, epsilon, verbose
    )
    if scores is None:
        # LP occasionally fails (numerical issues / solver edge cases). For DFU runs we
        # prefer a robust fallback instead of hard-crashing the experiment.
        if verbose:
            print("[WARN] LP relaxation failed; falling back to greedy selection ranking.")
        # Greedy ranking is deterministic given inputs and yields a usable ordering.
        # - If `count`/`ratio` is specified: rank ALL agents (disable epsilon stop), then take top-k.
        # - Otherwise: use epsilon-threshold stopping (default greedy_select_clients behavior).
        greedy_eps = -1.0 if (count is not None or ratio is not None) else float(epsilon)
        greedy_selected, greedy_scores = greedy_select_clients(
            surviving_agents,
            agent_dists,
            agent_counts,
            global_dist,
            epsilon=greedy_eps,
            max_ratio=1.0,
            verbose=verbose,
        )
        scores = greedy_scores
        if count is not None:
            num_select = min(int(count), len(surviving_agents))
            selected = greedy_selected[:num_select]
            if verbose:
                print(f"[WARN] Using greedy top-{num_select} selection (count={count}).")
        elif ratio is not None:
            num_select = max(1, math.ceil(len(surviving_agents) * float(ratio)))
            selected = greedy_selected[:num_select]
            if verbose:
                print(f"[WARN] Using greedy top-{num_select} selection (ratio={ratio}).")
        else:
            selected = greedy_selected
        error = compute_distribution_error(selected, agent_dists, agent_counts, global_dist)
        if verbose:
            print(f"[WARN] Greedy fallback selection: {len(selected)} agents, error={error:.4f}")
        return AgentSelectionResult(
            selected_agents=sorted(selected),
            strategy="ours",
            scores=scores,
            distribution_error=error,
        )

    # 3. 整数化：根据 count/ratio 或 epsilon 选择客户端
    if count is not None:
        # 选择分数最高的前 count 个客户端
        sorted_agents = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        num_select = min(count, len(surviving_agents))
        selected = [agent_id for agent_id, _ in sorted_agents[:num_select]]
        if verbose:
            print(f"Selecting top {num_select} agents by LP scores:")
            for i, (agent_id, score) in enumerate(sorted_agents[:num_select]):
                print(f"  {i+1}. Agent {agent_id}: score={score:.4f}")
    elif ratio is not None:
        # 选择分数最高的前 ratio 比例的客户端
        sorted_agents = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        num_select = max(1, math.ceil(len(surviving_agents) * ratio))
        selected = [agent_id for agent_id, _ in sorted_agents[:num_select]]
        if verbose:
            print(f"Selecting top {ratio*100:.0f}% ({num_select}) agents by LP scores")
    else:
        # 使用 epsilon 阈值选择
        selected = integerize_solution(
            surviving_agents, scores, agent_dists, agent_counts, global_dist,
            epsilon, verbose=verbose
        )

    # 计算最终分布误差
    error = compute_distribution_error(selected, agent_dists, agent_counts, global_dist)

    if verbose:
        print(f"Final selection: {len(selected)} agents, error={error:.4f}")

    return AgentSelectionResult(
        selected_agents=sorted(selected),
        strategy="ours",
        scores=scores,
        distribution_error=error
    )


def select_agents(
    strategy: str,
    surviving_agents: List[int],
    ratio: float = None,
    count: int = None,
    seed: int = 42,
    agent_indices: Optional[Dict[int, List[int]]] = None,
    all_labels: Optional[List[int]] = None,
    num_classes: Optional[int] = None,
    epsilon: float = 0.1,
    snapshot_loader=None,
    target_agent: int = 0,
    tdb_sketch_dim: int = 64,
    tdb_max_intervals: int = 3,
    tdb_round_stride: int = 1,
    tdb_alpha_u: float = 1.0,
    tdb_alpha_p: float = 1.0,
    tdb_alpha_q: float = 0.1,
    tdb_epsilon_u: Optional[float] = None,
    tdb_epsilon_p: Optional[float] = None,
    tdb_tau_q: float = 0.0,
    tdb_exposure_rho: float = 0.8,
    tdb_use_target_similarity: bool = False,
    tdb_time_limit: float = 30.0,
) -> AgentSelectionResult:
    """Main entry point for agent selection.

    Args:
        strategy: Selection strategy ("full", "random", "ours", "tdb")
        surviving_agents: List of surviving agent IDs
        ratio: Selection ratio for random strategy
        count: Exact number of agents to select for random strategy
        seed: Random seed for random strategy
        agent_indices: Required for "ours" and "tdb" strategies
        all_labels: Required for "ours" and "tdb" strategies
        num_classes: Required for "ours" and "tdb" strategies
        epsilon: LP error threshold for "ours" strategy
        snapshot_loader: Required for "tdb" strategy

    Returns:
        AgentSelectionResult
    """
    if strategy == "full":
        return select_agents_full(surviving_agents)

    elif strategy == "random":
        return select_agents_random(surviving_agents, ratio=ratio, count=count, seed=seed)

    elif strategy == "ours":
        if agent_indices is None or all_labels is None or num_classes is None:
            raise ValueError("ours strategy requires agent_indices, all_labels, and num_classes")
        return select_agents_ours(
            surviving_agents, ratio=ratio, count=count, 
            agent_indices=agent_indices, all_labels=all_labels, 
            num_classes=num_classes, epsilon=epsilon
        )

    elif strategy == "tdb":
        if agent_indices is None or all_labels is None or num_classes is None:
            raise ValueError("tdb strategy requires agent_indices, all_labels, and num_classes")
        if snapshot_loader is None:
            raise ValueError("tdb strategy requires snapshot_loader")
        return select_agents_tdb(
            surviving_agents,
            ratio=ratio,
            count=count,
            agent_indices=agent_indices,
            all_labels=all_labels,
            num_classes=num_classes,
            snapshot_loader=snapshot_loader,
            target_agent=target_agent,
            seed=seed,
            sketch_dim=tdb_sketch_dim,
            max_intervals=tdb_max_intervals,
            round_stride=tdb_round_stride,
            alpha_u=tdb_alpha_u,
            alpha_p=tdb_alpha_p,
            alpha_q=tdb_alpha_q,
            epsilon_u=tdb_epsilon_u,
            epsilon_p=tdb_epsilon_p,
            tau_q=tdb_tau_q,
            exposure_rho=tdb_exposure_rho,
            use_target_similarity=tdb_use_target_similarity,
            time_limit=tdb_time_limit,
        )

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
