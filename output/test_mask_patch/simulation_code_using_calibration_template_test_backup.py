import json
import math
import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    data_folder: str = "data_fitting/mask_adoption_data"
    seed: int = 42
    k_runs: int = 20
    l2_reg: float = 1.0
    max_iter: int = 400
    learning_rate: float = 0.1
    val_split_ratio: float = 0.8
    gov_intervention_day: int = 10
    gov_lam_factor_default: float = 1.5
    rho_info_decay_default: float = 0.5
    output_folder: str = "outputs"
    forecast_days: int = 10  # Days 30-39 if training is 0-29
    verbose: bool = True


@dataclass
class Parameters:
    # Decision model
    alpha: float
    gamma: float
    theta_f: float
    theta_w: float
    theta_c: float
    beta_r: float
    beta_i: float
    age_effects: Dict[str, float]
    occ_effects: Dict[str, float]
    tau: float
    # Layer weights derived and normalized
    w_family: float
    w_work: float
    w_community: float
    # Info propagation
    phi_family: float
    phi_work: float
    phi_community: float
    lambda_broadcast_base: float
    lambda_broadcast_factor_after_day10: float
    rho_info_decay: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FittedParams:
    """Container for all parameters needed by the simulator."""
    decision_weights: Dict[str, float]              # e.g., b0, b_prev, wF, wW, wC, b_info, b_risk, etc.
    layer_weights: Dict[str, float]                 # e.g., family, work_school, community
    info_params: Dict[str, float]                   # e.g., campaign_intensity, gamma_info, memory_decay
    noise_params: Dict[str, float]                  # e.g., temperature
    meta: Dict[str, Any]                            # e.g., seed, calibrator_name, training_window, notes
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_parameters(self) -> Parameters:
        """Convert FittedParams to legacy Parameters format."""
        # Extract values from structured dictionaries
        alpha = self.decision_weights.get('alpha', 0.0)
        gamma = self.decision_weights.get('gamma', 0.0)
        theta_f = self.decision_weights.get('theta_f', 1.0)
        theta_w = self.decision_weights.get('theta_w', 1.0)
        theta_c = self.decision_weights.get('theta_c', 1.0)
        beta_r = self.decision_weights.get('beta_r', 0.0)
        beta_i = self.decision_weights.get('beta_i', 0.0)
        tau = self.noise_params.get('tau', 1.0)
        
        # Layer weights
        w_family = self.layer_weights.get('family', 1.0)
        w_work = self.layer_weights.get('work_school', 1.0)
        w_community = self.layer_weights.get('community', 1.0)
        
        # Info params
        phi_family = self.info_params.get('phi_family', 0.1)
        phi_work = self.info_params.get('phi_work', 0.1)
        phi_community = self.info_params.get('phi_community', 0.1)
        lambda_broadcast_base = self.info_params.get('lambda_broadcast_base', 0.05)
        lambda_broadcast_factor_after_day10 = self.info_params.get('lambda_broadcast_factor_after_day10', 1.5)
        rho_info_decay = self.info_params.get('rho_info_decay', 0.5)
        
        # Age and occupation effects
        age_effects = self.decision_weights.get('age_effects', {})
        occ_effects = self.decision_weights.get('occ_effects', {})
        
        return Parameters(
            alpha=alpha, gamma=gamma, theta_f=theta_f, theta_w=theta_w, theta_c=theta_c,
            beta_r=beta_r, beta_i=beta_i, age_effects=age_effects, occ_effects=occ_effects,
            tau=tau, w_family=w_family, w_work=w_work, w_community=w_community,
            phi_family=phi_family, phi_work=phi_work, phi_community=phi_community,
            lambda_broadcast_base=lambda_broadcast_base, 
            lambda_broadcast_factor_after_day10=lambda_broadcast_factor_after_day10,
            rho_info_decay=rho_info_decay
        )


class Calibrator(ABC):
    """Pluggable calibrator interface with a stable evaluation callback signature."""
    
    @abstractmethod
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        pass


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_agent_attributes(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load agent_attributes from {path}: {e}") from e


def load_social_network(path: str) -> Dict[str, Dict[str, List[int]]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load social_network from {path}: {e}") from e


def load_train_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load train_data from {path}: {e}") from e


def align_ids(
    agents_df: pd.DataFrame,
    social: Dict[str, Dict[str, List[int]]],
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[int, int], pd.DataFrame, Dict[str, Dict[str, List[int]]], pd.DataFrame]:
    agents_ids = set(agents_df["agent_id"].astype(int).tolist())
    social_ids = set(int(k) for k in social.keys())
    train_ids = set(train_df["agent_id"].astype(int).unique().tolist())
    common = sorted(list(agents_ids & social_ids & train_ids))
    if len(common) == 0:
        raise RuntimeError("No common agent IDs across agent_attributes.csv, social_network.json, and train_data.csv")
    id2idx = {aid: i for i, aid in enumerate(common)}
    # Filter dataframes
    agents_df_f = agents_df[agents_df["agent_id"].isin(common)].copy()
    train_df_f = train_df[train_df["agent_id"].isin(common)].copy()
    # Filter social to only common ids and neighbors in common
    social_f: Dict[str, Dict[str, List[int]]] = {}
    for k, v in social.items():
        ik = int(k)
        if ik in id2idx:
            social_f[k] = {
                "family": [int(x) for x in v.get("family", []) if int(x) in id2idx],
                "work_school": [int(x) for x in v.get("work_school", []) if int(x) in id2idx],
                "community": [int(x) for x in v.get("community", []) if int(x) in id2idx],
            }
    return np.array(common, dtype=int), id2idx, agents_df_f, social_f, train_df_f


def build_multiplex_adjacency(
    social: Dict[str, Dict[str, List[int]]],
    id2idx: Dict[int, int],
    n: int,
) -> Dict[str, List[np.ndarray]]:
    layers = ["family", "work_school", "community"]
    adj: Dict[str, List[set]] = {layer: [set() for _ in range(n)] for layer in layers}
    # Symmetrize and deduplicate
    for k_str, v in social.items():
        i = id2idx[int(k_str)]
        for layer in layers:
            for nbr in v.get(layer, []):
                if nbr in id2idx:
                    j = id2idx[nbr]
                    if i != j:
                        adj[layer][i].add(j)
                        adj[layer][j].add(i)
    # Convert to numpy arrays
    adj_arrays: Dict[str, List[np.ndarray]] = {}
    for layer in layers:
        arr_list: List[np.ndarray] = []
        for i in range(n):
            if len(adj[layer][i]) == 0:
                arr_list.append(np.array([], dtype=int))
            else:
                arr_list.append(np.fromiter(adj[layer][i], dtype=int))
        adj_arrays[layer] = arr_list
    return adj_arrays


def compute_layer_neighbor_share(states: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    n = states.shape[0]
    shares = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = neighbors[i]
        if neigh.size == 0:
            shares[i] = 0.0
        else:
            shares[i] = float(np.mean(states[neigh]))
    return shares


def pivot_states(train_df: pd.DataFrame, id2idx: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    # Determine days
    days = sorted(train_df["day"].unique().tolist())
    n_days = len(days)
    n_agents = len(id2idx)
    wearing = np.zeros((n_days, n_agents), dtype=np.float64)
    received = np.zeros((n_days, n_agents), dtype=np.float64)
    # Sort train_df for efficient filling
    df_sorted = train_df.sort_values(["day", "agent_id"])
    day_to_idx = {d: i for i, d in enumerate(days)}
    for _, row in df_sorted.iterrows():
        d = int(row["day"])
        a = int(row["agent_id"])
        i_day = day_to_idx[d]
        i_agent = id2idx[a]
        wearing[i_day, i_agent] = 1.0 if bool(row["wearing_mask"]) else 0.0
        received[i_day, i_agent] = 1.0 if bool(row["received_info"]) else 0.0
    return wearing, received, days


def encode_demographics(agents_df: pd.DataFrame, common_ids: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    # Map to row order as common_ids
    idx_series = pd.Series(np.arange(len(common_ids)), index=common_ids)
    df_sorted = agents_df.set_index("agent_id").loc[common_ids]
    age_groups = df_sorted["age_group"].astype(str).tolist()
    occs = df_sorted["occupation"].astype(str).tolist()
    # Unique categories
    age_cats_all = sorted(list(pd.unique(df_sorted["age_group"].astype(str))))
    occ_cats_all = sorted(list(pd.unique(df_sorted["occupation"].astype(str))))
    # Baselines
    age_baseline = "Middle Age" if "Middle Age" in age_cats_all else age_cats_all[0]
    occ_baseline = "White Collar" if "White Collar" in occ_cats_all else occ_cats_all[0]
    age_cats = [c for c in age_cats_all if c != age_baseline]
    occ_cats = [c for c in occ_cats_all if c != occ_baseline]
    # Build one-hot excluding baseline
    n = len(common_ids)
    age_oh = np.zeros((n, len(age_cats)), dtype=np.float64)
    occ_oh = np.zeros((n, len(occ_cats)), dtype=np.float64)
    age_index_map = {c: idx for idx, c in enumerate(age_cats)}
    occ_index_map = {c: idx for idx, c in enumerate(occ_cats)}
    for i in range(n):
        ag = age_groups[i]
        oc = occs[i]
        if ag in age_index_map:
            age_oh[i, age_index_map[ag]] = 1.0
        if oc in occ_index_map:
            occ_oh[i, occ_index_map[oc]] = 1.0
    age_cat_names = age_cats  # column order
    occ_cat_names = occ_cats
    return age_oh, age_cat_names, occ_oh, occ_cat_names


def compute_mem_info(received_info: np.ndarray, rho: float) -> np.ndarray:
    # received_info: days x agents (0/1)
    T, N = received_info.shape
    mem = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        mem[t, :] = rho * mem[t - 1, :] + (1.0 - rho) * received_info[t, :]
    return mem


def build_train_validation_splits(days: List[int], ratio: float) -> Tuple[int, int, int]:
    # Returns indices for train_end_exclusive, val_start_inclusive, val_end_exclusive
    T = len(days)
    split_idx = int(math.floor(ratio * T))
    train_end = max(1, split_idx)  # at least 1 to allow t-1
    val_start = train_end
    val_end = T
    return train_end, val_start, val_end


def build_feature_matrix(
    wearing: np.ndarray,
    mem_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    day_start: int,
    day_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Build dataset for days [day_start, day_end) predicting wearing[t] from features at t-1 and mem_info[t]
    # wearing: T x N
    # share_*_by_day indexed by same T x N using wearing[t-1]
    T, N = wearing.shape
    assert 0 <= day_start < day_end <= T
    rows = []
    labels = []
    for t in range(day_start, day_end):
        wear_prev = wearing[t - 1, :]
        share_f = share_f_by_day[t - 1, :]
        share_w = share_w_by_day[t - 1, :]
        share_c = share_c_by_day[t - 1, :]
        mem_t = mem_info[t, :]
        # Constant features
        intercept = np.ones(N, dtype=np.float64)
        # risk is per-agent
        # concatenate: [intercept, wear_prev, share_f, share_w, share_c, risk, mem_t, age_oh, occ_oh]
        base = np.stack([intercept, wear_prev, share_f, share_w, share_c, risk, mem_t], axis=1)  # N x 7
        # Append demographics
        if age_oh.shape[1] > 0:
            base = np.concatenate([base, age_oh], axis=1)
        if occ_oh.shape[1] > 0:
            base = np.concatenate([base, occ_oh], axis=1)
        rows.append(base)
        labels.append(wearing[t, :])
    X = np.vstack(rows)  # (N*(day_end-day_start)) x n_features
    y = np.concatenate(labels, axis=0)  # (N*(day_end-day_start), )
    return X, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1.0,
    max_iter: int = 400,
    lr: float = 0.1,
    verbose: bool = False,
) -> np.ndarray:
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    # Exclude intercept from regularization (index 0)
    reg_mask = np.ones(n_features, dtype=np.float64)
    reg_mask[0] = 0.0
    # Adam optimizer
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for it in range(1, max_iter + 1):
        z = X @ w
        p = sigmoid(z)
        # Gradient of negative log-likelihood with L2
        grad = X.T @ (p - y) / n_samples + l2_reg * reg_mask * w / n_samples
        # Adam updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose and (it % 20 == 0 or it == max_iter):
            nll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) / n_samples
            reg_term = 0.5 * l2_reg * np.sum((reg_mask * w) ** 2) / n_samples
            total_loss = nll + reg_term
            print(f"[fit_logistic_l2] Iter {it:3d}: nll={nll:.5f} reg={reg_term:.5f} total={total_loss:.5f}")
    return w


def derive_layer_weights_and_betas(theta_f: float, theta_w: float, theta_c: float) -> Tuple[float, float, float, float, float, float]:
    # Convert layer coefficients into normalized weights and beta magnitudes
    coefs = np.array([theta_f, theta_w, theta_c], dtype=np.float64)
    abs_coefs = np.abs(coefs)
    total = np.sum(abs_coefs)
    if total <= 1e-12:
        # default equal weights
        w = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    else:
        w = abs_coefs / total
    beta_f, beta_w, beta_c = float(abs_coefs[0]), float(abs_coefs[1]), float(abs_coefs[2])
    return float(w[0]), float(w[1]), float(w[2]), beta_f, beta_w, beta_c


def calibrate_info_params_simple(
    received_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    gov_intervention_day: int,
    default_factor: float,
) -> Tuple[float, float, float, float, float]:
    # Use daily population averages to infer lambda_broadcast_base and factor, with heuristic phi's
    T, N = received_info.shape
    p_obs = received_info.mean(axis=1)  # per-day prevalence
    sf = share_f_by_day.mean(axis=1)
    sw = share_w_by_day.mean(axis=1)
    sc = share_c_by_day.mean(axis=1)

    # Heuristic phi values
    phi_f = 0.3
    phi_w = 0.2
    phi_c = 0.1

    # Pre and post intervention means
    pre_mask = np.arange(T) < gov_intervention_day
    post_mask = ~pre_mask
    if pre_mask.sum() == 0:
        pre_mask[:] = True
    # Compute means safely
    p0 = float(p_obs[pre_mask].mean()) if pre_mask.any() else float(p_obs.mean())
    s0 = float((phi_f * sf[pre_mask] + phi_w * sw[pre_mask] + phi_c * sc[pre_mask]).mean()) if pre_mask.any() else float((phi_f * sf + phi_w * sw + phi_c * sc).mean())
    lam_base = max(0.0, -math.log(max(1e-9, 1.0 - p0)) - s0)

    if post_mask.any():
        p1 = float(p_obs[post_mask].mean())
        s1 = float((phi_f * sf[post_mask] + phi_w * sw[post_mask] + phi_c * sc[post_mask]).mean())
        lam1 = max(0.0, -math.log(max(1e-9, 1.0 - p1)) - s1)
        lam_factor = (lam1 / lam_base) if lam_base > 1e-9 else default_factor
        lam_factor = max(1.0, min(5.0, lam_factor))
    else:
        lam_factor = default_factor

    # Clamp lambda
    lam_base = float(max(0.0, min(0.5, lam_base)))
    return phi_f, phi_w, phi_c, lam_base, lam_factor


def simulate_step_info(
    states_prev: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    phi_f: float,
    phi_w: float,
    phi_c: float,
    lambda_broadcast: float,
) -> np.ndarray:
    # Compute neighbor shares
    share_f = compute_layer_neighbor_share(states_prev, neighbors["family"])
    share_w = compute_layer_neighbor_share(states_prev, neighbors["work_school"])
    share_c = compute_layer_neighbor_share(states_prev, neighbors["community"])
    # Probability of receiving info
    u = phi_f * share_f + phi_w * share_w + phi_c * share_c + lambda_broadcast
    p_info = 1.0 - np.exp(-np.clip(u, 0.0, 50.0))
    rec = (np.random.rand(states_prev.shape[0]) < p_info).astype(np.float64)
    return rec


def compute_logit(
    prev_states: np.ndarray,
    share_f: np.ndarray,
    share_w: np.ndarray,
    share_c: np.ndarray,
    risk: np.ndarray,
    mem_info: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    params: Parameters,
    age_cat_names: List[str],
    occ_cat_names: List[str],
) -> np.ndarray:
    # Reconstruct age and occ effect vectors aligned to columns in age_oh and occ_oh
    age_effects_vec = np.array([params.age_effects.get(cat, 0.0) for cat in age_cat_names], dtype=np.float64)
    occ_effects_vec = np.array([params.occ_effects.get(cat, 0.0) for cat in occ_cat_names], dtype=np.float64)
    logits = (
        params.alpha
        + params.gamma * prev_states
        + params.theta_f * share_f
        + params.theta_w * share_w
        + params.theta_c * share_c
        + params.beta_r * risk
        + params.beta_i * mem_info
    )
    if age_oh.shape[1] > 0:
        logits += age_oh @ age_effects_vec
    if occ_oh.shape[1] > 0:
        logits += occ_oh @ occ_effects_vec
    return logits


def simulate_window(
    start_states: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    start_day_index: int,
    end_day_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns arrays: states over days [start_day_index+1..end_day_index], info_received, probabilities
    N = start_states.shape[0]
    days_count = end_day_index - start_day_index
    states = np.zeros((days_count, N), dtype=np.float64)
    info = np.zeros((days_count, N), dtype=np.float64)
    probs = np.zeros((days_count, N), dtype=np.float64)
    prev_states = start_states.copy()
    mem = np.zeros(N, dtype=np.float64)
    for d in range(days_count):
        global_day = start_day_index + d + 1
        # Determine lambda with intervention
        lam = params.lambda_broadcast_base * (params.lambda_broadcast_factor_after_day10 if global_day >= 10 else 1.0)
        # Info step
        rec = simulate_step_info(prev_states, neighbors, params.phi_family, params.phi_work, params.phi_community, lam)
        info[d, :] = rec
        mem = params.rho_info_decay * mem + (1.0 - params.rho_info_decay) * rec
        # Neighbor shares for decision
        share_f = compute_layer_neighbor_share(prev_states, neighbors["family"])
        share_w = compute_layer_neighbor_share(prev_states, neighbors["work_school"])
        share_c = compute_layer_neighbor_share(prev_states, neighbors["community"])
        logits = compute_logit(prev_states, share_f, share_w, share_c, risk, mem, age_oh, occ_oh, params, age_cat_names, occ_cat_names)
        if params.tau is not None and params.tau > 0:
            logits = logits / params.tau
        p = sigmoid(logits)
        probs[d, :] = p
        new_states = (np.random.rand(N) < p).astype(np.float64)
        states[d, :] = new_states
        prev_states = new_states
    return states, info, probs


def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).
    """
    # Convert FittedParams to Parameters for compatibility
    legacy_params = params.to_parameters()
    
    # Extract window information
    wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg = window
    train_start, train_end = train_window
    
    # Get age and occupation category names (hardcoded for now)
    age_cat_names = ['Youth', 'Young Adult', 'Middle Age']
    occ_cat_names = ['Blue Collar', 'White Collar', 'Student']
    
    # Run simulation on the training window
    result = evaluate_on_validation(
        wearing, neighbors, risk, age_oh, occ_oh,
        age_cat_names, occ_cat_names, legacy_params, 
        train_start, train_end, cfg.k_runs
    )
    
    return result


def evaluate_on_validation(
    wearing: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    val_start_idx: int,
    val_end_idx: int,
    k_runs: int,
) -> Dict[str, Any]:
    # Initial state is wearing[val_start_idx-1, :]
    init_states = wearing[val_start_idx - 1, :]
    T_val = val_end_idx - val_start_idx
    N = wearing.shape[1]
    # Observed daily rates
    obs_rates = wearing[val_start_idx:val_end_idx, :].mean(axis=1)
    # Observed transitions overall across val window
    prev_obs = wearing[val_start_idx - 1:val_end_idx - 1, :]
    curr_obs = wearing[val_start_idx:val_end_idx, :]
    # Construct observed transition probabilities
    def transition_probs(prev: np.ndarray, curr: np.ndarray) -> Dict[str, float]:
        # Flatten
        prev_f = prev.flatten()
        curr_f = curr.flatten()
        n = prev_f.shape[0]
        p01 = np.mean((prev_f == 0.0) & (curr_f == 1.0))
        p11 = np.mean((prev_f == 1.0) & (curr_f == 1.0))
        p10 = np.mean((prev_f == 1.0) & (curr_f == 0.0))
        p00 = np.mean((prev_f == 0.0) & (curr_f == 0.0))
        return {"P01": p01, "P11": p11, "P10": p10, "P00": p00}

    obs_trans = transition_probs(prev_obs, curr_obs)

    run_rmse = []
    run_mae = []
    run_brier = []
    run_trans_err = []

    daily_rate_runs = []

    for r in range(k_runs):
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,
            start_day_index=val_start_idx - 1,
            end_day_index=val_end_idx - 1,
        )
        # sim_states: T_val x N, sim_probs: T_val x N
        sim_rates = sim_states.mean(axis=1)
        daily_rate_runs.append(sim_rates)
        # RMSE and MAE on aggregate rates
        rmse = math.sqrt(float(np.mean((sim_rates - obs_rates) ** 2)))
        mae = float(np.mean(np.abs(sim_rates - obs_rates)))
        run_rmse.append(rmse)
        run_mae.append(mae)
        # Brier: use predicted probabilities vs observed wearing
        brier = float(np.mean((sim_probs - wearing[val_start_idx:val_end_idx, :]) ** 2))
        run_brier.append(brier)
        # Transitions from simulation
        prev_sim = np.vstack([init_states.reshape(1, -1), sim_states[:-1, :]])
        sim_trans = transition_probs(prev_sim, sim_states)
        trans_err = float(np.mean([abs(sim_trans[k] - obs_trans[k]) for k in ["P01", "P11", "P10", "P00"]]))
        run_trans_err.append(trans_err)

    def mean_ci(arr: List[float]) -> Tuple[float, float]:
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * s / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return m, ci

    rmse_mean, rmse_ci = mean_ci(run_rmse)
    mae_mean, mae_ci = mean_ci(run_mae)
    brier_mean, brier_ci = mean_ci(run_brier)
    trans_mean, trans_ci = mean_ci(run_trans_err)
    daily_rate_runs_arr = np.vstack(daily_rate_runs)  # k_runs x T_val
    daily_mean = daily_rate_runs_arr.mean(axis=0)
    daily_std = daily_rate_runs_arr.std(axis=0, ddof=1) if k_runs > 1 else np.zeros(T_val)
    daily_ci = 1.96 * daily_std / math.sqrt(k_runs) if k_runs > 1 else daily_std

    metrics = {
        "RMSE_aggregate_mean": rmse_mean,
        "RMSE_aggregate_CI95": rmse_ci,
        "MAE_aggregate_mean": mae_mean,
        "MAE_aggregate_CI95": mae_ci,
        "Brier_mean": brier_mean,
        "Brier_CI95": brier_ci,
        "TransitionFit_mean": trans_mean,
        "TransitionFit_CI95": trans_ci,
        "observed_daily_rates": obs_rates.tolist(),
        "predicted_daily_rates_mean": daily_mean.tolist(),
        "predicted_daily_rates_CI95": daily_ci.tolist(),
        "k_runs": k_runs,
    }
    return metrics


def simulate_forecast(
    wearing: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    last_train_day_idx: int,
    forecast_days: int,
    k_runs: int,
) -> Dict[str, Any]:
    # Start from last_train_day_idx state
    init_states = wearing[last_train_day_idx, :]
    start_day = last_train_day_idx
    end_day = last_train_day_idx + forecast_days
    run_rates = []
    for r in range(k_runs):
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,
            start_day_index=start_day,
            end_day_index=end_day,
        )
        run_rates.append(sim_states.mean(axis=1))
    run_rates_arr = np.vstack(run_rates)  # k x forecast_days
    mean_rates = run_rates_arr.mean(axis=0)
    std_rates = run_rates_arr.std(axis=0, ddof=1) if k_runs > 1 else np.zeros_like(mean_rates)
    ci95 = 1.96 * std_rates / math.sqrt(k_runs) if k_runs > 1 else std_rates
    days_forecast = [start_day + d + 1 for d in range(forecast_days)]
    forecast = {
        "days": days_forecast,
        "forecast_mean_rates": mean_rates.tolist(),
        "forecast_CI95": ci95.tolist(),
        "k_runs": k_runs,
    }
    return forecast


class LogitHeadCalibrator(Calibrator):
    """Fits a logistic decision head from micro-transitions on days_train (L2 regularized; intercept not regularized)."""
    
    def __init__(self, l2_reg: float = 1.0, max_iter: int = 400, learning_rate: float = 0.1):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        train_start, train_end = train_window
        
        # Set seed for reproducibility
        set_global_seed(seed)
        
        # Compute neighbor shares and memory info for feature matrix
        T = wearing.shape[0]
        N = wearing.shape[1]
        
        # Compute neighbor shares per day
        share_f_by_day = np.zeros_like(wearing)
        share_w_by_day = np.zeros_like(wearing)
        share_c_by_day = np.zeros_like(wearing)
        
        for t in range(T):
            s_prev = wearing[t, :]
            share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
            share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
            share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])
        
        # Compute memory info (assuming received_info is available or zero)
        received_info = np.zeros_like(wearing)  # Placeholder - should be extracted from bundle if available
        mem_info = compute_mem_info(received_info, cfg.rho_info_decay_default)
        
        # Build feature matrix for logistic regression
        X, y = build_feature_matrix(
            wearing=wearing,
            mem_info=mem_info,
            share_f_by_day=share_f_by_day,
            share_w_by_day=share_w_by_day,
            share_c_by_day=share_c_by_day,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            day_start=train_start,
            day_end=train_end
        )
        
        # Fit logistic regression with L2 regularization
        print(f"LogitHeadCalibrator: Starting optimization (L2={self.l2_reg}, max_iter={self.max_iter}, lr={self.learning_rate})")
        beta = fit_logistic_l2(X, y, self.l2_reg, self.max_iter, self.learning_rate, verbose=True)
        
        # Create FittedParams from beta coefficients
        decision_weights = {
            'alpha': beta[0],  # intercept
            'gamma': beta[1],  # previous state
            'theta_f': 1.0,    # family weight (normalized)
            'theta_w': 1.0,    # work weight (normalized)  
            'theta_c': 1.0,    # community weight (normalized)
            'beta_r': beta[2] if len(beta) > 2 else 0.0,  # risk perception
            'beta_i': beta[3] if len(beta) > 3 else 0.0,  # info received
            'age_effects': {},
            'occ_effects': {}
        }
        
        # Extract demographic effects if available
        if len(beta) > 4:
            n_age = age_oh.shape[1]
            n_occ = occ_oh.shape[1]
            age_effects = {f'age_{i}': beta[4+i] for i in range(n_age)}
            occ_effects = {f'occ_{i}': beta[4+n_age+i] for i in range(n_occ)}
            decision_weights['age_effects'] = age_effects
            decision_weights['occ_effects'] = occ_effects
        
        layer_weights = {
            'family': 1.0,
            'work_school': 1.0, 
            'community': 1.0
        }
        
        info_params = {
            'phi_family': 0.1,
            'phi_work': 0.1,
            'phi_community': 0.1,
            'lambda_broadcast_base': 0.05,
            'lambda_broadcast_factor_after_day10': cfg.gov_lam_factor_default,
            'rho_info_decay': cfg.rho_info_decay_default
        }
        
        noise_params = {
            'tau': 1.0
        }
        
        meta = {
            'seed': seed,
            'calibrator_name': 'logit_head',
            'training_window': train_window,
            'l2_reg': self.l2_reg,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate
        }
        
        # Create fitted parameters
        fitted_params = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta=meta
        )
        
        # Evaluate the fitted parameters for debugging
        print("LogitHeadCalibrator: Evaluating fitted parameters...")
        try:
            result = evaluator(simulator, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
            rmse = result.get('RMSE_aggregate_mean', float('inf'))
            mae = result.get('MAE_aggregate_mean', float('inf'))
            print(f"LogitHeadCalibrator: Training RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        except Exception as e:
            print(f"LogitHeadCalibrator: Evaluation failed: {e}")
        
        return fitted_params


class RandomSearchCalibrator(Calibrator):
    """Black-box search over selected simulator params (e.g., layer weights, info rates, memory, temperature)."""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        set_global_seed(seed)
        
        # Extract cfg from bundle
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        
        best_params = None
        best_score = float('inf')
        
        for trial in range(self.n_trials):
            # Random parameter sampling
            trial_seed = seed + trial
            np.random.seed(trial_seed)
            
            # Sample random parameters
            decision_weights = {
                'alpha': np.random.normal(0.0, 1.0),
                'gamma': np.random.uniform(0.5, 3.0),
                'theta_f': np.random.uniform(0.5, 2.0),
                'theta_w': np.random.uniform(0.5, 2.0),
                'theta_c': np.random.uniform(0.5, 2.0),
                'beta_r': np.random.normal(0.0, 0.5),
                'beta_i': np.random.normal(0.0, 0.5),
                'age_effects': {},
                'occ_effects': {}
            }
            
            layer_weights = {
                'family': np.random.uniform(0.1, 2.0),
                'work_school': np.random.uniform(0.1, 2.0),
                'community': np.random.uniform(0.1, 2.0)
            }
            
            info_params = {
                'phi_family': np.random.uniform(0.01, 0.3),
                'phi_work': np.random.uniform(0.01, 0.3),
                'phi_community': np.random.uniform(0.01, 0.3),
                'lambda_broadcast_base': np.random.uniform(0.01, 0.2),
                'lambda_broadcast_factor_after_day10': np.random.uniform(1.0, 3.0),
                'rho_info_decay': np.random.uniform(0.1, 0.9)
            }
            
            noise_params = {
                'tau': np.random.uniform(0.5, 2.0)
            }
            
            meta = {
                'seed': trial_seed,
                'calibrator_name': 'random_search',
                'training_window': train_window,
                'trial': trial
            }
            
            # Create candidate parameters
            candidate_params = FittedParams(
                decision_weights=decision_weights,
                layer_weights=layer_weights,
                info_params=info_params,
                noise_params=noise_params,
                meta=meta
            )
            
            # Evaluate on training window
            try:
                result = evaluator(simulator, candidate_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
                score = result.get('RMSE_aggregate_mean', float('inf'))
                
                if trial % 20 == 0:  # Print every 20th trial
                    print(f"Trial {trial}: RMSE = {score:.4f}, Best so far = {best_score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = candidate_params
                    print(f"New best at trial {trial}: RMSE = {score:.4f}")
                    
            except Exception as e:
                # Skip this trial if evaluation fails
                if trial % 20 == 0:
                    print(f"Trial {trial} failed: {e}")
                continue
        
        if best_params is None:
            # Fallback to default parameters
            best_params = self._get_default_params(seed, train_window)
            
        return best_params
    
    def _get_default_params(self, seed: int, train_window: Tuple[int, int]) -> FittedParams:
        """Get default parameters as fallback."""
        return FittedParams(
            decision_weights={
                'alpha': 0.0, 'gamma': 1.0, 'theta_f': 1.0, 'theta_w': 1.0, 'theta_c': 1.0,
                'beta_r': 0.0, 'beta_i': 0.0, 'age_effects': {}, 'occ_effects': {}
            },
            layer_weights={'family': 1.0, 'work_school': 1.0, 'community': 1.0},
            info_params={
                'phi_family': 0.1, 'phi_work': 0.1, 'phi_community': 0.1,
                'lambda_broadcast_base': 0.05, 'lambda_broadcast_factor_after_day10': 1.5,
                'rho_info_decay': 0.5
            },
            noise_params={'tau': 1.0},
            meta={'seed': seed, 'calibrator_name': 'random_search', 'training_window': train_window}
        )


# Calibrator registry
CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
}


def get_calibrator(name: str, config_path: str = None):
    """Get calibrator by name with optional configuration."""
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    
    # For now, use default parameters
    # TODO: Load optional config (JSON/YAML) into kwargs
    kwargs = {}
    
    return CALIBRATOR_REGISTRY[name](**kwargs)


def simulation_test(calibrator_type: str = "logit_head") -> None:
    """
    Test the simulation using pre-calibrated parameters on test data.
    
    Args:
        calibrator_type: "logit_head" or "random_search" to choose which calibrated parameters to use
    """
    print(f"Starting simulation test with {calibrator_type} calibrator...")
    
    # Load configuration and parameters
    if calibrator_type == "logit_head":
        param_dir = "data_fitting/mask_adoption_data/outputs_LogitHeadCalibrator"
    elif calibrator_type == "random_search":
        param_dir = "data_fitting/mask_adoption_data/outputs_RandomSearchCalibrator"
    else:
        raise ValueError(f"Unknown calibrator type: {calibrator_type}")
    
    # Load configuration
    config_path = os.path.join(param_dir, "config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    cfg = SimulationConfig(**config_data["config"])
    
    # Load calibrated parameters
    param_path = os.path.join(param_dir, "calibrated_parameters.json")
    with open(param_path, 'r') as f:
        param_data = json.load(f)
    
    # Convert FittedParams back to Parameters
    fitted_params = FittedParams(**param_data)
    params = fitted_params.to_parameters()
    
    print(f"Loaded parameters from {calibrator_type} calibrator")
    print(f"Calibrator meta: {fitted_params.meta}")
    
    # Set seed for reproducibility
    set_global_seed(cfg.seed)
    
    # Load test data
    test_path = os.path.join(cfg.data_folder, "test_data.csv")
    try:
        test_df = load_train_data(test_path)  # Reuse the same function
        print(f"Loaded test data: {len(test_df)} records")
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return
    
    # Load agent attributes and social network (same as training)
    agents_path = os.path.join(cfg.data_folder, "agent_attributes.csv")
    social_path = os.path.join(cfg.data_folder, "social_network.json")
    try:
        agents_df = load_agent_attributes(agents_path)
        social_raw = load_social_network(social_path)
        print(f"Loaded agent data: {len(agents_df)} agents")
    except Exception as e:
        print(f"Failed to load agent data: {e}")
        return
    
    # Align IDs (same as training)
    common_ids, id2idx, agents_df, social_f, test_df = align_ids(agents_df, social_raw, test_df)
    n_agents = len(common_ids)
    print(f"Aligned IDs: {n_agents} common agents")
    
    # Build network (multiplex)
    neighbors = build_multiplex_adjacency(social_f, id2idx, n_agents)
    
    # Pivot test time series
    wearing_test, received_test, days_test = pivot_states(test_df, id2idx)
    T_test = wearing_test.shape[0]
    print(f"Test data shape: {wearing_test.shape}, days: {days_test}")
    
    # Risk perception aligned
    risk_perception = agents_df.set_index("agent_id").loc[common_ids]["risk_perception"].to_numpy(dtype=np.float64)
    
    # Encode demographics
    age_oh, age_cat_names, occ_oh, occ_cat_names = encode_demographics(agents_df, common_ids)
    
    # Prepare neighbor shares per day using observed wearing states
    share_f_by_day = np.zeros_like(wearing_test)
    share_w_by_day = np.zeros_like(wearing_test)
    share_c_by_day = np.zeros_like(wearing_test)
    
    for t in range(T_test):
        s_prev = wearing_test[t, :]
        share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
        share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
        share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])
    
    # Compute mem_info array using calibrated rho
    rho_test = params.rho_info_decay
    mem_info_test = compute_mem_info(received_test, rho=rho_test)
    
    # Simulation-Based Inference    # Run Monte Carlo simulation on test data
    print(f"Running Monte Carlo simulation on test data with {cfg.k_runs} runs...")
    test_start_idx = 0
    test_end_idx = T_test
    
    # Store results from multiple runs
    run_rmse = []
    run_mae = []
    run_brier = []
    run_transition_fit = []
    daily_rate_runs = []
    
    # Use the first day as initial state
    initial_states = wearing_test[0, :]
    observed_rates = np.mean(wearing_test[test_start_idx:test_end_idx], axis=1)
    
    for r in range(cfg.k_runs):
        # Set different seed for each run to ensure variability
        run_seed = cfg.seed + r
        set_global_seed(run_seed)
        
        # Run simulation on test data
        states_test, info_test, probs_test = simulate_window(
            start_states=initial_states,
            neighbors=neighbors,
            risk=risk_perception,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,
            start_day_index=test_start_idx,
            end_day_index=test_end_idx,
        )
        
        # Calculate metrics for this run
        predicted_rates = np.mean(states_test, axis=1)
        daily_rate_runs.append(predicted_rates)
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean((observed_rates - predicted_rates) ** 2))
        mae = np.mean(np.abs(observed_rates - predicted_rates))
        
        # Calculate Brier score
        brier = np.mean((predicted_rates - observed_rates) ** 2)
        
        # Calculate transition fit
        if test_end_idx - test_start_idx > 1:
            prev_obs = wearing_test[test_start_idx:test_end_idx-1].flatten()
            curr_obs = wearing_test[test_start_idx+1:test_end_idx].flatten()
            prev_pred = states_test[:-1].flatten()
            curr_pred = states_test[1:].flatten()
            
            # Calculate transition probabilities
            obs_p01 = np.mean((prev_obs == 0.0) & (curr_obs == 1.0))
            obs_p11 = np.mean((prev_obs == 1.0) & (curr_obs == 1.0))
            pred_p01 = np.mean((prev_pred == 0.0) & (curr_pred == 1.0))
            pred_p11 = np.mean((prev_pred == 1.0) & (curr_pred == 1.0))
            
            transition_fit = 0.5 * (abs(obs_p01 - pred_p01) + abs(obs_p11 - pred_p11))
        else:
            transition_fit = 0.0
        
        # Store results
        run_rmse.append(rmse)
        run_mae.append(mae)
        run_brier.append(brier)
        run_transition_fit.append(transition_fit)
        
        if (r + 1) % 5 == 0 or r == 0:
            print(f"  Run {r+1}/{cfg.k_runs}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # Calculate statistics across runs
    run_rmse = np.array(run_rmse)
    run_mae = np.array(run_mae)
    run_brier = np.array(run_brier)
    run_transition_fit = np.array(run_transition_fit)
    daily_rate_runs = np.array(daily_rate_runs)
    
    # Calculate means and confidence intervals
    rmse_mean = np.mean(run_rmse)
    rmse_ci = 1.96 * np.std(run_rmse) / np.sqrt(cfg.k_runs)
    mae_mean = np.mean(run_mae)
    mae_ci = 1.96 * np.std(run_mae) / np.sqrt(cfg.k_runs)
    brier_mean = np.mean(run_brier)
    brier_ci = 1.96 * np.std(run_brier) / np.sqrt(cfg.k_runs)
    transition_fit_mean = np.mean(run_transition_fit)
    transition_fit_ci = 1.96 * np.std(run_transition_fit) / np.sqrt(cfg.k_runs)
    
    # Calculate daily rate statistics
    predicted_daily_rates_mean = np.mean(daily_rate_runs, axis=0)
    predicted_daily_rates_ci = 1.96 * np.std(daily_rate_runs, axis=0) / np.sqrt(cfg.k_runs)
    
    # Create test metrics dictionary
    test_metrics = {
        "RMSE_aggregate_mean": rmse_mean,
        "RMSE_aggregate_CI95": rmse_ci,
        "MAE_aggregate_mean": mae_mean,
        "MAE_aggregate_CI95": mae_ci,
        "Brier_mean": brier_mean,
        "Brier_CI95": brier_ci,
        "TransitionFit_mean": transition_fit_mean,
        "TransitionFit_CI95": transition_fit_ci,
        "observed_daily_rates": observed_rates.tolist(),
        "predicted_daily_rates_mean": predicted_daily_rates_mean.tolist(),
        "predicted_daily_rates_CI95": predicted_daily_rates_ci.tolist(),
        "k_runs": cfg.k_runs
    }
    
    # Create output directory for test results
    test_output_dir = os.path.join(cfg.data_folder, f"test_outputs_{calibrator_type}")
    ensure_dir(test_output_dir)
    
    # Save test results
    save_json(test_metrics, os.path.join(test_output_dir, "test_metrics.json"))
    save_json(fitted_params.to_dict(), os.path.join(test_output_dir, "calibrated_parameters.json"))
    save_json({"config": asdict(cfg)}, os.path.join(test_output_dir, "config.json"))
    
    # Save daily rates as CSV
    test_days = days_test[test_start_idx:test_end_idx]
    df_test = pd.DataFrame({
        "day": test_days,
        "observed_rate": test_metrics["observed_daily_rates"],
        "predicted_rate_mean": test_metrics["predicted_daily_rates_mean"],
        "predicted_rate_CI95": test_metrics["predicted_daily_rates_CI95"],
    })
    df_test.to_csv(os.path.join(test_output_dir, "test_daily_rates.csv"), index=False)
    
    # Print test results
    print(f"\n=== Test Results ({calibrator_type.upper()}) ===")
    print(f"Test RMSE: {test_metrics['RMSE_aggregate_mean']:.4f} ± {test_metrics['RMSE_aggregate_CI95']:.4f}")
    print(f"Test MAE: {test_metrics['MAE_aggregate_mean']:.4f} ± {test_metrics['MAE_aggregate_CI95']:.4f}")
    print(f"Test Brier: {test_metrics['Brier_mean']:.4f} ± {test_metrics['Brier_CI95']:.4f}")
    print(f"Test TransitionFit: {test_metrics['TransitionFit_mean']:.4f} ± {test_metrics['TransitionFit_CI95']:.4f}")
    print(f"Test data days: {len(test_days)} days")
    print(f"Test results saved to: {test_output_dir}")


def main() -> None:
    """
    Orchestrates the multi-agent simulation:
    - Load data
    - Align IDs and build multiplex network
    - Prepare features and neighbor shares
    - Calibrate info and decision parameters
    - Evaluate on temporal holdout with K-run simulation
    - Forecast days 30-39
    - Save outputs and configuration
    """
    cfg = SimulationConfig()
    set_global_seed(cfg.seed)

    # Prepare output directory
    out_dir = os.path.join(cfg.data_folder, cfg.output_folder)
    ensure_dir(out_dir)

    # Load data
    agents_path = os.path.join(cfg.data_folder, "agent_attributes.csv")
    social_path = os.path.join(cfg.data_folder, "social_network.json")
    train_path = os.path.join(cfg.data_folder, "train_data.csv")
    try:
        agents_df = load_agent_attributes(agents_path)
        social_raw = load_social_network(social_path)
        train_df = load_train_data(train_path)
    except Exception as e:
        # Print error and exit gracefully
        print(str(e))
        return

    # Align IDs
    common_ids, id2idx, agents_df, social_f, train_df = align_ids(agents_df, social_raw, train_df)
    n_agents = len(common_ids)

    # Build network (multiplex)
    neighbors = build_multiplex_adjacency(social_f, id2idx, n_agents)

    # Pivot time series
    wearing, received, days = pivot_states(train_df, id2idx)
    T = wearing.shape[0]

    # Risk perception aligned
    risk_perception = agents_df.set_index("agent_id").loc[common_ids]["risk_perception"].to_numpy(dtype=np.float64)

    # Encode demographics
    age_oh, age_cat_names, occ_oh, occ_cat_names = encode_demographics(agents_df, common_ids)

    # Prepare neighbor shares per day using observed wearing states
    share_f_by_day = np.zeros_like(wearing)
    share_w_by_day = np.zeros_like(wearing)
    share_c_by_day = np.zeros_like(wearing)
    for t in range(T):
        s_prev = wearing[t, :]
        share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
        share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
        share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])

    # Compute mem_info array using default rho for training features
    rho_train = cfg.rho_info_decay_default
    mem_info = compute_mem_info(received, rho=rho_train)

    # Split into train/validation
    train_end_idx, val_start_idx, val_end_idx = build_train_validation_splits(days, cfg.val_split_ratio)

    # Use new pluggable calibration architecture
    # Create calibrator
    calibrator = get_calibrator("logit_head")  # Switch back to LogitHeadCalibrator
    # calibrator = get_calibrator("random_search")  # Commented out RandomSearchCalibrator
    
    # Prepare bundle for calibrator
    bundle = (wearing, neighbors, risk_perception, age_oh, occ_oh, cfg)
    
    # Define training window
    train_window = (1, train_end_idx)
    
    # Fit parameters using calibrator
    fitted_params = calibrator.fit(
        bundle=bundle,
        simulator=None,  # Not used by LogitHeadCalibrator
        evaluator=evaluate_params,
        train_window=train_window,
        seed=cfg.seed
    )
    
    # Convert to legacy parameters for compatibility with existing code
    params = fitted_params.to_parameters()
    
    # Parameters are already available from fitted_params
    # No need for manual mapping - use the converted params directly

    # Evaluate on validation via K-run simulation
    metrics = evaluate_on_validation(
        wearing=wearing,
        neighbors=neighbors,
        risk=risk_perception,
        age_oh=age_oh,
        occ_oh=occ_oh,
        age_cat_names=age_cat_names,
        occ_cat_names=occ_cat_names,
        params=params,
        val_start_idx=val_start_idx,
        val_end_idx=val_end_idx,
        k_runs=cfg.k_runs,
    )

    # Forecast next cfg.forecast_days from last observed day (T-1)
    forecast = simulate_forecast(
        wearing=wearing,
        neighbors=neighbors,
        risk=risk_perception,
        age_oh=age_oh,
        occ_oh=occ_oh,
        age_cat_names=age_cat_names,
        occ_cat_names=occ_cat_names,
        params=params,
        last_train_day_idx=T - 1,
        forecast_days=cfg.forecast_days,
        k_runs=cfg.k_runs,
    )

    # Save outputs
    save_json({"config": asdict(cfg)}, os.path.join(out_dir, "config.json"))
    save_json(fitted_params.to_dict(), os.path.join(out_dir, "calibrated_parameters.json"))
    save_json(metrics, os.path.join(out_dir, "validation_metrics.json"))
    save_json(forecast, os.path.join(out_dir, "forecast.json"))

    # Also save daily predicted rates for validation as CSV
    val_days = days[val_start_idx:val_end_idx]
    df_val = pd.DataFrame({
        "day": val_days,
        "observed_rate": metrics["observed_daily_rates"],
        "predicted_rate_mean": metrics["predicted_daily_rates_mean"],
        "predicted_rate_CI95": metrics["predicted_daily_rates_CI95"],
    })
    df_val.to_csv(os.path.join(out_dir, "validation_daily_rates.csv"), index=False)

    # Save forecast CSV
    df_fore = pd.DataFrame({
        "day": forecast["days"],
        "forecast_mean_rate": forecast["forecast_mean_rates"],
        "forecast_CI95": forecast["forecast_CI95"],
    })
    df_fore.to_csv(os.path.join(out_dir, "forecast_daily_rates.csv"), index=False)

    if cfg.verbose:
        print("Calibration and evaluation complete.")
        print(f"Saved outputs to {out_dir}")


# Execute main for both direct execution and sandbox wrapper invocation
if __name__ == "__main__":
    import sys
    
    # Check if test mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            calibrator_type = sys.argv[2]
        else:
            calibrator_type = "logit_head"  # Default to logit_head
        
        print(f"Running in TEST mode with {calibrator_type} calibrator...")
        simulation_test(calibrator_type)
    else:
        print("Running in CALIBRATION mode...")
        main()