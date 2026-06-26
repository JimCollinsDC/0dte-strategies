from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def annualized_sharpe(x: pd.Series, periods: int = 252) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    vol = float(s.std(ddof=1))
    if (not np.isfinite(vol)) or vol <= 1e-12:
        return np.nan
    return float(np.sqrt(periods) * s.mean() / vol)


def annualized_sortino(x: pd.Series, periods: int = 252) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    downside = s[s < 0]
    dstd = float(downside.std(ddof=1)) if not downside.empty else np.nan
    if (not np.isfinite(dstd)) or dstd <= 1e-12:
        return np.nan
    return float(np.sqrt(periods) * s.mean() / dstd)


def max_drawdown(x: pd.Series) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    cum = s.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def cvar(x: pd.Series, q: float = 0.05) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    cutoff = float(s.quantile(q))
    tail = s[s <= cutoff]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def profit_factor(x: pd.Series) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    gains = float(s[s > 0].sum())
    losses = float(-s[s < 0].sum())
    if losses <= 1e-12:
        return np.nan
    return gains / losses


def expectancy(x: pd.Series) -> float:
    s = x.dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def simple_t_stat(x: pd.Series) -> float:
    s = x.dropna()
    n = int(s.shape[0])
    if n < 2:
        return np.nan
    std = float(s.std(ddof=1))
    if std <= 1e-12:
        return np.nan
    return float(s.mean() / (std / math.sqrt(n)))


def _stationary_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(1, int(block_size))
    idx = np.empty(n, dtype=int)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        take = min(b, n - filled)
        block = (start + np.arange(take)) % n
        idx[filled:filled + take] = block
        filled += take
    return idx


def bootstrap_pvalue_mean(
    x: pd.Series,
    n_bootstrap: int = 400,
    block_size: int = 5,
    seed: int = 42,
) -> float:
    s = x.dropna().astype(float)
    if s.shape[0] < 20:
        return np.nan

    arr = s.to_numpy(dtype=float)
    obs_t = simple_t_stat(pd.Series(arr))
    if not np.isfinite(obs_t):
        return np.nan

    centered = arr - float(np.mean(arr))
    rng = np.random.default_rng(seed)
    boot_t = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = _stationary_block_bootstrap_indices(centered.shape[0], block_size, rng)
        sample = centered[idx]
        boot_t[i] = simple_t_stat(pd.Series(sample))

    boot_t = boot_t[np.isfinite(boot_t)]
    if boot_t.size == 0:
        return np.nan
    return float(np.mean(np.abs(boot_t) >= abs(obs_t)))


def adjusted_t_stat_for_mining(t_stat: float, n_tests: int) -> float:
    if (not np.isfinite(t_stat)) or n_tests <= 1:
        return float(t_stat)
    penalty = math.sqrt(max(0.0, 2.0 * math.log(float(n_tests))))
    sign = 1.0 if t_stat >= 0 else -1.0
    return float(sign * max(0.0, abs(t_stat) - penalty))


def summarize_returns(x: pd.Series) -> dict[str, Any]:
    s = x.dropna()
    if s.empty:
        return {
            "expectancy": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "cvar_5": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "tail_risk_es1": np.nan,
            "t_stat": np.nan,
            "n_obs": 0,
        }

    return {
        "expectancy": expectancy(s),
        "sharpe": annualized_sharpe(s),
        "sortino": annualized_sortino(s),
        "max_drawdown": max_drawdown(s),
        "cvar_5": cvar(s, q=0.05),
        "win_rate": float((s > 0).mean()),
        "profit_factor": profit_factor(s),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "tail_risk_es1": cvar(s, q=0.01),
        "t_stat": simple_t_stat(s),
        "n_obs": int(s.shape[0]),
    }
