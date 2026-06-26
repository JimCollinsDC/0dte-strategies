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
