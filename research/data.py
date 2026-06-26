from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig


@dataclass(frozen=True)
class FeatureMeta:
    name: str
    category: str
    needs_external_data: bool
    economic_rationale: str
    compute_cost: str


def parse_levels(mnes_str: str) -> list[float]:
    return sorted(float(x) for x in str(mnes_str).split("/"))


def mnes_int(level: float) -> int:
    return int(round(level * 1e5))


def get_legs(strategy: str, mnes_str: str) -> list[tuple[str, int, int]]:
    levels = parse_levels(mnes_str)
    if len(levels) < 2:
        return []

    l = levels[0]
    h = levels[-1]

    if strategy == "strangle":
        return [("P", mnes_int(l), 1), ("C", mnes_int(h), 1)]
    if strategy == "risk_reversal":
        return [("P", mnes_int(l), -1), ("C", mnes_int(h), 1)]
    if strategy == "bull_call_spread":
        return [("C", mnes_int(l), 1), ("C", mnes_int(h), -1)]
    if strategy == "call_ratio_spread":
        return [("C", mnes_int(l), 1), ("C", mnes_int(h), -2)]
    if strategy == "bear_put_spread":
        return [("P", mnes_int(l), -1), ("P", mnes_int(h), 1)]
    if strategy == "put_ratio_spread":
        return [("P", mnes_int(l), -2), ("P", mnes_int(h), 1)]
    if strategy == "iron_condor":
        if len(levels) == 3:
            m = levels[1]
            return [("P", mnes_int(l), 1), ("P", mnes_int(m), -1), ("C", mnes_int(m), -1), ("C", mnes_int(h), 1)]
        if len(levels) == 4:
            ml = levels[1]
            mh = levels[2]
            return [("P", mnes_int(l), 1), ("P", mnes_int(ml), -1), ("C", mnes_int(mh), -1), ("C", mnes_int(h), 1)]
    return []


def _third_friday(dt: pd.Series) -> pd.Series:
    return (dt.dt.weekday == 4) & (dt.dt.day >= 15) & (dt.dt.day <= 21)


def _read_event_dates(path: Path, date_col: str = "date") -> set[pd.Timestamp]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if date_col not in df.columns and len(df.columns) > 0:
        date_col = df.columns[0]
    return set(pd.to_datetime(df[date_col]).dt.normalize().tolist())


def _build_spx_daily_features(data_dir: Path) -> pd.DataFrame:
    eod = pd.read_csv(data_dir / "ALL_eod.csv")
    eod["Date"] = pd.to_datetime(eod["Date"])
    spx = eod[eod["root"] == "SPX"].copy().sort_values("Date")

    spx["ret_1d"] = spx["Close"].pct_change()
    spx["overnight_gap"] = (spx["Open"] - spx["Close"].shift(1)) / spx["Close"].shift(1)

    tr = pd.concat(
        [
            (spx["High"] - spx["Low"]).abs(),
            (spx["High"] - spx["Close"].shift(1)).abs(),
            (spx["Low"] - spx["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    spx["atr14"] = tr.rolling(14).mean() / spx["Close"]

    spx["ma5"] = spx["Close"].rolling(5).mean()
    spx["ma20"] = spx["Close"].rolling(20).mean()
    spx["price_vs_ma20"] = (spx["Close"] / spx["ma20"]) - 1.0
    spx["mom_5d"] = spx["Close"].pct_change(5)
    spx["reversion_z20"] = (
        (spx["Close"] - spx["Close"].rolling(20).mean())
        / spx["Close"].rolling(20).std()
    )
    spx["realized_vol_5d"] = spx["ret_1d"].rolling(5).std() * np.sqrt(252)
    spx["realized_vol_20d"] = spx["ret_1d"].rolling(20).std() * np.sqrt(252)

    # Lag all daily features so they are known at 10:00 ET of t.
    lag_cols = [
        "overnight_gap",
        "atr14",
        "price_vs_ma20",
        "mom_5d",
        "reversion_z20",
        "realized_vol_5d",
        "realized_vol_20d",
    ]
    for c in lag_cols:
        spx[c] = spx[c].shift(1)

    out_cols = ["Date", *lag_cols]
    return spx[out_cols].rename(columns={"Date": "quote_date"})


def _half_spread_cost(strats: pd.DataFrame, opt: pd.DataFrame) -> pd.Series:
    opt = opt.copy()
    opt["quote_date"] = pd.to_datetime(opt["quote_date"])
    opt["quote_time"] = opt["quote_time"].astype(str)
    opt["option_type"] = opt["option_type"].astype(str)
    opt["mnes"] = pd.to_numeric(opt["mnes"], errors="coerce").astype("Int64")
    opt = opt.dropna(subset=["mnes"])  # defensive
    opt["mnes"] = opt["mnes"].astype(int)
    opt_lookup_df = opt.groupby(["quote_date", "quote_time", "option_type", "mnes"], as_index=False)["bas"].mean()
    bas_lookup = {
        (r.quote_date, r.quote_time, r.option_type, int(r.mnes)): float(r.bas)
        for r in opt_lookup_df.itertuples(index=False)
    }

    def calc(row: pd.Series) -> float:
        legs = get_legs(str(row["option_type"]), str(row["mnes"]))
        if not legs:
            return np.nan
        total = 0.0
        for side, m_int, qty in legs:
            key = (row["quote_date"], row["quote_time"], side, m_int)
            bas = bas_lookup.get(key, np.nan)
            if pd.isna(bas):
                return np.nan
            total += abs(qty) * bas
        return 0.5 * total

    return strats.apply(calc, axis=1)


def _dynamic_slippage_cost(panel: pd.DataFrame, cfg: ResearchConfig) -> pd.Series:
    # Start from baseline bps and scale up in stressed volatility/liquidity regimes.
    base = float(cfg.slippage_bps) * 0.01

    vix_ref = max(1e-6, float(cfg.slippage_vix_ref))
    vix = pd.to_numeric(panel.get("vix", np.nan), errors="coerce").fillna(vix_ref)
    vix_excess = np.maximum((vix - vix_ref) / vix_ref, 0.0)
    vix_mult = 1.0 + float(cfg.slippage_vix_beta) * vix_excess

    spread_proxy = pd.to_numeric(panel.get("half_spread_cost", np.nan), errors="coerce").fillna(0.0)
    spread_mult = 1.0 + float(cfg.slippage_spread_beta) * spread_proxy

    event_flag = (
        pd.to_numeric(panel.get("is_fomc", 0), errors="coerce").fillna(0).astype(int)
        | pd.to_numeric(panel.get("is_cpi", 0), errors="coerce").fillna(0).astype(int)
    )
    event_mult = np.where(event_flag > 0, float(cfg.slippage_event_mult), 1.0)

    return base * vix_mult * spread_mult * event_mult


def build_research_panel(cfg: ResearchConfig) -> tuple[pd.DataFrame, list[FeatureMeta]]:
    data_dir = cfg.data_dir
    strats = pd.read_parquet(data_dir / "data_structures.parquet")
    vix = pd.read_parquet(data_dir / "vix.parquet")
    slopes = pd.read_parquet(data_dir / "slopes.parquet")
    spx_mom = pd.read_parquet(data_dir / "future_moments_SPX.parquet")
    opt = pd.read_parquet(data_dir / "data_opt.parquet")

    strats = strats.copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats = strats[strats["quote_time"] == cfg.entry_time].copy()

    strats["mnes"] = strats["mnes"].astype(str)
    strats["max_moneyness_dev"] = strats["mnes"].map(lambda s: max(abs(x - 1.0) for x in parse_levels(s)))
    strats = strats[strats["max_moneyness_dev"] <= float(cfg.max_moneyness_dev)].copy()

    vix = vix.copy()
    vix["quote_date"] = pd.to_datetime(vix["quote_date"])
    vix["quote_time"] = vix["quote_time"].astype(str)
    vix10 = vix[vix["quote_time"] == cfg.entry_time].copy()
    if "root" in vix10.columns:
        vix10 = vix10[vix10["root"] == "SPXW"]
    if "dte" in vix10.columns:
        vix10 = vix10[vix10["dte"] == 0]
    vix10 = vix10.groupby("quote_date", as_index=False)[["vix", "vixup", "vixdn"]].mean()
    vix10["isk"] = vix10["vixup"] - vix10["vixdn"]

    slopes = slopes.copy()
    slopes["quote_date"] = pd.to_datetime(slopes["quote_date"])
    slopes["quote_time"] = slopes["quote_time"].astype(str)
    slopes = slopes[slopes["quote_time"] == cfg.entry_time].groupby("quote_date", as_index=False)[["slope_up", "slope_dn"]].mean()

    spx_mom = spx_mom.copy()
    spx_mom["date"] = pd.to_datetime(spx_mom["date"])
    spx_mom["time"] = spx_mom["time"].astype(str)
    spx10 = spx_mom[spx_mom["time"] == cfg.entry_time][["date", "SPX_lret", "SPX_lrv", "SPX_lrv_skew"]].copy()
    for c in ["SPX_lret", "SPX_lrv", "SPX_lrv_skew"]:
        spx10[c] = spx10[c].shift(1)
    spx10 = spx10.rename(columns={"date": "quote_date"})

    spx_daily = _build_spx_daily_features(data_dir)

    panel = strats.merge(vix10[["quote_date", "vix", "isk"]], on="quote_date", how="left")
    panel = panel.merge(slopes, on="quote_date", how="left")
    panel = panel.merge(spx10, on="quote_date", how="left")
    panel = panel.merge(spx_daily, on="quote_date", how="left")

    panel = panel.sort_values(["option_type", "mnes", "quote_date"]).reset_index(drop=True)
    grp = panel.groupby(["option_type", "mnes"], observed=True)
    panel["pnl_l1"] = grp["reth_und"].shift(1)
    panel["pnl_mean5_l1"] = grp["pnl_l1"].transform(lambda s: s.rolling(5).mean())
    panel["pnl_std5_l1"] = grp["pnl_l1"].transform(lambda s: s.rolling(5).std())

    panel["weekday"] = panel["quote_date"].dt.weekday
    panel["month"] = panel["quote_date"].dt.month
    panel["is_month_end"] = panel["quote_date"].dt.is_month_end.astype(int)
    panel["is_opex"] = _third_friday(panel["quote_date"]).astype(int)

    calendar_dir = data_dir / "calendars"
    fomc_dates = _read_event_dates(calendar_dir / "fomc.csv")
    cpi_dates = _read_event_dates(calendar_dir / "cpi.csv")
    panel["is_fomc"] = panel["quote_date"].dt.normalize().isin(fomc_dates).astype(int)
    panel["is_cpi"] = panel["quote_date"].dt.normalize().isin(cpi_dates).astype(int)

    if cfg.use_leg_spread_cost:
        panel["half_spread_cost"] = _half_spread_cost(panel, opt)
    else:
        panel["half_spread_cost"] = np.nan

    fixed_cost = float(cfg.fixed_cost_bps) * 0.01
    panel["dynamic_slippage_cost"] = _dynamic_slippage_cost(panel, cfg)
    panel["tc"] = panel["half_spread_cost"].fillna(0.0) + fixed_cost + panel["dynamic_slippage_cost"].fillna(0.0)
    panel["ret_gross"] = panel["reth_und"].astype(float)
    panel["ret_net"] = panel["ret_gross"] - panel["tc"]

    feature_meta = [
        FeatureMeta("vix", "volatility_regime", False, "Risk compensation can vary with variance state", "low"),
        FeatureMeta("isk", "skew", False, "Skew dislocations can proxy crash-hedging demand", "low"),
        FeatureMeta("overnight_gap", "gap", False, "Overnight inventory shocks may mean-revert intraday", "low"),
        FeatureMeta("SPX_lret", "intraday_trend", False, "Short-horizon trend/mean-reversion can persist from prior sessions", "low"),
        FeatureMeta("SPX_lrv", "realized_vol", False, "Vol clustering can shift intraday option risk premia", "low"),
        FeatureMeta("slope_dn", "skew", False, "Put-wing steepness reflects downside demand imbalance", "low"),
        FeatureMeta("atr14", "volatility_filter", False, "High ATR regimes often change execution quality and tails", "medium"),
        FeatureMeta("price_vs_ma20", "moving_average", False, "Distance from trend can proxy stretched positioning", "low"),
        FeatureMeta("mom_5d", "momentum", False, "Short-term continuation may alter same-day tail demand", "low"),
        FeatureMeta("reversion_z20", "mean_reversion", False, "Extremes vs rolling mean can revert", "low"),
        FeatureMeta("is_fomc", "macro_event", True, "Event risk reprices intraday vol/skew", "low"),
        FeatureMeta("is_cpi", "macro_event", True, "Inflation release shocks vol-of-vol and skew", "low"),
        FeatureMeta("is_opex", "seasonality", False, "Dealer positioning rolls around OPEX", "low"),
    ]

    return panel, feature_meta
