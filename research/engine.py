from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import FeatureMeta, build_research_panel
from .hypothesis import HypothesisTemplate, generate_templates, signal_from_template
from .metrics import summarize_returns


@dataclass
class SelectionRecord:
    strategy: str
    rebalance_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    feature: str
    rule: str
    direction: int
    q25: float
    q75: float
    train_sharpe: float
    train_expectancy: float
    train_obs: int


def _train_score(ret: pd.Series) -> tuple[float, float]:
    m = summarize_returns(ret)
    return float(m["sharpe"]), float(m["expectancy"])


def _simplicity_score(feature: str, external_features: set[str]) -> float:
    score = 1.0
    if feature in external_features:
        score -= 0.2
    return score


def _yearly_stability(signal_df: pd.DataFrame) -> tuple[float, float]:
    if signal_df.empty:
        return np.nan, np.nan
    work = signal_df.copy()
    work["year"] = pd.to_datetime(work["quote_date"]).dt.year
    yearly = work.groupby("year", observed=True)["strategy_ret"].mean()
    positive_share = float((yearly > 0).mean()) if len(yearly) > 0 else np.nan
    stability = float(1.0 / (1.0 + yearly.std(ddof=1))) if len(yearly) > 1 else np.nan
    return positive_share, stability


def _rank_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["rank_oos_sharpe"] = out["sharpe"].rank(ascending=False, method="min")
    out["rank_robustness"] = out["robustness_score"].rank(ascending=False, method="min")
    out["rank_stability"] = out["stability_score"].rank(ascending=False, method="min")
    out["rank_significance"] = out["t_stat"].rank(ascending=False, method="min")
    out["rank_simplicity"] = out["implementation_simplicity"].rank(ascending=False, method="min")
    out = out.sort_values(
        [
            "rank_oos_sharpe",
            "rank_robustness",
            "rank_stability",
            "rank_significance",
            "rank_simplicity",
        ],
        ascending=True,
    ).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out


def run_discovery(cfg: ResearchConfig, candidate_features: list[str]) -> dict[str, pd.DataFrame]:
    panel, feature_meta = build_research_panel(cfg)
    available = [f for f in candidate_features if f in panel.columns]
    templates = generate_templates(available)

    all_signals: list[pd.DataFrame] = []
    selections: list[SelectionRecord] = []

    external_features = {m.name for m in feature_meta if m.needs_external_data}

    for strategy, sdf in panel.groupby("option_type", observed=True):
        sdf = sdf.sort_values("quote_date").reset_index(drop=True)
        dates = np.array(sorted(sdf["quote_date"].unique()))
        if len(dates) <= cfg.min_train_days:
            continue

        for i in range(cfg.min_train_days, len(dates), cfg.rebalance_days):
            train_start_idx = max(0, i - cfg.rolling_window_days)
            tr_start = dates[train_start_idx]
            tr_end = dates[i - 1]
            te_end_idx = min(i + cfg.rebalance_days, len(dates)) - 1
            te_start = dates[i]
            te_end = dates[te_end_idx]

            train = sdf[(sdf["quote_date"] >= tr_start) & (sdf["quote_date"] <= tr_end)].copy()
            test = sdf[(sdf["quote_date"] >= te_start) & (sdf["quote_date"] <= te_end)].copy()
            if train.empty or test.empty:
                continue

            best = None
            best_score = (-np.inf, -np.inf)

            for t in templates:
                x_train = pd.to_numeric(train[t.feature], errors="coerce")
                if x_train.notna().sum() < cfg.min_trades_per_hypothesis:
                    continue

                q25 = float(x_train.quantile(0.25))
                q75 = float(x_train.quantile(0.75))
                sig_tr = signal_from_template(train, t, q25=q25, q75=q75)
                active = sig_tr != 0.0
                if int(active.sum()) < cfg.min_trades_per_hypothesis:
                    continue

                tr_ret = pd.Series(sig_tr * train["ret_net"].to_numpy(dtype=float), index=train.index)
                sharpe, expct = _train_score(tr_ret)
                if np.isnan(sharpe):
                    continue
                score = (sharpe, expct)
                if score > best_score:
                    best_score = score
                    best = (t, q25, q75, tr_ret)

            if best is None:
                continue

            t, q25, q75, tr_ret = best
            sig_te = signal_from_template(test, t, q25=q25, q75=q75)
            te_out = test[["quote_date", "option_type", "mnes", "ret_gross", "ret_net"]].copy()
            te_out["signal"] = sig_te
            te_out["strategy_ret"] = te_out["signal"] * te_out["ret_net"]
            te_out["template_feature"] = t.feature
            te_out["template_rule"] = t.rule
            te_out["template_direction"] = t.direction
            all_signals.append(te_out)

            selections.append(
                SelectionRecord(
                    strategy=str(strategy),
                    rebalance_date=pd.Timestamp(te_start),
                    train_start=pd.Timestamp(tr_start),
                    train_end=pd.Timestamp(tr_end),
                    feature=t.feature,
                    rule=t.rule,
                    direction=t.direction,
                    q25=q25,
                    q75=q75,
                    train_sharpe=float(best_score[0]),
                    train_expectancy=float(best_score[1]),
                    train_obs=int((tr_ret != 0.0).sum()),
                )
            )

    if not all_signals:
        raise RuntimeError("No OOS signals produced. Relax min trade thresholds or window settings.")

    signals = pd.concat(all_signals, axis=0, ignore_index=True)
    sel_df = pd.DataFrame([s.__dict__ for s in selections])

    summary_rows = []
    for strategy, sdf in signals.groupby("option_type", observed=True):
        m = summarize_returns(sdf["strategy_ret"])
        pos_year_share, stability = _yearly_stability(sdf)
        most_common_feature = (
            sdf["template_feature"].mode().iloc[0] if "template_feature" in sdf and not sdf["template_feature"].mode().empty else ""
        )
        simplicity = _simplicity_score(str(most_common_feature), external_features)

        summary_rows.append(
            {
                "option_type": str(strategy),
                **m,
                "robustness_score": float(pos_year_share),
                "stability_score": float(stability),
                "implementation_simplicity": float(simplicity),
            }
        )

    summary = pd.DataFrame(summary_rows)
    ranked = _rank_strategies(summary)

    feature_df = pd.DataFrame([m.__dict__ for m in feature_meta])
    return {
        "panel": panel,
        "signals": signals,
        "selections": sel_df,
        "summary": summary,
        "ranked": ranked,
        "feature_meta": feature_df,
    }


def save_experiment(results: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    paths = {
        "signals": output_dir / f"signals_{stamp}.parquet",
        "selections": output_dir / f"selections_{stamp}.csv",
        "summary": output_dir / f"summary_{stamp}.csv",
        "ranked": output_dir / f"ranked_{stamp}.csv",
        "feature_meta": output_dir / f"feature_meta_{stamp}.csv",
    }

    results["signals"].to_parquet(paths["signals"], index=False)
    results["selections"].to_csv(paths["selections"], index=False)
    results["summary"].to_csv(paths["summary"], index=False)
    results["ranked"].to_csv(paths["ranked"], index=False)
    results["feature_meta"].to_csv(paths["feature_meta"], index=False)
    return paths
