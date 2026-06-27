from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import build_research_panel
from .hypothesis import HypothesisTemplate, generate_templates, signal_from_template
from .metrics import adjusted_t_stat_for_mining, bootstrap_pvalue_mean, summarize_returns


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
    val_sharpe: float
    val_expectancy: float
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


def _split_train_validation(train: pd.DataFrame, validation_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train.empty:
        return train.copy(), train.copy()
    frac = min(max(float(validation_frac), 0.05), 0.5)
    cut = int(np.floor(len(train) * (1.0 - frac)))
    cut = max(1, min(cut, len(train) - 1))
    return train.iloc[:cut].copy(), train.iloc[cut:].copy()


def _apply_risk_overlays(test_df: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    if test_df.empty:
        return test_df

    out = test_df.sort_values("quote_date").reset_index(drop=True).copy()

    vix = pd.to_numeric(out.get("vix", np.nan), errors="coerce").fillna(0.0)
    hard_block = vix > float(cfg.risk_vix_hard_cap)

    event_flag = (
        pd.to_numeric(out.get("is_fomc", 0), errors="coerce").fillna(0).astype(int)
        | pd.to_numeric(out.get("is_cpi", 0), errors="coerce").fillna(0).astype(int)
    )

    scaled_signal = out["signal"].astype(float).to_numpy(copy=True)
    scaled_signal[event_flag.to_numpy() > 0] *= float(cfg.risk_event_scale)
    scaled_signal[hard_block.to_numpy()] = 0.0

    cooldown_remaining = 0
    final_signal = np.zeros_like(scaled_signal, dtype=float)
    kill_trigger = np.zeros_like(scaled_signal, dtype=int)

    for i in range(len(scaled_signal)):
        if cooldown_remaining > 0:
            final_signal[i] = 0.0
            cooldown_remaining -= 1
            continue

        final_signal[i] = scaled_signal[i]
        pnl_candidate = final_signal[i] * float(out.loc[i, "ret_net"])
        if pnl_candidate <= -float(cfg.risk_daily_loss_limit):
            kill_trigger[i] = 1
            cooldown_remaining = max(0, int(cfg.risk_cooldown_days))

    prev_signal = np.roll(final_signal, 1)
    prev_signal[0] = 0.0
    turnover = np.abs(final_signal - prev_signal)
    turnover_cost = turnover * float(cfg.turnover_cost_bps) * 0.01

    out["signal_raw"] = out["signal"].astype(float)
    out["signal"] = final_signal
    out["turnover"] = turnover
    out["turnover_cost"] = turnover_cost
    out["kill_switch_trigger"] = kill_trigger
    out["strategy_ret"] = out["signal"] * out["ret_net"] - out["turnover_cost"]
    return out


def _rank_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["rank_oos_sharpe"] = out["sharpe"].rank(ascending=False, method="min")
    out["rank_robustness"] = out["robustness_score"].rank(ascending=False, method="min")
    out["rank_stability"] = out["stability_score"].rank(ascending=False, method="min")
    out["rank_significance"] = out["adj_t_stat"].rank(ascending=False, method="min")
    out["rank_pvalue"] = out["bootstrap_pvalue"].rank(ascending=True, method="min")
    out["rank_simplicity"] = out["implementation_simplicity"].rank(ascending=False, method="min")
    out = out.sort_values(
        [
            "rank_oos_sharpe",
            "rank_robustness",
            "rank_stability",
            "rank_significance",
            "rank_pvalue",
            "rank_simplicity",
        ],
        ascending=True,
    ).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out


def _build_candidate_report(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    out["confidence_score"] = (
        out["sharpe"].fillna(0.0).clip(lower=-5, upper=5)
        + out["robustness_score"].fillna(0.0)
        + out["stability_score"].fillna(0.0)
        + out["implementation_simplicity"].fillna(0.0)
        + (1.0 - out["bootstrap_pvalue"].fillna(1.0))
    )
    out["risk_flag"] = np.where(
        (out["max_drawdown"].fillna(0.0) < -0.15) | (out["tail_risk_es1"].fillna(0.0) < -0.03),
        "tail-risk",
        "ok",
    )
    return out.sort_values(["confidence_score", "sharpe"], ascending=False).reset_index(drop=True)


def _build_live_paper_outputs(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel.empty or selections.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    panel = panel.copy()
    panel["quote_date"] = pd.to_datetime(panel["quote_date"])
    latest_date = panel["quote_date"].max()

    selections = selections.copy()
    selections["rebalance_date"] = pd.to_datetime(selections["rebalance_date"])
    latest_sel = selections.sort_values("rebalance_date").groupby("strategy", observed=True).tail(1)

    rows = []
    for sel in latest_sel.itertuples(index=False):
        today = panel[(panel["option_type"] == sel.strategy) & (panel["quote_date"] == latest_date)].copy()
        if today.empty:
            continue
        t = HypothesisTemplate(
            feature=str(sel.feature),
            rule=str(sel.rule),
            direction=int(sel.direction),
            description=f"Live signal from selected rule {sel.feature}/{sel.rule}",
        )
        sig = signal_from_template(today, t, q25=float(sel.q25), q75=float(sel.q75))
        alloc = float(cfg.live_notional) / max(1, len(today))
        block = today[["quote_date", "option_type", "mnes", "tc", "vix", "is_fomc", "is_cpi"]].copy()
        block["signal"] = sig
        block["alloc_notional"] = alloc
        block["hypothesis_feature"] = str(sel.feature)
        block["hypothesis_rule"] = str(sel.rule)
        block["hypothesis_direction"] = int(sel.direction)
        rows.append(block)

    daily_signal = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    trade_log_template = pd.DataFrame(
        columns=[
            "timestamp_utc",
            "trade_date",
            "option_type",
            "mnes",
            "signal",
            "qty",
            "entry_price",
            "exit_price",
            "fees",
            "slippage",
            "status",
            "pnl",
        ]
    )

    mon_rows = []
    if not signals.empty:
        work = signals.copy()
        work["quote_date"] = pd.to_datetime(work["quote_date"])
        for strategy, sdf in work.groupby("option_type", observed=True):
            sdf = sdf.sort_values("quote_date")
            roll = sdf.tail(30)
            m = summarize_returns(roll["strategy_ret"])
            mon_rows.append(
                {
                    "asof_date": latest_date,
                    "option_type": strategy,
                    "rolling30_sharpe": m["sharpe"],
                    "rolling30_max_drawdown": m["max_drawdown"],
                    "rolling30_win_rate": m["win_rate"],
                    "rolling30_trades": int((roll["signal"] != 0.0).sum()) if "signal" in roll else int(roll.shape[0]),
                }
            )
    monitor = pd.DataFrame(mon_rows)
    return daily_signal, trade_log_template, monitor


def _build_profitability_outputs(
    summary: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty or signals.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    eligible = summary[
        (summary["sharpe"].fillna(-np.inf) >= float(cfg.profit_min_sharpe))
        & (summary["bootstrap_pvalue"].fillna(1.0) <= float(cfg.profit_max_bootstrap_pvalue))
    ].copy()
    if eligible.empty:
        eligible = summary.copy()

    eligible = eligible.sort_values(["sharpe", "expectancy"], ascending=False).head(int(cfg.profit_top_k))
    selected = eligible["option_type"].astype(str).tolist()
    if not selected:
        empty = pd.DataFrame()
        return empty, empty, empty

    work = signals.copy()
    work["quote_date"] = pd.to_datetime(work["quote_date"])
    daily = (
        work[work["option_type"].isin(selected)]
        .groupby(["quote_date", "option_type"], observed=True, as_index=False)["strategy_ret"]
        .mean()
    )
    pivot = daily.pivot(index="quote_date", columns="option_type", values="strategy_ret").sort_index().fillna(0.0)

    vols = pivot.std(ddof=1).replace(0.0, np.nan)
    score_map = eligible.set_index("option_type")
    raw = {}
    for s in selected:
        sharpe = float(score_map.loc[s, "sharpe"]) if s in score_map.index else 0.0
        pval = float(score_map.loc[s, "bootstrap_pvalue"]) if s in score_map.index else 1.0
        quality = max(0.0, sharpe) * max(0.0, 1.0 - pval)
        vol = float(vols.get(s, np.nan))
        inv_vol = 0.0 if (not np.isfinite(vol) or vol <= 1e-8) else 1.0 / vol
        raw[s] = quality * inv_vol

    raw_s = pd.Series(raw, dtype=float)
    if float(raw_s.sum()) <= 1e-12:
        raw_s = pd.Series({s: 1.0 for s in selected}, dtype=float)

    w = raw_s / raw_s.sum()
    w = w.clip(upper=float(cfg.profit_max_weight))
    if float(w.sum()) <= 1e-12:
        w = pd.Series({s: 1.0 / len(selected) for s in selected}, dtype=float)
    else:
        w = w / w.sum()

    port = pivot[selected].dot(w.reindex(selected).fillna(0.0))
    realized_vol = float(port.std(ddof=1)) if port.shape[0] > 1 else np.nan
    if np.isfinite(realized_vol) and realized_vol > 1e-8:
        vol_scalar = min(2.0, float(cfg.profit_target_daily_vol) / realized_vol)
    else:
        vol_scalar = 1.0

    port_scaled = port * vol_scalar
    m = summarize_returns(port_scaled)

    daily_out = pd.DataFrame(
        {
            "quote_date": port_scaled.index,
            "portfolio_ret": port_scaled.values,
            "cum_ret": port_scaled.cumsum().values,
        }
    )

    weights_out = pd.DataFrame(
        {
            "option_type": selected,
            "base_weight": [float(w.get(s, 0.0)) for s in selected],
            "effective_weight": [float(w.get(s, 0.0) * vol_scalar) for s in selected],
            "sharpe": [float(score_map.loc[s, "sharpe"]) if s in score_map.index else np.nan for s in selected],
            "bootstrap_pvalue": [
                float(score_map.loc[s, "bootstrap_pvalue"]) if s in score_map.index else np.nan for s in selected
            ],
        }
    )

    summary_out = pd.DataFrame(
        [
            {
                "selected_count": int(len(selected)),
                "selected_strategies": "|".join(selected),
                "vol_target_daily": float(cfg.profit_target_daily_vol),
                "vol_scalar": float(vol_scalar),
                "portfolio_expectancy": float(m["expectancy"]),
                "portfolio_sharpe": float(m["sharpe"]),
                "portfolio_sortino": float(m["sortino"]),
                "portfolio_max_drawdown": float(m["max_drawdown"]),
                "portfolio_cvar_5": float(m["cvar_5"]),
                "portfolio_tail_risk_es1": float(m["tail_risk_es1"]),
                "portfolio_t_stat": float(m["t_stat"]),
                "portfolio_n_obs": int(m["n_obs"]),
                "portfolio_total_return": float(port_scaled.sum()),
            }
        ]
    )

    return summary_out, weights_out, daily_out


def run_discovery(cfg: ResearchConfig, candidate_features: list[str]) -> dict[str, pd.DataFrame]:
    panel, feature_meta = build_research_panel(cfg)
    available = [f for f in candidate_features if f in panel.columns]
    templates = generate_templates(available)

    all_signals: list[pd.DataFrame] = []
    selections: list[SelectionRecord] = []

    external_features = {m.name for m in feature_meta if m.needs_external_data}
    n_total_tests = max(1, len(templates))

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

            tr_core, tr_val = _split_train_validation(train, cfg.validation_frac)
            if tr_core.empty or tr_val.empty:
                continue

            best = None
            best_score = (-np.inf, -np.inf)

            for t in templates:
                x_train = pd.to_numeric(tr_core[t.feature], errors="coerce")
                if x_train.notna().sum() < cfg.min_trades_per_hypothesis:
                    continue

                q25 = float(x_train.quantile(0.25))
                q75 = float(x_train.quantile(0.75))
                sig_tr_core = signal_from_template(tr_core, t, q25=q25, q75=q75)
                active = sig_tr_core != 0.0
                if int(active.sum()) < cfg.min_trades_per_hypothesis:
                    continue

                tr_ret = pd.Series(sig_tr_core * tr_core["ret_net"].to_numpy(dtype=float), index=tr_core.index)
                sharpe, _ = _train_score(tr_ret)
                if np.isnan(sharpe):
                    continue

                sig_val = signal_from_template(tr_val, t, q25=q25, q75=q75)
                val_ret = pd.Series(sig_val * tr_val["ret_net"].to_numpy(dtype=float), index=tr_val.index)
                val_sharpe, val_expct = _train_score(val_ret)
                if np.isnan(val_sharpe):
                    continue

                score = (val_sharpe, val_expct)
                if score > best_score:
                    best_score = score
                    best = (t, q25, q75, tr_ret, val_ret)

            if best is None:
                continue

            t, q25, q75, tr_ret, val_ret = best
            sig_te = signal_from_template(test, t, q25=q25, q75=q75)
            te_out = test[
                [
                    "quote_date",
                    "option_type",
                    "mnes",
                    "ret_gross",
                    "ret_net",
                    "vix",
                    "is_fomc",
                    "is_cpi",
                    "tc",
                ]
            ].copy()
            te_out["signal"] = sig_te
            te_out["template_feature"] = t.feature
            te_out["template_rule"] = t.rule
            te_out["template_direction"] = t.direction
            te_out = _apply_risk_overlays(te_out, cfg)
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
                    val_sharpe=float(best_score[0]),
                    val_expectancy=float(best_score[1]),
                    train_sharpe=float(_train_score(tr_ret)[0]),
                    train_expectancy=float(tr_ret.mean()),
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
        boot_p = bootstrap_pvalue_mean(
            sdf["strategy_ret"],
            n_bootstrap=cfg.bootstrap_samples,
            block_size=cfg.bootstrap_block_size,
            seed=cfg.random_seed,
        )
        adj_t = adjusted_t_stat_for_mining(float(m.get("t_stat", np.nan)), n_total_tests)
        turnover_avg = float(pd.to_numeric(sdf.get("turnover", 0.0), errors="coerce").fillna(0.0).mean())
        active_share = float((pd.to_numeric(sdf.get("signal", 0.0), errors="coerce").fillna(0.0) != 0.0).mean())
        avg_tc = float(pd.to_numeric(sdf.get("tc", 0.0), errors="coerce").fillna(0.0).mean())
        capacity_proxy = active_share / max(1e-8, avg_tc)

        summary_rows.append(
            {
                "option_type": str(strategy),
                **m,
                "robustness_score": float(pos_year_share),
                "stability_score": float(stability),
                "implementation_simplicity": float(simplicity),
                "bootstrap_pvalue": float(boot_p),
                "adj_t_stat": float(adj_t),
                "avg_turnover": turnover_avg,
                "active_share": active_share,
                "avg_tc": avg_tc,
                "capacity_proxy": capacity_proxy,
            }
        )

    summary = pd.DataFrame(summary_rows)
    ranked = _rank_strategies(summary)
    candidate_report = _build_candidate_report(summary)

    feature_df = pd.DataFrame([m.__dict__ for m in feature_meta])
    live_signal, trade_log_template, monitor = _build_live_paper_outputs(panel, sel_df, signals, cfg)
    profitability_summary, profitability_weights, profitability_daily = _build_profitability_outputs(summary, signals, cfg)
    return {
        "panel": panel,
        "signals": signals,
        "selections": sel_df,
        "summary": summary,
        "ranked": ranked,
        "candidate_report": candidate_report,
        "feature_meta": feature_df,
        "live_signal": live_signal,
        "trade_log_template": trade_log_template,
        "monitor": monitor,
        "profitability_summary": profitability_summary,
        "profitability_weights": profitability_weights,
        "profitability_daily": profitability_daily,
    }


def save_experiment(results: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    paths = {
        "signals": output_dir / f"signals_{stamp}.parquet",
        "selections": output_dir / f"selections_{stamp}.csv",
        "summary": output_dir / f"summary_{stamp}.csv",
        "ranked": output_dir / f"ranked_{stamp}.csv",
        "candidate_report": output_dir / f"candidate_report_{stamp}.csv",
        "feature_meta": output_dir / f"feature_meta_{stamp}.csv",
        "live_signal": output_dir / f"live_signal_{stamp}.csv",
        "trade_log_template": output_dir / f"trade_log_template_{stamp}.csv",
        "monitor": output_dir / f"monitor_{stamp}.csv",
        "profitability_summary": output_dir / f"profitability_summary_{stamp}.csv",
        "profitability_weights": output_dir / f"profitability_weights_{stamp}.csv",
        "profitability_daily": output_dir / f"profitability_daily_{stamp}.csv",
    }

    results["signals"].to_parquet(paths["signals"], index=False)
    results["selections"].to_csv(paths["selections"], index=False)
    results["summary"].to_csv(paths["summary"], index=False)
    results["ranked"].to_csv(paths["ranked"], index=False)
    results["candidate_report"].to_csv(paths["candidate_report"], index=False)
    results["feature_meta"].to_csv(paths["feature_meta"], index=False)
    results["live_signal"].to_csv(paths["live_signal"], index=False)
    results["trade_log_template"].to_csv(paths["trade_log_template"], index=False)
    results["monitor"].to_csv(paths["monitor"], index=False)
    results["profitability_summary"].to_csv(paths["profitability_summary"], index=False)
    results["profitability_weights"].to_csv(paths["profitability_weights"], index=False)
    results["profitability_daily"].to_csv(paths["profitability_daily"], index=False)
    return paths
