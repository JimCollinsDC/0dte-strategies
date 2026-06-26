#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import ResearchConfig
from research.engine import run_discovery, save_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OOS hypothesis discovery for SPX 0DTE strategies.")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--entry-time", type=str, default="10:00:00")
    p.add_argument("--min-train-days", type=int, default=252)
    p.add_argument("--rolling-window-days", type=int, default=252)
    p.add_argument("--rebalance-days", type=int, default=21)
    p.add_argument("--fixed-cost-bps", type=float, default=0.5)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--no-leg-spread-cost", action="store_true", default=False)
    p.add_argument("--max-moneyness-dev", type=float, default=0.015)
    p.add_argument("--min-trades", type=int, default=40)
    p.add_argument(
        "--features",
        nargs="+",
        default=[
            "vix",
            "isk",
            "slope_up",
            "slope_dn",
            "overnight_gap",
            "SPX_lret",
            "SPX_lrv",
            "SPX_lrv_skew",
            "atr14",
            "price_vs_ma20",
            "mom_5d",
            "reversion_z20",
            "pnl_l1",
            "pnl_mean5_l1",
            "pnl_std5_l1",
            "weekday",
            "month",
            "is_month_end",
            "is_opex",
            "is_fomc",
            "is_cpi",
        ],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = ResearchConfig(
        project_root=args.project_root.resolve(),
        entry_time=str(args.entry_time),
        min_train_days=int(args.min_train_days),
        rolling_window_days=int(args.rolling_window_days),
        rebalance_days=int(args.rebalance_days),
        fixed_cost_bps=float(args.fixed_cost_bps),
        slippage_bps=float(args.slippage_bps),
        use_leg_spread_cost=not bool(args.no_leg_spread_cost),
        min_trades_per_hypothesis=int(args.min_trades),
        max_moneyness_dev=float(args.max_moneyness_dev),
    )

    results = run_discovery(cfg=cfg, candidate_features=list(args.features))
    paths = save_experiment(results=results, output_dir=cfg.output_dir)

    ranked = results["ranked"]
    print("Top discovered strategies (OOS):")
    print(ranked.head(10).to_string(index=False))
    print("\nArtifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
