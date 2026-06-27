# Research Framework

This directory is a separate, non-replication research layer focused on robust out-of-sample edge discovery.

## What It Does

- Builds a point-in-time feature panel from shipped data files.
- Auto-generates rule hypotheses from candidate features.
- Uses nested walk-forward selection: choose on train-core and validate on holdout, then apply only to next OOS block.
- Applies explicit transaction costs and dynamic slippage (volatility, spread, and macro-event sensitive).
- Computes bootstrap significance and data-mining-adjusted test statistics.
- Applies risk overlays (event scaling, VIX hard-cap block, daily loss kill-switch + cooldown, turnover costs).
- Computes risk/return diagnostics:
  - expectancy
  - sharpe
  - sortino
  - max drawdown
  - CVaR
  - win rate
  - profit factor
  - skew
  - kurtosis
  - tail risk (ES 1%)
- Ranks strategies using:
  1. OOS Sharpe
  2. robustness (positive-year share)
  3. stability (yearly dispersion transform)
  4. significance (adjusted t-stat + bootstrap p-value)
  5. implementation simplicity

## New Live-Paper Outputs

Each run also creates:

- `candidate_report_*.csv` (confidence score + risk flags + capacity proxy)
- `live_signal_*.csv` (latest-date model signals for paper/live wiring)
- `trade_log_template_*.csv` (append-only execution log schema)
- `monitor_*.csv` (rolling strategy health metrics)

## Run

```bash
c:/Projects/0dte-strategies/.venv/Scripts/python.exe research/run_research.py
```

## Optional Event Calendars

Place optional external macro calendars under `data/calendars/`:

- `fomc.csv`
- `cpi.csv`

Each file should include a date column (default column name: `date`).

## Experiment Artifacts

Each run writes timestamped files to `research/experiments/`:

- `signals_*.parquet`
- `selections_*.csv`
- `summary_*.csv`
- `ranked_*.csv`
- `candidate_report_*.csv`
- `feature_meta_*.csv`
- `live_signal_*.csv`
- `trade_log_template_*.csv`
- `monitor_*.csv`
