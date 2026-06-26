# Research Framework

This directory is a separate, non-replication research layer focused on robust out-of-sample edge discovery.

## What It Does

- Builds a point-in-time feature panel from shipped data files.
- Auto-generates rule hypotheses from candidate features.
- Uses walk-forward selection: choose best hypothesis on train windows, apply only to next OOS block.
- Applies explicit transaction costs and slippage.
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
  4. significance (t-stat)
  5. implementation simplicity

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
- `feature_meta_*.csv`
