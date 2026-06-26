# SPX 0DTE Research Platform Plan

## Purpose

This document converts the repository from a pure replication mindset into a production-oriented quantitative research process focused on discovering statistically significant, out-of-sample (OOS), economically plausible SPX 0DTE edges.

The objective is not code style optimization. The objective is robust edge discovery with strict anti-leakage controls, realistic costs, and deployable strategy logic.

---

## 1) Data Inventory and Feature Semantics

### 1.1 Core datasets

#### `data/data_opt.parquet`

- Granularity: option-leg level intraday panel.
- Typical role: raw option/Greek/liquidity surface for leg-level diagnostics and strategy cost reconstruction.
- Key columns:
  - Identifiers/time: `quote_date`, `quote_time`, `option_type`, `mnes`, `mnes_rel`
  - Pricing: `mid`, `intrinsic`, `tv`, `bas`, `payoff`
  - Returns: `reth`, `reth_und`, `sret`
  - Greeks: `implied_volatility`, `delta`, `gamma`, `theta`, `vega`
  - Underlying/liquidity: `active_underlying_price`, `bid_size`, `ask_size`, `trade_volume`, `open_interest`
  - Engineered flow/GEX fields: `trade_volume_delta`, `trade_volume_gamma`, `trade_volume_vega`, `trade_volume_delta_usd`, `trade_volume_gamma_usd`, `trade_volume_vega_usd`, `oi_gamma`, `oi_gamma_abs`, `oi_gamma_usd`, `oi_gamma_abs_usd`
- Inputs vs outputs:
  - Inputs: quotes, sizes, volume, OI, Greeks.
  - Outputs/targets: realized payoff/return columns (`payoff`, `reth`, `reth_und`) when used in backtests.
  - Engineered: intrinsic/time-value decomposition and notional flow/GEX transforms.

#### `data/data_structures.parquet`

- Granularity: strategy-level panel by strategy template and moneyness combination.
- Typical role: primary strategy return target dataset for unconditional and conditional analyses.
- Key columns:
  - `quote_date`, `quote_time`, `option_type`, `mnes`
  - `mid`, `tv`, `payoff`, `reth`, `reth_und`
  - aggregated Greeks: `delta`, `gamma`, `vega`
- Inputs vs outputs:
  - Inputs: strategy template, moneyness, market conditions.
  - Outputs: strategy returns (`reth_und`) and payoff metrics.
  - Engineered: aggregated strategy-level Greeks and payoff/return metrics.

#### `data/vix.parquet`

- Granularity: intraday implied moment snapshots.
- Typical role: ex-ante volatility/skew regime features.
- Key columns:
  - `quote_date`, `quote_time`, `root`, `dte`, `dts`, `expiration`, `minutes_to_mat`
  - implied moment proxies: `vix`, `vixup`, `vixdn`
- Inputs vs outputs:
  - Inputs: option strip state.
  - Outputs/features: implied variance/semivariance/skew proxies.
  - Engineered in analysis: `isk = vixup - vixdn`.

#### `data/slopes.parquet`

- Granularity: intraday volatility surface slope snapshots.
- Typical role: skew/smile conditioning signals.
- Key columns: `quote_date`, `quote_time`, `slope_up`, `slope_dn`.
- Inputs vs outputs:
  - Inputs: implied surface points.
  - Outputs/features: slope factors.

#### `data/future_moments_SPX.parquet`

- Granularity: forward realized SPX moments from anchor time to close.
- Typical role:
  - explanatory regressions (in-sample),
  - lagged predictors for strict OOS conditional models.
- Key columns:
  - IDs: `date`, `time`
  - returns/variance/skew: `SPX_lret`, `SPX_sret`, `SPX_lrv`, `SPX_srv`, `SPX_lrvup`, `SPX_lrvdn`, `SPX_lrv_skew`, `SPX_srvup`, `SPX_srvdn`, `SPX_srv_skew`
  - level: `SPX_close`
- Inputs vs outputs:
  - Inputs: high-frequency SPX path.
  - Outputs/features: realized moment targets and lagged features.

#### `data/future_moments_VIX.parquet`

- Same structure as SPX forward moments, with `VIX_*` columns.
- Typical role: auxiliary realized-moment diagnostics.

#### `data/ALL_eod.csv`

- Granularity: daily OHLCV for multiple roots (`SPX`, `VIX`, etc.).
- Key columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `root`.
- Typical role:
  - trend and context plotting,
  - daily engineered features (gap, ATR, MA, momentum, seasonality markers).

---

### 1.2 Feature families in current pipeline

#### Ex-ante implied features

- `vix`, `vixup`, `vixdn`, `isk`
- `slope_up`, `slope_dn`

#### Lagged realized features (safe for OOS when lagged)

- `SPX_lret`, `SPX_lrv`, `SPX_lrv_skew`

#### Strategy state features

- `mid`, `tv`, aggregated `delta/gamma/vega`
- lagged strategy performance: `pnl_l1`, `pnl_mean5_l1`, `pnl_std5_l1`

#### Liquidity/flow/GEX features (model-zoo extensions)

- liquidity: half-spread/depth/relative spread transforms
- flow: volume and Greek-notional flow proxies
- dealer-gamma proxies: OI gamma net/abs, normalized balances

#### Calendar/seasonality features

- `weekday`, `month`, month-end flags, OPEX proxies, optional event-day flags

### 1.3 Inputs, outputs, engineered summary

- Inputs (predictors): all ex-ante and lagged columns above.
- Outputs (targets):
  - return target: `reth_und` or net return (`pnl_net`)
  - binary target: `1(reth_und > 0)`
- Engineered fields: costs (`tc`), net returns, lagged moments, rolling moments, cross-sectional z-scores, flow/GEX transforms.

---

## 2) Script-by-Script Research Audit

### `code/analysis/option_strats_uncond_analysis.py`

- Hypothesis: unconditional 0DTE payoffs/returns and relation to implied/realized moments.
- Assumptions: moneyness templates are representative; pooled sample summaries are meaningful.
- Stats: descriptive moments, annualized SR, robust/HAC helper estimators, regression tables.
- IS/OOS: in-sample descriptive and explanatory.
- Risks:
  - extensive multiple comparisons,
  - explanatory use of ex-post realized quantities cannot be traded directly.

### `code/analysis/compute_implementable_pnl.py`

- Hypothesis: gross edge survives execution frictions.
- Assumptions: half-spread crossing + fixed 0.5 bp fee/slippage proxy.
- Stats: mean, Sharpe, ES1, turnover proxy, SR/turnover.
- IS/OOS: full-sample diagnostics.
- Risks:
  - simplified execution model (no queue/impact/latency model).

### `code/analysis/compute_tail_risk_diagnostics.py`

- Hypothesis: strategy tails are economically important.
- Assumptions: same net-PNL friction model as implementable table.
- Stats: skew, ES1, max DD, worst day, worst 5-day, loss probability.
- IS/OOS: full sample.

### `code/analysis/compute_clustered_inference_mht.py`

- Hypothesis: relationships persist under clustered inference and multiple testing correction.
- Assumptions: OLS with date clustering, selected specs.
- Stats: coefficients, clustered t-stats, p-values, BH q-values.
- IS/OOS: in-sample inference.
- Risks:
  - inference is explanatory, not direct tradable signal proof.

### `code/analysis/compute_structural_break_2022.py`

- Hypothesis: post-2022 expiry expansion changed expected returns and risk.
- Assumptions: fixed break window around known market-structure change.
- Stats: pre/post means, clustered post dummy effect, volatility ratio.
- IS/OOS: in-sample structural break test.

### `code/analysis/compute_vix_regime_conditioning.py`

- Hypothesis: strategy returns vary materially by VIX regime.
- Assumptions: terciles from sample distribution.
- Stats: full/low/mid/high means, high-minus-low t-test.
- IS/OOS: mostly in-sample regime conditioning.
- Risks:
  - static bins can drift over time.

### `code/analysis/compute_conditional_oos_protocol.py`

- Hypothesis: simple logistic timing signals improve net OOS outcomes.
- Assumptions:
  - rolling/expanding 252-day windows,
  - lag-safe realized features,
  - representative moneyness filter.
- Stats: hit rate, Brier, calibration slope, gross/net mean bps, gross/net SR.
- IS/OOS: strict walk-forward OOS.
- Risks:
  - representative moneyness chosen globally may introduce mild selection leakage.

### `code/analysis/compute_conditional_model_zoo.py`

- Hypothesis: model class, feature set, and scaling choice matter for OOS economics.
- Assumptions:
  - walk-forward retraining,
  - optional cross-sectional/time-series scaling,
  - optional hard/soft mapping.
- Stats:
  - binary: hit rate, Brier, calibration, logloss, AUC, mean net bps, SR net,
  - regression: directional mapping to economic PNL + SR net.
- IS/OOS: OOS walk-forward.
- Risks:
  - model/data-mining bias across large grids without full reality-check corrections.

### `code/analysis/compute_conditional_oos_investment_ts.py`

- Hypothesis: selected conditional sleeves and baskets produce persistent net cumulative returns.
- Assumptions: equal-weight basketing, top-k construction by chosen metric.
- Stats: cumulative PNL, mean/SR/hit/DD/ES summaries.
- IS/OOS: based on OOS signals.
- Risks:
  - potential ex-post basket selection if ranking uses full horizon.

### Supporting scripts

- `build_conditional_target_choice_table.py`: compares target formulations.
- `derive_binary_decision_summary.py`: remaps stored probabilities (hard/soft) without refit.
- `moneyness_selection.py`: representative strategy/moneyness key selection.
- `plot_conditional_topk_basket_legs.py`: payoff decomposition of top-k sleeves.
- `figs_strats.py`: static payoff profile and figure generation.

---

## 3) Dependency and Data-Flow Map

```text
Raw/derived data
  data_opt.parquet
  data_structures.parquet
  vix.parquet
  slopes.parquet
  future_moments_SPX.parquet
  future_moments_VIX.parquet
  ALL_eod.csv
      |
      +--> Unconditional stack
      |      option_strats_uncond_analysis.py
      |      compute_implementable_pnl.py
      |      compute_tail_risk_diagnostics.py
      |      compute_clustered_inference_mht.py
      |      compute_structural_break_2022.py
      |      compute_vix_regime_conditioning.py
      |
      +--> Conditional OOS stack
      |      compute_conditional_oos_protocol.py
      |      compute_conditional_model_zoo.py
      |      build_conditional_target_choice_table.py
      |      derive_binary_decision_summary.py
      |      compute_conditional_oos_investment_ts.py
      |      plot_conditional_topk_basket_legs.py
      |
      +--> Figures
             figs_strats.py

Outputs
  output/tables/*.tex
  output/figures/*.pdf
  data/conditional_* summaries
```

---

## 4) Proposed Feature Expansion for Edge Search

The table below emphasizes economic rationale, data availability, external dependency, and computational cost.

| Feature family | Why edge may exist | Existing data? | External data needed? | Compute cost |
|---|---|---|---|---|
| Volatility regime filters (`vix`, lagged `SPX_lrv`) | Risk premia and hedging demand are state-dependent | Yes | No | Low |
| Overnight gap filters | Overnight inventory shocks can mean-revert or trend intraday | Yes (`ALL_eod.csv`) | No | Low |
| Intraday trend filters (open to 10:00) | Opening flow imbalance can continue or reverse into close | Partially | Yes (intraday SPX bars) | Medium |
| Realized volatility filters | Vol clustering changes payoff convexity and spread behavior | Yes | No | Low |
| VIX level filters | Convexity demand and fear regime effects | Yes | No | Low |
| Skew filters (`isk`, `slope_dn`) | Crash-hedging demand over/under-prices downside wings | Yes | No | Low |
| Time-of-day effects | Microstructure and liquidity vary strongly intraday | Partially | Maybe (if more bars required) | Low-Med |
| Weekday effects | Dealer flow and expiry mechanics vary by weekday | Yes | No | Low |
| Monthly seasonality | Rebalance and hedging cycles around month boundaries | Yes | No | Low |
| Macro event days (generic) | Scheduled risk events reprice vol/skew | No (event calendar absent by default) | Yes | Low |
| FOMC | Policy uncertainty shocks intraday vol and skew | No (optional) | Yes | Low |
| CPI | Data-release shocks in rates/equity vol | No (optional) | Yes | Low |
| OPEX | Dealer inventory rolls and pinning effects | Yes (date derived) | No | Low |
| End-of-month | Institutional flow and vol supply/demand shifts | Yes | No | Low |
| Dealer gamma proxies (`oi_gamma*`) | Hedging reflexivity can amplify trend/reversion | Yes | No | Medium |
| ATR filters | Range state affects slippage and tail payoff quality | Yes (`ALL_eod.csv`) | No | Low |
| Moving averages | Medium-short trend state classification | Yes | No | Low |
| Momentum | Trend continuation in high flow regimes | Yes | No | Low |
| Mean reversion z-score | Over-extension often mean reverts | Yes | No | Low |

---

## 5) Research Framework Specification (Automated)

### 5.1 Required capabilities

The research framework must automatically:

1. generate hypotheses,
2. backtest hypotheses,
3. perform walk-forward validation,
4. avoid look-ahead bias,
5. include transaction costs,
6. include slippage,
7. measure expectancy, Sharpe, Sortino, max drawdown, CVaR, win rate, profit factor, skew, kurtosis, tail risk.

### 5.2 Implemented research layer

A dedicated non-replication package has been added under `research/`:

- `research/data.py`
  - builds point-in-time feature panel,
  - lags realized moments,
  - computes net returns with spread + fixed/slippage costs,
  - adds calendar and optional macro-event features.

- `research/hypothesis.py`
  - generates rule templates per feature (quartile-trigger momentum/reversion direction rules).

- `research/engine.py`
  - walk-forward train/OOS blocks,
  - hypothesis selection on train only,
  - OOS execution on subsequent block,
  - metric computation,
  - ranking by requested multi-criteria priority.

- `research/metrics.py`
  - expectancy, Sharpe, Sortino, max DD, CVaR, win rate, profit factor, skew, kurtosis, ES1, t-stat.

- `research/run_research.py`
  - CLI runner,
  - experiment artifact output.

- `research/experiments/`
  - timestamped artifacts for reproducibility and audit trail.

### 5.3 Anti-leakage controls

- Train/test split by time only.
- Feature thresholds estimated only on train windows.
- Lagged realized features only (no same-day realized leakage).
- Strategy selection re-evaluated at each rebalance date using train data only.

### 5.4 Cost model

- Net return includes:
  - half-spread estimate from option legs where available,
  - fixed cost in bps,
  - slippage in bps.

---

## 6) Ranking Framework for Discovered Strategies

Final ranking priority is:

1. OOS Sharpe,
2. robustness,
3. stability across years,
4. statistical significance,
5. implementation simplicity.

Current definitions in the framework:

- OOS Sharpe: annualized Sharpe of OOS strategy return series.
- Robustness: share of positive yearly mean returns.
- Stability: inverse transform of yearly return dispersion.
- Significance: simple t-stat on OOS mean return.
- Simplicity: penalty for signals requiring external dependencies.

---

## 7) Live-Trading Realism Rules

To keep strategies realistically tradable:

- No optimization on final test period.
- No curve fitting to isolated episodes.
- Every strategy must include an economic thesis, not just statistical fit.
- Capacity and execution quality must be considered (spread, turnover, gap/tail behavior).
- Avoid highly brittle feature conjunctions with very low event counts.

Recommended next additions:

- deflated Sharpe or White reality-check style corrections,
- block bootstrap confidence intervals,
- nested model selection for hyper-parameter/protocol decisions,
- turnover and max-tail-risk constraints in the selection objective.

---

## 8) Experiment Documentation Standard

Each run should persist:

- feature set and parameter config,
- train/test window definition,
- selected hypothesis per rebalance point,
- OOS per-trade/per-day returns,
- summary and ranked leaderboard,
- notes on economic rationale.

Artifacts produced by the framework:

- `signals_*.parquet`
- `selections_*.csv`
- `summary_*.csv`
- `ranked_*.csv`
- `feature_meta_*.csv`

under `research/experiments/`.

---

## 9) How to Run

From repository root:

```bash
c:/Projects/0dte-strategies/.venv/Scripts/python.exe research/run_research.py
```

Optional macro calendar support:

- `data/calendars/fomc.csv`
- `data/calendars/cpi.csv`

Each file should have a date column (default name: `date`).

---

## 10) Practical Interpretation

A discovered edge is actionable only if all of the following hold:

- positive OOS net expectancy after costs and slippage,
- acceptable OOS Sharpe and drawdown profile,
- consistent performance across years/regimes,
- statistical support after multiple-hypothesis safeguards,
- clear economic mechanism (inventory, convexity demand, dealer hedging, event-risk repricing, or structural flow effects).

This is the guiding standard for moving from replication to deployable SPX 0DTE research.
