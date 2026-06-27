from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    project_root: Path
    entry_time: str = "10:00:00"
    min_train_days: int = 252
    rolling_window_days: int = 252
    rebalance_days: int = 21
    fixed_cost_bps: float = 0.5
    slippage_bps: float = 0.5
    use_leg_spread_cost: bool = True
    max_hypotheses_per_feature: int = 4
    min_trades_per_hypothesis: int = 40
    max_moneyness_dev: float = 0.015
    validation_frac: float = 0.2
    bootstrap_samples: int = 400
    bootstrap_block_size: int = 5
    random_seed: int = 42

    # Dynamic execution model
    slippage_vix_ref: float = 20.0
    slippage_vix_beta: float = 1.0
    slippage_spread_beta: float = 1.0
    slippage_event_mult: float = 1.5

    # Risk overlays
    risk_daily_loss_limit: float = 0.03
    risk_cooldown_days: int = 2
    risk_vix_hard_cap: float = 45.0
    risk_event_scale: float = 0.5
    turnover_cost_bps: float = 0.5

    # Live-paper outputs
    live_notional: float = 100_000.0

    # Profitability layer
    profit_top_k: int = 3
    profit_min_sharpe: float = 0.75
    profit_max_bootstrap_pvalue: float = 0.15
    profit_target_daily_vol: float = 0.01
    profit_max_weight: float = 0.6

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "research" / "experiments"
