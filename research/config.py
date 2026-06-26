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

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "research" / "experiments"
