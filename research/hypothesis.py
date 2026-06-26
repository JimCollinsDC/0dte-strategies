from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HypothesisTemplate:
    feature: str
    rule: str
    direction: int
    description: str
    complexity: int = 1


def generate_templates(features: Iterable[str]) -> list[HypothesisTemplate]:
    templates: list[HypothesisTemplate] = []
    for f in features:
        templates.extend(
            [
                HypothesisTemplate(
                    feature=f,
                    rule="q25_reversion",
                    direction=1,
                    description=f"Long when {f} is in bottom quartile",
                ),
                HypothesisTemplate(
                    feature=f,
                    rule="q75_momentum",
                    direction=1,
                    description=f"Long when {f} is in top quartile",
                ),
                HypothesisTemplate(
                    feature=f,
                    rule="q25_reversion",
                    direction=-1,
                    description=f"Short when {f} is in bottom quartile",
                ),
                HypothesisTemplate(
                    feature=f,
                    rule="q75_momentum",
                    direction=-1,
                    description=f"Short when {f} is in top quartile",
                ),
            ]
        )
    return templates


def signal_from_template(
    df: pd.DataFrame,
    template: HypothesisTemplate,
    q25: float,
    q75: float,
) -> np.ndarray:
    x = pd.to_numeric(df[template.feature], errors="coerce").to_numpy(dtype=float)
    sig = np.zeros_like(x, dtype=float)

    if template.rule == "q25_reversion":
        cond = x <= q25
    elif template.rule == "q75_momentum":
        cond = x >= q75
    else:
        raise ValueError(f"Unsupported rule: {template.rule}")

    sig[cond] = float(template.direction)
    return sig
