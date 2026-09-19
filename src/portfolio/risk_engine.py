"""
Risk and Constraint Engine
Post-processes raw policy allocation weights to enforce enterprise portfolio risk constraints:
1. Max asset concentration limit (e.g. max 35% in any single equity)
2. Non-negative long-only bounds (0 <= w_i)
3. Simplex sum constraint (sum(w) == 1.0)
4. Turnover dampening / rebalancing thresholds to control transaction costs
"""

import numpy as np
from typing import Optional, List, Dict


class RiskEngine:
    """
    Applies regulatory and risk management constraints to recommended portfolio weights.
    """
    def __init__(
        self,
        max_asset_weight: float = 0.35,  # 35% cap
        min_asset_weight: float = 0.02,  # 2% minimum diversification floor
        max_turnover_limit: float = 0.40  # Max 40% portfolio turnover in single rebalance
    ):
        self.max_asset_weight = float(max_asset_weight)
        self.min_asset_weight = float(min_asset_weight)
        self.max_turnover_limit = float(max_turnover_limit)

    def apply_constraints(
        self,
        raw_weights: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Enforces asset concentration limits, minimum floor, turnover dampening,
        and guarantees sum(weights) == 1.0.
        """
        w = np.asarray(raw_weights, dtype=float).copy()
        n = len(w)

        # 1. Clean non-finite values
        w = np.nan_to_num(w, nan=1.0 / n, posinf=1.0, neginf=0.0)

        # 2. Normalize to sum to 1
        s = np.sum(w)
        w = w / s if s > 0 else np.ones(n) / n

        # 3. Iterative capping and redistribution
        for _ in range(10):
            exceeded = w > self.max_asset_weight
            if not np.any(exceeded):
                break
            excess_weight = np.sum(w[exceeded] - self.max_asset_weight)
            w[exceeded] = self.max_asset_weight

            non_exceeded = ~exceeded
            if np.any(non_exceeded):
                sub_sum = np.sum(w[non_exceeded])
                if sub_sum > 0:
                    w[non_exceeded] += excess_weight * (w[non_exceeded] / sub_sum)
                else:
                    w[non_exceeded] += excess_weight / np.sum(non_exceeded)

        # Enforce minimum floor
        w = np.maximum(w, self.min_asset_weight)
        w = w / np.sum(w)

        # 4. Turnover dampening (if previous weights provided)
        if current_weights is not None:
            prev_w = np.asarray(current_weights, dtype=float)
            turnover = np.sum(np.abs(w - prev_w))
            if turnover > self.max_turnover_limit:
                # Interpolate towards previous weight to satisfy turnover limit
                alpha = self.max_turnover_limit / turnover
                w = prev_w + alpha * (w - prev_w)
                w = np.maximum(w, 0.0)
                w = w / np.sum(w)

        return w


if __name__ == "__main__":
    risk = RiskEngine(max_asset_weight=0.30, min_asset_weight=0.05)
    unconstrained = np.array([0.55, 0.25, 0.10, 0.08, 0.02])
    constrained = risk.apply_constraints(unconstrained)
    print("Raw Weights:        ", np.round(unconstrained, 4))
    print("Constrained Weights:", np.round(constrained, 4))
    print("Max weight <= 0.30: ", np.max(constrained) <= 0.30001)
    print("Sum of weights:     ", np.round(np.sum(constrained), 6))
