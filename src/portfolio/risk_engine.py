"""
Risk and Constraint Engine
Enforces regulatory and institutional portfolio risk constraints:
1. Max asset concentration limit (e.g. max 35% in any single equity)
2. Minimum diversification floor (e.g. min 2% per asset)
3. Simplex budget constraint (sum(w) == 1.0)
4. Turnover dampening / rebalancing thresholds to control transaction costs

Uses mathematically proven Convex Projection onto Bounded Simplex (Water-Filling algorithm)
via exact bisection on the dual Lagrange multiplier. Guarantees simultaneous satisfaction
of budget and box bounds without iterative normalization drift.
"""

import numpy as np
from typing import Optional, List, Dict


def project_to_bounded_simplex(
    raw_weights: np.ndarray,
    lower_bound: float = 0.02,
    upper_bound: float = 0.35,
    max_iter: int = 100,
    tol: float = 1e-9
) -> np.ndarray:
    """
    Solves the convex quadratic program:
        min_w  0.5 * ||w - raw_weights||_2^2
        s.t.   sum(w_i) == 1.0
               lower_bound <= w_i <= upper_bound for all i=1..N

    Dual formulation:
        w_i*(lambda) = clip(raw_weights_i - lambda, lower_bound, upper_bound)
        where lambda solves sum(w_i*(lambda)) - 1.0 = 0.

    Guarantees:
    - sum(w*) == 1.0 to machine precision.
    - lower_bound <= w_i* <= upper_bound strictly satisfied for every asset.
    """
    w_raw = np.asarray(raw_weights, dtype=float)
    n = len(w_raw)

    if n * lower_bound > 1.0 + 1e-7 or n * upper_bound < 1.0 - 1e-7:
        raise ValueError(
            f"Infeasible constraint set: N={n}, min_sum={n * lower_bound:.3f}, "
            f"max_sum={n * upper_bound:.3f}. Simplex sum=1.0 is mathematically impossible."
        )

    # Initial bisection bounds for lambda
    low = np.min(w_raw) - upper_bound
    high = np.max(w_raw) - lower_bound

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        w = np.clip(w_raw - mid, lower_bound, upper_bound)
        s = np.sum(w)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            low = mid
        else:
            high = mid

    w = np.clip(w_raw - mid, lower_bound, upper_bound)
    diff = 1.0 - np.sum(w)
    if abs(diff) > 1e-12:
        can_adjust = (w < upper_bound - 1e-7) if diff > 0 else (w > lower_bound + 1e-7)
        if np.any(can_adjust):
            w[can_adjust] += diff / np.sum(can_adjust)

    return w


class RiskEngine:
    """
    Applies regulatory and risk management constraints to recommended portfolio weights.
    """
    def __init__(
        self,
        max_asset_weight: float = 0.35,  # 35% concentration cap
        min_asset_weight: float = 0.02,  # 2% diversification floor
        max_turnover_limit: float = 0.40  # Max 40% rebalancing turnover per step
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
        and guarantees sum(weights) == 1.0 via convex projection.
        """
        w = np.asarray(raw_weights, dtype=float).copy()
        n = len(w)

        # 1. Clean non-finite values
        w = np.nan_to_num(w, nan=1.0 / n, posinf=1.0, neginf=0.0)

        # 2. Exact convex water-filling projection onto bounded simplex
        w_proj = project_to_bounded_simplex(
            raw_weights=w,
            lower_bound=self.min_asset_weight,
            upper_bound=self.max_asset_weight
        )

        # 3. Turnover dampening (via convex interpolation which preserves box and simplex constraints)
        if current_weights is not None:
            prev_w = np.asarray(current_weights, dtype=float)
            turnover = np.sum(np.abs(w_proj - prev_w))
            if turnover > self.max_turnover_limit and turnover > 1e-6:
                alpha = self.max_turnover_limit / turnover
                # Convex combination of two points in the bounded simplex remains inside the bounded simplex!
                w_proj = prev_w + alpha * (w_proj - prev_w)

        # Post-condition verification
        assert np.isclose(np.sum(w_proj), 1.0, atol=1e-5), f"Budget constraint violated: sum={np.sum(w_proj)}"
        assert np.all(w_proj <= self.max_asset_weight + 1e-5), f"Upper bound violated: max={np.max(w_proj)}"
        assert np.all(w_proj >= self.min_asset_weight - 1e-5), f"Lower bound violated: min={np.min(w_proj)}"

        return w_proj
