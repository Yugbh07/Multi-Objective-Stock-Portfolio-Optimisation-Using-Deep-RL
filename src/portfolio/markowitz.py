"""
Markowitz Mean-Variance Optimization (MVO) Strategy
Implements classical Modern Portfolio Theory (MPT):
- Maximizes Sharpe ratio: (w^T * mu - Rf) / sqrt(w^T * Sigma * w)
- Or minimizes variance: w^T * Sigma * w
- Subject to full investment constraint: sum(w) = 1
- Long-only constraint: 0 <= w_i <= 1
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MarkowitzOptimizer:
    """
    Solves classical Mean-Variance Optimization using historical lookback windows.
    Strictly avoids lookahead bias by using only data available up to decision date.
    """
    def __init__(
        self,
        n_assets: int,
        lookback_window: int = 60,
        risk_free_rate: float = 0.02,
        objective: str = "max_sharpe"
    ):
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.objective = objective

    def optimize(
        self,
        historical_returns: np.ndarray
    ) -> np.ndarray:
        """
        historical_returns: 2D array of shape (lookback, n_assets)
        Returns: optimal weights vector of shape (n_assets,) summing to 1.0.
        """
        n = self.n_assets
        if len(historical_returns) < n:
            return np.ones(n) / n

        # Annualized expected returns and covariance matrix
        mu = np.mean(historical_returns, axis=0) * 252.0
        sigma = np.cov(historical_returns, rowvar=False) * 252.0

        # Regularize covariance matrix slightly to ensure positive definiteness
        sigma += np.eye(n) * 1e-6

        # Initial guess: equal weights
        init_weights = np.ones(n) / n
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        if self.objective == "max_sharpe":
            def neg_sharpe(w):
                port_ret = np.dot(w, mu)
                port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
                if port_vol <= 1e-8:
                    return 0.0
                return -(port_ret - self.risk_free_rate) / port_vol

            res = minimize(
                neg_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7}
            )

        elif self.objective == "min_variance":
            def port_variance(w):
                return np.dot(w.T, np.dot(sigma, w))

            res = minimize(
                port_variance,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7}
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        if res.success:
            weights = np.clip(res.x, 0.0, 1.0)
            s = np.sum(weights)
            return weights / s if s > 0 else np.ones(n) / n
        else:
            # Fallback to equal weights
            return np.ones(n) / n


if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = np.random.normal(0.0005, 0.015, size=(120, 5))
    opt = MarkowitzOptimizer(n_assets=5, lookback_window=120)
    w = opt.optimize(fake_returns)
    print("Optimal MVO Weights:", np.round(w, 4))
    print("Sum of weights:", np.sum(w))
