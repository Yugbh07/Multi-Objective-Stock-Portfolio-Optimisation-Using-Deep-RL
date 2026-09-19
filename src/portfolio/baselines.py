"""
Traditional Baseline Portfolio Strategies
1. Equal Weight Strategy: Rebalances periodically or daily to equal weights 1/N.
2. Buy & Hold Strategy: Allocates 1/N initially and lets weights drift naturally without trading.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class EqualWeightStrategy:
    """
    Fixed Equal Weight Strategy: Allocates 1/N capital equally across all N assets at each step.
    """
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.target_weight = np.ones(n_assets) / n_assets

    def get_weights(self, date: Optional[pd.Timestamp] = None, current_state: Optional[np.ndarray] = None) -> np.ndarray:
        return self.target_weight.copy()


class BuyAndHoldStrategy:
    """
    Buy & Hold Strategy:
    Initializes with 1/N allocation on day 0, then performs zero active rebalancing.
    Portfolio weights drift proportionally with asset price changes.
    """
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.initialized = False
        self.current_weights = np.ones(n_assets) / n_assets

    def reset(self):
        self.initialized = False
        self.current_weights = np.ones(self.n_assets) / self.n_assets

    def step_drift(self, asset_returns: np.ndarray) -> np.ndarray:
        """
        Updates weights based on market drift:
        w_i' = w_i * (1 + r_i) / sum(w_j * (1 + r_j))
        """
        if not self.initialized:
            self.initialized = True
            return self.current_weights.copy()

        grown = self.current_weights * (1.0 + asset_returns)
        total_grown = np.sum(grown)
        if total_grown > 0:
            self.current_weights = grown / total_grown
        return self.current_weights.copy()
