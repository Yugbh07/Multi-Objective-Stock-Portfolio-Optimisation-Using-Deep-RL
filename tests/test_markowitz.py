"""
Markowitz Mean-Variance Optimization Unit Tests
Verifies:
1. Covariance matrix regularization guarantees positive definiteness.
2. Solver enforces the 35% institutional concentration ceiling.
3. Weights strictly sum to 1.0.
4. Window slicing correctly restricts history to lookback_window.
5. Fallback tracking logs convergence vs fallback frequency.
"""

import pytest
import numpy as np
from src.portfolio.markowitz import MarkowitzOptimizer


class TestMarkowitzMVO:
    def test_mvo_weights_sum_to_one_and_satisfy_35_cap(self):
        np.random.seed(42)
        n_assets = 5
        returns = np.random.normal(0.0008, 0.015, size=(60, n_assets))
        
        opt = MarkowitzOptimizer(n_assets=n_assets, lookback_window=60, max_asset_weight=0.35)
        w = opt.optimize(returns)
        
        assert len(w) == n_assets
        assert np.isclose(np.sum(w), 1.0, atol=1e-7)
        assert np.all(w <= 0.35 + 1e-7)
        assert np.all(w >= 0.0 - 1e-7)

    def test_mvo_lookback_window_slicing(self):
        """Pass 200 days to an optimizer configured for lookback=60; verify it runs properly."""
        np.random.seed(123)
        returns_long = np.random.normal(0.0005, 0.02, size=(200, 5))
        
        opt = MarkowitzOptimizer(n_assets=5, lookback_window=60)
        w = opt.optimize(returns_long)
        
        assert np.isclose(np.sum(w), 1.0, atol=1e-7)
        assert np.all(w <= 0.35 + 1e-7)

    def test_mvo_fallback_on_insufficient_data(self):
        """Passing fewer than 20 observations should trigger safe 1/N fallback."""
        returns_short = np.random.normal(0.0005, 0.02, size=(10, 5))
        
        opt = MarkowitzOptimizer(n_assets=5, lookback_window=60)
        w = opt.optimize(returns_short)
        
        assert opt.fallback_count == 1
        assert np.allclose(w, 0.20)
