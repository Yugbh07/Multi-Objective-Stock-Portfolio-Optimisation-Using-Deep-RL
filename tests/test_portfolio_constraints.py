"""
Portfolio Constraint and Bounded Simplex Unit & Property Tests
Verifies that:
1. Weights always sum to 1.0 (to machine precision).
2. Weights are strictly non-negative (w_i >= lower_bound).
3. No single asset exceeds the 35% concentration cap (w_i <= upper_bound).
4. Handles extreme continuous actions (e.g. [+1000, -1000, 0, 0, 0]).
5. Handles identical/equal actions (e.g. [0, 0, 0, 0, 0]).
6. Works across varying asset universe sizes (N=3, N=5, N=10, N=20).
7. Explicitly raises ValueError on mathematically infeasible constraint sets.
8. Robust against NaNs and Infinities.
9. 10,000 randomized property tests over diverse Dirichlet and Gaussian distributions.
"""

import pytest
import numpy as np
from src.portfolio.risk_engine import project_to_bounded_simplex, RiskEngine
from src.environment.portfolio_env import PortfolioEnv
import pandas as pd


class TestBoundedSimplexProjection:
    def test_basic_5_asset_bounds(self):
        raw = np.array([0.65, 0.20, 0.10, 0.03, 0.02])
        w = project_to_bounded_simplex(raw, lower_bound=0.02, upper_bound=0.35)
        
        assert np.isclose(np.sum(w), 1.0, atol=1e-8)
        assert np.all(w >= 0.02 - 1e-8)
        assert np.all(w <= 0.35 + 1e-8)
        # Largest asset should be capped at 0.35
        assert np.isclose(np.max(w), 0.35, atol=1e-6)

    def test_extreme_actions(self):
        raw_extreme = np.array([1000.0, -1000.0, 50.0, 0.0, -50.0])
        w = project_to_bounded_simplex(raw_extreme, lower_bound=0.02, upper_bound=0.35)
        
        assert np.isclose(np.sum(w), 1.0, atol=1e-8)
        assert np.all(w >= 0.02 - 1e-8)
        assert np.all(w <= 0.35 + 1e-8)

    def test_equal_actions(self):
        raw_equal = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        w = project_to_bounded_simplex(raw_equal, lower_bound=0.02, upper_bound=0.35)
        
        assert np.isclose(np.sum(w), 1.0, atol=1e-8)
        assert np.allclose(w, 0.20, atol=1e-6)

    def test_different_asset_counts(self):
        # N = 3 (each must be in [0.02, 0.50])
        w3 = project_to_bounded_simplex(np.array([1.0, 0.0, 0.0]), lower_bound=0.02, upper_bound=0.50)
        assert len(w3) == 3
        assert np.isclose(np.sum(w3), 1.0, atol=1e-8)
        assert np.all(w3 <= 0.50 + 1e-8)
        assert np.all(w3 >= 0.02 - 1e-8)

        # N = 10 (each must be in [0.02, 0.20])
        raw10 = np.random.uniform(-5, 5, size=10)
        w10 = project_to_bounded_simplex(raw10, lower_bound=0.02, upper_bound=0.20)
        assert len(w10) == 10
        assert np.isclose(np.sum(w10), 1.0, atol=1e-8)
        assert np.all(w10 <= 0.20 + 1e-8)
        assert np.all(w10 >= 0.02 - 1e-8)

        # N = 20 (each must be in [0.01, 0.10])
        raw20 = np.random.uniform(-5, 5, size=20)
        w20 = project_to_bounded_simplex(raw20, lower_bound=0.01, upper_bound=0.10)
        assert len(w20) == 20
        assert np.isclose(np.sum(w20), 1.0, atol=1e-8)
        assert np.all(w20 <= 0.10 + 1e-8)
        assert np.all(w20 >= 0.01 - 1e-8)

    def test_infeasible_constraints_raises_error(self):
        # 5 assets with min floor 0.25 (5 * 0.25 = 1.25 > 1.0 -> IMPOSSIBLE)
        with pytest.raises(ValueError, match="Infeasible constraint set"):
            project_to_bounded_simplex(np.ones(5) / 5, lower_bound=0.25, upper_bound=0.35)

        # 5 assets with max cap 0.15 (5 * 0.15 = 0.75 < 1.0 -> IMPOSSIBLE)
        with pytest.raises(ValueError, match="Infeasible constraint set"):
            project_to_bounded_simplex(np.ones(5) / 5, lower_bound=0.02, upper_bound=0.15)

    def test_nan_and_inf_handling(self):
        raw_dirty = np.array([np.nan, np.inf, -np.inf, 0.5, 0.1])
        w = project_to_bounded_simplex(raw_dirty, lower_bound=0.02, upper_bound=0.35)
        
        assert not np.any(np.isnan(w))
        assert not np.any(np.isinf(w))
        assert np.isclose(np.sum(w), 1.0, atol=1e-8)
        assert np.all(w >= 0.02 - 1e-8)
        assert np.all(w <= 0.35 + 1e-8)

    def test_property_based_10000_random_portfolios(self):
        """Randomized stress test over 10,000 generated portfolios."""
        np.random.seed(1337)
        for _ in range(10000):
            # Mix of normal, uniform, and sparse Dirichlet distributions
            dist_type = np.random.choice(["normal", "uniform", "dirichlet", "extreme"])
            if dist_type == "normal":
                raw = np.random.normal(0, 10, size=5)
            elif dist_type == "uniform":
                raw = np.random.uniform(-100, 100, size=5)
            elif dist_type == "dirichlet":
                raw = np.random.dirichlet([0.1, 0.5, 1.0, 2.0, 5.0])
            else:
                raw = np.array([1e6, -1e6, 0.0, 100.0, -50.0])

            w = project_to_bounded_simplex(raw, lower_bound=0.02, upper_bound=0.35)

            assert np.isclose(np.sum(w), 1.0, atol=1e-7), f"Sum violated: {np.sum(w)} on raw={raw}"
            assert np.all(w >= 0.02 - 1e-7), f"Min bound violated: min={np.min(w)}"
            assert np.all(w <= 0.35 + 1e-7), f"Max bound violated: max={np.max(w)}"


class TestRiskEngineTurnoverDampening:
    def test_turnover_constraint_preserves_simplex_and_bounds(self):
        engine = RiskEngine(max_asset_weight=0.35, min_asset_weight=0.02, max_turnover_limit=0.20)
        
        w_prev = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        w_new_raw = np.array([0.35, 0.35, 0.10, 0.10, 0.10])
        
        w_final = engine.apply_constraints(raw_weights=w_new_raw, current_weights=w_prev)
        
        turnover = np.sum(np.abs(w_final - w_prev))
        assert turnover <= 0.20 + 1e-6
        assert np.isclose(np.sum(w_final), 1.0, atol=1e-7)
        assert np.all(w_final <= 0.35 + 1e-7)
        assert np.all(w_final >= 0.02 - 1e-7)
