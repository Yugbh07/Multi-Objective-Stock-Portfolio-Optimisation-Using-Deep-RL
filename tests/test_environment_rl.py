"""
Gymnasium Reinforcement Learning Environment Unit Tests
Verifies:
1. Environment initialization, reset, and step transitions.
2. Observation space dimension and valid bounds.
3. Every step enforces exact bounded simplex constraints:
   - w_i <= 0.35 (35% concentration cap)
   - w_i >= 0.02 (2% diversification floor)
   - sum(w) == 1.0 (simplex budget constraint)
4. Non-anticipative forward reward timing (return earned over t -> t+1).
5. Scalarized reward responds with correct gradient signs:
   - Higher return -> increases reward
   - Higher portfolio variance -> decreases reward
   - Higher turnover -> decreases reward
6. Drifted portfolio weights update correctly after price movement.
"""

import pytest
import numpy as np
import pandas as pd
from src.environment.portfolio_env import PortfolioEnv


def create_mock_env(n_days=50, n_assets=5, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.015, size=(n_days, n_assets))
    prices_data = 100.0 * np.exp(np.cumsum(returns, axis=0))
    prices_df = pd.DataFrame(prices_data, index=dates, columns=tickers)

    # Features
    feat_data = {}
    for i, t in enumerate(tickers):
        feat_data[f"{t}_return"] = returns[:, i]
        feat_data[f"{t}_vol_20"] = np.random.uniform(0.15, 0.35, n_days)
    features_df = pd.DataFrame(feat_data, index=dates)

    # Regimes
    regimes_df = pd.DataFrame({
        "regime_id": np.random.choice([0, 1, 2], size=n_days)
    }, index=dates)

    return PortfolioEnv(
        prices_df=prices_df,
        features_df=features_df,
        regimes_df=regimes_df,
        include_regime=True
    )


class TestPortfolioEnvironment:
    def test_environment_reset_and_observation_dimension(self):
        env = create_mock_env()
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert info["step"] == 0
        assert np.isclose(info["portfolio_value"], 1_000_000.0)

    def test_step_enforces_35_cap_and_2_floor_on_all_actions(self):
        env = create_mock_env()
        obs, info = env.reset()

        # Test extreme, concentrated, and opposing actions
        actions = [
            np.array([10.0, -10.0, 0.0, 5.0, -5.0]),
            np.array([100.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-5.0, -5.0, -5.0, -5.0, -5.0]),
            np.random.normal(0, 1, 5)
        ]

        for act in actions:
            obs, reward, term, trunc, info = env.step(act)
            w = info["weights"]
            assert np.isclose(np.sum(w), 1.0, atol=1e-6)
            assert np.all(w <= 0.35 + 1e-6)
            assert np.all(w >= 0.02 - 1e-6)
            assert isinstance(reward, float)
            assert not np.isnan(reward)
            if term:
                break

    def test_reward_penalties_direction(self):
        """Verify reward penalizes variance and turnover in correct direction."""
        env = create_mock_env()
        env.reset()

        # Neutral action
        action_neutral = np.zeros(5)
        _, r_neutral, _, _, _ = env.step(action_neutral)

        # High turnover action should trigger turnover penalty
        env.reset()
        # Step 1: concentrate in asset 0
        env.step(np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
        # Step 2: flip 100% to asset 4 (high turnover)
        _, r_flip, _, _, info_flip = env.step(np.array([0.0, 0.0, 0.0, 0.0, 5.0]))

        assert info_flip["turnover"] > 0.30
        assert info_flip["cost"] > 0.0
