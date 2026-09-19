"""
PPO Agent Inference Service
Loads trained PPO models and generates allocation weights for unseen market episodes.
"""

import os
import sys
import logging
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.portfolio_env import PortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Wrapper around trained Stable-Baselines3 PPO policy for inference and simulation.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        include_regime: bool = True
    ):
        if model_path is None:
            model_name = "ppo_portfolio.zip" if include_regime else "ppo_portfolio_no_regime.zip"
            model_path = os.path.join(PROJECT_ROOT, "models", model_name)

        self.model_path = model_path
        self.include_regime = include_regime
        logger.info(f"Loading PPO Agent from {self.model_path}...")
        self.model = PPO.load(self.model_path)

    def generate_weights_for_dataset(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regimes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rolls out the trained policy deterministically through a dataset slice,
        returning a DataFrame of portfolio weights indexed by Date.
        """
        env = PortfolioEnv(
            prices_df=prices_df,
            features_df=features_df,
            regimes_df=regimes_df,
            include_regime=self.include_regime
        )

        obs, info = env.reset()
        dates = env.dates
        tickers = env.tickers
        n_days = len(dates)

        weights_records = []
        # Day 0 initial equal weights
        weights_records.append(env.current_weights.copy())

        for step in range(n_days - 1):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            weights_records.append(np.array(info["weights"]))
            if terminated:
                break

        # Fill any remaining steps if terminated early
        while len(weights_records) < n_days:
            weights_records.append(weights_records[-1])

        weights_df = pd.DataFrame(
            weights_records[:n_days],
            index=dates[:n_days],
            columns=tickers
        )
        return weights_df


if __name__ == "__main__":
    prices = pd.read_csv("data/processed/clean_prices.csv", index_col=0, parse_dates=True)
    features = pd.read_csv("data/features/portfolio_features.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/features/market_regimes.csv", index_col=0, parse_dates=True)

    # Test test split slice
    test_p = prices.loc["2024-01-01":]
    test_f = features.loc["2024-01-01":]
    test_r = regimes.loc["2024-01-01":]

    if os.path.exists("models/ppo_portfolio.zip"):
        agent = PPOAgent()
        w_df = agent.generate_weights_for_dataset(test_p, test_f, test_r)
        print("Generated weights sample:")
        print(w_df.head())
    else:
        print("Model file not found yet. Run train_ppo.py first.")
