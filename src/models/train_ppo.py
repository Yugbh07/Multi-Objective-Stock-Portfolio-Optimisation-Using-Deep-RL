"""
PPO Agent Training Pipeline
Trains Deep Reinforcement Learning Agent using Stable-Baselines3:
- Enforces strict chronological data splitting (Train / Val / Test)
- Learns multi-objective asset rebalancing policy
- Trains standard PPO and ablation (PPO without regime)
- Saves checkpoints to models/ directory
"""

import os
import sys
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.portfolio_env import PortfolioEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_data(
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits prices, features, and regimes chronologically into:
    1. Training set (start to 2022-12-31)
    2. Validation set (2023-01-01 to 2023-12-31)
    3. Test / Out-of-Sample set (2024-01-01 to 2025-12-31)
    """
    prices = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "clean_prices.csv"), index_col=0, parse_dates=True)
    features = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "features", "portfolio_features.csv"), index_col=0, parse_dates=True)
    regimes = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "features", "market_regimes.csv"), index_col=0, parse_dates=True)

    common_index = prices.index.intersection(features.index).intersection(regimes.index)
    prices = prices.loc[common_index]
    features = features.loc[common_index]
    regimes = regimes.loc[common_index]

    # Chronological slices
    train_mask = common_index <= train_end
    val_mask = (common_index > train_end) & (common_index <= val_end)
    test_mask = common_index > val_end

    logger.info(f"Data Splits: Train={train_mask.sum()} days, Val={val_mask.sum()} days, Test={test_mask.sum()} days")

    return (
        (prices[train_mask], features[train_mask], regimes[train_mask]),
        (prices[val_mask], features[val_mask], regimes[val_mask]),
        (prices[test_mask], features[test_mask], regimes[test_mask])
    )


def train_ppo_agent(
    total_timesteps: int = 30000,
    include_regime: bool = True,
    model_save_path: Optional[str] = None,
    seed: int = 42
) -> PPO:
    """
    Trains PPO on the historical training slice.
    """
    (train_p, train_f, train_r), (val_p, val_f, val_r), (test_p, test_f, test_r) = split_data()

    if model_save_path is None:
        model_name = "ppo_portfolio.zip" if include_regime else "ppo_portfolio_no_regime.zip"
        model_save_path = os.path.join(PROJECT_ROOT, "models", model_name)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logger.info(f"Creating training environment (include_regime={include_regime})...")
    def make_env():
        return PortfolioEnv(
            prices_df=train_p,
            features_df=train_f,
            regimes_df=train_r,
            include_regime=include_regime
        )

    env = DummyVecEnv([make_env])

    logger.info("Initializing PPO with MlpPolicy...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed
    )

    logger.info(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)
    logger.info(f"Model saved successfully to {model_save_path}")

    return model


if __name__ == "__main__":
    # Train both standard (regime-aware) and ablation (no regime) models
    logger.info("--- Training Regime-Aware PPO ---")
    train_ppo_agent(total_timesteps=30000, include_regime=True)

    logger.info("\n--- Training Ablation: PPO Without Regime ---")
    train_ppo_agent(total_timesteps=30000, include_regime=False)
