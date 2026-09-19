"""
Financial Feature Engineering Engine
Extracts quantitative financial features:
- Daily arithmetic & log returns
- Rolling 20-day & 60-day annualized volatility
- Cross-asset rolling correlation & covariance matrices
- Momentum indicators (20-day, 60-day returns)
- Trend indicators (Price / 20-day SMA, Price / 50-day SMA, Price / 200-day SMA)
"""

import os
import logging
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_features(
    prices_path: Optional[str] = None,
    output_features_path: Optional[str] = None,
    rolling_vol_window: int = 20,
    annualization_factor: float = np.sqrt(252)
) -> pd.DataFrame:
    """
    Computes returns, volatilities, momentum, and technical indicators for all assets.
    Saves features dataframe to data/features/portfolio_features.csv.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if prices_path is None:
        prices_path = os.path.join(root_dir, "data", "processed", "clean_prices.csv")
    if output_features_path is None:
        output_features_path = os.path.join(root_dir, "data", "features", "portfolio_features.csv")

    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    tickers = list(prices.columns)
    logger.info(f"Computing features for {tickers} over {len(prices)} dates...")

    feature_dfs = []

    # 1. Daily Arithmetic Returns: (P_t / P_{t-1}) - 1
    returns = prices.pct_change()
    returns_cols = {ticker: f"{ticker}_return" for ticker in tickers}
    returns_df = returns.rename(columns=returns_cols)
    feature_dfs.append(returns_df)

    # 2. Rolling 20-day Annualized Volatility: std(returns) * sqrt(252)
    vol_20 = returns.rolling(window=rolling_vol_window).std() * annualization_factor
    vol_cols = {ticker: f"{ticker}_vol_20" for ticker in tickers}
    feature_dfs.append(vol_20.rename(columns=vol_cols))

    # 3. Rolling 60-day Annualized Volatility
    vol_60 = returns.rolling(window=60).std() * annualization_factor
    vol_60_cols = {ticker: f"{ticker}_vol_60" for ticker in tickers}
    feature_dfs.append(vol_60.rename(columns=vol_60_cols))

    # 4. Momentum: 20-day and 60-day cumulative returns
    mom_20 = prices.pct_change(periods=20)
    feature_dfs.append(mom_20.rename(columns={ticker: f"{ticker}_mom_20" for ticker in tickers}))

    mom_60 = prices.pct_change(periods=60)
    feature_dfs.append(mom_60.rename(columns={ticker: f"{ticker}_mom_60" for ticker in tickers}))

    # 5. Price / Moving Average Ratios (Trend Strength)
    sma_50 = prices.rolling(window=50).mean()
    sma_50_ratio = prices / sma_50
    feature_dfs.append(sma_50_ratio.rename(columns={ticker: f"{ticker}_price_to_sma50" for ticker in tickers}))

    sma_200 = prices.rolling(window=200).mean()
    sma_200_ratio = prices / sma_200
    feature_dfs.append(sma_200_ratio.rename(columns={ticker: f"{ticker}_price_to_sma200" for ticker in tickers}))

    # Combine all features
    features_df = pd.concat(feature_dfs, axis=1)

    # Drop leading NaN values resulting from rolling 200-day window
    features_clean = features_df.dropna()

    # Save to CSV
    features_clean.to_csv(output_features_path)
    logger.info(f"Saved feature dataset ({features_clean.shape}) to {output_features_path}")
    logger.info(f"Available date span: {features_clean.index.min().date()} to {features_clean.index.max().date()}")

    return features_clean


def get_covariance_matrix(
    returns_df: pd.DataFrame,
    as_of_date: Optional[str] = None,
    lookback_window: int = 60,
    annualize: bool = True
) -> np.ndarray:
    """
    Computes sample covariance matrix Sigma as of a specific date using strictly past data (no lookahead bias).
    """
    if as_of_date is not None:
        sub_returns = returns_df.loc[:as_of_date]
    else:
        sub_returns = returns_df

    recent_returns = sub_returns.iloc[-lookback_window:]
    cov = recent_returns.cov().values
    if annualize:
        cov = cov * 252.0
    return cov


def get_correlation_matrix(
    returns_df: pd.DataFrame,
    as_of_date: Optional[str] = None,
    lookback_window: int = 60
) -> pd.DataFrame:
    """
    Computes rolling pairwise correlation matrix.
    """
    if as_of_date is not None:
        sub_returns = returns_df.loc[:as_of_date]
    else:
        sub_returns = returns_df

    return sub_returns.iloc[-lookback_window:].corr()


if __name__ == "__main__":
    feat_df = compute_features()
    print("\n--- Features Sample ---")
    print(feat_df.head(3))
    print(f"\nTotal Engineered Columns: {len(feat_df.columns)}")
