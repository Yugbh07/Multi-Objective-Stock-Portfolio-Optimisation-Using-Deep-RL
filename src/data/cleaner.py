"""
Market Data Cleaner
Cleans and standardizes raw financial timeseries:
- Aligns calendar/trading dates across all assets
- Handles missing values (ffill, bfill)
- Ensures monotonic chronological ordering (strictly preventing lookahead bias)
- Outputs clean price matrix and tabular dataset.
"""

import os
import logging
from typing import Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_market_data(
    raw_path: Optional[str] = None,
    output_clean_path: Optional[str] = None,
    output_prices_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw market data from data/raw/market_data.csv.
    Produces:
    1. clean_prices.csv: DataFrame of clean Adjusted Close prices indexed by Date.
    2. clean_market_data.csv: Multi-level clean dataset.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if raw_path is None:
        raw_path = os.path.join(root_dir, "data", "raw", "market_data.csv")
    if output_clean_path is None:
        output_clean_path = os.path.join(root_dir, "data", "processed", "clean_market_data.csv")
    if output_prices_path is None:
        output_prices_path = os.path.join(root_dir, "data", "processed", "clean_prices.csv")

    os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)

    logger.info(f"Loading raw market data from {raw_path}...")
    df = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)

    # 1. Sort chronologically (time-awareness)
    df = df.sort_index()

    # 2. Check and remove duplicate dates
    if df.index.has_duplicates:
        logger.warning(f"Found {df.index.duplicated().sum()} duplicated dates. Retaining first.")
        df = df[~df.index.duplicated(keep="first")]

    # 3. Check missing values
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        logger.info(f"Filling {missing_count} missing values via forward fill then backward fill...")
        df = df.ffill().bfill()

    # 4. Extract Adjusted Close price matrix (splits & dividends accounted for)
    if "Adj Close" in df.columns.levels[0]:
        prices_df = df["Adj Close"].copy()
    elif "Close" in df.columns.levels[0]:
        prices_df = df["Close"].copy()
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in dataframe columns.")

    # Drop any remaining unaligned dates
    prices_df = prices_df.dropna()
    df = df.loc[prices_df.index]

    # Save processed outputs
    df.to_csv(output_clean_path)
    prices_df.to_csv(output_prices_path)

    logger.info(f"Cleaned dataset saved: {output_clean_path} (Shape: {df.shape})")
    logger.info(f"Clean price matrix saved: {output_prices_path} (Shape: {prices_df.shape})")
    logger.info(f"Trading date range: {prices_df.index.min().date()} to {prices_df.index.max().date()}")

    return df, prices_df


if __name__ == "__main__":
    clean_df, prices = clean_market_data()
    print("\n--- Price Matrix Sample ---")
    print(prices.head())
    print("\n--- Summary Statistics ---")
    print(prices.describe())
