"""
Market Data Collector
Fetches historical daily market data for target assets via yfinance with caching,
validation, and robust fallback mechanisms.
"""

import os
import sys
import logging
from typing import List, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_START = "2016-01-01"
DEFAULT_END = None  # None fetches up to the most recent trading session


def get_project_root() -> str:
    """Returns absolute path to the portfolio-optimizer project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def download_market_data(
    tickers: Optional[List[str]] = None,
    start_date: str = DEFAULT_START,
    end_date: Optional[str] = DEFAULT_END,
    save_path: Optional[str] = None,
    force_refresh: bool = False,
    allow_synthetic_fallback: bool = False
) -> pd.DataFrame:
    """
    Downloads historical OHLCV and Adjusted Close data from yfinance.
    Saves the multi-asset dataframe into data/raw/market_data.csv.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    root_dir = get_project_root()
    if save_path is None:
        save_path = os.path.join(root_dir, "data", "raw", "market_data.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if raw data already exists and force_refresh is False
    if os.path.exists(save_path) and not force_refresh:
        logger.info(f"Existing raw data found at {save_path}. Loading cached file.")
        try:
            df = pd.read_csv(save_path, header=[0, 1], index_col=0, parse_dates=True)
            if not df.empty:
                logger.info(f"Loaded {len(df)} records for tickers: {tickers}")
                return df
        except Exception as e:
            logger.warning(f"Could not load cached file: {e}. Re-downloading...")

    logger.info(f"Downloading historical data for {tickers} from {start_date} to {end_date}...")
    try:
        # Download all tickers at once with auto_adjust=False to keep standard Close and Adj Close
        raw_df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by="column",
            progress=True,
            auto_adjust=False
        )

        if raw_df is None or raw_df.empty:
            raise ValueError("yfinance returned an empty dataframe.")

        raw_df.to_csv(save_path)
        logger.info(f"Successfully saved raw data ({raw_df.shape}) to {save_path}")
        return raw_df

    except Exception as e:
        logger.error(f"Error downloading market data: {e}")
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"Failed to fetch market data from yfinance: {e}. "
                "Synthetic data fallback is disabled to prevent silent data fabrication."
            ) from e

        logger.warning("Generating realistic historical fallback dataset (DEMO ONLY)...")
        fallback_end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = pd.date_range(start=start_date, end=fallback_end, freq="B")
        import numpy as np
        np.random.seed(42)
        
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Adj Close", "Volume"], tickers],
            names=["Price", "Ticker"]
        )
        data = np.zeros((len(dates), len(columns)))
        fallback_df = pd.DataFrame(data, index=dates, columns=columns)
        
        base_prices = {"AAPL": 25.0, "MSFT": 50.0, "GOOGL": 35.0, "AMZN": 30.0, "NVDA": 8.0}
        drifts = {"AAPL": 0.0009, "MSFT": 0.0008, "GOOGL": 0.0007, "AMZN": 0.0007, "NVDA": 0.0015}
        vols = {"AAPL": 0.016, "MSFT": 0.015, "GOOGL": 0.015, "AMZN": 0.018, "NVDA": 0.025}
        
        for ticker in tickers:
            p0 = base_prices.get(ticker, 50.0)
            drift = drifts.get(ticker, 0.0008)
            vol = vols.get(ticker, 0.018)
            returns = np.random.normal(loc=drift, scale=vol, size=len(dates))
            price_path = p0 * np.exp(np.cumsum(returns))
            
            fallback_df[("Close", ticker)] = price_path
            fallback_df[("Adj Close", ticker)] = price_path
            fallback_df[("Open", ticker)] = price_path * (1 + np.random.normal(0, 0.003, len(dates)))
            fallback_df[("High", ticker)] = np.maximum(fallback_df[("Open", ticker)], price_path) * 1.005
            fallback_df[("Low", ticker)] = np.minimum(fallback_df[("Open", ticker)], price_path) * 0.995
            fallback_df[("Volume", ticker)] = np.random.randint(1000000, 50000000, size=len(dates))

        fallback_df.to_csv(save_path)
        logger.info(f"Fallback dataset saved to {save_path}")
        return fallback_df
    
def fetch_latest_market_data(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch the latest available market data directly from Yahoo Finance.
    Does not read from or overwrite the historical training CSV.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    logger.info(f"Fetching latest market data for {tickers}")

    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        group_by="column",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance returned no latest market data.")

    if isinstance(data.columns, pd.MultiIndex):
        available_tickers = set(data.columns.get_level_values(1))
    else:
        available_tickers = set(tickers)

    missing = set(tickers) - available_tickers

    if missing:
        raise RuntimeError(f"Missing market data for tickers: {sorted(missing)}")

    close_data = data["Close"]
    latest_data_timestamp = close_data.dropna(how="all").index[-1]

    now_utc = pd.Timestamp.now(tz="UTC")
    latest_timestamp = pd.Timestamp(latest_data_timestamp)
 
    if latest_timestamp.tzinfo is None:
       latest_timestamp = latest_timestamp.tz_localize("UTC")

    data_age_minutes = (now_utc - latest_timestamp).total_seconds() / 60

    logger.info(f"Latest market candle: {latest_timestamp}")
    logger.info(f"Data age: {data_age_minutes:.2f} minutes")

    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    close_data = close_data.dropna(how="all")

    if close_data.empty:
        raise RuntimeError("No valid closing prices returned.")
    # Validate intraday data freshness
    latest_timestamp = pd.Timestamp(close_data.dropna(how="all").index[-1])

    now_et = pd.Timestamp.now(tz="America/New_York")

    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("America/New_York")
    else:
        latest_timestamp = latest_timestamp.tz_convert("America/New_York")

    age_minutes = (now_et - latest_timestamp).total_seconds() / 60

    logger.info(f"Latest candle timestamp (ET): {latest_timestamp}")
    logger.info(f"Market data age: {age_minutes:.2f} minutes")

    if age_minutes > 30:
       raise RuntimeError(
          f"Market data is stale ({age_minutes:.1f} minutes old). "
          "Refusing to generate a recommendation."
       )
    

    latest_prices = close_data.iloc[-1].dropna()

    missing_prices = set(tickers) - set(latest_prices.index)

    if missing_prices:
        raise RuntimeError(f"Missing latest prices for: {sorted(missing_prices)}")

    if (latest_prices <= 0).any():
        raise RuntimeError("Invalid non-positive stock price detected.")

    result = pd.DataFrame({
        "Ticker": latest_prices.index,
        "Price": latest_prices.values,
        "Market_Data_Date": close_data.index[-1]
    })

    result["Fetched_At_UTC"] = datetime.now(timezone.utc).isoformat()
    result["Data_Status"] = "LAST MARKET CLOSE"

    logger.info("Latest prices successfully fetched and validated.")
    return result    


if __name__ == "__main__":
    df = fetch_latest_market_data()

    print("\nLatest Market Data:")
    print(df.to_string(index=False))
    print("\nFetched rows:", len(df))
