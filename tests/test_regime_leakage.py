"""
Market Regime Detector Causal / Zero-Leakage Unit Tests
Verifies that:
1. Historical regime classification uses strictly past data up to day t.
2. Modifying or injecting an extreme price shock in future dates has ZERO effect
   on past regime classifications (Look-ahead bias proof).
3. Expanding volatility quantile threshold is strictly non-anticipative.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.regimes.detector import detect_market_regimes


class TestRegimeDetectorCausality:
    def _create_synthetic_features(self, n_days=100, seed=42):
        np.random.seed(seed)
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        data = {}
        for ticker in tickers:
            data[f"{ticker}_return"] = np.random.normal(0.0005, 0.015, n_days)
            data[f"{ticker}_vol_20"] = np.random.uniform(0.15, 0.45, n_days)
            data[f"{ticker}_mom_20"] = np.random.normal(0.01, 0.05, n_days)
            data[f"{ticker}_price_to_sma50"] = np.random.uniform(0.95, 1.05, n_days)

        return pd.DataFrame(data, index=dates)

    def test_strictly_causal_expanding_threshold_no_future_leakage(self):
        """
        Proof of Zero Leakage:
        Classify a dataset of 80 days.
        Then, artificially modify days 60-80 with a massive market crash.
        Verify that classifications for days 0-59 DO NOT CHANGE AT ALL!
        """
        df_original = self._create_synthetic_features(n_days=80)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_orig_path = os.path.join(tmpdir, "features_orig.csv")
            reg_orig_path = os.path.join(tmpdir, "regimes_orig.csv")
            df_original.to_csv(feat_orig_path)
            
            res_orig = detect_market_regimes(
                features_path=feat_orig_path,
                output_path=reg_orig_path
            )

            # Now create a modified future: inject extreme shock in days 60 to 79
            df_modified = df_original.copy()
            for col in df_modified.columns:
                if "vol_20" in col:
                    df_modified.iloc[60:, df_modified.columns.get_loc(col)] = 2.50  # 250% volatility spike!
                if "mom_20" in col:
                    df_modified.iloc[60:, df_modified.columns.get_loc(col)] = -0.50

            feat_mod_path = os.path.join(tmpdir, "features_mod.csv")
            reg_mod_path = os.path.join(tmpdir, "regimes_mod.csv")
            df_modified.to_csv(feat_mod_path)

            res_mod = detect_market_regimes(
                features_path=feat_mod_path,
                output_path=reg_mod_path
            )

            # Days 0 to 59 MUST be 100% identical between original and modified future
            past_orig_regimes = res_orig["regime"].iloc[:60].values
            past_mod_regimes = res_mod["regime"].iloc[:60].values
            
            past_orig_threshold = res_orig["vol_threshold"].iloc[:60].values
            past_mod_threshold = res_mod["vol_threshold"].iloc[:60].values

            # Exact array equality
            np.testing.assert_array_equal(
                past_orig_regimes,
                past_mod_regimes,
                err_msg="Future volatility shock leaked into past regime classifications!"
            )
            np.testing.assert_allclose(
                past_orig_threshold,
                past_mod_threshold,
                rtol=1e-8,
                err_msg="Expanding volatility threshold leaked future data!"
            )
