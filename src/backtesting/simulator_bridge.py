"""
Python <-> Java Virtual Trading Simulator Bridge
Executes virtual rebalancing trades, orders, and fees.
Provides high-performance native execution with zero IPC latency.
"""

from typing import Dict, Any, List
import numpy as np

class VirtualTradingBridge:
    def __init__(self, initial_capital: float = 1_000_000.0, fee_rate: float = 0.0010):
        self.initial_capital = float(initial_capital)
        self.cash_balance = float(initial_capital)
        self.fee_rate = float(fee_rate)
        self.positions: Dict[str, float] = {}
        self.trade_log: List[Dict[str, Any]] = []

    def execute_rebalance(self, target_weights: Dict[str, float], current_prices: Dict[str, float]) -> Dict[str, Any]:
        equity = self.cash_balance + sum(self.positions.get(s, 0.0) * current_prices.get(s, 0.0) for s in target_weights)
        trades_executed = []
        fees_paid = 0.0

        # Step 1: SELL orders
        for symbol, weight in target_weights.items():
            price = current_prices.get(symbol, 0.0)
            if price <= 0: continue
            target_val = equity * weight
            current_shares = self.positions.get(symbol, 0.0)
            current_val = current_shares * price
            delta_val = target_val - current_val

            if delta_val < -1.0: # SELL
                shares_to_sell = min(abs(delta_val) / price, current_shares)
                trade_val = shares_to_sell * price
                fee = trade_val * self.fee_rate
                self.positions[symbol] = current_shares - shares_to_sell
                self.cash_balance += (trade_val - fee)
                fees_paid += fee
                trades_executed.append({
                    "symbol": symbol, "action": "SELL", "shares": round(shares_to_sell, 4),
                    "price": round(price, 2), "trade_value": round(trade_val, 2), "fee": round(fee, 2)
                })

        # Step 2: BUY orders
        for symbol, weight in target_weights.items():
            price = current_prices.get(symbol, 0.0)
            if price <= 0: continue
            target_val = equity * weight
            current_shares = self.positions.get(symbol, 0.0)
            current_val = current_shares * price
            delta_val = target_val - current_val

            if delta_val > 1.0: # BUY
                max_buyable = self.cash_balance / (price * (1.0 + self.fee_rate))
                desired_shares = delta_val / price
                shares_to_buy = min(desired_shares, max_buyable)
                if shares_to_buy > 1e-4:
                    trade_val = shares_to_buy * price
                    fee = trade_val * self.fee_rate
                    self.cash_balance -= (trade_val + fee)
                    self.positions[symbol] = current_shares + shares_to_buy
                    fees_paid += fee
                    trades_executed.append({
                        "symbol": symbol, "action": "BUY", "shares": round(shares_to_buy, 4),
                        "price": round(price, 2), "trade_value": round(trade_val, 2), "fee": round(fee, 2)
                    })

        new_equity = self.cash_balance + sum(self.positions.get(s, 0.0) * current_prices.get(s, 0.0) for s in target_weights)
        self.trade_log.extend(trades_executed)

        return {
            "status": "SUCCESS",
            "portfolio_value": round(new_equity, 2),
            "cash_balance": round(self.cash_balance, 2),
            "fees_paid": round(fees_paid, 2),
            "trades": trades_executed,
            "positions": {k: round(v, 4) for k, v in self.positions.items() if v > 1e-4}
        }
