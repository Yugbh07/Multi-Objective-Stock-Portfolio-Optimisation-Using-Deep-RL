"""
Python <-> Java Virtual Trading Simulator Bridge
High-performance Inter-Process Communication (IPC) Bridge:
- Primary Mode: Sub-millisecond HTTP/REST Socket connection to Java 17 TradingEngine (Port 8088).
- Fallback Mode: Seamless native Python OOP simulator when Java microservice is offline.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional


class VirtualTradingBridge:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        fee_rate: float = 0.0010,
        java_server_url: str = "http://localhost:8088"
    ):
        self.initial_capital = float(initial_capital)
        self.cash_balance = float(initial_capital)
        self.fee_rate = float(fee_rate)
        self.java_server_url = java_server_url.rstrip("/")
        self.positions: Dict[str, float] = {}
        self.trade_log: List[Dict[str, Any]] = []

    def check_java_status(self) -> Dict[str, Any]:
        """Probes the Java Trading Simulator Server health."""
        try:
            req = urllib.request.Request(
                f"{self.java_server_url}/health",
                headers={"User-Agent": "Python-VirtualTradingBridge"}
            )
            start_t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=1.0) as response:
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    data["online"] = True
                    data["ping_latency_ms"] = round(latency_ms, 2)
                    return data
        except Exception:
            pass

        return {
            "online": False,
            "status": "OFFLINE",
            "message": "Java 17 Simulator Server not reachable. Falling back to native execution."
        }

    def reset_account(self, capital: Optional[float] = None) -> Dict[str, Any]:
        """Resets virtual account to initial capital."""
        if capital is not None:
            self.initial_capital = float(capital)
        self.cash_balance = self.initial_capital
        self.positions = {}
        self.trade_log = []

        # Try resetting Java server
        try:
            req = urllib.request.Request(
                f"{self.java_server_url}/reset",
                data=b"{}",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass

        return {
            "status": "RESET_SUCCESS",
            "engine": "Python Native Simulator",
            "cash_balance": self.cash_balance
        }

    def execute_rebalance(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Executes portfolio rebalancing orders.
        Routes to Java 17 Engine if online; otherwise falls back to Python OOP simulator.
        """
        # Attempt 1: Fast Java HTTP IPC
        try:
            payload = json.dumps({
                "target_weights": {k: float(v) for k, v in target_weights.items()},
                "current_prices": {k: float(v) for k, v in current_prices.items()}
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.java_server_url}/rebalance",
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            start_t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=2.0) as response:
                total_ipc_ms = (time.perf_counter() - start_t) * 1000.0
                if response.status == 200:
                    res = json.loads(response.read().decode("utf-8"))
                    res["engine_type"] = "Java 17 Microservice"
                    res["total_ipc_latency_ms"] = round(total_ipc_ms, 2)
                    self.cash_balance = res.get("cash_balance", self.cash_balance)
                    self.positions = res.get("positions", self.positions)
                    self.trade_log.extend(res.get("trades", []))
                    return res
        except Exception:
            pass

        # Attempt 2: Fallback Python Native OOP Simulator
        return self._execute_python_fallback(target_weights, current_prices)

    def _execute_python_fallback(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        start_t = time.perf_counter()
        equity = self.cash_balance + sum(
            self.positions.get(s, 0.0) * current_prices.get(s, 0.0) for s in target_weights
        )
        trades_executed = []
        fees_paid = 0.0

        # Phase 1: SELL orders
        for symbol, weight in target_weights.items():
            price = current_prices.get(symbol, 0.0)
            if price <= 0:
                continue
            target_val = equity * weight
            current_shares = self.positions.get(symbol, 0.0)
            current_val = current_shares * price
            delta_val = target_val - current_val

            if delta_val < -1.0:
                shares_to_sell = min(abs(delta_val) / price, current_shares)
                trade_val = shares_to_sell * price
                fee = trade_val * self.fee_rate
                self.positions[symbol] = current_shares - shares_to_sell
                self.cash_balance += (trade_val - fee)
                fees_paid += fee
                trades_executed.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "shares": round(shares_to_sell, 4),
                    "price": round(price, 2),
                    "fee": round(fee, 2)
                })

        # Phase 2: BUY orders
        for symbol, weight in target_weights.items():
            price = current_prices.get(symbol, 0.0)
            if price <= 0:
                continue
            target_val = equity * weight
            current_shares = self.positions.get(symbol, 0.0)
            current_val = current_shares * price
            delta_val = target_val - current_val

            if delta_val > 1.0:
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
                        "symbol": symbol,
                        "action": "BUY",
                        "shares": round(shares_to_buy, 4),
                        "price": round(price, 2),
                        "fee": round(fee, 2)
                    })

        new_equity = self.cash_balance + sum(
            self.positions.get(s, 0.0) * current_prices.get(s, 0.0) for s in target_weights
        )
        self.trade_log.extend(trades_executed)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "status": "SUCCESS",
            "engine": "Python Native Simulator (Fallback)",
            "engine_type": "Python Fallback",
            "latency_ms": round(latency_ms, 3),
            "portfolio_value": round(new_equity, 2),
            "cash_balance": round(self.cash_balance, 2),
            "fees_paid": round(fees_paid, 2),
            "trades_count": len(trades_executed),
            "trades": trades_executed,
            "positions": {k: round(v, 4) for k, v in self.positions.items() if v > 1e-4}
        }
