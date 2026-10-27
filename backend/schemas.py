from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PortfolioGenerateRequest(BaseModel):
    capital: float = Field(default=1000000.0, description="Investment capital in INR")
    risk_profile: str = Field(default="moderate", description="conservative, moderate, aggressive")
    assets: List[str] = Field(default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    horizon: str = Field(default="1Y")

class PortfolioGenerateResponse(BaseModel):
    regime: str
    risk_score: float
    recommended_allocation: Dict[str, float]
    expected_cagr: float
    expected_sharpe: float
    expected_volatility: float
    expected_max_drawdown: float
    explanation: Dict[str, Any]

class BacktestRequest(BaseModel):
    capital: float = 1000000.0
    start_date: str = "2024-01-01"
    transaction_cost_bps: float = 10.0
