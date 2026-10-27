package simulator;

import java.util.*;

public class TradingEngine {
    private Account account;
    private Portfolio portfolio;
    private RiskManager riskManager;
    private List<Trade> tradeHistory;
    private double transactionFeeRate = 0.0010;

    public TradingEngine(double initialCapital) {
        this.account = new Account("ACC-001", initialCapital);
        this.portfolio = new Portfolio("PORT-001");
        this.riskManager = new RiskManager();
        this.tradeHistory = new ArrayList<>();
    }

    public Account getAccount() { return account; }
    public Portfolio getPortfolio() { return portfolio; }
    public List<Trade> getTradeHistory() { return tradeHistory; }

    public synchronized Map<String, Object> executeRebalance(
        Map<String, Double> targetWeights,
        Map<String, Double> currentPrices
    ) {
        double totalEquity = portfolio.calculateTotalValue(currentPrices, account.getCashBalance());
        List<Map<String, Object>> executedTrades = new ArrayList<>();
        double totalFees = 0.0;

        // Step 1: SELL orders
        for (Map.Entry<String, Double> entry : targetWeights.entrySet()) {
            String symbol = entry.getKey();
            double targetWeight = entry.getValue();
            double price = currentPrices.getOrDefault(symbol, 0.0);
            if (price <= 0) continue;

            double targetValue = totalEquity * targetWeight;
            double currentShares = portfolio.getPosition(symbol);
            double currentValue = currentShares * price;
            double deltaValue = targetValue - currentValue;

            if (deltaValue < -1.0) {
                double sharesToSell = Math.abs(deltaValue) / price;
                sharesToSell = Math.min(sharesToSell, currentShares);
                double tradeValue = sharesToSell * price;
                double fee = tradeValue * transactionFeeRate;

                portfolio.updatePosition(symbol, -sharesToSell);
                account.credit(tradeValue - fee);
                totalFees += fee;

                Trade trade = new Trade("TRD-" + System.currentTimeMillis(), "ORD-S", symbol, sharesToSell, price, fee);
                tradeHistory.add(trade);

                Map<String, Object> tMap = new HashMap<>();
                tMap.put("symbol", symbol);
                tMap.put("action", "SELL");
                tMap.put("shares", sharesToSell);
                tMap.put("price", price);
                tMap.put("fee", fee);
                executedTrades.add(tMap);
            }
        }

        // Step 2: BUY orders
        for (Map.Entry<String, Double> entry : targetWeights.entrySet()) {
            String symbol = entry.getKey();
            double targetWeight = entry.getValue();
            double price = currentPrices.getOrDefault(symbol, 0.0);
            if (price <= 0) continue;

            double targetValue = totalEquity * targetWeight;
            double currentShares = portfolio.getPosition(symbol);
            double currentValue = currentShares * price;
            double deltaValue = targetValue - currentValue;

            if (deltaValue > 1.0) {
                double maxBuyable = account.getCashBalance() / (price * (1.0 + transactionFeeRate));
                double desiredShares = deltaValue / price;
                double sharesToBuy = Math.min(desiredShares, maxBuyable);

                if (sharesToBuy > 1e-4) {
                    double tradeValue = sharesToBuy * price;
                    double fee = tradeValue * transactionFeeRate;

                    account.debit(tradeValue + fee);
                    portfolio.updatePosition(symbol, sharesToBuy);
                    totalFees += fee;

                    Trade trade = new Trade("TRD-" + System.currentTimeMillis(), "ORD-B", symbol, sharesToBuy, price, fee);
                    tradeHistory.add(trade);

                    Map<String, Object> tMap = new HashMap<>();
                    tMap.put("symbol", symbol);
                    tMap.put("action", "BUY");
                    tMap.put("shares", sharesToBuy);
                    tMap.put("price", price);
                    tMap.put("fee", fee);
                    executedTrades.add(tMap);
                }
            }
        }

        double updatedEquity = portfolio.calculateTotalValue(currentPrices, account.getCashBalance());
        Map<String, Object> result = new HashMap<>();
        result.put("status", "SUCCESS");
        result.put("portfolioValue", updatedEquity);
        result.put("cashBalance", account.getCashBalance());
        result.put("totalFeesPaid", totalFees);
        result.put("tradesExecuted", executedTrades.size());
        result.put("positions", portfolio.getPositions());
        result.put("trades", executedTrades);
        return result;
    }
}
