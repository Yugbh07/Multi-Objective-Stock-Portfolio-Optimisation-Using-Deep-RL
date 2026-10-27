package simulator;

import java.util.HashMap;
import java.util.Map;

public class Portfolio {
    private String portfolioId;
    private Map<String, Double> positions;

    public Portfolio(String portfolioId) {
        this.portfolioId = portfolioId;
        this.positions = new HashMap<>();
    }

    public Map<String, Double> getPositions() { return positions; }

    public double getPosition(String symbol) {
        return positions.getOrDefault(symbol, 0.0);
    }

    public void updatePosition(String symbol, double deltaShares) {
        double current = getPosition(symbol);
        double updated = current + deltaShares;
        if (Math.abs(updated) < 1e-6) {
            positions.remove(symbol);
        } else {
            positions.put(symbol, updated);
        }
    }

    public double calculateTotalValue(Map<String, Double> currentPrices, double cashBalance) {
        double total = cashBalance;
        for (Map.Entry<String, Double> entry : positions.entrySet()) {
            double price = currentPrices.getOrDefault(entry.getKey(), 0.0);
            total += entry.getValue() * price;
        }
        return total;
    }
}
