package simulator;

import java.io.*;
import java.net.*;
import java.util.*;

public class SimulatorServer {
    public static void main(String[] args) {
        System.out.println("=== Java Virtual Trading Simulator Initialized ===");
        TradingEngine engine = new TradingEngine(1000000.0);

        Map<String, Double> prices = new HashMap<>();
        prices.put("AAPL", 225.5);
        prices.put("MSFT", 430.0);
        prices.put("GOOGL", 180.2);
        prices.put("AMZN", 195.0);
        prices.put("NVDA", 125.8);

        Map<String, Double> weights = new HashMap<>();
        weights.put("AAPL", 0.20);
        weights.put("MSFT", 0.25);
        weights.put("GOOGL", 0.20);
        weights.put("AMZN", 0.15);
        weights.put("NVDA", 0.20);

        Map<String, Object> res = engine.executeRebalance(weights, prices);
        System.out.println("Execution Status: " + res.get("status"));
        System.out.println("Initial Capital: INR 1,000,000.0");
        System.out.println("Portfolio Value: INR " + res.get("portfolioValue"));
        System.out.println("Cash Balance: INR " + res.get("cashBalance"));
        System.out.println("Fees Incurred: INR " + res.get("totalFeesPaid"));
        System.out.println("Trades Executed: " + res.get("tradesExecuted"));
        System.out.println("Asset Positions: " + res.get("positions"));
    }
}
