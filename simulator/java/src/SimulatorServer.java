package simulator;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * Java 17 Virtual Trading Simulator Server
 * High-performance, low-latency microservice for virtual trade execution,
 * order validation, position management, and transaction fee accounting.
 * Zero external JAR dependencies - uses built-in Java 17 HTTP microserver.
 */
public class SimulatorServer {
    private static final int PORT = 8088;
    private static TradingEngine engine = new TradingEngine(1000000.0);

    public static void main(String[] args) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.setExecutor(Executors.newFixedThreadPool(4));

        server.createContext("/health", new HealthHandler());
        server.createContext("/status", new HealthHandler());
        server.createContext("/rebalance", new RebalanceHandler());
        server.createContext("/reset", new ResetHandler());

        server.start();

        System.out.println("====================================================");
        System.out.println("🚀 Java 17 OOP Trading Simulator Server Online!");
        System.out.println("Listening on: http://localhost:" + PORT);
        System.out.println("Initial Virtual Capital: INR 1,000,000.00");
        System.out.println("Endpoints Available:");
        System.out.println("  - GET  /health    : Engine status & cash balance");
        System.out.println("  - POST /rebalance : Execute order matching & rebalance");
        System.out.println("  - POST /reset     : Reset account to initial balance");
        System.out.println("====================================================");
    }

    private static void setCorsHeaders(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
    }

    static class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCorsHeaders(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            double cash = engine.getAccount().getCashBalance();
            int tradesCount = engine.getTradeHistory().size();
            Map<String, Double> positions = engine.getPortfolio().getPositions();

            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"status\":\"ONLINE\",");
            sb.append("\"engine\":\"Java 17 OOP Trading Simulator\",");
            sb.append("\"port\":").append(PORT).append(",");
            sb.append("\"cash_balance\":").append(String.format(Locale.US, "%.2f", cash)).append(",");
            sb.append("\"trades_count\":").append(tradesCount).append(",");
            sb.append("\"positions\":{");
            int i = 0;
            for (Map.Entry<String, Double> entry : positions.entrySet()) {
                if (i++ > 0) sb.append(",");
                sb.append("\"").append(entry.getKey()).append("\":").append(String.format(Locale.US, "%.4f", entry.getValue()));
            }
            sb.append("}");
            sb.append("}");

            byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            OutputStream os = exchange.getResponseBody();
            os.write(bytes);
            os.close();
        }
    }

    static class ResetHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCorsHeaders(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            engine.reset(1000000.0);
            String response = "{\"status\":\"RESET_SUCCESS\",\"message\":\"Account capital reset to INR 1,000,000.00\",\"cash_balance\":1000000.0}";
            byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            OutputStream os = exchange.getResponseBody();
            os.write(bytes);
            os.close();
        }
    }

    static class RebalanceHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setCorsHeaders(exchange);
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            long startTime = System.nanoTime();

            InputStream is = exchange.getRequestBody();
            String body = new String(is.readAllBytes(), StandardCharsets.UTF_8);

            Map<String, Double> targetWeights = parseDoubleMap(body, "target_weights");
            Map<String, Double> currentPrices = parseDoubleMap(body, "current_prices");

            if (targetWeights.isEmpty() || currentPrices.isEmpty()) {
                String err = "{\"status\":\"ERROR\",\"message\":\"Missing target_weights or current_prices in request body\"}";
                byte[] errBytes = err.getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(400, errBytes.length);
                OutputStream os = exchange.getResponseBody();
                os.write(errBytes);
                os.close();
                return;
            }

            Map<String, Object> execResult = engine.executeRebalance(targetWeights, currentPrices);
            double latencyMs = (System.nanoTime() - startTime) / 1_000_000.0;

            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"status\":\"SUCCESS\",");
            sb.append("\"engine\":\"Java 17 OOP Trading Simulator\",");
            sb.append("\"latency_ms\":").append(String.format(Locale.US, "%.3f", latencyMs)).append(",");
            sb.append("\"portfolio_value\":").append(String.format(Locale.US, "%.2f", (Double) execResult.get("portfolioValue"))).append(",");
            sb.append("\"cash_balance\":").append(String.format(Locale.US, "%.2f", (Double) execResult.get("cashBalance"))).append(",");
            sb.append("\"fees_paid\":").append(String.format(Locale.US, "%.2f", (Double) execResult.get("totalFeesPaid"))).append(",");
            sb.append("\"trades_count\":").append(execResult.get("tradesExecuted")).append(",");

            // Positions
            @SuppressWarnings("unchecked")
            Map<String, Double> posMap = (Map<String, Double>) execResult.get("positions");
            sb.append("\"positions\":{");
            int pCount = 0;
            if (posMap != null) {
                for (Map.Entry<String, Double> entry : posMap.entrySet()) {
                    if (pCount++ > 0) sb.append(",");
                    sb.append("\"").append(entry.getKey()).append("\":").append(String.format(Locale.US, "%.4f", entry.getValue()));
                }
            }
            sb.append("},");

            // Executed trades
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> trades = (List<Map<String, Object>>) execResult.get("trades");
            sb.append("\"trades\":[");
            int tCount = 0;
            if (trades != null) {
                for (Map<String, Object> t : trades) {
                    if (tCount++ > 0) sb.append(",");
                    sb.append("{");
                    sb.append("\"symbol\":\"").append(t.get("symbol")).append("\",");
                    sb.append("\"action\":\"").append(t.get("action")).append("\",");
                    sb.append("\"shares\":").append(String.format(Locale.US, "%.4f", (Double) t.get("shares"))).append(",");
                    sb.append("\"price\":").append(String.format(Locale.US, "%.2f", (Double) t.get("price"))).append(",");
                    sb.append("\"fee\":").append(String.format(Locale.US, "%.2f", (Double) t.get("fee")));
                    sb.append("}");
                }
            }
            sb.append("]");
            sb.append("}");

            byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            OutputStream os = exchange.getResponseBody();
            os.write(bytes);
            os.close();
        }

        private Map<String, Double> parseDoubleMap(String json, String key) {
            Map<String, Double> map = new LinkedHashMap<>();
            int keyIdx = json.indexOf("\"" + key + "\"");
            if (keyIdx == -1) return map;
            int braceStart = json.indexOf("{", keyIdx);
            if (braceStart == -1) return map;
            int braceEnd = json.indexOf("}", braceStart);
            if (braceEnd == -1) return map;

            String sub = json.substring(braceStart + 1, braceEnd);
            String[] parts = sub.split(",");
            for (String part : parts) {
                String[] kv = part.split(":");
                if (kv.length == 2) {
                    String k = kv[0].replaceAll("[\"\\s]", "");
                    try {
                        double v = Double.parseDouble(kv[1].replaceAll("[\"\\s]", ""));
                        map.put(k, v);
                    } catch (Exception ignored) {}
                }
            }
            return map;
        }
    }
}
