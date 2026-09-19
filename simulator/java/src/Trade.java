package simulator;

public class Trade {
    private String tradeId;
    private String orderId;
    private String symbol;
    private double executedQuantity;
    private double executedPrice;
    private double fee;
    private long timestamp;

    public Trade(String tradeId, String orderId, String symbol, double executedQuantity, double executedPrice, double fee) {
        this.tradeId = tradeId;
        this.orderId = orderId;
        this.symbol = symbol;
        this.executedQuantity = executedQuantity;
        this.executedPrice = executedPrice;
        this.fee = fee;
        this.timestamp = System.currentTimeMillis();
    }

    public String getTradeId() { return tradeId; }
    public String getOrderId() { return orderId; }
    public String getSymbol() { return symbol; }
    public double getExecutedQuantity() { return executedQuantity; }
    public double getExecutedPrice() { return executedPrice; }
    public double getFee() { return fee; }
    public long getTimestamp() { return timestamp; }
}
