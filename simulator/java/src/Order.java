package simulator;

public class Order {
    public enum Type { BUY, SELL }
    public enum Status { PENDING, FILLED, REJECTED }

    private String orderId;
    private String symbol;
    private double quantity;
    private double price;
    private Type type;
    private Status status;

    public Order(String orderId, String symbol, double quantity, double price, Type type) {
        this.orderId = orderId;
        this.symbol = symbol;
        this.quantity = quantity;
        this.price = price;
        this.type = type;
        this.status = Status.PENDING;
    }

    public String getOrderId() { return orderId; }
    public String getSymbol() { return symbol; }
    public double getQuantity() { return quantity; }
    public double getPrice() { return price; }
    public Type getType() { return type; }
    public Status getStatus() { return status; }
    public void setStatus(Status status) { this.status = status; }
}
