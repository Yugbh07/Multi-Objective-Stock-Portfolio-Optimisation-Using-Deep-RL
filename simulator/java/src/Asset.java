package simulator;

public class Asset {
    private String symbol;
    private double lastPrice;

    public Asset(String symbol, double lastPrice) {
        this.symbol = symbol;
        this.lastPrice = lastPrice;
    }

    public String getSymbol() { return symbol; }
    public double getLastPrice() { return lastPrice; }
    public void setLastPrice(double price) { this.lastPrice = price; }
}
