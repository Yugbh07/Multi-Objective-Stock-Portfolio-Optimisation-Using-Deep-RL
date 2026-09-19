package simulator;

public class Transaction {
    private String id;
    private String type;
    private double amount;
    private double balanceAfter;
    private long timestamp;

    public Transaction(String id, String type, double amount, double balanceAfter) {
        this.id = id;
        this.type = type;
        this.amount = amount;
        this.balanceAfter = balanceAfter;
        this.timestamp = System.currentTimeMillis();
    }

    public String getId() { return id; }
    public String getType() { return type; }
    public double getAmount() { return amount; }
    public double getBalanceAfter() { return balanceAfter; }
    public long getTimestamp() { return timestamp; }
}
