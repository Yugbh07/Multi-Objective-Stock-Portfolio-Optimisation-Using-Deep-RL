package simulator;

public class Account {
    private String accountId;
    private double cashBalance;
    private double initialCapital;

    public Account(String accountId, double initialCapital) {
        this.accountId = accountId;
        this.initialCapital = initialCapital;
        this.cashBalance = initialCapital;
    }

    public String getAccountId() { return accountId; }
    public double getCashBalance() { return cashBalance; }
    public double getInitialCapital() { return initialCapital; }

    public void debit(double amount) { this.cashBalance -= amount; }
    public void credit(double amount) { this.cashBalance += amount; }
}
