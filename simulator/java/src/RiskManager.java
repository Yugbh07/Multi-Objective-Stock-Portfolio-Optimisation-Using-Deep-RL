package simulator;

public class RiskManager {
    public boolean validateOrder(Order order, Account account, Portfolio portfolio, double currentPrice) {
        if (order.getType() == Order.Type.BUY) {
            return account.getCashBalance() >= order.getQuantity() * currentPrice;
        } else if (order.getType() == Order.Type.SELL) {
            return portfolio.getPosition(order.getSymbol()) >= order.getQuantity();
        }
        return true;
    }
}
