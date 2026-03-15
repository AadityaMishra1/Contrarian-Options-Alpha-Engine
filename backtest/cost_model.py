"""Cost model for options backtesting.

Accounts for commissions, slippage, and bid-ask spread on options fills.
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    """Models realistic transaction costs for options trades.

    Attributes:
        commission_per_contract: Flat commission charged per options contract.
        slippage_ticks: Number of ticks of slippage assumed on each fill.
        tick_size: Minimum price increment for the options contract (dollars).
    """

    commission_per_contract: float = 0.65
    slippage_ticks: int = 1
    tick_size: float = 0.01

    def commission(self, contracts: int) -> float:
        """Calculate total commission for a given number of contracts.

        Args:
            contracts: Number of options contracts traded.

        Returns:
            Total commission in dollars.
        """
        return abs(contracts) * self.commission_per_contract

    def slippage(self, contracts: int) -> float:
        """Calculate total slippage cost for a given number of contracts.

        Slippage is modeled as a fixed number of ticks per contract, each
        contract controlling 100 shares.

        Args:
            contracts: Number of options contracts traded.

        Returns:
            Total slippage cost in dollars.
        """
        return abs(contracts) * self.slippage_ticks * self.tick_size * 100

    def half_spread(self, contracts: int, spread: float) -> float:
        """Calculate cost of crossing half the bid-ask spread.

        Args:
            contracts: Number of options contracts traded.
            spread: Full bid-ask spread in dollars per share.

        Returns:
            Half-spread cost in dollars.
        """
        return abs(contracts) * (spread / 2.0) * 100

    def total_cost(self, contracts: int, price: float, spread: float) -> float:
        """Compute the total round-trip transaction cost for an options trade.

        Combines commission, slippage, and half-spread crossing into a single
        dollar figure deducted from P&L on entry (exit costs are handled
        separately on the closing fill).

        Args:
            contracts: Number of options contracts traded (positive for buy,
                       negative for sell; absolute value is used internally).
            price: Mid-market price of the option in dollars per share.
            spread: Full bid-ask spread in dollars per share.

        Returns:
            Total one-way transaction cost in dollars.
        """
        _ = price  # price retained for future premium-scaled models
        return (
            self.commission(contracts)
            + self.slippage(contracts)
            + self.half_spread(contracts, spread)
        )


if __name__ == "__main__":
    model = CostModel()
    contracts = 2
    price = 0.15
    spread = 0.05
    cost = model.total_cost(contracts, price, spread)
    print(f"Total cost for {contracts} contracts @ ${price:.2f} (spread ${spread:.2f}): ${cost:.4f}")
    print(f"  Commission : ${model.commission(contracts):.4f}")
    print(f"  Slippage   : ${model.slippage(contracts):.4f}")
    print(f"  Half-spread: ${model.half_spread(contracts, spread):.4f}")
