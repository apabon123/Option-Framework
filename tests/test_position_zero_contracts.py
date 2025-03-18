"""
Test script to verify that positions with 0 contracts are skipped in display.
"""

from core.position import Position, OptionPosition
from core.position_inventory import PositionInventory
from core.trading_engine import TradingEngine
from core.portfolio import Portfolio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('test')

# Create a test portfolio with a position that has 0 contracts
def test_zero_contracts():
    logger.info("Creating test portfolio...")
    
    # Create a portfolio
    portfolio = Portfolio(initial_capital=100000, logger=logger)
    
    # Create a position with contracts > 0
    normal_position = OptionPosition(
        symbol="SPY240329C00450000",
        option_data={"Strike": 450, "Type": "call", "Expiration": "2024-03-29"},
        contracts=10,
        entry_price=1.0,
        is_short=True,
        logger=logger
    )
    
    # Create a position with 0 contracts
    zero_position = OptionPosition(
        symbol="SPY240329C00460000",
        option_data={"Strike": 460, "Type": "call", "Expiration": "2024-03-29"},
        contracts=0,
        entry_price=1.0,
        is_short=True,
        logger=logger
    )
    
    # Add positions to inventory
    portfolio.inventory.add_position(normal_position)
    portfolio.inventory.add_position(zero_position)
    
    # Log the portfolio positions before filtering
    logger.info(f"Total positions in portfolio: {len(portfolio.positions)}")
    logger.info(f"Position symbols: {', '.join(portfolio.positions.keys())}")
    
    # Create a dummy trading engine just to test logging
    class DummyStrategy:
        def __init__(self):
            self.name = "DummyStrategy"
    
    engine = TradingEngine({"portfolio": {}}, DummyStrategy())
    engine.portfolio = portfolio
    engine.logger = logger
    
    # Call the log_open_positions method which should skip the 0 contracts position
    logger.info("Logging open positions (should skip the position with 0 contracts):")
    engine._log_open_positions()
    
    # Manual verification
    logger.info("\nManual verification of skipping:")
    for symbol, position in portfolio.positions.items():
        logger.info(f"Position {symbol} has {position.contracts} contracts")
        if position.contracts == 0:
            logger.info(f"  This position should have been skipped in the table above")

if __name__ == "__main__":
    test_zero_contracts() 