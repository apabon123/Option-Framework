# Fix for the margin calculator to handle the case where portfolio is passed instead of positions dictionary
# The error is: 'int' object has no attribute 'contracts'

def calculate_portfolio_margin_fixed(self, positions_or_portfolio):
    """
    Calculate margin for a portfolio of positions, with safety checks.
    
    This improved version handles both dictionary of positions and Portfolio objects.
    
    Args:
        positions_or_portfolio: Dictionary of positions by symbol or Portfolio object
        
    Returns:
        dict: Dictionary with total margin and margin by position
    """
    # Handle the case where a Portfolio object is passed instead of a dictionary
    positions = {}
    
    # Check if positions_or_portfolio is a Portfolio object
    if hasattr(positions_or_portfolio, 'positions'):
        positions = positions_or_portfolio.positions
    # Check if it's already a dictionary
    elif isinstance(positions_or_portfolio, dict):
        positions = positions_or_portfolio
    else:
        # Return zero margin if input is invalid
        self.logger.warning(f"Invalid input to calculate_portfolio_margin: {type(positions_or_portfolio)}")
        return {'total_margin': 0, 'margin_by_position': {}}
    
    if not positions:
        return {'total_margin': 0, 'margin_by_position': {}}
    
    # Calculate margin for each position with error handling
    margin_by_position = {}
    total_margin = 0
    
    for symbol, position in positions.items():
        try:
            # Ensure position is a valid Position object
            if not hasattr(position, 'contracts'):
                self.logger.warning(f"Invalid position object for {symbol}: {type(position)}")
                continue
                
            position_margin = self.calculate_position_margin(position)
            margin_by_position[symbol] = position_margin
            total_margin += position_margin
        except Exception as e:
            self.logger.error(f"Error calculating margin for {symbol}: {e}")
            # Continue processing other positions
            continue
    
    # Log the portfolio margin
    if self.logger:
        self.logger.debug(f"[Margin] Portfolio of {len(positions)} positions")
        self.logger.debug(f"  Total margin: ${total_margin:.2f}")
    
    return {
        'total_margin': total_margin,
        'margin_by_position': margin_by_position
    }

def calculate_position_margin_fixed(self, position):
    """
    Calculate margin requirement for a position with better error handling.
    
    Args:
        position: Position to calculate margin for
        
    Returns:
        float: Margin requirement in dollars
    """
    # Validate position object
    if not hasattr(position, 'contracts'):
        if self.logger:
            self.logger.warning(f"Invalid position object in calculate_position_margin: {type(position)}")
        return 0
        
    if position.contracts <= 0:
        return 0
    
    # For null positions, return zero margin
    if not hasattr(position, 'current_price') or position.current_price is None:
        return 0
        
    # Basic margin calculation
    leverage = getattr(self, 'max_leverage', 1.0)
    
    # Use either avg_entry_price or current_price
    position_price = position.current_price
    if hasattr(position, 'avg_entry_price') and position.avg_entry_price is not None:
        position_price = position.avg_entry_price
    
    # For options, multiply by 100 (contract multiplier)
    contract_multiplier = 100 if hasattr(position, 'option_type') else 1
    initial_margin = position_price * position.contracts * contract_multiplier / leverage
    
    # Adjust margin for unrealized PnL if position is short
    adjusted_margin = initial_margin
    if hasattr(position, 'unrealized_pnl') and position.is_short:
        adjusted_margin = initial_margin + position.unrealized_pnl
    
    # Log the margin calculation
    if self.logger:
        self.logger.debug(f"[Margin] Position {position.symbol}: {position.contracts} contracts")
        self.logger.debug(f"  Initial margin: ${initial_margin:.2f}")
        self.logger.debug(f"  Adjusted margin: ${adjusted_margin:.2f}")
    
    return max(adjusted_margin, 0)  # Ensure non-negative margin 