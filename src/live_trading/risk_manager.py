"""
Risk Manager

Handles position sizing and risk management (matches GPU kernel logic).
"""

from typing import Dict
import numpy as np


class RiskManager:
    """Risk management for position sizing and safety checks."""
    
    def __init__(self, leverage: int, max_risk_per_trade: float = 0.02):
        """
        Initialize risk manager.
        
        Args:
            leverage: Trading leverage
            max_risk_per_trade: Maximum % of balance to risk per trade (default 2%)
        """
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        
        # Safety limits
        self.max_leverage = 125
        self.min_position_size = 0.001  # Minimum position size in contracts
    
    def calculate_position_size(self, balance: float, price: float, leverage: int) -> float:
        """
        Calculate position size based on balance and risk.
        
        Args:
            balance: Available balance
            price: Current price
            leverage: Leverage to use
        
        Returns:
            Position size in contracts
        """
        # Calculate risk amount
        risk_amount = balance * self.max_risk_per_trade
        
        # Position value we can control with this risk
        position_value = risk_amount * leverage
        
        # Convert to contracts
        size = position_value / price
        
        # Apply minimum
        if size < self.min_position_size:
            return 0.0
        
        return size
    
    def check_liquidation_risk(
        self,
        entry_price: float,
        current_price: float,
        side: int,
        leverage: int
    ) -> bool:
        """
        Check if position is at risk of liquidation (matches GPU kernel).
        
        Args:
            entry_price: Entry price
            current_price: Current price
            side: 1 (long) or -1 (short)
            leverage: Position leverage
        
        Returns:
            True if at liquidation risk
        """
        # Calculate PnL percentage
        if side == 1:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Liquidation threshold: -100% / leverage
        liquidation_threshold = -1.0 / leverage
        
        return pnl_pct <= liquidation_threshold
    
    def validate_leverage(self, leverage: int) -> bool:
        """Validate leverage is within safe limits."""
        return 1 <= leverage <= self.max_leverage
