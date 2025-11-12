"""
Enhanced Risk Manager for live trading.
Implements comprehensive pre-order risk checks, position limits, and liquidation warnings.
"""

import time
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_trading.exceptions import (
    RiskLimitError,
    LiquidationRiskError,
    InsufficientMarginError,
    ValidationError
)


class RiskConfig:
    """Risk management configuration."""
    
    def __init__(self):
        # Position limits
        self.max_position_size_btc = 10.0  # Maximum position size
        self.max_position_size_eth = 100.0
        self.max_leverage = 3  # Maximum allowed leverage
        self.min_leverage = 1
        
        # Liquidation thresholds
        self.liquidation_warning_pct = 5.0  # Warn if <5% from liquidation
        self.liquidation_critical_pct = 2.0  # Critical if <2% from liquidation
        
        # Daily limits
        self.daily_loss_limit_usd = 500.0  # Stop trading if daily loss exceeds
        self.max_daily_trades = 50  # Maximum trades per day
        
        # Position management
        self.max_open_positions = 3  # Maximum concurrent positions
        self.require_stop_loss = True  # Always require stop loss
        self.require_take_profit = True  # Always require take profit
        
        # Order size limits
        self.min_order_size = 1  # Minimum order size in contracts
        self.max_order_size = 100  # Maximum order size in contracts


class EnhancedRiskManager:
    """
    Enhanced risk manager for pre-trade checks and monitoring.
    
    Responsibilities:
    - Validate orders before submission
    - Check position size limits
    - Monitor liquidation risk
    - Track daily P&L and limits
    - Enforce risk rules
    """
    
    def __init__(self, config: RiskConfig = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration (uses defaults if None)
        """
        self.config = config or RiskConfig()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = time.time()
        self.open_positions = {}  # symbol -> position data
    
    def _reset_daily_limits_if_needed(self):
        """Reset daily counters if new day."""
        current_time = time.time()
        # Reset if more than 24 hours
        if current_time - self.last_reset > 86400:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = current_time
            print(f"📊 Daily limits reset")
    
    def validate_symbol(self, symbol: str):
        """Validate trading symbol format."""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string", field="symbol", value=symbol)
        
        # Futures symbols should end with M (perpetual) or contain USDT
        if not ('USDT' in symbol.upper()):
            raise ValidationError(
                f"Invalid futures symbol: {symbol}. Must contain USDT",
                field="symbol",
                value=symbol
            )
    
    def validate_side(self, side: str):
        """Validate order side."""
        if side not in ['buy', 'sell']:
            raise ValidationError(
                f"Invalid side: {side}. Must be 'buy' or 'sell'",
                field="side",
                value=side
            )
    
    def validate_size(self, size: int):
        """Validate order size."""
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValidationError(
                f"Invalid size: {size}. Must be positive number",
                field="size",
                value=size
            )
        
        if size < self.config.min_order_size:
            raise ValidationError(
                f"Size {size} below minimum {self.config.min_order_size}",
                field="size",
                value=size
            )
        
        if size > self.config.max_order_size:
            raise RiskLimitError(
                f"Size {size} exceeds maximum {self.config.max_order_size}",
                limit_type="order_size",
                current_value=size,
                max_value=self.config.max_order_size
            )
    
    def validate_leverage(self, leverage: int):
        """Validate leverage setting."""
        if not isinstance(leverage, (int, float)):
            raise ValidationError(
                f"Invalid leverage: {leverage}. Must be number",
                field="leverage",
                value=leverage
            )
        
        if leverage < self.config.min_leverage or leverage > self.config.max_leverage:
            raise RiskLimitError(
                f"Leverage {leverage} outside allowed range {self.config.min_leverage}-{self.config.max_leverage}",
                limit_type="leverage",
                current_value=leverage,
                max_value=self.config.max_leverage
            )
    
    def check_position_limits(self, symbol: str, new_size: int, current_position: Optional[Dict] = None):
        """
        Check if adding new size would exceed position limits.
        
        Args:
            symbol: Trading symbol
            new_size: Size to add (positive for long, negative for short)
            current_position: Current position data
        
        Raises:
            RiskLimitError: If position limit would be exceeded
        """
        current_qty = 0
        if current_position and current_position.get('currentQty'):
            current_qty = current_position['currentQty']
        
        projected_qty = current_qty + new_size
        projected_abs = abs(projected_qty)
        
        # Check symbol-specific limits
        if 'BTC' in symbol.upper() or 'XBT' in symbol.upper():
            max_size = self.config.max_position_size_btc
        elif 'ETH' in symbol.upper():
            max_size = self.config.max_position_size_eth
        else:
            max_size = self.config.max_position_size_btc  # Default
        
        if projected_abs > max_size:
            raise RiskLimitError(
                f"Position size {projected_abs} would exceed limit {max_size} for {symbol}",
                limit_type="position_size",
                current_value=projected_abs,
                max_value=max_size
            )
    
    def check_liquidation_risk(
        self,
        symbol: str,
        current_price: float,
        position: Optional[Dict] = None
    ):
        """
        Check if position is near liquidation.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position: Position data including liquidation price
        
        Raises:
            LiquidationRiskError: If position is critically close to liquidation
        """
        if not position or position.get('currentQty', 0) == 0:
            return  # No position, no risk
        
        liquidation_price = position.get('liquidationPrice') or position.get('liquidation_price')
        if not liquidation_price:
            return  # Can't check without liquidation price
        
        current_qty = position.get('currentQty', 0)
        
        # Calculate distance to liquidation
        if current_qty > 0:  # Long position
            distance_pct = ((liquidation_price - current_price) / current_price) * 100
        else:  # Short position
            distance_pct = ((current_price - liquidation_price) / current_price) * 100
        
        # Check thresholds
        if abs(distance_pct) < self.config.liquidation_critical_pct:
            raise LiquidationRiskError(
                f"🚨 CRITICAL: Position {distance_pct:.2f}% from liquidation! "
                f"Current: ${current_price:.2f}, Liquidation: ${liquidation_price:.2f}",
                distance_pct=distance_pct
            )
        
        if abs(distance_pct) < self.config.liquidation_warning_pct:
            print(f"⚠️  WARNING: Position {distance_pct:.2f}% from liquidation")
            print(f"   Current: ${current_price:.2f}, Liquidation: ${liquidation_price:.2f}")
    
    def check_daily_limits(self):
        """
        Check if daily trading limits have been hit.
        
        Raises:
            RiskLimitError: If daily limits exceeded
        """
        self._reset_daily_limits_if_needed()
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.daily_loss_limit_usd:
            raise RiskLimitError(
                f"Daily loss limit exceeded: ${self.daily_pnl:.2f} < -${self.config.daily_loss_limit_usd:.2f}",
                limit_type="daily_loss",
                current_value=self.daily_pnl,
                max_value=-self.config.daily_loss_limit_usd
            )
        
        # Check daily trade count
        if self.daily_trades >= self.config.max_daily_trades:
            raise RiskLimitError(
                f"Daily trade limit exceeded: {self.daily_trades} >= {self.config.max_daily_trades}",
                limit_type="daily_trades",
                current_value=self.daily_trades,
                max_value=self.config.max_daily_trades
            )
    
    def pre_order_check(
        self,
        symbol: str,
        side: str,
        size: int,
        leverage: int,
        current_price: float = None,
        position: Optional[Dict] = None
    ):
        """
        Comprehensive pre-order validation.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            size: Order size
            leverage: Position leverage
            current_price: Current market price (optional)
            position: Current position data (optional)
        
        Raises:
            ValidationError: If basic validation fails
            RiskLimitError: If risk limits would be exceeded
            LiquidationRiskError: If position near liquidation
        """
        # Basic validation
        self.validate_symbol(symbol)
        self.validate_side(side)
        self.validate_size(size)
        self.validate_leverage(leverage)
        
        # Risk checks
        self.check_daily_limits()
        
        # Position-specific checks
        order_qty = size if side == 'buy' else -size
        self.check_position_limits(symbol, order_qty, position)
        
        # Check liquidation risk if position exists and price provided
        if position and current_price:
            self.check_liquidation_risk(symbol, current_price, position)
        
        print(f"✅ Risk check passed for {side} {size} {symbol} @ {leverage}x leverage")
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking."""
        self._reset_daily_limits_if_needed()
        self.daily_pnl += pnl
    
    def record_trade(self):
        """Record that a trade was executed."""
        self._reset_daily_limits_if_needed()
        self.daily_trades += 1
    
    def update_position(self, symbol: str, position: Dict):
        """Update tracked position data."""
        self.open_positions[symbol] = position
    
    def get_risk_stats(self) -> dict:
        """Get risk management statistics."""
        self._reset_daily_limits_if_needed()
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_loss_limit': self.config.daily_loss_limit_usd,
            'daily_trades_limit': self.config.max_daily_trades,
            'open_positions': len([p for p in self.open_positions.values() if p.get('currentQty', 0) != 0]),
            'max_leverage': self.config.max_leverage
        }
