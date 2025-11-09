"""
Position Manager - Paper and Live Trading

Handles position tracking for both paper trading (fake) and live trading (real).
Replicates GPU kernel position logic exactly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import time

from ..utils.validation import log_info, log_error, log_warning


@dataclass
class Position:
    """Single position (matches GPU kernel Position struct)."""
    entry_price: float
    size: float  # Position size in contracts
    side: int  # 1 = long, -1 = short
    leverage: int
    tp_price: float  # Take profit price
    sl_price: float  # Stop loss price
    entry_time: float  # Timestamp
    pnl: float = 0.0
    fees_paid: float = 0.0
    is_liquidated: bool = False
    exit_price: float = 0.0
    exit_time: float = 0.0
    
    # Live trading specific
    order_id: Optional[str] = None  # Exchange order ID
    position_id: Optional[str] = None  # Exchange position ID


@dataclass
class PositionSummary:
    """Real-time position summary for dashboard."""
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    win_rate: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    current_balance: float = 0.0


class PositionManager(ABC):
    """Abstract base class for position management."""
    
    def __init__(self, initial_balance: float, max_positions: int = 100):
        """Initialize position manager."""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_positions = max_positions
        
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Stats
        self.total_fees = 0.0
        self.total_pnl = 0.0
    
    @abstractmethod
    def open_position(
        self,
        side: int,
        price: float,
        size: float,
        leverage: int,
        tp_price: float,
        sl_price: float
    ) -> Optional[Position]:
        """Open a new position."""
        pass
    
    @abstractmethod
    def close_position(self, position: Position, price: float, reason: str = "manual") -> bool:
        """Close an existing position."""
        pass
    
    @abstractmethod
    def update_positions(self, current_price: float) -> List[Position]:
        """Update all positions with current price, check TP/SL."""
        pass
    
    def get_summary(self, current_price: float) -> PositionSummary:
        """Get position summary for dashboard."""
        summary = PositionSummary()
        
        summary.total_positions = len(self.open_positions) + len(self.closed_positions)
        summary.open_positions = len(self.open_positions)
        summary.closed_positions = len(self.closed_positions)
        summary.total_fees = self.total_fees
        summary.current_balance = self.current_balance
        
        # Calculate unrealized PnL
        unrealized = 0.0
        for pos in self.open_positions:
            if pos.side == 1:  # Long
                pnl = (current_price - pos.entry_price) / pos.entry_price * pos.leverage
            else:  # Short
                pnl = (pos.entry_price - current_price) / pos.entry_price * pos.leverage
            
            pnl_value = pnl * (pos.size * pos.entry_price)
            unrealized += pnl_value
        
        summary.unrealized_pnl = unrealized
        
        # Calculate realized PnL
        realized = sum(pos.pnl for pos in self.closed_positions)
        summary.realized_pnl = realized
        summary.total_pnl = realized + unrealized
        
        # Win rate
        if self.closed_positions:
            wins = sum(1 for pos in self.closed_positions if pos.pnl > 0)
            summary.win_rate = wins / len(self.closed_positions)
            
            winning_positions = [pos.pnl for pos in self.closed_positions if pos.pnl > 0]
            losing_positions = [pos.pnl for pos in self.closed_positions if pos.pnl < 0]
            
            if winning_positions:
                summary.largest_win = max(winning_positions)
            if losing_positions:
                summary.largest_loss = min(losing_positions)
        
        return summary
    
    def calculate_position_size(self, price: float, leverage: int, risk_percent: float = 0.02) -> float:
        """
        Calculate position size based on available balance and risk.
        
        Args:
            price: Current price
            leverage: Leverage to use
            risk_percent: Percent of balance to risk (default 2%)
        
        Returns:
            Position size in contracts
        """
        risk_amount = self.current_balance * risk_percent
        size = (risk_amount * leverage) / price
        return size


class PaperPositionManager(PositionManager):
    """Paper trading position manager (simulated positions)."""
    
    def __init__(self, initial_balance: float, max_positions: int = 100):
        """Initialize paper trading manager."""
        super().__init__(initial_balance, max_positions)
        
        # Kucoin Futures fees
        self.maker_fee = 0.0002  # 0.02%
        self.taker_fee = 0.0006  # 0.06%
        self.slippage = 0.001  # 0.1%
        
        log_info(f"📄 Paper Trading initialized with ${initial_balance:,.2f}")
    
    def open_position(
        self,
        side: int,
        price: float,
        size: float,
        leverage: int,
        tp_price: float,
        sl_price: float
    ) -> Optional[Position]:
        """Open simulated position."""
        if len(self.open_positions) >= self.max_positions:
            log_warning(f"Cannot open position: max {self.max_positions} positions reached")
            return None
        
        # Apply slippage (worse fill)
        fill_price = price * (1 + self.slippage * side)
        
        # Calculate fees (taker fee for market order)
        position_value = size * fill_price
        entry_fee = position_value * self.taker_fee
        
        # Check if enough balance
        margin_required = position_value / leverage
        if margin_required + entry_fee > self.current_balance:
            log_warning(f"Insufficient balance: need ${margin_required + entry_fee:.2f}, have ${self.current_balance:.2f}")
            return None
        
        # Create position
        position = Position(
            entry_price=fill_price,
            size=size,
            side=side,
            leverage=leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            entry_time=time.time(),
            fees_paid=entry_fee
        )
        
        # Update balance
        self.current_balance -= entry_fee
        self.total_fees += entry_fee
        
        self.open_positions.append(position)
        
        side_str = "LONG" if side == 1 else "SHORT"
        log_info(f"📄 PAPER {side_str}: {size:.4f} @ ${fill_price:.2f} (Lev: {leverage}x, TP: ${tp_price:.2f}, SL: ${sl_price:.2f})")
        
        return position
    
    def close_position(self, position: Position, price: float, reason: str = "manual") -> bool:
        """Close simulated position."""
        if position not in self.open_positions:
            return False
        
        # Apply slippage (worse fill)
        fill_price = price * (1 - self.slippage * position.side)
        
        # Calculate PnL
        if position.side == 1:  # Long
            pnl_pct = (fill_price - position.entry_price) / position.entry_price
        else:  # Short
            pnl_pct = (position.entry_price - fill_price) / position.entry_price
        
        # Apply leverage
        pnl_pct *= position.leverage
        
        # Calculate PnL value
        position_value = position.size * position.entry_price
        pnl_value = pnl_pct * position_value
        
        # Exit fee
        exit_fee = (position.size * fill_price) * self.taker_fee
        total_fees = position.fees_paid + exit_fee
        
        # Net PnL
        net_pnl = pnl_value - exit_fee
        
        # Update position
        position.exit_price = fill_price
        position.exit_time = time.time()
        position.pnl = net_pnl
        position.fees_paid = total_fees
        
        # Check liquidation
        if pnl_pct <= (-1.0 / position.leverage):
            position.is_liquidated = True
            net_pnl = -position_value / position.leverage  # Lose all margin
            reason = "liquidation"
        
        # Update balance
        self.current_balance += net_pnl
        self.total_fees += exit_fee
        self.total_pnl += net_pnl
        
        # Move to closed
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        
        side_str = "LONG" if position.side == 1 else "SHORT"
        pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
        log_info(f"📄 CLOSE {side_str} @ ${fill_price:.2f}: {pnl_str} ({reason})")
        
        return True
    
    def update_positions(self, current_price: float) -> List[Position]:
        """Update positions and check TP/SL."""
        closed = []
        
        for position in self.open_positions[:]:  # Copy list to allow modification
            # Check TP/SL
            hit_tp = False
            hit_sl = False
            
            if position.side == 1:  # Long
                if current_price >= position.tp_price:
                    hit_tp = True
                elif current_price <= position.sl_price:
                    hit_sl = True
            else:  # Short
                if current_price <= position.tp_price:
                    hit_tp = True
                elif current_price >= position.sl_price:
                    hit_sl = True
            
            if hit_tp:
                self.close_position(position, current_price, "take_profit")
                closed.append(position)
            elif hit_sl:
                self.close_position(position, current_price, "stop_loss")
                closed.append(position)
        
        return closed


class LivePositionManager(PositionManager):
    """Live trading position manager (real positions on exchange)."""
    
    def __init__(self, initial_balance: float, kucoin_client, max_positions: int = 100):
        """
        Initialize live trading manager.
        
        Args:
            initial_balance: Starting balance
            kucoin_client: Authenticated Kucoin client
            max_positions: Max concurrent positions
        """
        super().__init__(initial_balance, max_positions)
        self.client = kucoin_client
        
        # Kucoin Futures fees (will be fetched from exchange)
        self.maker_fee = 0.0002
        self.taker_fee = 0.0006
        
        log_info(f"💰 LIVE Trading initialized with ${initial_balance:,.2f}")
        log_warning("⚠️  REAL MONEY MODE - All trades will use real funds!")
    
    def open_position(
        self,
        side: int,
        price: float,
        size: float,
        leverage: int,
        tp_price: float,
        sl_price: float
    ) -> Optional[Position]:
        """Open real position on exchange."""
        if len(self.open_positions) >= self.max_positions:
            log_warning(f"Cannot open position: max {self.max_positions} positions reached")
            return None
        
        try:
            # Place market order
            side_str = "buy" if side == 1 else "sell"
            
            order = self.client.create_market_order(
                symbol='XBTUSDTM',  # Will be configurable
                side=side_str,
                size=int(size),
                leverage=leverage
            )
            
            # Set TP/SL orders
            # TODO: Implement TP/SL order placement
            
            # Create position tracking
            position = Position(
                entry_price=float(order.get('price', price)),
                size=size,
                side=side,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_time=time.time(),
                order_id=order.get('orderId'),
                position_id=order.get('id')
            )
            
            self.open_positions.append(position)
            
            log_info(f"💰 LIVE {side_str.upper()}: {size:.4f} @ ${position.entry_price:.2f} [Order: {position.order_id}]")
            
            return position
            
        except Exception as e:
            log_error(f"Failed to open position: {e}")
            return None
    
    def close_position(self, position: Position, price: float, reason: str = "manual") -> bool:
        """Close real position on exchange."""
        if position not in self.open_positions:
            return False
        
        try:
            # Place closing order (opposite side)
            side_str = "sell" if position.side == 1 else "buy"
            
            order = self.client.create_market_order(
                symbol='XBTUSDTM',
                side=side_str,
                size=int(position.size),
                reduce_only=True
            )
            
            fill_price = float(order.get('price', price))
            
            # Calculate PnL (will fetch from exchange)
            # TODO: Get actual PnL from exchange
            
            position.exit_price = fill_price
            position.exit_time = time.time()
            
            self.open_positions.remove(position)
            self.closed_positions.append(position)
            
            log_info(f"💰 CLOSE {side_str.upper()} @ ${fill_price:.2f} ({reason}) [Order: {order.get('orderId')}]")
            
            return True
            
        except Exception as e:
            log_error(f"Failed to close position: {e}")
            return False
    
    def update_positions(self, current_price: float) -> List[Position]:
        """Update positions from exchange."""
        # TODO: Fetch actual positions from exchange
        # TODO: Check TP/SL triggers
        
        return []
