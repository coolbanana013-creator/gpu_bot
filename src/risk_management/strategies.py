"""
Position sizing and risk management strategies.
Implements 15+ strategies with parameter generation and validation.
"""
import numpy as np
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from ..utils.validation import validate_float, log_debug


class RiskStrategyType(Enum):
    """Enumeration of all risk management strategy types."""
    FIXED_PERCENT = "FIXED_PERCENT"
    FIXED_AMOUNT = "FIXED_AMOUNT"
    KELLY_FULL = "KELLY_FULL"
    KELLY_HALF = "KELLY_HALF"
    KELLY_FRACTIONAL = "KELLY_FRACTIONAL"
    VOLATILITY_ATR = "VOLATILITY_ATR"
    VOLATILITY_STDDEV = "VOLATILITY_STDDEV"
    MARTINGALE = "MARTINGALE"
    ANTI_MARTINGALE = "ANTI_MARTINGALE"
    EQUITY_CURVE = "EQUITY_CURVE"
    WIN_STREAK = "WIN_STREAK"
    LOSS_STREAK = "LOSS_STREAK"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    RISK_REWARD_RATIO = "RISK_REWARD_RATIO"
    PROGRESSIVE = "PROGRESSIVE"


@dataclass
class RiskStrategyParams:
    """Container for risk strategy parameters."""
    strategy_type: RiskStrategyType
    params: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.strategy_type.value,
            "params": self.params
        }


class BaseRiskStrategy:
    """Base class for all risk strategies."""
    
    def __init__(self, strategy_type: RiskStrategyType):
        self.strategy_type = strategy_type
    
    def generate_params(self) -> Dict[str, Any]:
        """Generate random valid parameters. Override in subclasses."""
        raise NotImplementedError
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Calculate position size based on strategy.
        Override in subclasses.
        
        Returns:
            Position size as fraction of balance (0.0-1.0+)
        """
        raise NotImplementedError


# ============================================================================
# FIXED SIZE STRATEGIES
# ============================================================================

class FixedPercentStrategy(BaseRiskStrategy):
    """Fixed percentage of balance per trade."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.FIXED_PERCENT)
    
    def generate_params(self) -> Dict[str, Any]:
        """Fixed percent: 1-10% of balance."""
        percent = np.random.uniform(1.0, 10.0)
        return {"percent": float(percent)}
    
    def calculate_position_size(self, balance: float, params: Dict[str, Any], **kwargs) -> float:
        """Calculate position size."""
        return params["percent"] / 100.0


class FixedAmountStrategy(BaseRiskStrategy):
    """Fixed dollar amount per trade."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.FIXED_AMOUNT)
    
    def generate_params(self) -> Dict[str, Any]:
        """Fixed amount: 10-100 units."""
        amount = np.random.uniform(10.0, 100.0)
        return {"amount": float(amount)}
    
    def calculate_position_size(self, balance: float, params: Dict[str, Any], **kwargs) -> float:
        """Calculate position size as fraction of balance."""
        if balance <= 0:
            return 0.0
        return min(params["amount"] / balance, 1.0)


# ============================================================================
# KELLY CRITERION STRATEGIES
# ============================================================================

class KellyFullStrategy(BaseRiskStrategy):
    """Full Kelly Criterion."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.KELLY_FULL)
    
    def generate_params(self) -> Dict[str, Any]:
        """Kelly has no additional params (uses win rate and avg win/loss)."""
        return {}
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        **kwargs
    ) -> float:
        """
        Calculate Kelly fraction: f = (p*b - q) / b
        where p = win rate, q = loss rate, b = avg_win/avg_loss
        """
        if avg_loss == 0:
            return 0.05  # Fallback
        
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_fraction = (win_rate * b - q) / b
        
        # Clamp to reasonable range
        return max(0.0, min(kelly_fraction, 0.25))


class KellyHalfStrategy(BaseRiskStrategy):
    """Half Kelly Criterion (more conservative)."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.KELLY_HALF)
    
    def generate_params(self) -> Dict[str, Any]:
        return {}
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        **kwargs
    ) -> float:
        """Half Kelly fraction."""
        if avg_loss == 0:
            return 0.025
        
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_fraction = (win_rate * b - q) / b
        
        return max(0.0, min(kelly_fraction * 0.5, 0.125))


class KellyFractionalStrategy(BaseRiskStrategy):
    """Fractional Kelly Criterion."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.KELLY_FRACTIONAL)
    
    def generate_params(self) -> Dict[str, Any]:
        """Fraction: 0.1-1.0 of Kelly."""
        fraction = np.random.uniform(0.1, 1.0)
        return {"fraction": float(fraction)}
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        **kwargs
    ) -> float:
        """Fractional Kelly."""
        if avg_loss == 0:
            return 0.05 * params["fraction"]
        
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly_fraction = (win_rate * b - q) / b
        
        return max(0.0, min(kelly_fraction * params["fraction"], 0.25))


# ============================================================================
# VOLATILITY-BASED STRATEGIES
# ============================================================================

class VolatilityATRStrategy(BaseRiskStrategy):
    """Position size based on ATR volatility."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.VOLATILITY_ATR)
    
    def generate_params(self) -> Dict[str, Any]:
        """ATR multiplier: 0.5-5.0."""
        multiplier = np.random.uniform(0.5, 5.0)
        base_percent = np.random.uniform(2.0, 10.0)
        return {
            "multiplier": float(multiplier),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        atr: float = 1.0,
        price: float = 100.0,
        **kwargs
    ) -> float:
        """Size inversely proportional to ATR."""
        if atr == 0 or price == 0:
            return params["base_percent"] / 100.0
        
        # Higher ATR = smaller position
        atr_percent = (atr / price) * 100.0
        adjusted_percent = params["base_percent"] / (1.0 + atr_percent / params["multiplier"])
        
        return max(0.01, min(adjusted_percent / 100.0, 0.20))


class VolatilityStdDevStrategy(BaseRiskStrategy):
    """Position size based on standard deviation."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.VOLATILITY_STDDEV)
    
    def generate_params(self) -> Dict[str, Any]:
        """StdDev multiplier: 1.0-5.0."""
        multiplier = np.random.uniform(1.0, 5.0)
        base_percent = np.random.uniform(2.0, 10.0)
        return {
            "multiplier": float(multiplier),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        stddev: float = 1.0,
        **kwargs
    ) -> float:
        """Size inversely proportional to volatility."""
        adjusted_percent = params["base_percent"] / (1.0 + stddev * params["multiplier"])
        return max(0.01, min(adjusted_percent / 100.0, 0.20))


# ============================================================================
# MARTINGALE STRATEGIES
# ============================================================================

class MartingaleStrategy(BaseRiskStrategy):
    """Increase position after losses (dangerous!)."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.MARTINGALE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Multiplier: 1.1-3.0, max steps: 3-10."""
        multiplier = np.random.uniform(1.1, 3.0)
        max_steps = np.random.randint(3, 11)
        base_percent = np.random.uniform(1.0, 5.0)
        return {
            "multiplier": float(multiplier),
            "max_steps": int(max_steps),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        consecutive_losses: int = 0,
        **kwargs
    ) -> float:
        """Increase size after losses."""
        steps = min(consecutive_losses, params["max_steps"])
        size_percent = params["base_percent"] * (params["multiplier"] ** steps)
        return min(size_percent / 100.0, 0.50)


class AntiMartingaleStrategy(BaseRiskStrategy):
    """Increase position after wins."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.ANTI_MARTINGALE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Multiplier: 1.1-3.0, max steps: 3-10."""
        multiplier = np.random.uniform(1.1, 3.0)
        max_steps = np.random.randint(3, 11)
        base_percent = np.random.uniform(1.0, 5.0)
        return {
            "multiplier": float(multiplier),
            "max_steps": int(max_steps),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        consecutive_wins: int = 0,
        **kwargs
    ) -> float:
        """Increase size after wins."""
        steps = min(consecutive_wins, params["max_steps"])
        size_percent = params["base_percent"] * (params["multiplier"] ** steps)
        return min(size_percent / 100.0, 0.50)


# ============================================================================
# EQUITY CURVE STRATEGIES
# ============================================================================

class EquityCurveStrategy(BaseRiskStrategy):
    """Adjust size based on equity curve."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.EQUITY_CURVE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Smoothing factor: 0.1-0.9."""
        smoothing = np.random.uniform(0.1, 0.9)
        base_percent = np.random.uniform(2.0, 10.0)
        return {
            "smoothing": float(smoothing),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        initial_balance: float = 100.0,
        **kwargs
    ) -> float:
        """Size based on balance change."""
        if initial_balance <= 0:
            return params["base_percent"] / 100.0
        
        equity_ratio = balance / initial_balance
        smoothed_ratio = 1.0 + (equity_ratio - 1.0) * params["smoothing"]
        
        adjusted_percent = params["base_percent"] * smoothed_ratio
        return max(0.01, min(adjusted_percent / 100.0, 0.25))


class WinStreakStrategy(BaseRiskStrategy):
    """Increase size on win streaks."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.WIN_STREAK)
    
    def generate_params(self) -> Dict[str, Any]:
        """Streak threshold: 2-10."""
        threshold = np.random.randint(2, 11)
        boost_percent = np.random.uniform(10.0, 50.0)
        base_percent = np.random.uniform(2.0, 8.0)
        return {
            "threshold": int(threshold),
            "boost_percent": float(boost_percent),
            "base_percent": float(base_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        consecutive_wins: int = 0,
        **kwargs
    ) -> float:
        """Boost size on win streak."""
        if consecutive_wins >= params["threshold"]:
            adjusted = params["base_percent"] * (1.0 + params["boost_percent"] / 100.0)
        else:
            adjusted = params["base_percent"]
        
        return min(adjusted / 100.0, 0.30)


class ProgressiveStrategy(BaseRiskStrategy):
    """Progressive position sizing."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.PROGRESSIVE)
    
    def generate_params(self) -> Dict[str, Any]:
        """Start/end percentages."""
        start_percent = np.random.uniform(1.0, 5.0)
        end_percent = np.random.uniform(5.0, 15.0)
        return {
            "start_percent": float(start_percent),
            "end_percent": float(end_percent)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        trade_number: int = 0,
        total_trades: int = 100,
        **kwargs
    ) -> float:
        """Gradually increase size over time."""
        if total_trades <= 0:
            return params["start_percent"] / 100.0
        
        progress = min(trade_number / total_trades, 1.0)
        current_percent = params["start_percent"] + (params["end_percent"] - params["start_percent"]) * progress
        
        return min(current_percent / 100.0, 0.25)


class LossStreakStrategy(BaseRiskStrategy):
    """Reduce position size during losing streaks."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.LOSS_STREAK)
    
    def generate_params(self) -> Dict[str, Any]:
        """Base percent and reduction factor."""
        base_percent = np.random.uniform(3.0, 8.0)
        reduction_factor = np.random.uniform(0.3, 0.7)  # Reduce by 30-70%
        max_losses = int(np.random.randint(3, 8))  # Consider last 3-7 trades
        return {
            "base_percent": float(base_percent),
            "reduction_factor": float(reduction_factor),
            "max_losses": int(max_losses)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        consecutive_losses: int = 0,
        **kwargs
    ) -> float:
        """Reduce size based on consecutive losses."""
        max_losses = params.get("max_losses", 5)
        loss_ratio = min(consecutive_losses / max_losses, 1.0)
        reduction = 1.0 - (loss_ratio * (1.0 - params["reduction_factor"]))
        adjusted = params["base_percent"] * reduction
        return min(adjusted / 100.0, 0.30)


class MaxDrawdownStrategy(BaseRiskStrategy):
    """Adjust position size based on current drawdown."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.MAX_DRAWDOWN)
    
    def generate_params(self) -> Dict[str, Any]:
        """Base percent and drawdown threshold."""
        base_percent = np.random.uniform(3.0, 8.0)
        max_drawdown_pct = np.random.uniform(10.0, 30.0)  # Max allowed drawdown 10-30%
        min_size_pct = np.random.uniform(1.0, 3.0)  # Minimum size during high drawdown
        return {
            "base_percent": float(base_percent),
            "max_drawdown_pct": float(max_drawdown_pct),
            "min_size_pct": float(min_size_pct)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        peak_balance: float = None,
        **kwargs
    ) -> float:
        """Reduce size during drawdown periods."""
        if peak_balance is None or peak_balance <= 0 or balance >= peak_balance:
            return params["base_percent"] / 100.0
        
        # Calculate current drawdown
        drawdown_pct = ((peak_balance - balance) / peak_balance) * 100.0
        max_dd = params["max_drawdown_pct"]
        
        if drawdown_pct >= max_dd:
            # At or beyond max drawdown, use minimum size
            return params["min_size_pct"] / 100.0
        else:
            # Scale linearly between base and min based on drawdown
            dd_ratio = drawdown_pct / max_dd
            adjusted = params["base_percent"] - (params["base_percent"] - params["min_size_pct"]) * dd_ratio
            return min(adjusted / 100.0, 0.30)


class RiskRewardRatioStrategy(BaseRiskStrategy):
    """Position size based on risk-reward ratio."""
    
    def __init__(self):
        super().__init__(RiskStrategyType.RISK_REWARD_RATIO)
    
    def generate_params(self) -> Dict[str, Any]:
        """Base risk percent and minimum risk-reward ratio."""
        base_risk_pct = np.random.uniform(1.0, 3.0)  # Risk per trade
        min_rr_ratio = np.random.uniform(1.5, 3.0)  # Minimum R:R
        max_position_pct = np.random.uniform(5.0, 15.0)  # Max position size
        return {
            "base_risk_pct": float(base_risk_pct),
            "min_rr_ratio": float(min_rr_ratio),
            "max_position_pct": float(max_position_pct)
        }
    
    def calculate_position_size(
        self,
        balance: float,
        params: Dict[str, Any],
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        **kwargs
    ) -> float:
        """Calculate size based on R:R ratio."""
        if stop_loss_pct <= 0:
            return params["base_risk_pct"] / 100.0
        
        # Calculate actual R:R ratio
        rr_ratio = take_profit_pct / stop_loss_pct
        
        # Only take trade if R:R meets minimum
        if rr_ratio < params["min_rr_ratio"]:
            return 0.01  # Minimal position if R:R is poor
        
        # Calculate position size based on risk amount
        risk_amount = balance * (params["base_risk_pct"] / 100.0)
        position_size = risk_amount / (stop_loss_pct / 100.0)
        
        # Convert to fraction and cap
        size_fraction = position_size / balance
        return min(size_fraction, params["max_position_pct"] / 100.0)


# ============================================================================
# RISK STRATEGY FACTORY
# ============================================================================

class RiskStrategyFactory:
    """
    Factory for creating and managing risk strategies.
    """
    
    STRATEGY_CLASSES = {
        RiskStrategyType.FIXED_PERCENT: FixedPercentStrategy,
        RiskStrategyType.FIXED_AMOUNT: FixedAmountStrategy,
        RiskStrategyType.KELLY_FULL: KellyFullStrategy,
        RiskStrategyType.KELLY_HALF: KellyHalfStrategy,
        RiskStrategyType.KELLY_FRACTIONAL: KellyFractionalStrategy,
        RiskStrategyType.VOLATILITY_ATR: VolatilityATRStrategy,
        RiskStrategyType.VOLATILITY_STDDEV: VolatilityStdDevStrategy,
        RiskStrategyType.MARTINGALE: MartingaleStrategy,
        RiskStrategyType.ANTI_MARTINGALE: AntiMartingaleStrategy,
        RiskStrategyType.EQUITY_CURVE: EquityCurveStrategy,
        RiskStrategyType.WIN_STREAK: WinStreakStrategy,
        RiskStrategyType.PROGRESSIVE: ProgressiveStrategy,
        RiskStrategyType.LOSS_STREAK: LossStreakStrategy,
        RiskStrategyType.MAX_DRAWDOWN: MaxDrawdownStrategy,
        RiskStrategyType.RISK_REWARD_RATIO: RiskRewardRatioStrategy,
    }
    
    @classmethod
    def get_all_strategy_types(cls) -> list:
        """Get list of all available strategy types."""
        return list(cls.STRATEGY_CLASSES.keys())
    
    @classmethod
    def create_strategy(cls, strategy_type: RiskStrategyType) -> BaseRiskStrategy:
        """Create a strategy instance."""
        if strategy_type not in cls.STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return cls.STRATEGY_CLASSES[strategy_type]()
    
    @classmethod
    def generate_random_strategy_params(cls, strategy_type: RiskStrategyType) -> RiskStrategyParams:
        """Generate random parameters for a strategy."""
        strategy = cls.create_strategy(strategy_type)
        params = strategy.generate_params()
        return RiskStrategyParams(strategy_type=strategy_type, params=params)
    
    @classmethod
    def calculate_position_size(
        cls,
        strategy_params: RiskStrategyParams,
        balance: float,
        **kwargs
    ) -> float:
        """Calculate position size using a strategy."""
        strategy = cls.create_strategy(strategy_params.strategy_type)
        return strategy.calculate_position_size(balance, strategy_params.params, **kwargs)
    
    @classmethod
    def get_strategy_count(cls) -> int:
        """Get total number of available strategies."""
        return len(cls.STRATEGY_CLASSES)


def calculate_average_position_size(
    strategies: list,
    balance: float,
    **kwargs
) -> float:
    """
    Calculate average position size from multiple strategies.
    
    Args:
        strategies: List of RiskStrategyParams
        balance: Current balance
        **kwargs: Additional parameters for strategies
        
    Returns:
        Average position size (0.0-1.0)
    """
    if not strategies:
        return 0.05  # Default 5%
    
    sizes = []
    for strategy_params in strategies:
        size = RiskStrategyFactory.calculate_position_size(strategy_params, balance, **kwargs)
        sizes.append(size)
    
    return np.mean(sizes)


log_debug(f"Risk strategy factory initialized with {RiskStrategyFactory.get_strategy_count()} strategies")
