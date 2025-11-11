"""
Bot Loader - Improved Bot Loading with Fitness Sorting and Validation

Loads saved bots from evolution results with:
- Fitness-based sorting (best first)
- Comprehensive validation
- Detailed bot statistics
- Multiple selection methods
"""

import glob
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from ..bot_generator.compact_generator import CompactBotConfig
from .validation import log_info, log_warning, log_error


class BotLoadError(Exception):
    """Custom exception for bot loading errors."""
    pass


class BotLoader:
    """
    Load and manage saved bot configurations.
    
    Features:
    - Automatic fitness sorting (highest first)
    - Bot validation (check for required fields)
    - Filter by minimum fitness threshold
    - Search by bot ID
    - Display detailed statistics
    """
    
    def __init__(self, bot_directory: str = "bots"):
        """
        Initialize bot loader.
        
        Args:
            bot_directory: Root directory containing saved bots
        """
        self.bot_directory = Path(bot_directory)
        self.bots: List[Dict] = []
        self.bot_files: List[Path] = []
    
    def discover_bots(self, pair: str = "BTC_USDT", timeframe: str = "1m") -> int:
        """
        Discover all saved bot files for a specific pair/timeframe.
        
        Args:
            pair: Trading pair (e.g., "BTC_USDT")
            timeframe: Timeframe (e.g., "1m")
        
        Returns:
            Number of bots discovered
        """
        # Search pattern: bots/{pair}/{timeframe}/bot_*.json
        search_path = self.bot_directory / pair / timeframe / "bot_*.json"
        self.bot_files = list(Path().glob(str(search_path)))
        
        log_info(f"Discovered {len(self.bot_files)} bot files in {pair}/{timeframe}")
        return len(self.bot_files)
    
    def load_all_bots(self, min_fitness: float = 0.0) -> List[Dict]:
        """
        Load all discovered bots with fitness filtering.
        
        Args:
            min_fitness: Minimum fitness threshold (default: 0.0 = no filter)
        
        Returns:
            List of bot data dicts, sorted by fitness (highest first)
        """
        self.bots = []
        
        for bot_file in self.bot_files:
            try:
                with open(bot_file, 'r') as f:
                    bot_data = json.load(f)
                
                # Validate required fields
                if not self._validate_bot_data(bot_data):
                    log_warning(f"Invalid bot file: {bot_file.name}")
                    continue
                
                # Filter by fitness
                fitness = bot_data.get('fitness_score', 0.0)
                if fitness < min_fitness:
                    continue
                
                # Add file path for reference
                bot_data['_file_path'] = str(bot_file)
                bot_data['_file_name'] = bot_file.name
                
                self.bots.append(bot_data)
                
            except Exception as e:
                log_warning(f"Failed to load {bot_file.name}: {e}")
        
        # Sort by fitness (highest first)
        self.bots.sort(key=lambda x: x.get('fitness_score', 0.0), reverse=True)
        
        log_info(f"Loaded {len(self.bots)} valid bots (min fitness: {min_fitness:.2f})")
        return self.bots
    
    def _validate_bot_data(self, bot_data: Dict) -> bool:
        """
        Validate bot data has all required fields.
        
        Args:
            bot_data: Bot data dict
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'bot_id',
            'num_indicators',
            'indicator_indices',
            'indicator_params',
            'risk_strategy',
            'risk_param',
            'tp_multiplier',
            'sl_multiplier',
            'leverage'
        ]
        
        for field in required_fields:
            if field not in bot_data:
                return False
        
        # Validate data types
        if not isinstance(bot_data['num_indicators'], int):
            return False
        if bot_data['num_indicators'] < 1 or bot_data['num_indicators'] > 8:
            return False
        if not isinstance(bot_data['indicator_indices'], list):
            return False
        if not isinstance(bot_data['indicator_params'], list):
            return False
        if not isinstance(bot_data['leverage'], int):
            return False
        if bot_data['leverage'] < 1 or bot_data['leverage'] > 125:
            return False
        
        return True
    
    def get_top_bots(self, count: int = 10) -> List[Dict]:
        """
        Get top N bots by fitness.
        
        Args:
            count: Number of bots to return
        
        Returns:
            List of top N bots
        """
        return self.bots[:count]
    
    def find_bot_by_id(self, bot_id: int) -> Optional[Dict]:
        """
        Find bot by ID.
        
        Args:
            bot_id: Bot ID to search for
        
        Returns:
            Bot data dict or None if not found
        """
        for bot in self.bots:
            if bot.get('bot_id') == bot_id:
                return bot
        return None
    
    def display_bot_list(self, max_display: int = 20):
        """
        Display formatted list of bots.
        
        Args:
            max_display: Maximum number of bots to display
        """
        if len(self.bots) == 0:
            log_warning("No bots loaded")
            return
        
        print("\n" + "="*100)
        print(f"{'#':<5} {'Bot ID':<10} {'Fitness':<12} {'Indicators':<12} {'Lev':<6} {'TP%':<8} {'SL%':<8} {'Risk Strategy':<20}")
        print("-"*100)
        
        for i, bot in enumerate(self.bots[:max_display], 1):
            bot_id = bot.get('bot_id', 0)
            fitness = bot.get('fitness_score', 0.0)
            num_indicators = bot.get('num_indicators', 0)
            leverage = bot.get('leverage', 0)
            tp_pct = bot.get('tp_multiplier', 0.0) * 100
            sl_pct = bot.get('sl_multiplier', 0.0) * 100
            risk_strategy = bot.get('risk_strategy', 0)
            survival = bot.get('survival_generations', 0)
            
            # Get risk strategy name
            from ..live_trading.gpu_kernel_port import RISK_STRATEGY_NAMES
            risk_name = RISK_STRATEGY_NAMES.get(risk_strategy, f"Strategy {risk_strategy}")[:19]
            
            # Color code by fitness
            fitness_str = f"{fitness:>10.2f}"
            if fitness > 2.0:
                fitness_str = f"🟢 {fitness_str}"
            elif fitness > 1.0:
                fitness_str = f"🟡 {fitness_str}"
            else:
                fitness_str = f"🔴 {fitness_str}"
            
            print(f"{i:<5} {bot_id:<10} {fitness_str:<12} {num_indicators:<12} {leverage:<6}x {tp_pct:<8.2f} {sl_pct:<8.2f} {risk_name:<20}")
            
            # Show survival if significant
            if survival > 0:
                print(f"      └─ Survived {survival} generations")
        
        if len(self.bots) > max_display:
            print(f"\n... and {len(self.bots) - max_display} more bots")
        
        print("="*100 + "\n")
    
    def load_bot_config(self, bot_data: Dict) -> CompactBotConfig:
        """
        Load bot configuration from data dict.
        
        Args:
            bot_data: Bot data dict
        
        Returns:
            CompactBotConfig object
        
        Raises:
            BotLoadError: If loading fails
        """
        try:
            bot_config = CompactBotConfig.from_dict(bot_data)
            
            log_info(f"✅ Loaded bot {bot_config.bot_id}")
            log_info(f"   Fitness: {bot_data.get('fitness_score', 0.0):.4f}")
            log_info(f"   Indicators: {bot_config.num_indicators}")
            log_info(f"   Leverage: {bot_config.leverage}x")
            log_info(f"   TP: {bot_config.tp_multiplier*100:.2f}%, SL: {bot_config.sl_multiplier*100:.2f}%")
            
            from ..live_trading.gpu_kernel_port import RISK_STRATEGY_NAMES
            risk_name = RISK_STRATEGY_NAMES.get(bot_config.risk_strategy, "Unknown")
            log_info(f"   Risk Strategy: {risk_name} (param: {bot_config.risk_param:.4f})")
            
            return bot_config
            
        except Exception as e:
            raise BotLoadError(f"Failed to load bot config: {e}")
    
    def interactive_selection(self) -> Optional[CompactBotConfig]:
        """
        Interactive bot selection with user input.
        
        Returns:
            Selected CompactBotConfig or None if cancelled
        """
        if len(self.bots) == 0:
            log_error("No bots available for selection")
            return None
        
        # Display bots
        self.display_bot_list()
        
        # Get user selection
        print("Selection Options:")
        print("  1. Select by number (1-N)")
        print("  2. Select by bot ID")
        print("  3. Use top bot (highest fitness)")
        print("  0. Cancel")
        
        choice = input("\nSelect option [0-3]: ").strip()
        
        if choice == "0":
            log_info("Bot selection cancelled")
            return None
        
        elif choice == "1":
            # Select by number
            max_num = min(len(self.bots), 20)
            num = int(input(f"Enter bot number [1-{max_num}]: ").strip())
            
            if num < 1 or num > max_num:
                log_error(f"Invalid selection: {num}")
                return None
            
            bot_data = self.bots[num - 1]
            return self.load_bot_config(bot_data)
        
        elif choice == "2":
            # Select by bot ID
            bot_id = int(input("Enter bot ID: ").strip())
            bot_data = self.find_bot_by_id(bot_id)
            
            if bot_data is None:
                log_error(f"Bot ID {bot_id} not found")
                return None
            
            return self.load_bot_config(bot_data)
        
        elif choice == "3":
            # Use top bot
            bot_data = self.bots[0]
            log_info(f"Selected top bot (ID: {bot_data['bot_id']}, Fitness: {bot_data['fitness_score']:.4f})")
            return self.load_bot_config(bot_data)
        
        else:
            log_error(f"Invalid choice: {choice}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded bots.
        
        Returns:
            Dict with statistics
        """
        if len(self.bots) == 0:
            return {}
        
        fitnesses = [b.get('fitness_score', 0.0) for b in self.bots]
        leverages = [b.get('leverage', 0) for b in self.bots]
        num_indicators = [b.get('num_indicators', 0) for b in self.bots]
        
        return {
            'total_bots': len(self.bots),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'avg_leverage': np.mean(leverages),
            'avg_indicators': np.mean(num_indicators),
            'top_bot_id': self.bots[0].get('bot_id') if len(self.bots) > 0 else None
        }
    
    def display_statistics(self):
        """Display bot statistics."""
        stats = self.get_statistics()
        
        if not stats:
            log_warning("No statistics available (no bots loaded)")
            return
        
        print("\n" + "="*60)
        print("BOT STATISTICS")
        print("="*60)
        print(f"Total Bots:           {stats['total_bots']}")
        print(f"Average Fitness:      {stats['avg_fitness']:.4f}")
        print(f"Max Fitness:          {stats['max_fitness']:.4f}")
        print(f"Min Fitness:          {stats['min_fitness']:.4f}")
        print(f"Average Leverage:     {stats['avg_leverage']:.1f}x")
        print(f"Average Indicators:   {stats['avg_indicators']:.1f}")
        print(f"Top Bot ID:           {stats['top_bot_id']}")
        print("="*60 + "\n")


# Convenience functions for quick loading
def load_best_bot(pair: str = "BTC_USDT", timeframe: str = "1m", min_fitness: float = 0.0) -> Optional[CompactBotConfig]:
    """
    Load the best bot for a given pair/timeframe.
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        min_fitness: Minimum fitness threshold
    
    Returns:
        CompactBotConfig of best bot or None
    """
    loader = BotLoader()
    loader.discover_bots(pair, timeframe)
    loader.load_all_bots(min_fitness)
    
    if len(loader.bots) == 0:
        log_error(f"No bots found for {pair}/{timeframe} with fitness >= {min_fitness}")
        return None
    
    bot_data = loader.bots[0]
    log_info(f"Loaded best bot: ID {bot_data['bot_id']}, Fitness {bot_data['fitness_score']:.4f}")
    
    return loader.load_bot_config(bot_data)


def load_bot_by_id(bot_id: int, pair: str = "BTC_USDT", timeframe: str = "1m") -> Optional[CompactBotConfig]:
    """
    Load bot by ID.
    
    Args:
        bot_id: Bot ID
        pair: Trading pair
        timeframe: Timeframe
    
    Returns:
        CompactBotConfig or None
    """
    loader = BotLoader()
    loader.discover_bots(pair, timeframe)
    loader.load_all_bots()
    
    bot_data = loader.find_bot_by_id(bot_id)
    if bot_data is None:
        log_error(f"Bot ID {bot_id} not found")
        return None
    
    return loader.load_bot_config(bot_data)
