"""
List available trained bots and their metadata.
Helps select the best bot for live trading.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def scan_bots_directory(base_dir: str = "bots") -> Dict[str, List[str]]:
    """Scan bots directory and categorize bots."""
    bots = {}
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Bots directory not found: {base_dir}")
        return bots
    
    # Scan for bot files
    for symbol_dir in base_path.iterdir():
        if not symbol_dir.is_dir():
            continue
        
        for timeframe_dir in symbol_dir.iterdir():
            if not timeframe_dir.is_dir():
                continue
            
            bot_files = list(timeframe_dir.glob("*.pkl"))
            if bot_files:
                key = f"{symbol_dir.name}/{timeframe_dir.name}"
                bots[key] = [str(f.relative_to(base_path)) for f in bot_files]
    
    return bots


def get_bot_metadata(bot_path: str) -> Optional[Dict]:
    """Try to extract metadata from bot file."""
    try:
        with open(bot_path, 'rb') as f:
            bot = pickle.load(f)
        
        # Try to get attributes
        metadata = {
            'type': type(bot).__name__,
            'file_size': os.path.getsize(bot_path),
        }
        
        # Try to get performance metrics if available
        if hasattr(bot, 'score_'):
            metadata['score'] = float(bot.score_)
        if hasattr(bot, 'n_estimators'):
            metadata['n_estimators'] = bot.n_estimators
        if hasattr(bot, 'max_depth'):
            metadata['max_depth'] = bot.max_depth
        
        return metadata
    except Exception as e:
        return {'error': str(e)}


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    """Main function."""
    print("="*80)
    print("🤖 TRAINED BOTS INVENTORY")
    print("="*80)
    
    # Scan bots directory
    bots = scan_bots_directory()
    
    if not bots:
        print("\n❌ No bots found in 'bots/' directory")
        print("\nℹ️  Make sure you have trained bots in:")
        print("   bots/BTC_USDT/1m/")
        print("   bots/ETH_USDT/1m/")
        print("   etc.")
        return
    
    # Display bots by category
    total_bots = 0
    for category, bot_files in sorted(bots.items()):
        print(f"\n📁 {category}")
        print("-" * 80)
        print(f"   Total Bots: {len(bot_files)}")
        total_bots += len(bot_files)
        
        # Show first few bots as examples
        for i, bot_file in enumerate(bot_files[:5]):
            bot_path = Path("bots") / bot_file
            metadata = get_bot_metadata(str(bot_path))
            
            if metadata:
                size = format_size(metadata.get('file_size', 0))
                bot_type = metadata.get('type', 'Unknown')
                print(f"   [{i+1}] {bot_file}")
                print(f"       Type: {bot_type} | Size: {size}")
                
                if 'score' in metadata:
                    print(f"       Score: {metadata['score']:.4f}")
        
        if len(bot_files) > 5:
            print(f"   ... and {len(bot_files) - 5} more bots")
    
    # Summary
    print("\n" + "="*80)
    print(f"📊 SUMMARY")
    print("="*80)
    print(f"Total Categories: {len(bots)}")
    print(f"Total Bots: {total_bots}")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    print("1. Start with BTC_USDT/1m bots (most liquid market)")
    print("2. Test multiple bots in paper trading")
    print("3. Compare performance over 24-48 hours")
    print("4. Select bot with best risk-adjusted returns")
    print("5. Start with 1x leverage (no amplification)")
    
    # Next steps
    print("\n" + "="*80)
    print("📝 NEXT STEPS")
    print("="*80)
    print("1. Create bot loader script:")
    print("   python scripts/load_bot.py --bot bots/BTC_USDT/1m/bot_001.pkl")
    print("\n2. Test bot signals:")
    print("   python scripts/test_bot_signals.py --bot bots/BTC_USDT/1m/bot_001.pkl")
    print("\n3. Run paper trading:")
    print("   python src/live_trading/run_bot.py --bot bot_001.pkl --test-mode")


if __name__ == "__main__":
    main()
