"""Verify bot diversity after a test run."""
import os
import pandas as pd

def check_diversity(gen_file):
    """Check diversity in a generation CSV file."""
    try:
        df = pd.read_csv(gen_file, sep=';', on_bad_lines='skip')
        
        print(f"\n{os.path.basename(gen_file)}:")
        print(f"  Total bots: {len(df)}")
        
        # Check unique bot IDs
        bot_col = 'BotID'
        unique_bots = df[bot_col].nunique()
        print(f"  Unique BotIDs: {unique_bots}")
        
        # Check indicator diversity
        ind_col = 'IndicatorsUsed'
        unique_indicators = df[ind_col].nunique()
        print(f"  Unique indicator combinations: {unique_indicators}")
        
        # Show diversity percentage
        diversity_pct = (unique_indicators / len(df)) * 100
        
        if diversity_pct > 50:
            print(f"  ✅ EXCELLENT diversity: {diversity_pct:.1f}%")
        elif diversity_pct > 10:
            print(f"  ⚠️  MODERATE diversity: {diversity_pct:.1f}%")
        else:
            print(f"  ❌ POOR diversity: {diversity_pct:.1f}%")
            
        # Sample some indicators
        print(f"  Sample combinations:")
        for combo in df[ind_col].head(10).tolist():
            print(f"    - {combo}")
            
        return diversity_pct
        
    except Exception as e:
        print(f"Error reading {gen_file}: {e}")
        return 0

# Check newest generation files
print("="*60)
print("BOT DIVERSITY VERIFICATION")
print("="*60)

log_files = sorted([f for f in os.listdir('logs') if f.startswith('generation_') and f.endswith('.csv')])

if not log_files:
    print("\nNo generation logs found. Run main.py first.")
else:
    print(f"\nFound {len(log_files)} generation logs")
    
    # Check first generation (should be diverse)
    if 'generation_0.csv' in log_files:
        check_diversity('logs/generation_0.csv')
    
    # Check a middle generation
    if len(log_files) > 2:
        mid_gen = log_files[len(log_files)//2]
        check_diversity(f'logs/{mid_gen}')
    
    # Check last generation
    last_gen = log_files[-1]
    check_diversity(f'logs/{last_gen}')

print("\n" + "="*60)
