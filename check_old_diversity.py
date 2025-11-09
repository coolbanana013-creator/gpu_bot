"""Check bot diversity in old generation logs."""
import pandas as pd

try:
    # Check generation 45 (near end of old run)
    df = pd.read_csv('logs/generation_45.csv', sep=';', on_bad_lines='skip')
    
    print("Generation 45 (OLD RUN - before fix):")
    print(f"  Total bots: {len(df)}")
    
    bot_col = 'BotID'
    unique_bots = df[bot_col].nunique()
    print(f"  Unique BotIDs: {unique_bots}")
    
    if unique_bots < len(df) / 2:
        most_common = df[bot_col].mode()[0]
        count = (df[bot_col] == most_common).sum()
        print(f"  ⚠️ PROBLEM: BotID {most_common} appears {count} times ({count/len(df)*100:.1f}%)")
        print(f"  Sample BotIDs: {df[bot_col].head(20).tolist()}")
    else:
        print(f"  ✅ Bots are diverse")
        
except Exception as e:
    print(f"Error reading file: {e}")

print("\n" + "="*60)
print("The fix has been applied. Next run should show diverse bots.")
print("="*60)
