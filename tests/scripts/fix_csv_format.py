"""
Fix corrupted CSV files by converting comma decimal separators to dots.

This script repairs CSV files that have commas as decimal separators,
making numeric columns parseable again.
"""
import pandas as pd
import os
import glob
from pathlib import Path


def fix_csv_file(filepath: str, output_path: str = None):
    """
    Fix a CSV file by converting comma decimals to dot decimals.
    
    Args:
        filepath: Path to the CSV file to fix
        output_path: Optional output path (defaults to overwriting original)
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"📂 Processing: {filepath}")
    
    try:
        # Read with no type inference to keep everything as strings initially
        df = pd.read_csv(filepath, sep=';', dtype=str)
        
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Columns that should be numeric (excluding Generation, BotID, and string columns)
        numeric_columns = [col for col in df.columns 
                          if col not in ['Generation', 'BotID', 'IndicatorsUsed', 
                                        'AllCyclesHaveTrades', 'AllCyclesProfitable']]
        
        # Convert comma decimals to dot decimals for numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Replace comma with dot in numeric strings
                df[col] = df[col].str.replace(',', '.', regex=False)
                # Convert to numeric (will handle conversion properly)
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not convert {col} to numeric: {e}")
        
        # Save fixed file
        output = output_path or filepath
        df.to_csv(output, sep=';', index=False)
        
        print(f"   ✅ Fixed and saved to: {output}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing file: {e}")
        return False


def fix_all_generation_files(logs_dir: str = 'logs'):
    """
    Fix all generation CSV files in the logs directory.
    
    Args:
        logs_dir: Path to logs directory
    """
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        return
    
    # Find all generation CSV files
    pattern = os.path.join(logs_dir, 'generation_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No generation CSV files found in {logs_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"CSV FORMAT FIX UTILITY")
    print(f"{'='*60}")
    print(f"Found {len(csv_files)} CSV file(s) to fix\n")
    
    success_count = 0
    for filepath in sorted(csv_files):
        if fix_csv_file(filepath):
            success_count += 1
        print()
    
    print(f"{'='*60}")
    print(f"✅ Successfully fixed {success_count}/{len(csv_files)} files")
    print(f"{'='*60}\n")


def verify_csv_format(filepath: str):
    """
    Verify that a CSV file has correct numeric format.
    
    Args:
        filepath: Path to CSV file
    """
    print(f"\n📊 Verifying: {filepath}")
    
    try:
        df = pd.read_csv(filepath, sep=';', nrows=5)
        
        # Check if numeric columns are actually numeric
        numeric_cols = ['ProfitPct', 'WinRate', 'FitnessScore', 'SharpeRatio']
        
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"\n   Data type check:")
        
        all_good = True
        for col in numeric_cols:
            if col in df.columns:
                dtype = df[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(dtype)
                status = "✅" if is_numeric else "❌"
                print(f"   {status} {col:20s} -> {dtype}")
                if not is_numeric:
                    all_good = False
                    print(f"      Sample value: {df[col].iloc[0]}")
        
        if all_good:
            print(f"\n   ✅ All numeric columns have correct format!")
        else:
            print(f"\n   ❌ Some columns need fixing (run fix_csv_file)")
        
        return all_good
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Fix specific file
        filepath = sys.argv[1]
        fix_csv_file(filepath)
        verify_csv_format(filepath)
    else:
        # Fix all generation files
        fix_all_generation_files()
        
        # Verify a sample file
        sample_file = 'logs/generation_1.csv'
        if os.path.exists(sample_file):
            verify_csv_format(sample_file)
