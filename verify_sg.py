import pandas as pd
import os
import glob

# Find all generation CSVs
gen_files = sorted(glob.glob('logs/generation_*.csv'))
print(f"Found {len(gen_files)} generation files\n")

# Check ALL files
for gen_file in gen_files:
    print(f"=== {os.path.basename(gen_file)} ===")
    df = pd.read_csv(gen_file, delimiter=';')
    if 'SurvivedGenerations' in df.columns:
        sg_values = df['SurvivedGenerations'].unique()
        sg_values = sorted(sg_values)
        print(f"Unique SurvivedGenerations values: {sg_values}")
        print(f"Min: {df['SurvivedGenerations'].min()}, Max: {df['SurvivedGenerations'].max()}")
        print()
