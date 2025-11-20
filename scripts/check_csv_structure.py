import pandas as pd

# Read the CSV file
df = pd.read_csv('logs/generation_0.csv', sep=';', nrows=10)

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nData types:")
print(df.dtypes.value_counts())

print(f"\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print(f"\nSample data from first row:")
print(df.iloc[0])

print(f"\n\nChecking for any string contamination in numeric columns:")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols[:5]:  # Check first 5 numeric columns
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Sample values: {df[col].head(3).tolist()}")
