import pandas as pd
df = pd.read_csv('logs/generation_10.csv', sep=';', nrows=1)
print('Columns in CSV:')
for i, col in enumerate(df.columns):
    print(f'{i:2d}. {col}')
