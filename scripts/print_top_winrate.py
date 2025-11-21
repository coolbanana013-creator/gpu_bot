import pandas as pd
import os
path='logs/generation_0.csv'
if not os.path.exists(path):
    print('generation_0.csv not found')
    exit(1)
df=pd.read_csv(path, sep=';')
col='TotalWinRate'
df[col]=df[col].astype(str).str.replace(',','.').astype(float)
print(df[['BotID','TotalWinRate','AvgProfitPctPerCycle','AvgWinRatePerCycle']].sort_values('TotalWinRate',ascending=False).head(20))
