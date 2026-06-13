import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")
engine = create_engine(db_url)

print("Pulling raw data from cloud database...")
df = pd.read_sql("SELECT * FROM raw_market_data ORDER BY ticker, date", engine)

df.columns = [col.lower() for col in df.columns]
df['date'] = pd.to_datetime(df['date'])

print("Engineering data science features...")

df['log_return'] = np.log(df['close'] / df.groupby('ticker')['close'].shift(1))
df['volatility_5d'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(window=5).std())
df['ma_5d'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=5).mean())
df['ma_20d'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20).mean())
df['ma_ratio'] = df['ma_5d'] / df['ma_20d']

df = df.dropna().copy()

df['target_high_volatility'] = df.groupby('ticker')['volatility_5d'].transform(
    lambda x: (x > x.median()).astype(int)
)

final_features_df = df.drop(columns=['ma_5d', 'ma_20d'])

print("Uploading engineered features back to Supabase...")
final_features_df.to_sql("engineered_market_features", engine, if_exists="replace", index=False)

print("Success! Feature engineering table is live in the cloud.")