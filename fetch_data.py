import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")

if not db_url:
    raise ValueError("Database URL not found. Check yout .env file")

tickers = ["SPY", "AAPL", "GOOG"]

print("Fetching historical market data...")
all_data = []

for ticker in tickers:
    df = yf.download(ticker, period="5y", interval="1d")    
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["ticker"]=ticker
    all_data.append(df)

final_df = pd.concat(all_data,ignore_index=True)
final_df.columns = [col.lower().replace(" ", "_") for col in final_df.columns]

print("Connecting to Supabase and uploading data...")
engine = create_engine(db_url)

final_df.to_sql("raw_market_data", engine, if_exists="replace", index=False)

print("Success! Data successfully piped tp the cloud")