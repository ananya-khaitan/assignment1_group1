import yfinance as yf
import pandas as pd
import os

print("Downloading bulk historical data from Yahoo Finance...")

# List of the 8 Rare Earth companies from your Excel file
company_tickers = ['UUUU', 'ARU.AX', 'NEO.TO', 'ILU.AX', '600392.SS', 'MP', 'LYC.AX', '600111.SS']

# List of the missing macroeconomic indicators
indicator_tickers = [
    '^GSPC',      # S&P 500
    '000001.SS',  # Shanghai Composite Index
    'CL=F',       # Crude Oil Futures
    'CNY=X'       # USD/CNY Exchange Rate
]

all_tickers = company_tickers + indicator_tickers

# Download all historical data spanning back to 2001 (or earliest available for new companies like MP)
# 'group_by="ticker"' organizes columns by company
bulk_data = yf.download(all_tickers, start="2001-01-01", group_by="ticker", progress=True)

# Define output path
output_file = "Bulk_Yahoo_Historical_Data.csv"
bulk_data.to_csv(output_file)

print(f"\nSuccess! Downloaded {len(all_tickers)} tickers.")
print(f"Data saved to: {os.path.abspath(output_file)}")
