import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def preprocess():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'Data')
    
    yahoo_file = os.path.join(data_dir, 'Bulk_Yahoo_Historical_Data.csv')
    trends_file = os.path.join(data_dir, 'rare_earth_trends.csv')
    output_file = os.path.join(data_dir, 'aligned_dataset.csv')

    print("Loading Bulk Yahoo Data...")
    df_yahoo = pd.read_csv(yahoo_file, header=[0, 1], index_col=0, parse_dates=True)
    df_yahoo.index.name = 'Date'
    
    companies = ['UUUU', 'ARU.AX', 'NEO.TO', 'ILU.AX', '600392.SS', 'MP', 'LYC.AX', '600111.SS']
    indicators = {'^GSPC': 'SP500', '000001.SS': 'Shanghai_Index', 'CL=F': 'Crude_Oil', 'CNY=X': 'USD_CNY'}

    # 1. Target Variables
    # yfinance group_by="ticker" results in tuples like ('UUUU', 'Close')
    close_cols = [c for c in df_yahoo.columns if c[1] == 'Close' and c[0] in companies]
    high_cols = [c for c in df_yahoo.columns if c[1] == 'High' and c[0] in companies]
    low_cols = [c for c in df_yahoo.columns if c[1] == 'Low' and c[0] in companies]

    df_target = pd.DataFrame(index=df_yahoo.index)
    
    # We use nanmean to average the available values
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df_target['Target_Close'] = df_yahoo[close_cols].mean(axis=1)
        df_target['Target_High'] = df_yahoo[high_cols].mean(axis=1)
        df_target['Target_Low'] = df_yahoo[low_cols].mean(axis=1)
    
    # 2. Structured Indicators
    for tick, name in indicators.items():
        try:
            df_target[name] = df_yahoo[(tick, 'Close')]
        except KeyError:
            print(f"Indicator {tick} not found under (Ticker, 'Close').")

    df_target.dropna(subset=['Target_Close'], inplace=True)
    df_target.ffill(inplace=True)
    df_target.bfill(inplace=True)

    # 3. Unstructured Data
    print("Loading Google Trends Data (Worldwide)...")
    try:
        df_trends = pd.read_csv(trends_file, parse_dates=['Time'])
        trend_col = next((c for c in df_trends.columns if c != 'Time'), None)
        df_trends.rename(columns={'Time': 'Date', trend_col: 'Search_Index'}, inplace=True)
        df_trends.set_index('Date', inplace=True)
        
        df_trends_daily = df_trends.resample('D').ffill()
        df_final = df_target.join(df_trends_daily, how='left')
        df_final['Search_Index'].fillna(0, inplace=True)
        
        # Clean <1 strings
        df_final['Search_Index'] = pd.to_numeric(df_final['Search_Index'].astype(str).str.replace("<1", "0"), errors='coerce').fillna(0)
    except Exception as e:
        print("Could not load trends data. Assuming Structured-Only mode.")
        print(f"Error: {e}")
        df_final = df_target.copy()
        df_final['Search_Index'] = 0.0

    # 4. Cleaning & Normalization
    cols_to_clip = [c for c in ['SP500', 'Shanghai_Index', 'Crude_Oil', 'USD_CNY', 'Search_Index'] if c in df_final.columns]
    
    if len(df_final) > 0:
        for col in cols_to_clip:
            lower = df_final[col].quantile(0.01)
            upper = df_final[col].quantile(0.99)
            df_final[col] = np.clip(df_final[col], lower, upper)

        scaler = MinMaxScaler()
        df_final[cols_to_clip] = scaler.fit_transform(df_final[cols_to_clip])
    else:
        print("Warning: DataFrame is empty after merging!")

    df_final.reset_index(inplace=True)
    df_final.to_csv(output_file, index=False)
    print(f"Dataset completely preprocessed and organized. Shape: {df_final.shape}")

if __name__ == '__main__':
    preprocess()
