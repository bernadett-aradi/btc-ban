import collections
import collections.abc

# ==========================================
# 0. COMPATIBILITY FIX
# ==========================================
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence

# ==========================================
# IMPORTS
# ==========================================
import pandas as pd
import yfinance as yf
import pageviewapi
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import random
import sys

# ==========================================
# CONFIGURATION
# ==========================================
START_DATE = "2017-01-01"
END_DATE = "2025-12-31"
OUTPUT_FILE = "btc_macro_dataset.csv"

KEYWORDS = ["Bitcoin", "Ethereum", "Inflation", "Recession"]

# We keep the mapping for the other assets, but BTC is handled specially now
TICKERS = {
    "BTC-USD": "BTC_Price", # Will serve as Close price
    "^IXIC": "MKT_Nasdaq",
    "^GSPC": "MKT_SP500",
    "GC=F": "FUND_Gold",
    "CL=F": "FUND_Oil",
    "DX-Y.NYB": "FUND_DXY",
    "^VIX": "MKT_VIX"
}

# ==========================================
# 1. ROBUST GOOGLE TRENDS FETCHER
# ==========================================
def get_google_trends_stitched(kw_list, start_str, end_str):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Starting Google Trends Fetch (Stitched) ---")
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    full_df = pd.DataFrame()
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")

    total_kws = len(kw_list)
    for i, kw in enumerate(kw_list, 1):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ({i}/{total_kws}) Processing Keyword: '{kw}'")
        
        chunk_days = 180
        overlap_days = 30
        
        current_series = pd.Series(dtype=float)
        curr_start = start_dt
        
        while curr_start < end_dt:
            curr_end = curr_start + timedelta(days=chunk_days)
            
            is_last_window = False
            if curr_end >= end_dt: 
                curr_end = end_dt
                is_last_window = True
            
            tf = f"{curr_start.strftime('%Y-%m-%d')} {curr_end.strftime('%Y-%m-%d')}"
            
            attempts = 0
            max_retries = 5
            chunk = pd.DataFrame()
            success = False
            
            while attempts < max_retries:
                try:
                    time.sleep(random.uniform(1.5, 3.0)) 
                    
                    pytrends.build_payload([kw], cat=0, timeframe=tf, geo='', gprop='')
                    chunk = pytrends.interest_over_time()
                    
                    if 'isPartial' in chunk.columns: 
                        chunk = chunk.drop(columns=['isPartial'])
                    
                    if not chunk.empty:
                        success = True
                        break
                    else:
                        print(f"    > Empty response for {tf}. Retrying...")
                        attempts += 1
                        time.sleep(2 ** attempts)
                        
                except Exception as e:
                    wait = (2 ** attempts) + random.uniform(0, 1)
                    print(f"    > Error fetching {tf}: {e}")
                    print(f"    > Sleeping {wait:.1f}s before retry {attempts+1}/{max_retries}...")
                    time.sleep(wait)
                    attempts += 1
            
            if not success or chunk.empty:
                print(f"    ! FAILED to fetch {tf}. Skipping window.")
                if is_last_window: break
                curr_start = curr_end - timedelta(days=overlap_days)
                continue

            if current_series.empty:
                current_series = chunk[kw]
                print(f"    + Initialized: {len(chunk)} rows ({tf})")
            else:
                overlap_idx = current_series.index.intersection(chunk.index)
                
                if len(overlap_idx) > 5:
                    prev_mean = current_series[overlap_idx].replace(0, 0.01).mean()
                    curr_mean = chunk.loc[overlap_idx, kw].replace(0, 0.01).mean()
                    
                    if curr_mean == 0: scale_factor = 1.0
                    else: scale_factor = prev_mean / curr_mean
                    
                    chunk[kw] = chunk[kw] * scale_factor
                    
                    new_data = chunk.loc[chunk.index > current_series.index[-1], kw]
                    current_series = pd.concat([current_series, new_data])
                    print(f"    + Stitched: {len(new_data)} new rows (Scale: {scale_factor:.2f})")
                else:
                    new_data = chunk.loc[chunk.index > current_series.index[-1], kw]
                    current_series = pd.concat([current_series, new_data])
                    print(f"    + Appended: {len(new_data)} rows (No Overlap)")

            if is_last_window:
                print(f"    > Completed timeline for '{kw}'.")
                break

            curr_start = curr_end - timedelta(days=overlap_days)
            sys.stdout.flush()

        if not current_series.empty:
            current_series = (current_series - current_series.min()) / (current_series.max() - current_series.min()) * 100
            s_df = current_series.to_frame(name=f"GT_{kw}")
            s_df = s_df[~s_df.index.duplicated(keep='first')]
            
            if full_df.empty: full_df = s_df
            else: full_df = full_df.join(s_df, how='outer')

    return full_df

# ==========================================
# 2. WIKI FETCHER
# ==========================================
def get_wiki_data(topics, start, end):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Starting Wikipedia Fetch ---")
    wiki_df = pd.DataFrame()
    s_fmt = start.replace("-", "")
    e_fmt = end.replace("-", "")
    
    total = len(topics)
    for i, topic in enumerate(topics, 1):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ({i}/{total}) Fetching Wiki: '{topic}'")
        try:
            views = pageviewapi.per_article('en.wikipedia', topic, s_fmt, e_fmt, 
                                            access='all-access', agent='user', granularity='daily')
            dates, counts = [], []
            if 'items' in views:
                for item in views['items']:
                    dates.append(datetime.strptime(item['timestamp'][:8], "%Y%m%d"))
                    counts.append(item['views'])
            
            if counts:
                series = pd.Series(data=counts, index=dates, name=f"WIKI_{topic}")
                series = series[~series.index.duplicated(keep='first')]
                
                if wiki_df.empty: wiki_df = pd.DataFrame(series)
                else: wiki_df = wiki_df.join(series, how='outer')
                print(f"    > Success: {len(counts)} records.")
            else:
                print(f"    > Warning: No data found.")
                
        except Exception as e:
            print(f"    ! Wiki Error {topic}: {e}")
            
    return wiki_df

# ==========================================
# 3. MARKET DATA (MODIFIED FOR ROGERS-SATCHELL)
# ==========================================
def get_market_data(ticker_map, start, end):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Starting Market Data Fetch (yfinance) ---")
    tickers_list = list(ticker_map.keys())
    
    print(f"  > Downloading: {', '.join(tickers_list)}")
    try:
        # We need OHLC for Rogers-Satchell, so we ensure auto_adjust=True (Open/High/Low/Close)
        df = yf.download(tickers_list, start=start, end=end, auto_adjust=True, threads=True)
    except Exception as e:
        print(f"  ! Critical YF Error: {e}")
        return pd.DataFrame()

    print(f"  > Processing response shape: {df.shape}")
    
    # Containers
    price_df = pd.DataFrame()
    btc_ohlc = pd.DataFrame() # Separate container for Rogers-Satchell inputs
    vol_df = pd.DataFrame()
    
    # Handle YF MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # 1. Extract BTC OHLC specifically
            # YFinance structure can vary, usually Level 0 is PriceType, Level 1 is Ticker OR vice versa
            if 'BTC-USD' in df.columns.get_level_values(1):
                # Structure: (PriceType, Ticker)
                btc_ohlc['Open'] = df.xs('Open', axis=1, level=0)['BTC-USD']
                btc_ohlc['High'] = df.xs('High', axis=1, level=0)['BTC-USD']
                btc_ohlc['Low'] = df.xs('Low', axis=1, level=0)['BTC-USD']
                btc_ohlc['Close'] = df.xs('Close', axis=1, level=0)['BTC-USD']
            elif 'BTC-USD' in df.columns.get_level_values(0):
                 # Structure: (Ticker, PriceType)
                 btc_ohlc = df['BTC-USD'][['Open', 'High', 'Low', 'Close']]

            # 2. Extract General 'Close' prices for all tickers
            if 'Close' in df.columns.get_level_values(0):
                price_df = df.xs('Close', axis=1, level=0)
            elif 'Price' in df.columns.get_level_values(0):
                 price_df = df.xs('Price', axis=1, level=0)
            else:
                 # Fallback
                 cols = [c for c in df.columns if c[0] == 'Close']
                 if cols:
                     price_df = df[cols]
                     price_df.columns = price_df.columns.droplevel(0)
            
            # 3. Try fetching 'Volume'
            if 'Volume' in df.columns.get_level_values(0):
                 vol_df = df.xs('Volume', axis=1, level=0)
                 
        except Exception as e:
            print(f"  ! Error flattening YF MultiIndex: {e}")
            return pd.DataFrame()
    else:
        # Fallback for single ticker or flat structure (unlikely with this list)
        price_df = df
        if 'BTC-USD' in tickers_list:
            btc_ohlc = df[['Open', 'High', 'Low', 'Close']]

    final_df = price_df.rename(columns=ticker_map)
    
    if 'BTC-USD' in vol_df.columns:
        final_df['FUND_BTC_Volume'] = vol_df['BTC-USD']
        print("  > Added BTC Volume.")
        
    # Append BTC OHLC to final_df temporarily or calculate RS here. 
    # We will pass the OHLC columns to main via final_df for calculation.
    if not btc_ohlc.empty:
        final_df['BTC_Open'] = btc_ohlc['Open']
        final_df['BTC_High'] = btc_ohlc['High']
        final_df['BTC_Low'] = btc_ohlc['Low']
        # BTC_Price is already set via ticker_map logic (Close)
        print("  > Captured BTC OHLC for Rogers-Satchell.")

    print(f"  > Market Data Ready: {len(final_df)} rows.")
    return final_df

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"=== SCRIPT START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # 1. Fetch
    gt_df = get_google_trends_stitched(KEYWORDS, START_DATE, END_DATE)
    wiki_df = get_wiki_data(KEYWORDS, START_DATE, END_DATE)
    mkt_df = get_market_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Merge
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Merging Datasets ---")
    df = mkt_df.join(wiki_df, how='outer').join(gt_df, how='outer')
    
    # 3. Cleaning
    # Forward fill financial data over weekends
    ffill_cols = [c for c in df.columns if 'MKT_' in c or 'FUND_' in c or 'BTC_' in c]
    df[ffill_cols] = df[ffill_cols].ffill()
    
    # Drop rows where BTC didn't trade
    df = df.dropna(subset=['BTC_Price'])
    
    # 4. Feature Engineering
    df['BTC_Ret'] = np.log(df['BTC_Price']).diff()
    
    # --- ROGERS-SATCHELL VOLATILITY CALCULATION ---
    # Formula: V_rs = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    print("  > Calculating Rogers-Satchell Volatility...")
    try:
        # Ensure positive values for logs
        h = df['BTC_High']
        l = df['BTC_Low']
        c = df['BTC_Price'] # Close
        o = df['BTC_Open']
        
        term1 = np.log(h / c) * np.log(h / o)
        term2 = np.log(l / c) * np.log(l / o)
        
        df['BTC_RogersSatchell'] = term1 + term2
        
        # Cleanup OHLC columns if not needed for output
        df = df.drop(columns=['BTC_Open', 'BTC_High', 'BTC_Low'])
        
    except KeyError as e:
        print(f"  ! Error computing Rogers-Satchell: Missing columns {e}")
        print("  ! Falling back to Squared Returns.")
        df['BTC_RogersSatchell'] = df['BTC_Ret']**2

    # Original Volume Metric (kept for compatibility)
    df['BTC_Vol'] = df['BTC_Ret'].abs() * 100
    
    print(f"  > Final Data Shape: {df.shape}")
    df.to_csv(OUTPUT_FILE)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] SUCCESS. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
