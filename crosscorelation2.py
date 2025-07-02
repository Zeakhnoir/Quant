import pandas as pd
import numpy as np

# 1) File names (make sure they're in your working dir)
fut_file  = 'trb_usdt_futures_export.csv'
spot_file = 'trb_usdt_spot_export.csv'

# 2) Loader function
def load_book_ticker(path):
    df = pd.read_csv(
        path,
        usecols=['time','symbol','bid_price','ask_price'],
        dtype={'symbol': str},
        na_values=['', 'None', 'NULL']
    )
    df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
    df['ask_price'] = pd.to_numeric(df['ask_price'], errors='coerce')
    df.dropna(subset=['bid_price','ask_price'], inplace=True)

    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]

    return df

# 3) Load both feeds
fut  = load_book_ticker(fut_file)
spot = load_book_ticker(spot_file)

# 4) Compute mid-prices
fut['mid']  = (fut['bid_price']  + fut['ask_price'])  / 2
spot['mid'] = (spot['bid_price'] + spot['ask_price']) / 2

# 5) Build a uniform 1 ms UTC grid over the overlapping times
start = max(fut.index.min(), spot.index.min())
end   = min(fut.index.max(), spot.index.max())
grid = pd.date_range(start=start, end=end, freq='1ms', tz='UTC')

# 6) Forward-fill (LOCF) onto that grid
fut_mid  = fut['mid'].reindex(grid, method='ffill')
spot_mid = spot['mid'].reindex(grid, method='ffill')

# 7) Compute log-returns
fut_ret  = np.log(fut_mid).diff().dropna()
spot_ret = np.log(spot_mid).diff().dropna()

# 8) Shift futures by τ =  ms (since spot leads)
tau_ms = 3
fut_ret_lagged = fut_ret.shift(-tau_ms)

# 9) Align the two series at zero lag now that fut is shifted
common_idx = spot_ret.index.intersection(fut_ret_lagged.index)
x = spot_ret.loc[common_idx]
y = fut_ret_lagged.loc[common_idx]

# 10) Compute the simple Pearson correlation at τ = ms
corr_at_tau = x.corr(y)

print(f"Cross‐correlation at τ = {tau_ms} ms (spot leads futures): {corr_at_tau:.6f}")
