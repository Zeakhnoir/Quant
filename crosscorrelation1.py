import pandas as pd
import numpy as np

# 1) File names (make sure they're in your working dir)
fut_file  = 'trb_usdt_futures_export.csv'
spot_file = 'trb_usdt_spot_export.csv'

# 2) Loader function
def load_book_ticker(path):
    # read only the four columns, coerce bad prices to NaN
    df = pd.read_csv(
        path,
        usecols=['time','symbol','bid_price','ask_price'],
        dtype={'symbol': str},
        na_values=['', 'None', 'NULL']
    )
    # convert prices to floats, drop rows where that fails
    df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
    df['ask_price'] = pd.to_numeric(df['ask_price'], errors='coerce')
    df.dropna(subset=['bid_price','ask_price'], inplace=True)

    # parse ISO-8601 timestamps into a tz-aware DatetimeIndex
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # if there are multiple updates in the same ms, keep only the last
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
if start >= end:
    raise RuntimeError(f"No overlap between futures ({fut.index.min()}→{fut.index.max()}) "
                       f"and spot ({spot.index.min()}→{spot.index.max()}).")
grid = pd.date_range(start=start, end=end, freq='1ms', tz='UTC')

# 6) Forward-fill (LOCF) onto that grid
fut_mid  = fut['mid'].reindex(grid, method='ffill')
spot_mid = spot['mid'].reindex(grid, method='ffill')

# 7) Compute log-returns
fut_ret  = np.log(fut_mid).diff().dropna()
spot_ret = np.log(spot_mid).diff().dropna()

# 8) Align the two return series
common_idx = fut_ret.index.intersection(spot_ret.index)
x = spot_ret.loc[common_idx].values
y = fut_ret.loc[common_idx].values

# 9) FFT‐based cross‐correlation
n    = len(x)
nfft = 1 << ((2*n - 1).bit_length())   # next power of two ≥ 2n
x0   = x - x.mean()
y0   = y - y.mean()
X    = np.fft.rfft(x0,   nfft)
Y    = np.fft.rfft(y0,   nfft)
corr = np.fft.irfft(X * np.conj(Y), nfft)

# 10) Shift so lag 0 is in the center; lags in ms
corr = np.concatenate([corr[-n+1:], corr[:n]])
lags = np.arange(-n+1, n)

# 11) Find peak correlation
peak_idx   = np.argmax(corr)
lead_lag_ms = lags[peak_idx]

# 12) Report
direction = "spot leads"    if lead_lag_ms > 0 \
          else "futures lead" if lead_lag_ms < 0 \
          else "no lead"
print(f"Peak cross-correlation at lag = {lead_lag_ms} ms ({direction}).")
