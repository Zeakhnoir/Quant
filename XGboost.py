import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# === Configuration ===
spot_file     = 'trb_usdt_spot_export.csv'
fut_file      = 'trb_usdt_futures_export.csv'
trades_file   = 'trb_usdt_trades_export.csv'
grid_freq     = '1ms'
window_ms     = 3
threshold     = 0.0007     # 0.07%
lookahead_ms  = 5
n_per_class   = 1000       # number of examples per class for training

# === 1) Load book-ticker data ===
def load_book(path):
    df = pd.read_csv(path, usecols=['time','bid_price','ask_price'])
    df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
    df['ask_price'] = pd.to_numeric(df['ask_price'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time','bid_price','ask_price'])
    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df

spot_book = load_book(spot_file)
fut_book  = load_book(fut_file)

# === 2) Load and tag trades with bid/ask from spot ===
trades = pd.read_csv(trades_file, usecols=['time','price','quantity'])
trades['time'] = pd.to_datetime(trades['time'], utc=True, errors='coerce')
trades = trades.dropna(subset=['time']).set_index('time').sort_index()

# Merge each trade with most recent spot quote
spot_quotes = spot_book.reset_index().rename(columns={'time':'quote_time'})
trades_df = trades.reset_index().rename(columns={'time':'trade_time'})
merged_trades = pd.merge_asof(
    trades_df.sort_values('trade_time'),
    spot_quotes.sort_values('quote_time'),
    left_on='trade_time',
    right_on='quote_time',
    direction='backward'
).dropna(subset=['bid_price','ask_price'])

# Infer trade side and signed quantity
merged_trades['side'] = np.where(
    merged_trades['price'] >= merged_trades['ask_price'],  1,
    np.where(merged_trades['price'] <= merged_trades['bid_price'], -1, 0)
)
merged_trades['signed_qty'] = merged_trades['side'] * merged_trades['quantity']
merged_trades = merged_trades.set_index('trade_time')

# === 3) Build common 1 ms grid and ffill mid-prices ===
start = max(spot_book.index.min(), fut_book.index.min())
end   = min(spot_book.index.max(), fut_book.index.max())
grid  = pd.date_range(start, end, freq=grid_freq, tz='UTC')

spot_mid = ((spot_book['bid_price'] + spot_book['ask_price']) / 2).reindex(grid, method='ffill')
fut_mid  = ((fut_book['bid_price']  + fut_book['ask_price'])  / 2).reindex(grid, method='ffill')
spot_ask = spot_book['ask_price'].reindex(grid, method='ffill')
spot_bid = spot_book['bid_price'].reindex(grid, method='ffill')

# === 4) Resample signed volume and trade count onto grid ===
trade_signed = merged_trades['signed_qty'].resample(grid_freq).sum().reindex(grid, fill_value=0)
trade_count  = merged_trades['signed_qty'].resample(grid_freq).count().reindex(grid, fill_value=0)

# === 5) Compute returns and rolling features ===
spot_ret = spot_mid.pct_change(periods=window_ms)
fut_ret  = fut_mid.pct_change(periods=window_ms)

events = pd.DataFrame(index=grid)
events['spot_ret']      = spot_ret
events['spread']        = (spot_ask - spot_bid)
events['net_signed_vol']= trade_signed.rolling(f'{window_ms}ms').sum()
events['trade_count']   = trade_count.rolling(f'{window_ms}ms').sum()

# Detect spot trigger events
events['trigger'] = (events['spot_ret'].abs() >= threshold).astype(int)

# === 6) Label events by perp response within lookahead_ms ===
events['fut_lead_ret'] = fut_ret.shift(-lookahead_ms)
mask = events['trigger'] == 1
events = events.loc[mask].copy()
events['label'] = (events['fut_lead_ret'].abs() >= threshold).astype(int)
events = events.dropna(subset=['label'])

# === 7) Sample balanced dataset ===
pos = events[events['label']==1]
neg = events[events['label']==0]
pos_samp = pos.sample(n=min(n_per_class, len(pos)), random_state=42)
neg_samp = neg.sample(n=min(n_per_class, len(neg)), random_state=42)
balanced = pd.concat([pos_samp, neg_samp]).sample(frac=1, random_state=42)

# === 8) Train/test split ===
features = ['spot_ret','spread','net_signed_vol','trade_count']
X = balanced[features]
y = balanced['label']
# 70% train, 30% test
train_size = int(0.7 * len(balanced))
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test,  y_test  = X.iloc[train_size:], y.iloc[train_size:]

# === 9) Fit XGBoost classifier ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# === 10) Evaluate ===
y_pred = model.predict(X_test)
print("XGBoost Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# Feature importances
importances = model.feature_importances_
imp_df = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(imp_df)
