# Complete script with data loading and threshold calibration:

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score

# Configuration
spot_file     = 'trb_usdt_spot_export.csv'
fut_file      = 'trb_usdt_futures_export.csv'
trades_file   = 'trb_usdt_trades_export.csv'
grid_freq     = '1ms'
window_ms     = 3
threshold     = 0.0007
lookahead_ms  = 5
n_per_class   = 1000

# Function to load book-ticker
def load_book(path):
    df = pd.read_csv(path, usecols=['time','bid_price','ask_price'])
    df['bid_price'] = pd.to_numeric(df['bid_price'], errors='coerce')
    df['ask_price'] = pd.to_numeric(df['ask_price'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time','bid_price','ask_price'])
    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df

# Load book data
spot_book = load_book(spot_file)
fut_book  = load_book(fut_file)

# Load trades and infer side
trades = pd.read_csv(trades_file, usecols=['time','price','quantity'])
trades['time'] = pd.to_datetime(trades['time'], utc=True, errors='coerce')
trades = trades.dropna(subset=['time']).set_index('time').sort_index()
spot_quotes = spot_book.reset_index().rename(columns={'time':'quote_time'})
trades_df = trades.reset_index().rename(columns={'time':'trade_time'})
merged_trades = pd.merge_asof(trades_df.sort_values('trade_time'),
                              spot_quotes.sort_values('quote_time'),
                              left_on='trade_time', right_on='quote_time',
                              direction='backward').dropna(subset=['bid_price','ask_price'])
merged_trades['side'] = np.where(merged_trades['price'] >= merged_trades['ask_price'],1,
                          np.where(merged_trades['price'] <= merged_trades['bid_price'],-1,0))
merged_trades['signed_qty'] = merged_trades['side'] * merged_trades['quantity']
merged_trades = merged_trades.set_index('trade_time')

# Build grid
start = max(spot_book.index.min(), fut_book.index.min())
end   = min(spot_book.index.max(), fut_book.index.max())
grid  = pd.date_range(start, end, freq=grid_freq, tz='UTC')

# Mid-prices & trade aggregates
spot_mid = ((spot_book['bid_price'] + spot_book['ask_price'])/2).reindex(grid,method='ffill')
fut_mid  = ((fut_book['bid_price']  + fut_book['ask_price'])/2).reindex(grid,method='ffill')
spot_ask = spot_book['ask_price'].reindex(grid,method='ffill')
spot_bid = spot_book['bid_price'].reindex(grid,method='ffill')

trade_signed = merged_trades['signed_qty'].resample(grid_freq).sum().reindex(grid, fill_value=0)
trade_count  = merged_trades['signed_qty'].resample(grid_freq).count().reindex(grid, fill_value=0)

spot_ret = spot_mid.pct_change(periods=window_ms)
fut_ret  = fut_mid.pct_change(periods=window_ms)

events = pd.DataFrame(index=grid)
events['spot_ret'] = spot_ret
events['spread']   = (spot_ask - spot_bid)
events['net_signed_vol'] = trade_signed.rolling(f'{window_ms}ms').sum()
events['trade_count']    = trade_count.rolling(f'{window_ms}ms').sum()
events['trigger']  = (events['spot_ret'].abs() >= threshold).astype(int)

events['fut_lead_ret'] = fut_ret.shift(-lookahead_ms)
events = events[events['trigger']==1].copy()
events['label'] = (events['fut_lead_ret'].abs() >= threshold).astype(int)
events = events.dropna(subset=['label'])

# Balanced training set
pos_train = events[events['label']==1].sample(n=min(n_per_class, len(events[events['label']==1])), random_state=42)
neg_train = events[events['label']==0].sample(n=min(n_per_class, len(events[events['label']==0])), random_state=42)
train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)
test  = events.drop(train.index)

features = ['spot_ret','spread','net_signed_vol','trade_count']

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(train[features], train['label'])

# Predict probabilities
probs = model.predict_proba(test[features])[:,1]
y_true = test['label'].values

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
avg_prec = average_precision_score(y_true, probs)
print(f"Average precision (PR AUC): {avg_prec:.4f}\n")

# Choose threshold for ~50% recall
target_recall = 0.5
idx = np.argmin(np.abs(recalls - target_recall))
chosen_thr = thresholds[idx]
print(f"Chosen threshold for recall~0.5: {chosen_thr:.4f}, precision={precisions[idx]:.4f}\n")

# Evaluate at chosen threshold
y_pred = (probs >= chosen_thr).astype(int)
print("Classification report at chosen threshold:\n")
print(classification_report(y_true, y_pred, digits=4))
