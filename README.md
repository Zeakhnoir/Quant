# Quant
# Lead–Lag Microstructure Analysis

This repository contains code and a write‐up for detecting and predicting sub-5 ms lead–lag relationships between Binance spot and perpetual (futures) markets for the TRB/USDT pair.

## Repository Contents

- **`submission.pdf`**  
  Full write‐up with problem setup, data preprocessing, feature engineering, modeling, results, and conclusions.

- **`*.py`** scripts  
  - **`quant*.py`**: exploratory analyses (jump‐pairing, cross‐correlation, AR residuals, Kalman filter).  
  - **`svm.py`**: SVM classification pipeline.  
  - **`xgb_threshold.py`**: XGBoost with PR‐AUC threshold calibration.  
  - **`xgb_split.py`**: XGBoost on balanced 70/30 train/test split.  
  - **`xgb_kf.py`**: XGBoost including Kalman‐filtered velocity features.  
  - **`ar_model.py`**: AR(10) unsupervised residual threshold detection.

- **`requirements.txt`**  
  Python dependencies required to run all scripts.

## Data

Place the following CSV exports in the project root:

- `trb_usdt_spot_export.csv`  
- `trb_usdt_futures_export.csv`  
- `trb_usdt_trades_export.csv`

These contain tick‐level best bid/ask and trade data with UTC timestamps.

## Feature Engineering

1. **Grid alignment**  
   - 1 ms or 5 ms uniform grid, LOCF forward‐fill of book quotes.  
2. **Trade‐side inference**  
   - Aggressor side (+1/–1) by comparing trade price to bid/ask.  
3. **Core features**  
   - **3 ms spot return**  
   - **20 ms spot volatility**  
   - **Spread** (ask–bid)  
   - **Net signed volume** and **trade count** over 3 ms  
   - *(Optional)* Kalman‐filtered spot & perp velocity  

4. **Unsupervised option**  
   - AR(10) on 5 ms log‐returns, flag residuals > 5σ as “jumps.”

## Supervised Pipelines

- **SVM**  
  - RBF kernel with balanced class weights.  
- **XGBoost**  
  - Two evaluation modes:  
    1. **Threshold calibration** via PR‐AUC (realistic imbalance).  
    2. **Balanced 70/30 split** (symmetric performance).  
  - Optionally include Kalman velocities.

## Running the Code

1. Clone this repo and `cd` into it.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
