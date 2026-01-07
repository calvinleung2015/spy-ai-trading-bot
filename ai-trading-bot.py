from ib_insync import *
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# ========================
# --- Configuration ---
# ========================
SYMBOL, EXCHANGE, CURRENCY = 'SPY', 'SMART', 'USD'
POSITION_SIZE = 150
TRADE_START, TRADE_END = time(9, 50), time(10, 45) 

CSV_RESULTS = 'spy_v8_results.csv'
TRAIN_DATA_CACHE = 'train_cache_v8.csv'
MODEL_PATH = 'spy_model_v8.pkl'

# --- TIGHTENED PARAMETERS ---
ML_PROB_THRESHOLD = 0.60  # Only high-conviction trades
MIN_ADX = 22              # Ensure market isn't sideways
MAX_TRADES_PER_DAY = 1    

# ========================
# --- Helper: Indicators ---
# ========================
def get_indicators(bars):
    """Calculate technicals needed for both training and execution."""
    df = pd.DataFrame([(b.date, b.open, b.high, b.low, b.close, b.volume) 
                       for b in bars], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # ATR for Volatility-Adjusted Stops
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Trend Regime
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    
    # Momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    return df

def extract_features(df, idx):
    """Feature engineering for the ML model."""
    row = df.iloc[idx]
    # Use relative metrics so the model understands 'context'
    f1 = row['close'] / row['ema9']     
    f2 = row['ema9'] / row['ema21']    
    f3 = row['rsi'] / 100.0            
    f4 = row['volume'] / df['volume'].iloc[max(0, idx-10):idx].mean() 
    f5 = (row['close'] - df['open'].iloc[0]) / df['open'].iloc[0] 
    return [f1, f2, f3, f4, f5]

FEAT_COLS = ['ema_rel', 'ema_spread', 'rsi', 'vol_spike', 'day_trend']

# ========================
# --- Core Logic ---
# ========================
def get_training_data(ib, contract):
    if os.path.exists(TRAIN_DATA_CACHE): return pd.read_csv(TRAIN_DATA_CACHE)
    all_rows = []
    print("📡 Fetching 2023 history for training...")
    # Fetching in chunks to avoid IBKR rate limits
    for m in range(1, 13):
        end_dt = f"2023{m:02d}28 00:00:00"
        bars = ib.reqHistoricalData(contract, endDateTime=end_dt, durationStr='1 M', 
                                    barSizeSetting='1 min', whatToShow='TRADES', useRTH=True)
        if bars:
            df = get_indicators(bars)
            for i in range(25, len(df) - 20):
                feats = extract_features(df, i)
                # Label: 1 if price moves up 0.15% in next 15 mins
                label = 1 if df['close'].iloc[i+15] > df['close'].iloc[i] * 1.0015 else 0
                all_rows.append(feats + [label])
    
    df_train = pd.DataFrame(all_rows, columns=FEAT_COLS + ['label'])
    df_train.to_csv(TRAIN_DATA_CACHE, index=False)
    return df_train

def backtest_day(ib, contract, model, date_str):
    bars = ib.reqHistoricalData(contract, endDateTime=f"{date_str} 16:00:00 US/Eastern", 
                                durationStr='1 D', barSizeSetting='1 min', 
                                whatToShow='TRADES', useRTH=True, formatDate=1)
    if len(bars) < 60: return None
    
    df = get_indicators(bars)
    pnl, trades = 0, 0
    
    for i in range(25, len(df)):
        row = df.iloc[i]
        curr_time = row['date'].time() if isinstance(row['date'], datetime) else row['date'].to_pydatetime().time()
        
        if not (TRADE_START <= curr_time <= TRADE_END): continue
        if trades >= MAX_TRADES_PER_DAY: break

        feats = extract_features(df, i)
        prob = model.predict_proba(pd.DataFrame([feats], columns=FEAT_COLS))[0][1]
        
        # --- ENTRY FILTERS ---
        entry_type = None
        # Long: ML High Prob + Price above EMAs + RSI not overbought
        if prob > ML_PROB_THRESHOLD and row['close'] > row['ema9'] > row['ema21'] and row['rsi'] < 65:
            entry_type = 'LONG'
        # Short: ML Low Prob + Price below EMAs + RSI not oversold
        elif prob < (1 - ML_PROB_THRESHOLD) and row['close'] < row['ema9'] < row['ema21'] and row['rsi'] > 35:
            entry_type = 'SHORT'

        if entry_type:
            entry_p = row['close']
            atr = row['atr']
            # Risk Management: 1.5x ATR Stop, 3x ATR Target (2:1 Reward/Risk)
            stop_dist = max(atr * 1.5, 0.20) # Minimum 20 cent stop for SPY
            target_dist = stop_dist * 2.0
            
            stop = entry_p - stop_dist if entry_type == 'LONG' else entry_p + stop_dist
            target = entry_p + target_dist if entry_type == 'LONG' else entry_p - target_dist

            # Exit Logic (Trailing or Time-based)
            for j in range(i + 1, min(i + 90, len(df))): # Max hold 90 mins
                exit_row = df.iloc[j]
                if entry_type == 'LONG':
                    if exit_row['high'] >= target: pnl = (target - entry_p) * POSITION_SIZE; trades = 1; break
                    if exit_row['low'] <= stop: pnl = (stop - entry_p) * POSITION_SIZE; trades = 1; break
                else:
                    if exit_row['low'] <= target: pnl = (entry_p - target) * POSITION_SIZE; trades = 1; break
                    if exit_row['high'] >= stop: pnl = (entry_p - stop) * POSITION_SIZE; trades = 1; break
            if trades > 0: break
            
    return {'date': date_str, 'pnl': pnl, 'trades': trades}

def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=125)
    except:
        print("❌ Could not connect to TWS/Gateway.")
        return

    contract = Stock(SYMBOL, EXCHANGE, CURRENCY)
    ib.qualifyContracts(contract)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        df = get_training_data(ib, contract)
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=5)
        model.fit(df[FEAT_COLS], df['label'])
        joblib.dump(model, MODEL_PATH)

    results = []
    curr, end = datetime(2022, 1, 1), datetime(2022, 12, 31)
    
    while curr <= end:
        if curr.weekday() < 5:
            res = backtest_day(ib, contract, model, curr.strftime('%Y%m%d'))
            if res and res['trades'] > 0:
                results.append(res)
                icon = "✅" if res['pnl'] > 0 else "❌"
                print(f"{icon} {res['date']} | PnL: ${res['pnl']:>8.2f}")
        curr += timedelta(days=1)

    if results:
        df_res = pd.DataFrame(results)
        print(f"\n💰 FINAL PnL: ${df_res['pnl'].sum():.2f} | Win Rate: {(len(df_res[df_res['pnl']>0])/len(df_res))*100:.1f}%")
    
    ib.disconnect()

if __name__ == "__main__":
    main()

