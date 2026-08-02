import yfinance as yf
import numpy as np
import pandas as pd

def load_spy_data(start = '2015-01-01', end='2026-08-02'):
    spy = yf.download('SPY', start = start, end = end, auto_adjust = False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Garman-Klass realized vol estimator (daily, annualized)
    log_hl = np.log(spy['High'] / spy['Low'])
    log_co = np.log(spy['Close'] / spy['Open'])
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    spy['realized_vol'] = np.sqrt(gk_var * 252) * 100 # annualized %

    returns = np.log(spy['Close'] / spy['Close'].shift(1)).dropna() # daily returns 

    return spy, returns


