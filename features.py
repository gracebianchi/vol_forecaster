import numpy as np
import pandas as pd

def build_features(spy, returns):
    rv = spy['realized_vol'] # realized volatility
    
    features = pd.DataFrame({
        'rv': rv,
        'rv_lag1': rv.shift(1),
        'rv_lag5': rv.shift(1).rolling(5).mean(),
        'rv_lag22': rv.shift(1).rolling(22).mean(),
        'rv_vol_of_vol': rv.shift(1).rolling(22).std(),
        'ret_skew_22': returns.shift(1).rolling(22).skew(),
        'ret_kurt_22': returns.shift(1).rolling(22).kurt(),
        'day_of_week': rv.index.dayofweek

    })
    
    return features.dropna()