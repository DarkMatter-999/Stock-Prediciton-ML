import ta
import pandas as pd
import numpy as np

def atr_crossover(df, sensitivity=2, atr_period=1):
    xATR = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)

    nLoss = sensitivity * xATR

    src = df['close']

    xATRTrailingStop = pd.Series(np.zeros(len(src)))
    for i in range(1, len(src)):
        if src[i] > xATRTrailingStop[i-1] and src[i-1] > xATRTrailingStop[i-1]:
            xATRTrailingStop[i] = max(xATRTrailingStop[i-1], src[i] - nLoss[i])
        elif src[i] < xATRTrailingStop[i-1] and src[i-1] < xATRTrailingStop[i-1]:
            xATRTrailingStop[i] = min(xATRTrailingStop[i-1], src[i] + nLoss[i])
        elif src[i] > xATRTrailingStop[i-1]:
            xATRTrailingStop[i] = src[i] - nLoss[i]
        else:
            xATRTrailingStop[i] = src[i] + nLoss[i]

    pos = pd.Series(np.zeros(len(src)))
    for i in range(1, len(src)):
        if src[i-1] < xATRTrailingStop[i-1] and src[i] > xATRTrailingStop[i-1]:
            pos[i] = 1
        elif src[i-1] > xATRTrailingStop[i-1] and src[i] < xATRTrailingStop[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]

    ema = src.ewm(span=1, adjust=False).mean()
    above = ema > xATRTrailingStop
    below = ema < xATRTrailingStop

    buy = (src > xATRTrailingStop) & above
    sell = (src < xATRTrailingStop) & below

    df["atr_buy"] = buy
    df["atr_sell"] = sell


    # barbuy = src > xATRTrailingStop
    # barsell = src < xATRTrailingStop

    return df
