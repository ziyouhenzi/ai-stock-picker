"""
factors/technical.py - 技术因子计算模块
计算 30+ 种技术指标作为选股因子
"""
import pandas as pd
import numpy as np


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入单股日线 OHLCV 数据，返回包含所有技术因子的 DataFrame
    """
    result = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ──────────────────────────────────────────────
    # 1. 动量因子
    # ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        result[f"momentum_{n}d"] = close.pct_change(n)

    result["momentum_reversal"] = -close.pct_change(5)   # 短期反转

    # ──────────────────────────────────────────────
    # 2. 均线偏离因子
    # ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        ma = close.rolling(n).mean()
        result[f"ma{n}_ratio"] = close / ma - 1

    # ──────────────────────────────────────────────
    # 3. RSI
    # ──────────────────────────────────────────────
    for period in [6, 14, 24]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        result[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # ──────────────────────────────────────────────
    # 4. MACD
    # ──────────────────────────────────────────────
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    result["macd_diff"] = macd_line - signal_line
    result["macd_signal"] = np.sign(result["macd_diff"])

    # ──────────────────────────────────────────────
    # 5. 布林带位置
    # ──────────────────────────────────────────────
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    result["bollinger_pos"] = (close - lower) / (upper - lower + 1e-9)

    # ──────────────────────────────────────────────
    # 6. 成交量因子
    # ──────────────────────────────────────────────
    result["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    result["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    result["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

    if "turnover" in df.columns:
        result["turnover_rate"] = df["turnover"]
        result["turnover_ma5"] = df["turnover"].rolling(5).mean()

    # ──────────────────────────────────────────────
    # 7. 波动率因子
    # ──────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    result["volatility_5d"] = log_ret.rolling(5).std() * np.sqrt(252)
    result["volatility_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    result["atr_14"] = tr.rolling(14).mean() / close

    # ──────────────────────────────────────────────
    # 8. 价格形态因子
    # ──────────────────────────────────────────────
    result["high_low_ratio"] = (high - low) / (close + 1e-9)
    result["close_position"] = (close - low) / (high - low + 1e-9)

    # 52周高低点
    result["dist_52w_high"] = close / close.rolling(252).max() - 1
    result["dist_52w_low"] = close / close.rolling(252).min() - 1

    return result


def compute_factor_matrix(price_dict: dict, factor_name: str) -> pd.DataFrame:
    """
    输入多只股票的价格字典，返回因子矩阵 (date x stock)
    """
    factor_series = {}
    for code, df in price_dict.items():
        try:
            factors = compute_all_factors(df)
            if factor_name in factors.columns:
                factor_series[code] = factors[factor_name]
        except Exception:
            continue
    return pd.DataFrame(factor_series)
