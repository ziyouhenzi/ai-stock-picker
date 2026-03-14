"""
data/fetcher.py - 数据获取模块
支持 akshare / tushare 双数据源，自动缓存
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_stock_list(universe="hs300"):
    """获取股票池列表"""
    cache_file = f"{CACHE_DIR}/stock_list_{universe}.csv"
    if os.path.exists(cache_file):
        age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        if age < 86400:  # 缓存1天
            return pd.read_csv(cache_file, dtype={"code": str})

    try:
        import akshare as ak
        if universe == "hs300":
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            df = df.rename(columns={"成分券代码": "code", "成分券名称": "name", "权重": "weight"})
        elif universe == "zz500":
            df = ak.index_stock_cons_weight_csindex(symbol="000905")
            df = df.rename(columns={"成分券代码": "code", "成分券名称": "name", "权重": "weight"})
        else:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "code", "name": "name"})
            df["weight"] = 1.0
        df[["code", "name"]].to_csv(cache_file, index=False)
        return df[["code", "name"]]
    except Exception as e:
        print(f"[Warning] 获取股票列表失败: {e}，使用模拟数据")
        return _mock_stock_list()


def get_daily_data(code, start_date, end_date):
    """获取个股日线数据"""
    cache_file = f"{CACHE_DIR}/{code}_{start_date}_{end_date}.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"], index_col="date")

    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="hfq"
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.to_csv(cache_file)
        return df
    except Exception as e:
        return _mock_daily_data(code, start_date, end_date)


def get_index_data(index_code, start_date, end_date):
    """获取指数数据作为基准"""
    cache_file = f"{CACHE_DIR}/index_{index_code}_{start_date}_{end_date}.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"], index_col="date")

    try:
        import akshare as ak
        df = ak.index_zh_a_hist(
            symbol=index_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "")
        )
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["close"]].sort_index()
        df.to_csv(cache_file)
        return df
    except Exception as e:
        return _mock_index_data(start_date, end_date)


def _mock_stock_list():
    """模拟股票列表（用于无网络环境测试）"""
    stocks = [
        ("600000", "浦发银行"), ("600036", "招商银行"), ("600519", "贵州茅台"),
        ("601318", "中国平安"), ("000001", "平安银行"), ("000858", "五粮液"),
        ("300750", "宁德时代"), ("600900", "长江电力"), ("601899", "紫金矿业"),
        ("002594", "比亚迪"), ("600276", "恒瑞医药"), ("601888", "中国国旅"),
        ("000333", "美的集团"), ("600309", "万华化学"), ("601166", "兴业银行"),
        ("000725", "京东方A"), ("002415", "海康威视"), ("600030", "中信证券"),
        ("601628", "中国人寿"), ("000568", "泸州老窖"),
    ]
    return pd.DataFrame(stocks, columns=["code", "name"])


def _mock_daily_data(code, start_date, end_date):
    """生成模拟日线数据"""
    np.random.seed(hash(code) % 2**32)
    dates = pd.date_range(start_date, end_date, freq="B")
    n = len(dates)
    returns = np.random.normal(0.0003, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "open": close * np.random.uniform(0.98, 1.02, n),
        "high": close * np.random.uniform(1.00, 1.05, n),
        "low": close * np.random.uniform(0.95, 1.00, n),
        "close": close,
        "volume": np.random.randint(1000000, 50000000, n).astype(float),
        "amount": close * np.random.randint(1000000, 50000000, n),
        "turnover": np.random.uniform(0.5, 5.0, n),
    }, index=dates)
    df.index.name = "date"
    return df


def _mock_index_data(start_date, end_date):
    """生成模拟指数数据"""
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq="B")
    n = len(dates)
    returns = np.random.normal(0.0002, 0.015, n)
    close = 4000 * np.cumprod(1 + returns)
    df = pd.DataFrame({"close": close}, index=dates)
    df.index.name = "date"
    return df
