# ====================================================
# config.py - 全局配置
# ====================================================

# 数据源配置
DATA_SOURCE = "akshare"   # akshare | tushare
TUSHARE_TOKEN = ""        # 如使用 tushare 请填入 token

# 股票池
UNIVERSE = "hs300"        # hs300 | zz500 | all_a

# 回测参数
BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 1_000_000,   # 初始资金（元）
    "commission": 0.0003,           # 佣金率
    "slippage": 0.001,              # 滑点
    "rebalance_freq": "monthly",    # 调仓频率: weekly | monthly | quarterly
    "top_n": 20,                    # 每次持仓股票数
    "benchmark": "000300",          # 基准指数：沪深300
}

# 模型参数
MODEL_CONFIG = {
    "model_type": "random_forest",  # random_forest | xgboost | lgbm
    "n_estimators": 200,
    "max_depth": 6,
    "train_window": 252,            # 训练窗口（交易日）
    "predict_horizon": 20,          # 预测周期（交易日）
    "label_threshold": 0.05,        # 涨幅阈值，超过即为正样本
}

# 因子列表
FACTOR_LIST = [
    # 技术因子
    "momentum_5d", "momentum_20d", "momentum_60d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "rsi_14", "macd_signal", "bollinger_pos",
    "volume_ratio_5d", "turnover_rate",
    "atr_14", "volatility_20d",
    # 基本面因子
    "pe_ratio", "pb_ratio", "roe", "roa",
    "revenue_growth", "profit_growth",
    "debt_ratio", "current_ratio",
]

# 输出路径
OUTPUT_DIR = "output"
MODEL_DIR = "output/models"
REPORT_DIR = "output/reports"
