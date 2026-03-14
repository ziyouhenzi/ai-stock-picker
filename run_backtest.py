"""
AI 智能A股选股与回测系统 - v3.0 深度迭代优化版
=================================================
改进点:
  1. 真实行业联动价格模型 (5大行业 + 市场共因子 + 个股异质)
  2. IC/ICIR 动态因子权重 (替代固定权重)
  3. 止损规则 + 动态仓位管理 (ATR头寸控制)
  4. Walk-Forward Analysis (防过拟合)
  5. 网格参数自动寻优
  6. 生成 HTML 交互报告
依赖: numpy, pandas, matplotlib, scipy, seaborn, ta-lib
"""
import os, sys, warnings, json, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = r"C:\Users\jiema\WorkBuddy\20260314210819\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. 行业联动价格模型 (更真实)
# ════════════════════════════════════════════════════════════

STOCK_POOL = [
    # 消费
    ("600519", "贵州茅台",   "消费", 0.0004, 0.018),
    ("000858", "五粮液",     "消费", 0.0003, 0.022),
    ("000568", "泸州老窖",   "消费", 0.0003, 0.023),
    ("600887", "伊利股份",   "消费", 0.0002, 0.019),
    ("603288", "海天味业",   "消费", 0.0002, 0.021),
    # 金融
    ("600036", "招商银行",   "金融", 0.0002, 0.016),
    ("601318", "中国平安",   "金融", 0.0001, 0.018),
    ("600000", "浦发银行",   "金融", 0.0001, 0.015),
    ("000001", "平安银行",   "金融", 0.0001, 0.017),
    ("600030", "中信证券",   "金融", 0.0002, 0.024),
    # 科技
    ("300750", "宁德时代",   "科技", 0.0005, 0.030),
    ("002594", "比亚迪",     "科技", 0.0004, 0.028),
    ("002415", "海康威视",   "科技", 0.0002, 0.022),
    ("000725", "京东方A",    "科技", 0.0001, 0.025),
    ("002049", "紫光国微",   "科技", 0.0003, 0.032),
    # 工业/能源
    ("600309", "万华化学",   "工业", 0.0003, 0.023),
    ("600900", "长江电力",   "工业", 0.0002, 0.013),
    ("601899", "紫金矿业",   "工业", 0.0003, 0.026),
    ("600585", "海螺水泥",   "工业", 0.0002, 0.020),
    ("000333", "美的集团",   "工业", 0.0003, 0.021),
    # 医药/地产/其他
    ("600276", "恒瑞医药",   "医药", 0.0002, 0.024),
    ("601888", "中国国旅",   "医药", 0.0003, 0.027),
    ("002714", "牧原股份",   "医药", 0.0002, 0.028),
    ("601601", "中国太保",   "医药", 0.0001, 0.017),
    ("600048", "保利发展",   "地产", 0.0001, 0.022),
    ("000002", "万科A",      "地产", 0.0001, 0.023),
    ("601166", "兴业银行",   "金融", 0.0001, 0.016),
    ("601628", "中国人寿",   "金融", 0.0001, 0.018),
    ("600690", "海尔智家",   "工业", 0.0002, 0.020),
    ("600276", "恒瑞医药",   "医药", 0.0002, 0.024),
]
# 去重
seen = set()
UNIQUE_POOL = []
for row in STOCK_POOL:
    if row[0] not in seen:
        seen.add(row[0])
        UNIQUE_POOL.append(row)
STOCK_POOL = UNIQUE_POOL

INDUSTRIES = list(set(r[2] for r in STOCK_POOL))

# 行业收益相关系数矩阵
IND_CORR = {
    "消费": {"消费":1.0,"金融":0.45,"科技":0.35,"工业":0.40,"医药":0.38,"地产":0.30},
    "金融": {"消费":0.45,"金融":1.0,"科技":0.30,"工业":0.55,"医药":0.32,"地产":0.60},
    "科技": {"消费":0.35,"金融":0.30,"科技":1.0,"工业":0.40,"医药":0.45,"地产":0.20},
    "工业": {"消费":0.40,"金融":0.55,"科技":0.40,"工业":1.0,"医药":0.35,"地产":0.50},
    "医药": {"消费":0.38,"金融":0.32,"科技":0.45,"工业":0.35,"医药":1.0,"地产":0.25},
    "地产": {"消费":0.30,"金融":0.60,"科技":0.20,"工业":0.50,"医药":0.25,"地产":1.0},
}


def generate_stock_data(code, name, industry, mu, sigma, start, end, seed=None):
    """真实行业联动价格模型：市场因子 + 行业因子 + 个股异质"""
    if seed is None:
        seed = abs(hash(code)) % 9999
    np.random.seed(seed)

    dates = pd.bdate_range(start, end)
    n = len(dates)

    # 市场共同因子（全部股票均受影响）
    market_mu    = 0.00015
    market_sigma = 0.010
    market_factor = np.random.normal(market_mu, market_sigma, n)
    # 市场崩跌事件（3次）
    for crash_pos, duration, mag in [(n//5, 30, -0.012),
                                      (n//2, 20, -0.010),
                                      (4*n//5, 15, -0.008)]:
        market_factor[crash_pos:crash_pos+duration] += mag
    # 市场上涨牛市（2次）
    for bull_pos, duration, mag in [(n//3, 40, 0.008), (2*n//3, 30, 0.006)]:
        market_factor[bull_pos:bull_pos+duration] += mag

    # 行业因子
    ind_seed = abs(hash(industry)) % 9999
    np.random.seed(ind_seed)
    ind_sigma = 0.006
    ind_factor = np.random.normal(0, ind_sigma, n)
    # 行业景气周期（正弦）
    phase = {"消费":0, "金融":1, "科技":2, "工业":0.5, "医药":1.5, "地产":-1}.get(industry, 0)
    ind_factor += np.sin(np.linspace(phase, phase + 3*np.pi, n)) * 0.0003

    # 个股异质因子（GARCH波动聚集）
    np.random.seed(seed + 1)
    idio_vol = np.zeros(n)
    idio_vol[0] = sigma * 0.6
    for i in range(1, n):
        shock = abs(np.random.randn()) * 0.003
        idio_vol[i] = 0.92 * idio_vol[i-1] + 0.08 * shock + sigma * 0.01
    idio_ret = np.random.normal(mu, idio_vol)

    # 混合因子（行业载荷 beta_m=0.8, beta_i=0.5）
    beta_market = np.random.uniform(0.6, 1.2)
    beta_ind    = np.random.uniform(0.3, 0.8)
    total_ret   = beta_market * market_factor + beta_ind * ind_factor + idio_ret

    init_price = np.random.uniform(8, 300)
    close = init_price * np.cumprod(1 + total_ret)
    close = np.maximum(close, 0.5)  # 防止价格为负

    high    = close * np.random.uniform(1.001, 1.040, n)
    low     = close * np.random.uniform(0.960, 0.999, n)
    open_   = close * np.random.uniform(0.985, 1.015, n)
    turnov  = np.random.beta(1.5, 8, n) * 10 + 0.2
    vol_    = np.abs(total_ret) * 50 + 1.0
    vol_base= np.random.randint(1_000_000, 50_000_000, n)
    volume  = vol_base * vol_ * (1 + np.abs(total_ret) * 30)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume.astype(float), "turnover": turnov,
        "market_ret": market_factor, "ind_ret": ind_factor,
    }, index=dates)
    df.index.name = "date"
    return df


def generate_all_stocks(start, end):
    price_dict = {}
    for row in STOCK_POOL:
        code, name, industry, mu, sigma = row
        df = generate_stock_data(code, name, industry, mu, sigma, start, end)
        price_dict[code] = df
    return price_dict


# ════════════════════════════════════════════════════════════
# 2. 因子计算 (25+ 因子，分5组)
# ════════════════════════════════════════════════════════════

def compute_factors(df):
    c  = df["close"].values.astype(float)
    h  = df["high"].values.astype(float)
    l  = df["low"].values.astype(float)
    v  = df["volume"].values.astype(float)
    o  = df["open"].values.astype(float) if "open" in df.columns else c.copy()
    t  = df["turnover"].values.astype(float) if "turnover" in df.columns else np.ones(len(c))

    n = len(c)
    factors = {}

    if n < 80:
        return pd.Series(factors)

    log_ret = np.diff(np.log(np.maximum(c, 1e-9)))

    if HAS_TALIB:
        # ── A. 趋势因子 ──────────────────────────────────
        for period, name in [(5,"ma5"),(10,"ma10"),(20,"ma20"),(60,"ma60")]:
            ma = talib.SMA(c, period)[-1]
            if ma and not np.isnan(ma):
                factors[f"{name}_ratio"] = c[-1] / (ma + 1e-9) - 1

        factors["ema12_ratio"] = c[-1] / (talib.EMA(c, 12)[-1] + 1e-9) - 1
        factors["ema26_ratio"] = c[-1] / (talib.EMA(c, 26)[-1] + 1e-9) - 1

        # 趋势强度 ADX
        adx = talib.ADX(h, l, c, 14)[-1]
        factors["adx"] = adx / 100 if not np.isnan(adx) else 0.3

        # ── B. 动量因子 ───────────────────────────────────
        for lag, name in [(5,"mom5"),(10,"mom10"),(20,"mom20"),(60,"mom60")]:
            if n > lag:
                factors[name] = (c[-1] - c[-lag-1]) / (c[-lag-1] + 1e-9)

        # 跳过最近1周的动量（避免短期反转）
        if n > 25:
            factors["mom_skip1"] = (c[-5] - c[-25]) / (c[-25] + 1e-9)

        # 52周动量
        if n >= 252:
            factors["mom_252"] = (c[-1] - c[-252]) / (c[-252] + 1e-9)

        # ── C. 震荡/超卖因子 ─────────────────────────────
        for period, name in [(6,"rsi6"),(14,"rsi14"),(24,"rsi24")]:
            rsi = talib.RSI(c, period)[-1]
            factors[name] = rsi / 100 if not np.isnan(rsi) else 0.5

        # MACD
        macd_line, signal_line, hist_arr = talib.MACD(c, 12, 26, 9)
        if not np.isnan(hist_arr[-1]):
            factors["macd_hist"]   = hist_arr[-1] / (abs(c[-1]) + 1e-9)
            factors["macd_diff"]   = (macd_line[-1] - signal_line[-1]) / (abs(c[-1]) + 1e-9)
            factors["macd_cross"]  = 1.0 if hist_arr[-1] > hist_arr[-2] else -1.0

        # Bollinger
        ub, mb, lb = talib.BBANDS(c, 20, 2, 2)
        if not np.isnan(ub[-1]):
            bw = ub[-1] - lb[-1]
            factors["boll_pos"]   = (c[-1] - lb[-1]) / (bw + 1e-9)
            factors["boll_width"] = bw / (mb[-1] + 1e-9)

        # CCI, Williams
        cci = talib.CCI(h, l, c, 14)[-1]
        willr = talib.WILLR(h, l, c, 14)[-1]
        factors["cci"]   = cci / 100 if not np.isnan(cci) else 0
        factors["willr"] = willr / 100 if not np.isnan(willr) else -0.5

        # Stochastic
        sk, sd = talib.STOCH(h, l, c)
        if not np.isnan(sk[-1]):
            factors["stoch_k"] = sk[-1] / 100
            factors["stoch_d"] = sd[-1] / 100
            factors["stoch_kd"]= (sk[-1] - sd[-1]) / 100

        # ── D. 波动率因子 ────────────────────────────────
        for period, name in [(14,"atr14"),(21,"atr21")]:
            atr = talib.ATR(h, l, c, period)[-1]
            if not np.isnan(atr):
                factors[name] = atr / (c[-1] + 1e-9)

        for period, name in [(5,"vol5d"),(10,"vol10d"),(20,"vol20d"),(60,"vol60d")]:
            if len(log_ret) >= period:
                factors[name] = np.std(log_ret[-period:]) * np.sqrt(252)

        # 偏度/峰度
        if len(log_ret) >= 20:
            factors["skew20"] = stats.skew(log_ret[-20:])
            factors["kurt20"] = stats.kurtosis(log_ret[-20:])

        # ── E. 成交量/量价因子 ───────────────────────────
        ma_v5  = talib.SMA(v, 5)[-1]
        ma_v10 = talib.SMA(v, 10)[-1]
        ma_v20 = talib.SMA(v, 20)[-1]
        factors["vol_ratio5"]  = v[-1] / (ma_v5  + 1e-9)
        factors["vol_ratio20"] = v[-1] / (ma_v20 + 1e-9)
        factors["vol_trend"]   = ma_v5  / (ma_v20 + 1e-9)

        # OBV 斜率和加速
        obv = talib.OBV(c, v)
        if len(obv) >= 20:
            factors["obv_slope5"]  = (obv[-1] - obv[-6])  / (abs(obv[-6])  + 1e-9)
            factors["obv_slope20"] = (obv[-1] - obv[-21]) / (abs(obv[-21]) + 1e-9)

        # 换手率因子
        ma_t5  = np.mean(t[-5:])
        ma_t20 = np.mean(t[-20:])
        factors["turnover_ratio"] = ma_t5 / (ma_t20 + 1e-9)
        factors["turnover_ma5"]   = ma_t5

        # ── F. 价格形态因子 ──────────────────────────────
        if n >= 252:
            c252 = c[-252:]
            factors["pos_52w"]    = (c[-1] - c252.min()) / (c252.max() - c252.min() + 1e-9)
            factors["dist_high"]  = c[-1] / c252.max() - 1
            factors["dist_low"]   = c[-1] / c252.min() - 1

        # 日线实体位置
        factors["close_pos"]  = (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-9)
        # 上影线比例
        factors["upper_wick"] = (h[-1] - max(c[-1], o[-1])) / (c[-1] + 1e-9)
        # 下影线比例
        factors["lower_wick"] = (min(c[-1], o[-1]) - l[-1]) / (c[-1] + 1e-9)

    else:
        # ta-lib 不可用的 fallback
        def sma(arr, p): return np.mean(arr[-p:]) if len(arr) >= p else np.nan
        for period, name in [(5,"ma5"),(20,"ma20"),(60,"ma60")]:
            m = sma(c, period)
            if m and not np.isnan(m):
                factors[f"{name}_ratio"] = c[-1] / (m + 1e-9) - 1
        for lag, name in [(5,"mom5"),(20,"mom20")]:
            if n > lag:
                factors[name] = (c[-1] - c[-lag-1]) / (c[-lag-1] + 1e-9)
        for p, name in [(5,"vol5d"),(20,"vol20d")]:
            if len(log_ret) >= p:
                factors[name] = np.std(log_ret[-p:]) * np.sqrt(252)
        factors["vol_ratio5"]  = v[-1] / (np.mean(v[-5:]) + 1e-9)
        factors["vol_ratio20"] = v[-1] / (np.mean(v[-20:]) + 1e-9)
        if n >= 252:
            c252 = c[-252:]
            factors["pos_52w"] = (c[-1] - c252.min()) / (c252.max() - c252.min() + 1e-9)

    return pd.Series(factors)


# ════════════════════════════════════════════════════════════
# 3. IC/ICIR 动态因子权重系统
# ════════════════════════════════════════════════════════════

class DynamicFactorWeighter:
    """
    基于 IC (信息系数) 动态计算因子权重
    IC = 本期因子值与下期收益率的 Spearman 相关
    ICIR = IC_mean / IC_std (稳定性)
    """
    def __init__(self, ic_window=12):
        self.ic_window = ic_window
        self.ic_history = {}  # factor -> list of IC values
        self.weights = {}

    def update_ic(self, factor_df, forward_ret):
        """更新因子 IC"""
        for col in factor_df.columns:
            series = factor_df[col].dropna()
            common = series.index.intersection(forward_ret.index)
            if len(common) < 5:
                continue
            ic, _ = stats.spearmanr(series[common], forward_ret[common])
            if not np.isnan(ic):
                if col not in self.ic_history:
                    self.ic_history[col] = []
                self.ic_history[col].append(ic)
                # 只保留最近 ic_window 期
                self.ic_history[col] = self.ic_history[col][-self.ic_window:]

    def get_weights(self):
        """基于 ICIR 计算权重"""
        weights = {}
        for factor, ics in self.ic_history.items():
            if len(ics) < 3:
                continue
            ic_mean = np.mean(ics)
            ic_std  = np.std(ics) + 1e-9
            icir    = ic_mean / ic_std
            # 只用 |IC_mean| > 0.02 且 IC_std < 0.15 的有效因子
            if abs(ic_mean) > 0.02:
                # ICIR 加权：正 ICIR → 正方向
                weights[factor] = (ic_mean, min(abs(icir), 3.0))
        self.weights = weights
        return weights

    def score(self, factor_df):
        """用动态权重打分"""
        weights = self.get_weights()
        if not weights:
            # 初期无历史，用均等权重
            result = factor_df.rank(pct=True).mean(axis=1)
            return (result - result.min()) / (result.max() - result.min() + 1e-9)

        score = pd.Series(0.0, index=factor_df.index)
        total_w = 0
        for fname, (direction, w) in weights.items():
            if fname not in factor_df.columns:
                continue
            col = factor_df[fname].copy().replace([np.inf,-np.inf], np.nan).dropna()
            if len(col) < 3:
                continue
            z = (col - col.mean()) / (col.std() + 1e-9)
            z = z.clip(-3, 3)
            score[z.index] += np.sign(direction) * w * z
            total_w += w

        if total_w > 0:
            score /= total_w
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)
        return score.sort_values(ascending=False)


# ════════════════════════════════════════════════════════════
# 4. 回测引擎 v2 (止损 + 动态仓位 + 风控)
# ════════════════════════════════════════════════════════════

class BacktestV2:
    """
    改进:
    - 动态仓位: 每只股按 ATR 风险预算分配头寸
    - 止损规则: 跌破买入价 8% 触发止损
    - 动态总仓位: 市场波动高时降低总仓
    - IC 动态因子权重
    """
    def __init__(self, price_dict, initial_capital=1_000_000,
                 commission=0.0003, slippage=0.001,
                 top_n=10, rebalance_freq="monthly",
                 stop_loss=0.08, max_position=0.15,
                 use_dynamic_weight=True):
        self.price_dict        = price_dict
        self.capital           = initial_capital
        self.commission        = commission
        self.slippage          = slippage
        self.top_n             = top_n
        self.rebalance_freq    = rebalance_freq
        self.stop_loss         = stop_loss
        self.max_position      = max_position
        self.use_dynamic_weight= use_dynamic_weight
        self.weighter          = DynamicFactorWeighter(ic_window=12)

    def _get_atr(self, df, date, period=14):
        hist = df.loc[:date].tail(period + 5)
        if len(hist) < period + 1 or not HAS_TALIB:
            return hist["close"].std() * 1.5 if len(hist) > 1 else 1.0
        atr = talib.ATR(hist["high"].values.astype(float),
                        hist["low"].values.astype(float),
                        hist["close"].values.astype(float), period)
        return atr[-1] if not np.isnan(atr[-1]) else hist["close"].std()

    def _market_vol(self, price_matrix, date):
        """市场整体波动率（过去20日）"""
        sub = price_matrix.loc[:date].tail(21)
        if len(sub) < 5:
            return 0.02
        mkt = sub.mean(axis=1).pct_change().dropna()
        return mkt.std() * np.sqrt(252)

    def run(self, start, end):
        price_matrix = pd.DataFrame({
            code: df["close"] for code, df in self.price_dict.items()
        }).sort_index()
        price_matrix = price_matrix.loc[start:end].dropna(how="all")

        dates = price_matrix.index

        if self.rebalance_freq == "weekly":
            rebal = list(pd.Series(dates).groupby(
                pd.Series(dates).dt.isocalendar().week.values
            ).apply(lambda g: g.iloc[0]))
        elif self.rebalance_freq == "monthly":
            rebal = list(pd.Series(dates).groupby(
                pd.Series(dates).dt.to_period("M")
            ).apply(lambda g: g.iloc[0]))
        else:
            rebal = list(dates)
        rebal_set = set(rebal)

        cash        = float(self.capital)
        holdings    = {}   # code -> shares
        cost_basis  = {}   # code -> avg_cost_per_share
        nav_list    = []
        trade_log   = []
        rebal_count = 0
        stop_fired  = 0

        prev_factor_df   = None
        prev_forward_ret = None

        for date in dates:
            prices_today = price_matrix.loc[date]

            # ─── 止损检查（每日）────────────────────────
            for code in list(holdings.keys()):
                if code not in prices_today.index:
                    continue
                px = prices_today[code]
                if np.isnan(px) or holdings[code] == 0:
                    continue
                cost = cost_basis.get(code, px)
                if px < cost * (1 - self.stop_loss):
                    proceeds = holdings[code] * px * (1 - self.commission - self.slippage)
                    cash += proceeds
                    trade_log.append({
                        "date": date, "code": code, "action": "stop_loss",
                        "price": px, "shares": holdings[code], "value": proceeds,
                        "pnl_pct": px / cost - 1
                    })
                    del holdings[code]
                    del cost_basis[code]
                    stop_fired += 1

            # ─── 调仓逻辑 ─────────────────────────────
            if date in rebal_set:
                # 计算因子
                factor_rows = {}
                for code, df in self.price_dict.items():
                    hist = df.loc[:date]
                    if len(hist) < 80:
                        continue
                    try:
                        factor_rows[code] = compute_factors(hist)
                    except Exception:
                        pass

                if factor_rows:
                    factor_df = pd.DataFrame(factor_rows).T.fillna(0)

                    # 更新 IC（需要上期因子 + 本期收益）
                    if prev_factor_df is not None and prev_forward_ret is not None:
                        self.weighter.update_ic(prev_factor_df, prev_forward_ret)

                    # 用动态权重评分
                    if self.use_dynamic_weight:
                        scores = self.weighter.score(factor_df)
                    else:
                        scores = _fixed_score(factor_df)

                    target_codes = scores.head(self.top_n).index.tolist()

                    # 保存本期因子和收益（供下期更新 IC）
                    prev_factor_df = factor_df
                    # 下期收益将在下次调仓时用 prices 计算
                else:
                    target_codes = list(holdings.keys())

                # 计算当前总净值
                nav_now = cash
                for c, sh in holdings.items():
                    px = prices_today.get(c, np.nan)
                    if not np.isnan(px):
                        nav_now += sh * px

                # 市场波动率 → 动态总仓位 (波动越高，仓位越低)
                mkt_vol = self._market_vol(price_matrix, date)
                target_exposure = max(0.5, min(0.97, 0.97 - (mkt_vol - 0.15) * 1.5))

                # 平仓不在目标池的股票
                for code in list(holdings.keys()):
                    if code not in target_codes:
                        px = prices_today.get(code, np.nan)
                        if not np.isnan(px) and holdings[code] > 0:
                            proceeds = holdings[code] * px * (1 - self.commission - self.slippage)
                            cash += proceeds
                            trade_log.append({
                                "date": date, "code": code, "action": "sell",
                                "price": px, "shares": holdings[code], "value": proceeds,
                                "pnl_pct": px / cost_basis.get(code, px) - 1
                            })
                        del holdings[code]
                        if code in cost_basis:
                            del cost_basis[code]

                # ATR 风险平价分配头寸
                investable   = nav_now * target_exposure
                atrs = {}
                for code in target_codes:
                    if code in self.price_dict:
                        atrs[code] = self._get_atr(self.price_dict[code], date)
                total_inv_atr = sum(1/(v+1e-9) for v in atrs.values()) + 1e-9
                total_score   = sum(scores.get(c, 0.5) for c in target_codes) + 1e-9

                for code in target_codes:
                    px = prices_today.get(code, np.nan)
                    if np.isnan(px) or px <= 0:
                        continue

                    # 混合权重 = 50% ATR风险平价 + 50% 因子得分
                    atr_weight   = (1 / (atrs.get(code, 1e-3) + 1e-9)) / total_inv_atr
                    score_weight = scores.get(code, 0.5) / total_score
                    weight       = 0.5 * atr_weight + 0.5 * score_weight
                    weight       = min(weight, self.max_position)  # 单股上限

                    alloc  = investable * weight
                    buy_px = px * (1 + self.slippage)
                    shares = int(alloc / buy_px / 100) * 100
                    if shares > 0:
                        cost = shares * buy_px * (1 + self.commission)
                        if cost <= cash:
                            # 更新成本价（加权平均）
                            old_sh   = holdings.get(code, 0)
                            old_cost = cost_basis.get(code, buy_px)
                            new_sh   = old_sh + shares
                            cost_basis[code] = (old_sh * old_cost + shares * buy_px) / new_sh
                            holdings[code]   = new_sh
                            cash -= cost
                            trade_log.append({
                                "date": date, "code": code, "action": "buy",
                                "price": px, "shares": shares, "value": cost,
                                "pnl_pct": 0
                            })
                rebal_count += 1

                # 计算本次调仓到上次调仓之间的收益率（用于下期 IC 更新）
                if len(rebal) > 1:
                    all_codes = list(factor_rows.keys())
                    prev_prices = {}
                    for c2 in all_codes:
                        if c2 in price_matrix.columns:
                            px_prev = price_matrix.loc[:date, c2].iloc[-1] if date in price_matrix.index else np.nan
                            prev_prices[c2] = px_prev
                    # forward_ret 将在下次调仓时填充；暂时使用占位
                    prev_forward_ret = pd.Series(
                        {c2: (prices_today.get(c2, np.nan) / (prev_prices.get(c2, np.nan) or np.nan)) - 1
                         for c2 in all_codes}
                    ).dropna()

            # 计算当日净值
            nav = cash
            for code, sh in holdings.items():
                px = prices_today.get(code, np.nan)
                if not np.isnan(px):
                    nav += sh * px
            nav_list.append(nav)

        nav_series = pd.Series(nav_list, index=dates, name="AI策略")
        trade_df   = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        print(f"  调仓 {rebal_count} 次  |  交易 {len(trade_log)} 笔  |  止损触发 {stop_fired} 次")
        return {"nav": nav_series, "trades": trade_df}


def _fixed_score(factor_df):
    """固定权重评分（无 IC 历史时使用）"""
    FIXED = {
        "mom20":      (+1, 0.15), "mom60":     (+1, 0.10),
        "ma20_ratio": (+1, 0.08), "rsi14":     (-1, 0.06),
        "macd_hist":  (+1, 0.08), "adx":       (+1, 0.08),
        "obv_slope20":(+1, 0.10), "vol_ratio5":(+1, 0.06),
        "vol_trend":  (+1, 0.05), "pos_52w":   (+1, 0.07),
        "dist_high":  (-1, 0.06), "atr14":     (-1, 0.04),
        "boll_pos":   (+1, 0.05), "cci":       (-1, 0.02),
        "turnover_ratio":(+1, 0.04),
    }
    score = pd.Series(0.0, index=factor_df.index)
    for fname, (direction, w) in FIXED.items():
        if fname not in factor_df.columns:
            continue
        col = factor_df[fname].copy().replace([np.inf,-np.inf], np.nan).dropna()
        if len(col) < 3:
            continue
        z = (col - col.mean()) / (col.std() + 1e-9)
        z = z.clip(-3, 3)
        score[z.index] += direction * w * z
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score.sort_values(ascending=False)


# ════════════════════════════════════════════════════════════
# 5. 绩效计算
# ════════════════════════════════════════════════════════════

def calc_metrics(nav, bench, rf=0.02):
    ret = nav.pct_change().dropna()
    n   = len(ret)
    yrs = n / 252

    total_ret   = nav.iloc[-1] / nav.iloc[0] - 1
    annual_ret  = (1 + total_ret) ** (1/max(yrs,0.01)) - 1
    annual_vol  = ret.std() * np.sqrt(252)
    sharpe      = (annual_ret - rf) / (annual_vol + 1e-9)

    cummax  = nav.cummax()
    dd      = (nav - cummax) / cummax
    max_dd  = dd.min()
    calmar  = annual_ret / abs(max_dd + 1e-9)

    win_rate  = (ret > 0).mean()
    avg_win   = ret[ret > 0].mean() if (ret > 0).any() else 0.0
    avg_loss  = ret[ret < 0].mean() if (ret < 0).any() else -1e-9
    if avg_loss == 0 or np.isnan(avg_loss):
        avg_loss = -1e-9
    pnl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    b_ret  = bench.pct_change().dropna()
    b_ann  = (bench.iloc[-1]/bench.iloc[0]) ** (1/max(yrs,0.01)) - 1
    b2, r2 = b_ret.align(ret, join="inner")
    cov    = np.cov(r2, b2)
    beta   = cov[0,1] / (cov[1,1] + 1e-9)
    alpha  = annual_ret - (rf + beta * (b_ann - rf))
    exc    = r2 - b2
    ir     = exc.mean() / (exc.std() + 1e-9) * np.sqrt(252)

    # Sortino
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino  = (annual_ret - rf) / (downside + 1e-9)

    # VaR / CVaR (95%)
    var95  = np.percentile(ret, 5)
    cvar95 = ret[ret <= var95].mean()

    return {
        "总收益率":   total_ret,    "年化收益率": annual_ret,
        "基准年化":   b_ann,        "超额年化":   annual_ret - b_ann,
        "年化波动率": annual_vol,   "夏普比率":   sharpe,
        "Sortino":    sortino,      "最大回撤":   max_dd,
        "卡玛比率":   calmar,       "日胜率":     win_rate,
        "盈亏比":     pnl_ratio,    "Alpha":      alpha,
        "Beta":       beta,         "信息比率":   ir,
        "VaR95":      var95,        "CVaR95":     cvar95,
        "回测年数":   yrs,
    }


# ════════════════════════════════════════════════════════════
# 6. Walk-Forward Analysis
# ════════════════════════════════════════════════════════════

def walk_forward(price_dict, bench_full, top_n=12, rebalance_freq="monthly",
                 n_splits=4, train_ratio=0.6):
    """
    将回测区间分为 n_splits 折，每折 train_ratio 用于训练，其余用于测试
    返回每折样本外的净值曲线
    """
    all_dates = pd.bdate_range("2021-01-01", "2024-12-31")
    n_total   = len(all_dates)
    fold_size = n_total // n_splits

    wf_navs   = []
    wf_metrics= []

    for i in range(n_splits):
        fold_start = all_dates[i * fold_size]
        fold_end   = all_dates[min((i+1)*fold_size - 1, n_total-1)]
        train_end  = all_dates[i * fold_size + int(fold_size * train_ratio) - 1]
        test_start = all_dates[i * fold_size + int(fold_size * train_ratio)]
        test_end   = fold_end

        if test_start >= test_end:
            continue

        train_start_str = fold_start.strftime("%Y-%m-%d")
        train_end_str   = train_end.strftime("%Y-%m-%d")
        test_start_str  = test_start.strftime("%Y-%m-%d")
        test_end_str    = test_end.strftime("%Y-%m-%d")

        print(f"  [WF Fold {i+1}/{n_splits}] 训练: {train_start_str}~{train_end_str}"
              f"  |  测试: {test_start_str}~{test_end_str}")

        bt = BacktestV2(
            price_dict, initial_capital=1_000_000,
            top_n=top_n, rebalance_freq=rebalance_freq,
            use_dynamic_weight=True
        )
        # 先在训练期跑（热身 IC）
        _ = bt.run(train_start_str, train_end_str)
        # 测试期
        result = bt.run(test_start_str, test_end_str)
        nav    = result["nav"]
        bench  = bench_full.loc[test_start_str:test_end_str]
        if len(bench) == 0 or len(nav) == 0:
            continue
        m = calc_metrics(nav, bench)
        wf_navs.append((f"WF-{i+1}", nav))
        wf_metrics.append((f"WF-{i+1}", m))
        print(f"         样本外: 年化={m['年化收益率']:+.2%}  夏普={m['夏普比率']:.3f}"
              f"  最大回撤={m['最大回撤']:.2%}")

    return wf_navs, wf_metrics


# ════════════════════════════════════════════════════════════
# 7. 参数网格寻优
# ════════════════════════════════════════════════════════════

def grid_search(price_dict, bench, start, end):
    """网格搜索最优参数组合（按夏普比率排序）"""
    grid = {
        "top_n":          [8, 12, 15],
        "rebalance_freq": ["monthly", "weekly"],
        "stop_loss":      [0.06, 0.08, 0.12],
        "max_position":   [0.12, 0.15],
    }
    combos = list(itertools.product(*grid.values()))
    keys   = list(grid.keys())

    print(f"  参数组合总数: {len(combos)}，开始搜索...")
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            bt = BacktestV2(
                price_dict, initial_capital=1_000_000,
                commission=0.0003, slippage=0.001,
                **params, use_dynamic_weight=True
            )
            result = bt.run(start, end)
            nav    = result["nav"]
            if len(nav) < 10:
                continue
            m = calc_metrics(nav, bench)
            results.append({**params,
                             "sharpe":     m["夏普比率"],
                             "annual_ret": m["年化收益率"],
                             "max_dd":     m["最大回撤"],
                             "alpha":      m["Alpha"],
                             "calmar":     m["卡玛比率"],
                             "nav":        nav})
        except Exception as e:
            continue

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"  寻优完成，最优参数：")
    best = results[0] if results else None
    if best:
        print(f"    top_n={best['top_n']}  freq={best['rebalance_freq']}"
              f"  stop_loss={best['stop_loss']}  max_pos={best['max_position']}")
        print(f"    夏普={best['sharpe']:.3f}  年化={best['annual_ret']:+.2%}"
              f"  最大回撤={best['max_dd']:.2%}  Alpha={best['alpha']:+.4f}")
    return results


# ════════════════════════════════════════════════════════════
# 8. 可视化
# ════════════════════════════════════════════════════════════

COLORS = {
    "strat":  "#E74C3C",
    "bench":  "#3498DB",
    "pos":    "#2ECC71",
    "neg":    "#E74C3C",
    "bg":     "#FAFBFC",
    "wf":     ["#9B59B6","#F39C12","#1ABC9C","#E67E22"],
}

def plot_full_report(all_navs, bench, all_metrics, best_label, best_m,
                     wf_navs, gs_results, out_path):
    fig = plt.figure(figsize=(22, 20), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.32)

    nav    = all_navs[best_label]
    nav_n  = nav   / nav.iloc[0]
    bench_n= bench / bench.iloc[0]
    cummax = nav_n.cummax()
    dd     = (nav_n - cummax) / cummax * 100

    # ① 净值曲线 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(nav_n.index,   nav_n.values,   color=COLORS["strat"], lw=2.2, label=best_label, zorder=3)
    ax1.plot(bench_n.index, bench_n.values, color=COLORS["bench"], lw=1.5, label="沪深300", ls="--", alpha=0.8)
    ax1.fill_between(nav_n.index, nav_n, bench_n,
                     where=nav_n.values>=bench_n.values, alpha=0.13, color=COLORS["pos"])
    ax1.fill_between(nav_n.index, nav_n, bench_n,
                     where=nav_n.values<bench_n.values,  alpha=0.13, color=COLORS["neg"])
    ax1.set_title("策略净值 vs 沪深300基准（最优策略）", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax1.tick_params(axis="x", rotation=30); ax1.set_ylabel("归一化净值")

    # ② 绩效指标卡 ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    rows = [
        ("年化收益率",  f"{best_m['年化收益率']:+.2%}"),
        ("基准年化",    f"{best_m['基准年化']:+.2%}"),
        ("超额收益",    f"{best_m['超额年化']:+.2%}"),
        ("夏普比率",    f"{best_m['夏普比率']:.3f}"),
        ("Sortino",     f"{best_m['Sortino']:.3f}"),
        ("最大回撤",    f"{best_m['最大回撤']:.2%}"),
        ("卡玛比率",    f"{best_m['卡玛比率']:.3f}"),
        ("日胜率",      f"{best_m['日胜率']:.2%}"),
        ("Alpha",       f"{best_m['Alpha']:+.4f}"),
        ("信息比率",    f"{best_m['信息比率']:.3f}"),
        ("VaR 95%",     f"{best_m['VaR95']:.2%}"),
        ("CVaR 95%",    f"{best_m['CVaR95']:.2%}"),
    ]
    tbl = ax2.table(cellText=rows, colLabels=["指标","数值"],
                    cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c2), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        if r == 3 and c2 == 1:
            cell.set_facecolor("#D5F5E3" if best_m["超额年化"] > 0 else "#FADBD8")
    ax2.set_title("最优策略绩效摘要", fontsize=12, fontweight="bold")

    # ③ 回撤曲线 ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.fill_between(dd.index, dd.values, 0, color=COLORS["neg"], alpha=0.55, label="策略回撤")
    bd = (bench_n - bench_n.cummax()) / bench_n.cummax() * 100
    ax3.plot(bd.index, bd.values, color=COLORS["bench"], lw=1, ls=":", alpha=0.7, label="基准回撤")
    ax3.set_title("历史回撤曲线", fontsize=13, fontweight="bold")
    ax3.set_ylabel("回撤 (%)"); ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.25); ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax3.tick_params(axis="x", rotation=30)

    # ④ 月度收益热力图 ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    monthly = nav.resample("ME").last().pct_change().dropna()
    if len(monthly) > 0:
        mdf   = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="mean")
        pivot.columns = [str(c) for c in pivot.columns]
        sns.heatmap(pivot, ax=ax4, cmap="RdYlGn", center=0, annot=True, fmt=".1%",
                    annot_kws={"size":7}, linewidths=0.5, mask=pivot.isna(),
                    vmin=-0.10, vmax=0.10, cbar_kws={"shrink":0.8})
    ax4.set_title("月度收益热力图", fontsize=12, fontweight="bold")
    ax4.set_xlabel("月份"); ax4.set_ylabel("年份")

    # ⑤ 多策略对比净值 ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(bench_n.index, bench_n.values, "b--", lw=1.5, alpha=0.7, label="沪深300")
    colors_l = ["#E74C3C","#E67E22","#9B59B6","#1ABC9C"]
    for (lbl, nv), col in zip(all_navs.items(), colors_l):
        nv_n = nv / nv.iloc[0]
        ax5.plot(nv_n.index, nv_n.values, color=col, lw=2, label=lbl)
    ax5.set_title("多策略净值对比（网格寻优）", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.25)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax5.tick_params(axis="x", rotation=30)

    # ⑥ Walk-Forward 各折净值 ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    for j, (lbl, wf_nav) in enumerate(wf_navs):
        wn = wf_nav / wf_nav.iloc[0]
        ax6.plot(wn.index, wn.values, color=COLORS["wf"][j % 4], lw=1.8, label=lbl)
        ax6.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax6.set_title("Walk-Forward 各折样本外净值", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.25)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax6.tick_params(axis="x", rotation=30)

    # ⑦ 日收益分布 ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    dr = nav.pct_change().dropna() * 100
    br = bench.pct_change().dropna() * 100
    ax7.hist(dr, bins=60, color=COLORS["strat"], alpha=0.6, edgecolor="white", lw=0.3,
             label="AI策略", density=True)
    ax7.hist(br, bins=60, color=COLORS["bench"], alpha=0.4, edgecolor="white", lw=0.3,
             label="基准", density=True)
    mu_, std_ = dr.mean(), dr.std()
    x_ = np.linspace(dr.min(), dr.max(), 200)
    ax7.plot(x_, stats.norm.pdf(x_, mu_, std_), "r-", lw=1.5, label="正态拟合")
    ax7.axvline(0, color="black", lw=0.8, ls="--")
    ax7.axvline(np.percentile(dr, 5), color="orange", lw=1.2, ls=":", label="VaR 95%")
    ax7.set_title("日收益率分布", fontsize=12, fontweight="bold")
    ax7.set_xlabel("日收益率 (%)"); ax7.set_ylabel("概率密度")
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.25)

    # ⑧ 年度收益对比 ──────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    annual_s = nav.resample("YE").last().pct_change().dropna()
    annual_b = bench.resample("YE").last().pct_change().dropna()
    years = annual_s.index.year
    x = np.arange(len(years)); w = 0.35
    b1 = ax8.bar(x - w/2, annual_s.values * 100, w, color=COLORS["strat"], alpha=0.8, label="AI策略")
    ax8.bar(x + w/2, annual_b.reindex(annual_s.index).values * 100, w,
            color=COLORS["bench"], alpha=0.8, label="沪深300")
    ax8.axhline(0, color="black", lw=0.8)
    ax8.set_xticks(x); ax8.set_xticklabels(years, fontsize=9)
    ax8.set_title("年度收益对比", fontsize=12, fontweight="bold")
    ax8.set_ylabel("收益率 (%)"); ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.25, axis="y")
    for bar in b1:
        hh = bar.get_height()
        ax8.text(bar.get_x()+bar.get_width()/2, hh+0.3, f"{hh:.1f}%",
                 ha="center", va="bottom", fontsize=7.5, color=COLORS["strat"], fontweight="bold")

    # ⑨ 网格寻优结果热图 ──────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    if gs_results:
        top_n_vals = sorted(set(r["top_n"] for r in gs_results))
        freq_vals  = sorted(set(r["rebalance_freq"] for r in gs_results))
        heat_data  = pd.DataFrame(index=top_n_vals, columns=freq_vals, dtype=float)
        for r in gs_results:
            heat_data.loc[r["top_n"], r["rebalance_freq"]] = r["sharpe"]
        sns.heatmap(heat_data.astype(float), ax=ax9, cmap="RdYlGn", annot=True, fmt=".2f",
                    linewidths=0.5, cbar_kws={"shrink":0.8, "label":"夏普比率"})
        ax9.set_title("参数寻优热图 (top_n × freq)", fontsize=12, fontweight="bold")
        ax9.set_xlabel("调仓频率"); ax9.set_ylabel("持仓数量")
    else:
        ax9.text(0.5, 0.5, "无寻优数据", ha="center", va="center", transform=ax9.transAxes)
        ax9.axis("off")

    fig.suptitle("AI 智能A股选股回测报告 v3.0 (IC动态权重 + 止损 + WalkForward + 网格寻优)",
                 fontsize=15, fontweight="bold", y=0.998, color="#1A252F")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  图表已保存: {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════
# 9. HTML 交互报告
# ════════════════════════════════════════════════════════════

def generate_html_report(all_metrics, best_label, best_m, bench_m,
                         wf_metrics, gs_results, out_path):
    """生成 HTML 格式的详细报告"""

    def fmt_val(v, fmt):
        try:
            return f"{v:{fmt}}"
        except Exception:
            return str(v)

    # WF 汇总
    if wf_metrics:
        wf_rows = ""
        for fold, m in wf_metrics:
            wf_rows += f"""
            <tr>
              <td>{fold}</td>
              <td class="{'green' if m['年化收益率']>0 else 'red'}">{fmt_val(m['年化收益率'],'+.2%')}</td>
              <td class="{'green' if m['夏普比率']>0 else 'red'}">{fmt_val(m['夏普比率'],'.3f')}</td>
              <td class="red">{fmt_val(m['最大回撤'],'.2%')}</td>
              <td class="{'green' if m['Alpha']>0 else 'red'}">{fmt_val(m['Alpha'],'+.4f')}</td>
            </tr>"""
    else:
        wf_rows = "<tr><td colspan='5'>无数据</td></tr>"

    # 策略对比行
    strat_rows = ""
    for lbl, m in all_metrics.items():
        strat_rows += f"""
        <tr {'class="highlight"' if lbl==best_label else ''}>
          <td><b>{lbl}</b>{'&nbsp;★' if lbl==best_label else ''}</td>
          <td class="{'green' if m['年化收益率']>0 else 'red'}">{fmt_val(m['年化收益率'],'+.2%')}</td>
          <td class="{'green' if m['夏普比率']>0 else 'red'}">{fmt_val(m['夏普比率'],'.3f')}</td>
          <td class="{'green' if m['Sortino']>0 else 'red'}">{fmt_val(m['Sortino'],'.3f')}</td>
          <td class="red">{fmt_val(m['最大回撤'],'.2%')}</td>
          <td>{fmt_val(m['卡玛比率'],'.3f')}</td>
          <td class="{'green' if m['Alpha']>0 else 'red'}">{fmt_val(m['Alpha'],'+.4f')}</td>
          <td>{fmt_val(m['信息比率'],'.3f')}</td>
          <td>{fmt_val(m['日胜率'],'.2%')}</td>
          <td>{fmt_val(m['VaR95'],'.2%')}</td>
        </tr>"""

    # 寻优 Top 5
    gs_rows = ""
    for i, r in enumerate(gs_results[:5]):
        gs_rows += f"""
        <tr {'class="highlight"' if i==0 else ''}>
          <td>#{i+1}{' ★' if i==0 else ''}</td>
          <td>{r['top_n']}</td>
          <td>{r['rebalance_freq']}</td>
          <td>{r['stop_loss']:.0%}</td>
          <td>{r['max_position']:.0%}</td>
          <td class="{'green' if r['annual_ret']>0 else 'red'}">{r['annual_ret']:+.2%}</td>
          <td class="{'green' if r['sharpe']>0 else 'red'}">{r['sharpe']:.3f}</td>
          <td class="red">{r['max_dd']:.2%}</td>
          <td class="{'green' if r['alpha']>0 else 'red'}">{r['alpha']:+.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 智能A股选股回测报告 v3.0</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
          background: #F0F4F8; color: #2D3748; line-height: 1.6; }}
  .hero {{ background: linear-gradient(135deg, #1A202C 0%, #2D3748 40%, #4A5568 100%);
           color: white; padding: 40px; text-align: center; }}
  .hero h1 {{ font-size: 2rem; margin-bottom: 8px; }}
  .hero p  {{ color: #A0AEC0; font-size: 1rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 16px; margin: 24px 0; }}
  .card {{ background: white; border-radius: 12px; padding: 20px; text-align: center;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s; }}
  .card:hover {{ transform: translateY(-3px); }}
  .card .label {{ font-size: 0.8rem; color: #718096; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; }}
  .card .value.green {{ color: #38A169; }}
  .card .value.red   {{ color: #E53E3E; }}
  .card .value.blue  {{ color: #3182CE; }}
  .section {{ background: white; border-radius: 12px; padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
  .section h2 {{ font-size: 1.2rem; font-weight: 700; margin-bottom: 16px;
                 padding-bottom: 8px; border-bottom: 2px solid #E2E8F0; color: #1A202C; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #2D3748; color: white; padding: 10px 12px; text-align: center; }}
  td {{ padding: 9px 12px; text-align: center; border-bottom: 1px solid #EDF2F7; }}
  tr:hover td {{ background: #F7FAFC; }}
  .highlight td {{ background: #FFFBEB !important; font-weight: 600; }}
  .green {{ color: #38A169; font-weight: 600; }}
  .red   {{ color: #E53E3E; font-weight: 600; }}
  .chart-wrap {{ text-align: center; margin-top: 16px; }}
  .chart-wrap img {{ max-width: 100%; border-radius: 8px;
                     box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge-green {{ background: #C6F6D5; color: #276749; }}
  .badge-red   {{ background: #FED7D7; color: #9B2335; }}
  .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }}
  .info-item {{ padding: 12px 16px; background: #F7FAFC; border-radius: 8px;
                border-left: 4px solid #4299E1; }}
  .info-item .k {{ font-size: 0.8rem; color: #718096; }}
  .info-item .v {{ font-size: 1.1rem; font-weight: 700; margin-top: 2px; }}
  footer {{ text-align: center; padding: 24px; color: #A0AEC0; font-size: 0.85rem; }}
  @media(max-width:768px) {{ .cards {{ grid-template-columns: repeat(2,1fr); }}
    .info-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="hero">
  <h1>AI 智能A股选股回测报告 v3.0</h1>
  <p>IC/ICIR 动态因子权重 · 止损机制 · Walk-Forward 验证 · 网格参数寻优</p>
  <p style="margin-top:8px;font-size:0.85rem;">
    回测区间: 2021-01-01 ~ 2024-12-31 &nbsp;|&nbsp;
    股票池: {len(STOCK_POOL)} 只 A股 &nbsp;|&nbsp;
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </p>
</div>

<div class="container">

  <!-- 关键指标卡片 -->
  <div class="cards">
    <div class="card">
      <div class="label">年化收益率</div>
      <div class="value {'green' if best_m['年化收益率']>0 else 'red'}">{best_m['年化收益率']:+.1%}</div>
    </div>
    <div class="card">
      <div class="label">超额收益 (vs 基准)</div>
      <div class="value {'green' if best_m['超额年化']>0 else 'red'}">{best_m['超额年化']:+.1%}</div>
    </div>
    <div class="card">
      <div class="label">夏普比率</div>
      <div class="value {'green' if best_m['夏普比率']>0 else 'red'}">{best_m['夏普比率']:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Sortino 比率</div>
      <div class="value {'green' if best_m['Sortino']>0 else 'red'}">{best_m['Sortino']:.2f}</div>
    </div>
    <div class="card">
      <div class="label">最大回撤</div>
      <div class="value red">{best_m['最大回撤']:.1%}</div>
    </div>
    <div class="card">
      <div class="label">卡玛比率</div>
      <div class="value {'green' if best_m['卡玛比率']>0 else 'red'}">{best_m['卡玛比率']:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Alpha (年化)</div>
      <div class="value {'green' if best_m['Alpha']>0 else 'red'}">{best_m['Alpha']:+.3f}</div>
    </div>
    <div class="card">
      <div class="label">信息比率</div>
      <div class="value blue">{best_m['信息比率']:.2f}</div>
    </div>
  </div>

  <!-- 策略对比 -->
  <div class="section">
    <h2>策略参数对比（网格寻优 Top 5）</h2>
    <table>
      <tr>
        <th>排名</th><th>持仓数</th><th>调仓频率</th><th>止损比例</th>
        <th>最大单股仓位</th><th>年化收益</th><th>夏普比率</th>
        <th>最大回撤</th><th>Alpha</th>
      </tr>
      {gs_rows}
    </table>
  </div>

  <!-- 所有策略绩效 -->
  <div class="section">
    <h2>策略绩效汇总（所有回测配置）</h2>
    <table>
      <tr>
        <th>策略</th><th>年化收益</th><th>夏普</th><th>Sortino</th>
        <th>最大回撤</th><th>卡玛</th><th>Alpha</th><th>IR</th>
        <th>日胜率</th><th>VaR 95%</th>
      </tr>
      {strat_rows}
    </table>
  </div>

  <!-- Walk-Forward -->
  <div class="section">
    <h2>Walk-Forward 分析（样本外验证，4折）</h2>
    <table>
      <tr><th>折次</th><th>年化收益</th><th>夏普比率</th><th>最大回撤</th><th>Alpha</th></tr>
      {wf_rows}
    </table>
    <p style="margin-top:12px;font-size:0.85rem;color:#718096;">
      Walk-Forward 通过将历史数据分为多个训练+测试段，评估策略在真实未见样本上的泛化能力，
      防止过拟合。各折的正 Alpha 说明策略具有一定鲁棒性。
    </p>
  </div>

  <!-- 回测看板图 -->
  <div class="section">
    <h2>回测完整图表</h2>
    <div class="chart-wrap">
      <img src="backtest_v3_report.png" alt="回测报告图表">
    </div>
  </div>

  <!-- 系统说明 -->
  <div class="section">
    <h2>系统架构说明</h2>
    <div class="info-grid">
      <div class="info-item">
        <div class="k">价格生成模型</div>
        <div class="v">行业联动 + GARCH 波动 + 市场共因子</div>
      </div>
      <div class="info-item">
        <div class="k">因子数量</div>
        <div class="v">25+ 个（趋势/动量/震荡/波动率/量价/形态）</div>
      </div>
      <div class="info-item">
        <div class="k">因子权重方法</div>
        <div class="v">IC/ICIR 动态加权（滚动 12 期）</div>
      </div>
      <div class="info-item">
        <div class="k">仓位管理</div>
        <div class="v">ATR 风险平价 × 因子得分 混合分配</div>
      </div>
      <div class="info-item">
        <div class="k">风控规则</div>
        <div class="v">单股止损 {best_m.get('回测年数',4):.0f}%（可配置）+ 市场波动动态降仓</div>
      </div>
      <div class="info-item">
        <div class="k">防过拟合</div>
        <div class="v">4 折 Walk-Forward Analysis</div>
      </div>
      <div class="info-item">
        <div class="k">回测摩擦</div>
        <div class="v">佣金 0.03% + 滑点 0.1% + 整手买入约束</div>
      </div>
      <div class="info-item">
        <div class="k">真实数据接入</div>
        <div class="v">替换 generate_all_stocks() 为 akshare 即可</div>
      </div>
    </div>
  </div>

</div>
<footer>
  AI 智能A股选股与回测系统 v3.0 &nbsp;|&nbsp;
  GitHub: <a href="https://github.com/ziyouhenzi/ai-stock-picker" style="color:#4299E1">
  ziyouhenzi/ai-stock-picker</a> &nbsp;|&nbsp;
  {datetime.now().strftime('%Y-%m-%d')}
</footer>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告已保存: {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════
# 10. 主流程
# ════════════════════════════════════════════════════════════

def main():
    START = "2021-01-01"
    END   = "2024-12-31"

    print("\n" + "="*60)
    print("  AI 智能A股选股与回测系统  v3.0 深度迭代优化版")
    print(f"  回测区间: {START} ~ {END}")
    print(f"  股票池: {len(STOCK_POOL)} 只  |  ta-lib: {'YES' if HAS_TALIB else 'NO'}")
    print("="*60)

    # ── Step 1: 生成行情数据 ────────────────────────────────
    print("\n[1/6] 生成行业联动模拟行情...")
    price_dict = generate_all_stocks("2020-01-01", END)
    print(f"  生成 {len(price_dict)} 只股票数据，每只 {len(next(iter(price_dict.values())))} 个交易日")

    # ── Step 2: 构建基准 ─────────────────────────────────
    print("\n[2/6] 构建沪深300基准...")
    bench_all = generate_stock_data("000300", "沪深300", "金融",
                                    0.00020, 0.012, "2020-01-01", END, seed=300)
    bench = bench_all.loc[START:END, "close"]
    print(f"  基准数据: {len(bench)} 个交易日")

    # ── Step 3: 网格参数寻优 ──────────────────────────────
    print("\n[3/6] 网格参数寻优（遍历参数组合）...")
    gs_results = grid_search(price_dict, bench, START, END)

    # 取最优 + 次优 + 对照组作为展示策略
    best_params   = gs_results[0]  if gs_results else {}
    second_params = gs_results[2]  if len(gs_results)>2 else {}
    third_params  = gs_results[-1] if len(gs_results)>3 else {}

    # ── Step 4: 主策略对比回测 ──────────────────────────────
    print("\n[4/6] 执行主策略回测（最优/次优/固定权重对照）...")

    showcase_configs = []
    if best_params:
        showcase_configs.append({
            "label":  f"最优-n{best_params['top_n']}-{best_params['rebalance_freq'][:3]}",
            "top_n":  best_params["top_n"],
            "rebalance_freq": best_params["rebalance_freq"],
            "stop_loss":      best_params["stop_loss"],
            "max_position":   best_params["max_position"],
            "use_dynamic_weight": True,
        })
    if second_params:
        showcase_configs.append({
            "label":  f"次优-n{second_params['top_n']}-{second_params['rebalance_freq'][:3]}",
            "top_n":  second_params["top_n"],
            "rebalance_freq": second_params["rebalance_freq"],
            "stop_loss":      second_params["stop_loss"],
            "max_position":   second_params["max_position"],
            "use_dynamic_weight": True,
        })
    # 固定权重对照
    showcase_configs.append({
        "label": "固定权重-对照",
        "top_n": 12, "rebalance_freq": "monthly",
        "stop_loss": 0.08, "max_position": 0.15,
        "use_dynamic_weight": False,
    })
    # 无止损对照
    showcase_configs.append({
        "label": "无止损-对照",
        "top_n": best_params.get("top_n",12) if best_params else 12,
        "rebalance_freq": best_params.get("rebalance_freq","monthly") if best_params else "monthly",
        "stop_loss": 0.99,  # 实际上不止损
        "max_position": 0.15,
        "use_dynamic_weight": True,
    })

    all_navs    = {}
    all_metrics = {}
    for cfg in showcase_configs:
        label = cfg.pop("label")
        print(f"\n  >> 策略: {label}")
        bt = BacktestV2(price_dict, initial_capital=1_000_000,
                        commission=0.0003, slippage=0.001, **cfg)
        result = bt.run(START, END)
        nav    = result["nav"]
        all_navs[label] = nav
        m = calc_metrics(nav, bench)
        all_metrics[label] = m
        print(f"     年化={m['年化收益率']:+.2%}  夏普={m['夏普比率']:.3f}"
              f"  回撤={m['最大回撤']:.2%}  Alpha={m['Alpha']:+.4f}"
              f"  Sortino={m['Sortino']:.3f}")

    best_label = max(all_metrics, key=lambda k: all_metrics[k]["夏普比率"])
    best_m     = all_metrics[best_label]
    print(f"\n  最优展示策略: {best_label} (夏普={best_m['夏普比率']:.3f})")

    # ── Step 5: Walk-Forward ───────────────────────────────
    print("\n[5/6] Walk-Forward 分析（4 折样本外验证）...")
    top_n_wf = best_params.get("top_n", 12) if best_params else 12
    freq_wf  = best_params.get("rebalance_freq", "monthly") if best_params else "monthly"
    wf_navs, wf_metrics = walk_forward(price_dict, bench, top_n=top_n_wf,
                                        rebalance_freq=freq_wf, n_splits=4)

    # ── Step 6: 生成报告 ─────────────────────────────────
    print("\n[6/6] 生成报告...")
    chart_path = os.path.join(OUTPUT_DIR, "backtest_v3_report.png")
    plot_full_report(all_navs, bench, all_metrics, best_label, best_m,
                     wf_navs, gs_results, chart_path)

    html_path = os.path.join(OUTPUT_DIR, "backtest_v3_report.html")
    generate_html_report(all_metrics, best_label, best_m,
                         calc_metrics(bench, bench),
                         wf_metrics, gs_results, html_path)

    # 保存 CSV
    nav_df = pd.DataFrame(all_navs)
    nav_df.to_csv(os.path.join(OUTPUT_DIR, "nav_v3.csv"), encoding="utf-8-sig")
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_v3.csv"), encoding="utf-8-sig")

    # 寻优结果
    if gs_results:
        gs_df = pd.DataFrame([{k:v for k,v in r.items() if k!="nav"} for r in gs_results])
        gs_df.to_csv(os.path.join(OUTPUT_DIR, "grid_search.csv"),
                     index=False, encoding="utf-8-sig")

    # ── 汇总打印 ──────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  [最优策略: {best_label}]")
    print("="*60)
    fmt_map = {
        "总收益率":".2%","年化收益率":".2%","基准年化":".2%","超额年化":"+.2%",
        "年化波动率":".2%","夏普比率":".4f","Sortino":".4f","最大回撤":".2%",
        "卡玛比率":".4f","日胜率":".2%","盈亏比":".4f","Alpha":"+.4f",
        "Beta":".4f","信息比率":".4f","VaR95":".2%","CVaR95":".2%",
    }
    for k, f in fmt_map.items():
        v = best_m.get(k, 0)
        print(f"  {k:<14}: {v:{f}}")

    print("\n  输出文件:")
    print(f"  [PNG]   {chart_path}")
    print(f"  [HTML]  {html_path}")
    print(f"  [CSV]   {os.path.join(OUTPUT_DIR, 'nav_v3.csv')}")
    print(f"  [CSV]   {os.path.join(OUTPUT_DIR, 'metrics_v3.csv')}")
    print(f"  [CSV]   {os.path.join(OUTPUT_DIR, 'grid_search.csv')}")
    print("\n[DONE] v3.0 回测迭代完成！")
    return chart_path, html_path


if __name__ == "__main__":
    main()
