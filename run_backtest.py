"""
AI 智能A股选股与回测系统 - 优化独立运行版
===========================================
依赖: numpy, pandas, matplotlib, scipy, seaborn, ta-lib (均已安装)
无需 scikit-learn / xgboost
运行: python run_backtest.py
"""
import os
import sys
import warnings
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

# ── 尝试导入 ta-lib ────────────────────────────────────────
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
# 1. 数据模拟（真实数据需联网）
# ════════════════════════════════════════════════════════════

STOCK_POOL = [
    ("600519", "贵州茅台"), ("600036", "招商银行"), ("000858", "五粮液"),
    ("601318", "中国平安"), ("300750", "宁德时代"), ("600900", "长江电力"),
    ("000333", "美的集团"), ("600309", "万华化学"), ("002594", "比亚迪"),
    ("600276", "恒瑞医药"), ("601888", "中国国旅"), ("000568", "泸州老窖"),
    ("002415", "海康威视"), ("600030", "中信证券"), ("601899", "紫金矿业"),
    ("000725", "京东方A"),  ("601166", "兴业银行"), ("600000", "浦发银行"),
    ("000001", "平安银行"), ("600309", "万华化学"), ("601628", "中国人寿"),
    ("600048", "保利发展"), ("000002", "万科A"),    ("600887", "伊利股份"),
    ("002714", "牧原股份"), ("601601", "中国太保"), ("600585", "海螺水泥"),
    ("603288", "海天味业"), ("002049", "紫光国微"), ("600690", "海尔智家"),
]

def generate_stock_data(code: str, start: str, end: str,
                         mu: float = 0.0003, sigma: float = 0.022,
                         seed: int = None) -> pd.DataFrame:
    """生成带趋势和波动聚集的模拟股价"""
    if seed is None:
        seed = int(code) % 9999 if code.isdigit() else hash(code) % 9999
    np.random.seed(seed)

    dates = pd.bdate_range(start, end)
    n = len(dates)

    # GARCH 风格波动：低波动持续低，高波动持续高
    vol = np.zeros(n)
    vol[0] = sigma
    for i in range(1, n):
        shock = abs(np.random.randn()) * 0.003
        vol[i] = 0.94 * vol[i-1] + 0.06 * shock + 0.001

    # 加入行业周期（正弦波动）
    cycle = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0002

    returns = np.random.normal(mu + cycle, vol)

    # 模拟市场崩溃（2次小回调）
    crash1 = n // 4
    crash2 = 3 * n // 4
    returns[crash1:crash1+20] -= 0.008
    returns[crash2:crash2+15] -= 0.006

    init_price = np.random.uniform(10, 200)
    close = init_price * np.cumprod(1 + returns)
    high  = close * np.random.uniform(1.001, 1.035, n)
    low   = close * np.random.uniform(0.965, 0.999, n)
    open_ = close * np.random.uniform(0.985, 1.015, n)
    vol_  = np.random.randint(500_000, 30_000_000, n) * (1 + np.abs(returns) * 20)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol_.astype(float),
        "turnover": np.random.uniform(0.3, 8.0, n),
    }, index=dates)
    df.index.name = "date"
    return df


# ════════════════════════════════════════════════════════════
# 2. 因子计算（使用 ta-lib 精准指标）
# ════════════════════════════════════════════════════════════

def compute_factors(df: pd.DataFrame) -> pd.Series:
    """计算单股全因子，返回最新截面 Series"""
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)

    factors = {}

    if HAS_TALIB:
        # ── 趋势类 ────────────────────────────────────
        factors["ma5_ratio"]  = c[-1] / talib.SMA(c, 5)[-1]  - 1
        factors["ma20_ratio"] = c[-1] / talib.SMA(c, 20)[-1] - 1
        factors["ma60_ratio"] = c[-1] / talib.SMA(c, 60)[-1] - 1
        factors["ema12_ratio"]= c[-1] / talib.EMA(c, 12)[-1] - 1

        # ── 动量类 ────────────────────────────────────
        factors["mom_5"]  = talib.MOM(c, 5)[-1]  / (c[-6]  + 1e-9)
        factors["mom_20"] = talib.MOM(c, 20)[-1] / (c[-21] + 1e-9)
        factors["mom_60"] = talib.MOM(c, 60)[-1] / (c[-61] + 1e-9) if len(c) > 60 else 0

        # ── 震荡指标 ──────────────────────────────────
        factors["rsi_6"]  = talib.RSI(c, 6)[-1]  / 100
        factors["rsi_14"] = talib.RSI(c, 14)[-1] / 100
        factors["rsi_24"] = talib.RSI(c, 24)[-1] / 100

        # MACD
        macd, signal, hist = talib.MACD(c, 12, 26, 9)
        factors["macd_hist"]   = hist[-1]  / (abs(c[-1]) + 1e-9)
        factors["macd_signal"] = 1 if hist[-1] > 0 else -1

        # Bollinger Band 位置
        upper, mid, lower = talib.BBANDS(c, 20, 2, 2)
        factors["boll_pos"] = (c[-1] - lower[-1]) / (upper[-1] - lower[-1] + 1e-9)
        factors["boll_width"] = (upper[-1] - lower[-1]) / (mid[-1] + 1e-9)

        # CCI
        factors["cci_14"] = talib.CCI(h, l, c, 14)[-1] / 100

        # Williams %R
        factors["willr_14"] = talib.WILLR(h, l, c, 14)[-1] / 100

        # Stochastic
        slowk, slowd = talib.STOCH(h, l, c)
        factors["stoch_k"] = slowk[-1] / 100
        factors["stoch_d"] = slowd[-1] / 100

        # ADX
        factors["adx_14"] = talib.ADX(h, l, c, 14)[-1] / 100

        # ── 波动率类 ──────────────────────────────────
        factors["atr_ratio"]  = talib.ATR(h, l, c, 14)[-1] / c[-1]
        factors["natr"]       = talib.NATR(h, l, c, 14)[-1] / 100

        log_ret = np.diff(np.log(c + 1e-9))
        factors["vol_5d"]  = np.std(log_ret[-5:])  * np.sqrt(252)
        factors["vol_20d"] = np.std(log_ret[-20:]) * np.sqrt(252)

        # ── 成交量类 ──────────────────────────────────
        ma_v5  = talib.SMA(v, 5)[-1]
        ma_v20 = talib.SMA(v, 20)[-1]
        factors["vol_ratio_5"]  = v[-1] / (ma_v5  + 1e-9)
        factors["vol_ratio_20"] = v[-1] / (ma_v20 + 1e-9)
        factors["vol_trend"]    = ma_v5 / (ma_v20 + 1e-9)

        # OBV 斜率
        obv = talib.OBV(c, v)
        factors["obv_slope"] = (obv[-1] - obv[-20]) / (abs(obv[-20]) + 1e-9) if len(obv) > 20 else 0

        # ── 形态类 ────────────────────────────────────
        factors["price_pos_52w"] = (c[-1] - np.min(c[-252:])) / (np.max(c[-252:]) - np.min(c[-252:]) + 1e-9)
        factors["dist_52w_high"] = c[-1] / np.max(c[-252:]) - 1
        factors["close_pos"]     = (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-9)

    else:
        # 无 ta-lib 的 fallback（纯 numpy）
        def sma(arr, n): return np.mean(arr[-n:])
        def ema(arr, n):
            k = 2/(n+1)
            e = arr[0]
            for x in arr[1:]: e = x*k + e*(1-k)
            return e

        for n, name in [(5,"ma5"),(20,"ma20"),(60,"ma60")]:
            if len(c) >= n:
                factors[f"{name}_ratio"] = c[-1]/sma(c,n) - 1
        if len(c) >= 5:
            factors["mom_5"] = (c[-1]-c[-6])/(c[-6]+1e-9)
        if len(c) >= 20:
            factors["mom_20"] = (c[-1]-c[-21])/(c[-21]+1e-9)
        log_ret = np.diff(np.log(c+1e-9))
        factors["vol_5d"]  = np.std(log_ret[-5:])*np.sqrt(252)
        factors["vol_20d"] = np.std(log_ret[-20:])*np.sqrt(252)
        factors["vol_ratio_5"]  = v[-1]/(np.mean(v[-5:])+1e-9)
        factors["vol_ratio_20"] = v[-1]/(np.mean(v[-20:])+1e-9)
        if len(c) >= 252:
            factors["price_pos_52w"] = (c[-1]-np.min(c[-252:]))/(np.max(c[-252:])-np.min(c[-252:])+1e-9)

    return pd.Series(factors)


# ════════════════════════════════════════════════════════════
# 3. 综合评分模型（多因子加权打分）
# ════════════════════════════════════════════════════════════

# 因子方向和权重（正=越大越好，负=越小越好）
FACTOR_WEIGHTS = {
    "mom_20":       (+1, 0.15),  # 中期动量
    "mom_60":       (+1, 0.10),  # 长期动量
    "ma20_ratio":   (+1, 0.08),  # 均线偏离
    "rsi_14":       (-1, 0.06),  # RSI 逆势（不能过热）
    "macd_signal":  (+1, 0.08),  # MACD 趋势
    "boll_pos":     (+1, 0.05),  # 布林带位置
    "adx_14":       (+1, 0.08),  # 趋势强度
    "obv_slope":    (+1, 0.10),  # 量价配合
    "vol_ratio_5":  (+1, 0.06),  # 短期放量
    "vol_trend":    (+1, 0.05),  # 量能趋势
    "price_pos_52w":(+1, 0.07),  # 52周相对位置
    "dist_52w_high":(-1, 0.06),  # 距高点距离（远离高点更安全）
    "atr_ratio":    (-1, 0.04),  # 波动率（低波动优先）
    "cci_14":       (-1, 0.02),  # CCI 逆势
}

def score_stocks(factor_df: pd.DataFrame) -> pd.Series:
    """
    多因子综合评分
    1. 对每个因子做截面 Z-score 标准化
    2. 按权重加权求和
    3. 最终得分映射到 [0,1]
    """
    score = pd.Series(0.0, index=factor_df.index)
    used = 0
    for fname, (direction, weight) in FACTOR_WEIGHTS.items():
        if fname not in factor_df.columns:
            continue
        col = factor_df[fname].copy()
        col = col.replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) < 3:
            continue
        # Z-score 标准化
        z = (col - col.mean()) / (col.std() + 1e-9)
        z = z.clip(-3, 3)
        score[z.index] += direction * weight * z
        used += 1

    # 归一化到 0~1
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score.sort_values(ascending=False)


# ════════════════════════════════════════════════════════════
# 4. 回测引擎
# ════════════════════════════════════════════════════════════

class Backtest:
    def __init__(self, price_dict, initial_capital=1_000_000,
                 commission=0.0003, slippage=0.001,
                 top_n=10, rebalance_freq="monthly"):
        self.price_dict = price_dict
        self.capital    = initial_capital
        self.commission = commission
        self.slippage   = slippage
        self.top_n      = top_n
        self.rebalance_freq = rebalance_freq

    def run(self, start: str, end: str) -> dict:
        print(f"\n{'='*55}")
        print(f"  开始回测: {start} → {end}")
        print(f"  初始资金: {self.capital:,.0f} 元")
        print(f"  持仓数量: {self.top_n} 只  |  调仓频率: {self.rebalance_freq}")
        print(f"{'='*55}")

        # 构建价格矩阵
        price_matrix = pd.DataFrame({
            code: df["close"] for code, df in self.price_dict.items()
        }).sort_index()
        price_matrix = price_matrix.loc[start:end].dropna(how="all")

        dates = price_matrix.index
        # 获取调仓日
        if self.rebalance_freq == "weekly":
            rebal = list(pd.Series(dates).groupby(
                pd.Series(dates).dt.isocalendar().week.values).apply(lambda g: g.iloc[0]))
        elif self.rebalance_freq == "monthly":
            rebal = list(pd.Series(dates).groupby(
                pd.Series(dates).dt.to_period("M")).apply(lambda g: g.iloc[0]))
        else:
            rebal = list(dates)
        rebal_set = set(rebal)

        cash = float(self.capital)
        holdings = {}  # code → shares
        nav_list = []
        trade_log = []
        rebal_count = 0

        for date in dates:
            # 调仓逻辑
            if date in rebal_set:
                # 计算截面因子
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
                    factor_df = pd.DataFrame(factor_rows).T
                    scores = score_stocks(factor_df)
                    target_codes = scores.head(self.top_n).index.tolist()
                else:
                    target_codes = list(holdings.keys())

                # 计算当前净值
                nav_now = cash
                for c, sh in holdings.items():
                    px = price_matrix.loc[date, c] if c in price_matrix.columns else np.nan
                    if not np.isnan(px):
                        nav_now += sh * px

                # 平仓不在目标池的股票
                for code in list(holdings.keys()):
                    if code not in target_codes:
                        if code in price_matrix.columns:
                            px = price_matrix.loc[date, code]
                            if not np.isnan(px):
                                proceeds = holdings[code] * px * (1 - self.commission - self.slippage)
                                cash += proceeds
                                trade_log.append({"date": date, "code": code,
                                                   "action": "sell", "price": px,
                                                   "shares": holdings[code], "value": proceeds})
                        del holdings[code]

                # 按评分权重分配资金
                total_score = sum(scores[c] for c in target_codes if c in scores.index)
                investable = nav_now * 0.97
                for code in target_codes:
                    if code not in price_matrix.columns:
                        continue
                    px = price_matrix.loc[date, code]
                    if np.isnan(px) or px <= 0:
                        continue
                    sc = scores.get(code, 1/self.top_n)
                    alloc = investable * (sc / (total_score + 1e-9))
                    buy_px = px * (1 + self.slippage)
                    shares = int(alloc / buy_px / 100) * 100
                    if shares > 0:
                        cost = shares * buy_px * (1 + self.commission)
                        if cost <= cash:
                            cash -= cost
                            holdings[code] = holdings.get(code, 0) + shares
                            trade_log.append({"date": date, "code": code,
                                               "action": "buy", "price": px,
                                               "shares": shares, "value": cost})
                rebal_count += 1

            # 计算当日净值
            nav = cash
            for code, shares in holdings.items():
                if code in price_matrix.columns:
                    px = price_matrix.loc[date, code]
                    if not np.isnan(px):
                        nav += shares * px
            nav_list.append(nav)

        nav_series = pd.Series(nav_list, index=dates, name="AI策略")
        print(f"  完成 {rebal_count} 次调仓，共 {len(trade_log)} 笔交易")
        return {"nav": nav_series, "trades": pd.DataFrame(trade_log), "price_matrix": price_matrix}


# ════════════════════════════════════════════════════════════
# 5. 绩效计算
# ════════════════════════════════════════════════════════════

def calc_metrics(nav: pd.Series, bench: pd.Series, rf=0.02) -> dict:
    ret = nav.pct_change().dropna()
    n   = len(ret)
    yrs = n / 252

    total_ret    = nav.iloc[-1] / nav.iloc[0] - 1
    annual_ret   = (1 + total_ret) ** (1/yrs) - 1 if yrs > 0 else 0
    annual_vol   = ret.std() * np.sqrt(252)
    sharpe       = (annual_ret - rf) / (annual_vol + 1e-9)

    cummax       = nav.cummax()
    dd           = (nav - cummax) / cummax
    max_dd       = dd.min()
    calmar       = annual_ret / abs(max_dd + 1e-9)

    win_rate     = (ret > 0).mean()
    avg_win      = ret[ret > 0].mean()
    avg_loss     = ret[ret < 0].mean()
    pnl_ratio    = abs(avg_win / (avg_loss + 1e-9))

    # 相对基准
    b_ret = bench.pct_change().dropna()
    b_ann = (bench.iloc[-1]/bench.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    b_ret_a, r_a = b_ret.align(ret, join="inner")
    cov   = np.cov(r_a, b_ret_a)
    beta  = cov[0,1] / (cov[1,1] + 1e-9)
    alpha = annual_ret - (rf + beta * (b_ann - rf))
    exc   = r_a - b_ret_a
    ir    = exc.mean() / (exc.std() + 1e-9) * np.sqrt(252)

    return {
        "总收益率":     total_ret,
        "年化收益率":   annual_ret,
        "基准年化":     b_ann,
        "超额年化":     annual_ret - b_ann,
        "年化波动率":   annual_vol,
        "夏普比率":     sharpe,
        "最大回撤":     max_dd,
        "卡玛比率":     calmar,
        "日胜率":       win_rate,
        "盈亏比":       pnl_ratio,
        "Alpha":        alpha,
        "Beta":         beta,
        "信息比率":     ir,
        "回测年数":     yrs,
    }


# ════════════════════════════════════════════════════════════
# 6. 可视化
# ════════════════════════════════════════════════════════════

COLORS = {"strat":"#E74C3C","bench":"#3498DB","pos":"#2ECC71","neg":"#E74C3C","bg":"#FAFBFC"}

def plot_dashboard(nav, bench, metrics, scores_history, trade_df, out_path):
    fig = plt.figure(figsize=(20, 16), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    nav_n   = nav   / nav.iloc[0]
    bench_n = bench / bench.iloc[0]
    cummax  = nav_n.cummax()
    dd      = (nav_n - cummax) / cummax * 100

    # ① 净值曲线 ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(nav_n.index,   nav_n.values,   color=COLORS["strat"], lw=2,   label="AI策略",  zorder=3)
    ax1.plot(bench_n.index, bench_n.values, color=COLORS["bench"], lw=1.5, label="沪深300基准",
             linestyle="--", alpha=0.85, zorder=2)
    ax1.fill_between(nav_n.index, nav_n, bench_n,
                     where=nav_n.values >= bench_n.values,
                     alpha=0.12, color=COLORS["pos"], label="跑赢基准")
    ax1.fill_between(nav_n.index, nav_n, bench_n,
                     where=nav_n.values < bench_n.values,
                     alpha=0.12, color=COLORS["neg"])
    ax1.set_title("策略净值 vs 沪深300基准", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_ylabel("净值")

    # ② 绩效指标卡 ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    rows = [
        ("年化收益率",  f"{metrics['年化收益率']:+.2%}"),
        ("基准年化",    f"{metrics['基准年化']:+.2%}"),
        ("超额收益",    f"{metrics['超额年化']:+.2%}"),
        ("夏普比率",    f"{metrics['夏普比率']:.3f}"),
        ("最大回撤",    f"{metrics['最大回撤']:.2%}"),
        ("卡玛比率",    f"{metrics['卡玛比率']:.3f}"),
        ("日胜率",      f"{metrics['日胜率']:.2%}"),
        ("盈亏比",      f"{metrics['盈亏比']:.3f}"),
        ("Alpha",       f"{metrics['Alpha']:+.4f}"),
        ("Beta",        f"{metrics['Beta']:.3f}"),
        ("信息比率",    f"{metrics['信息比率']:.3f}"),
    ]
    tbl = ax2.table(cellText=rows, colLabels=["指标","数值"],
                    cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c2), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        # 高亮超额收益行
        if r == 3 and c2 == 1:
            val = metrics["超额年化"]
            cell.set_facecolor("#D5F5E3" if val > 0 else "#FADBD8")
    ax2.set_title("绩效指标摘要", fontsize=12, fontweight="bold")

    # ③ 回撤曲线 ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.fill_between(dd.index, dd.values, 0, color=COLORS["neg"], alpha=0.55, label="策略回撤")
    bench_dd = (bench_n - bench_n.cummax()) / bench_n.cummax() * 100
    ax3.plot(bench_dd.index, bench_dd.values, color=COLORS["bench"],
             lw=1, linestyle=":", alpha=0.7, label="基准回撤")
    ax3.set_title("历史回撤曲线", fontsize=13, fontweight="bold")
    ax3.set_ylabel("回撤 (%)")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax3.tick_params(axis="x", rotation=30)

    # ④ 月度收益热力图 ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    monthly = nav.resample("ME").last().pct_change().dropna()
    if len(monthly) > 0:
        mdf = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values
        })
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="mean")
        mn = ["1","2","3","4","5","6","7","8","9","10","11","12"]
        pivot.columns = [mn[c-1] for c in pivot.columns if 1 <= c <= 12]
        mask = pivot.isna()
        sns.heatmap(pivot, ax=ax4, cmap="RdYlGn", center=0,
                    annot=True, fmt=".1%", annot_kws={"size":7},
                    linewidths=0.5, mask=mask,
                    vmin=-0.10, vmax=0.10,
                    cbar_kws={"shrink":0.8, "label":"月度收益"})
    ax4.set_title("月度收益热力图", fontsize=12, fontweight="bold")
    ax4.set_xlabel("月份"); ax4.set_ylabel("年份")

    # ⑤ 日收益分布 ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    daily_ret = nav.pct_change().dropna() * 100
    bench_ret = bench.pct_change().dropna() * 100
    ax5.hist(daily_ret, bins=60, color=COLORS["strat"], alpha=0.6,
             edgecolor="white", lw=0.3, label="AI策略", density=True)
    ax5.hist(bench_ret, bins=60, color=COLORS["bench"], alpha=0.4,
             edgecolor="white", lw=0.3, label="基准", density=True)
    # 正态拟合
    mu_, std_ = daily_ret.mean(), daily_ret.std()
    x_ = np.linspace(daily_ret.min(), daily_ret.max(), 200)
    ax5.plot(x_, stats.norm.pdf(x_, mu_, std_), "r-", lw=1.5, label="正态拟合")
    ax5.axvline(0, color="black", lw=0.8, linestyle="--")
    ax5.set_title("日收益率分布", fontsize=12, fontweight="bold")
    ax5.set_xlabel("日收益率 (%)"); ax5.set_ylabel("概率密度")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.25)

    # ⑥ 策略 vs 基准年度收益对比 ──────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    annual_strat = nav.resample("YE").last().pct_change().dropna()
    annual_bench = bench.resample("YE").last().pct_change().dropna()
    years = annual_strat.index.year
    x = np.arange(len(years))
    w = 0.35
    bars1 = ax6.bar(x - w/2, annual_strat.values * 100, w,
                    color=COLORS["strat"], alpha=0.8, label="AI策略")
    bars2 = ax6.bar(x + w/2, annual_bench.reindex(annual_strat.index).values * 100, w,
                    color=COLORS["bench"], alpha=0.8, label="沪深300")
    ax6.axhline(0, color="black", lw=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(years, fontsize=9)
    ax6.set_title("年度收益对比", fontsize=12, fontweight="bold")
    ax6.set_ylabel("收益率 (%)"); ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax6.text(bar.get_x()+bar.get_width()/2, h+0.3,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5,
                 color=COLORS["strat"], fontweight="bold")

    # ⑦ 因子得分排行（最新一期）──────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    if scores_history:
        last_scores = scores_history[-1].head(15)
        colors_ = ["#E74C3C" if i < 5 else "#E67E22" if i < 10 else "#95A5A6"
                   for i in range(len(last_scores))]
        ax7.barh(range(len(last_scores)-1, -1, -1),
                 last_scores.values, color=colors_, alpha=0.85, edgecolor="white")
        ax7.set_yticks(range(len(last_scores)-1, -1, -1))
        labels = []
        for c in last_scores.index:
            name = dict(STOCK_POOL).get(c, c)
            labels.append(f"{name}({c})")
        ax7.set_yticklabels(labels, fontsize=8.5)
        ax7.set_title("最新期选股评分 Top15", fontsize=12, fontweight="bold")
        ax7.set_xlabel("综合评分")
        ax7.grid(True, alpha=0.25, axis="x")
        # 标注分值
        for i, (val, code) in enumerate(zip(last_scores.values, last_scores.index)):
            ax7.text(val + 0.005, len(last_scores)-1-i, f"{val:.3f}",
                     va="center", fontsize=7.5)

    fig.suptitle("AI 智能A股选股回测报告", fontsize=18,
                 fontweight="bold", y=0.995, color="#1A252F")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  图表已保存: {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════
# 7. 主流程
# ════════════════════════════════════════════════════════════

def main():
    START = "2021-01-01"
    END   = "2024-12-31"

    print("\n" + "="*55)
    print("    AI A股选股与回测系统  v2.0-优化版")
    print(f"    回测区间: {START} -> {END}")
    print(f"    股票池: {len(STOCK_POOL)} 只  |  ta-lib: {'YES' if HAS_TALIB else 'NO'}")
    print("="*55)

    # ── Step 1: 生成数据 ──────────────────────────────────
    print("\n[1/5] 生成/加载行情数据...")
    price_dict = {}
    for code, name in STOCK_POOL:
        # 向前多取1年用于因子预热
        df = generate_stock_data(code, "2020-01-01", END)
        price_dict[code] = df
        print(f"  {code} {name}: {len(df)} 个交易日")

    # ── Step 2: 构建基准 ──────────────────────────────────
    print("\n[2/5] 构建基准 (沪深300模拟)...")
    np.random.seed(42)
    bench_all = generate_stock_data("000300", "2020-01-01", END, mu=0.00020, sigma=0.016, seed=300)
    bench = bench_all.loc[START:END, "close"]
    print(f"  基准数据: {len(bench)} 个交易日")

    # ── Step 3: 记录调仓因子得分（用于可视化） ─────────────
    print("\n[3/5] 计算因子截面得分...")
    dates_bt = pd.bdate_range(START, END)
    monthly_dates = [g.iloc[0] for _, g in
                     pd.Series(dates_bt).groupby(pd.Series(dates_bt).dt.to_period("M"))]

    scores_history = []
    for i, d in enumerate(monthly_dates):
        factor_rows = {}
        for code, df in price_dict.items():
            hist = df.loc[:d]
            if len(hist) < 80:
                continue
            try:
                factor_rows[code] = compute_factors(hist)
            except Exception:
                pass
        if factor_rows:
            fdf = pd.DataFrame(factor_rows).T
            sc  = score_stocks(fdf)
            scores_history.append(sc)
            if (i+1) % 6 == 0:
                top3 = ", ".join([f"{dict(STOCK_POOL).get(c,c)}({sc:.2f})"
                                  for c, sc in sc.head(3).items()])
                print(f"  {d.strftime('%Y-%m')}  Top3: {top3}")

    # ── Step 4: 回测（不同参数对比） ─────────────────────
    print("\n[4/5] 执行回测...")

    configs = [
        dict(top_n=10, rebalance_freq="monthly",  label="AI-月调-10只"),
        dict(top_n=15, rebalance_freq="monthly",  label="AI-月调-15只"),
        dict(top_n=10, rebalance_freq="weekly",   label="AI-周调-10只"),
    ]

    all_navs = {}
    all_metrics = {}
    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n  >> 策略: {label}")

        bt = Backtest(price_dict, initial_capital=1_000_000,
                      commission=0.0003, slippage=0.001, **cfg)
        result = bt.run(START, END)
        nav = result["nav"]
        all_navs[label] = nav

        m = calc_metrics(nav, bench)
        all_metrics[label] = m
        print(f"     年化收益: {m['年化收益率']:+.2%}  |  夏普: {m['夏普比率']:.3f}"
              f"  |  最大回撤: {m['最大回撤']:.2%}  |  Alpha: {m['Alpha']:+.4f}")

    # 选表现最好的策略出图
    best_label = max(all_metrics, key=lambda k: all_metrics[k]["夏普比率"])
    best_nav   = all_navs[best_label]
    best_m     = all_metrics[best_label]
    print(f"\n  最优策略: {best_label}  (夏普比率 {best_m['夏普比率']:.3f})")

    # ── Step 5: 生成报告 ──────────────────────────────────
    print("\n[5/5] 生成可视化报告...")
    chart_path = os.path.join(OUTPUT_DIR, "backtest_dashboard.png")
    plot_dashboard(
        nav=best_nav,
        bench=bench,
        metrics=best_m,
        scores_history=scores_history,
        trade_df=pd.DataFrame(),
        out_path=chart_path
    )

    # 多策略对比图
    comp_path = os.path.join(OUTPUT_DIR, "strategy_comparison.png")
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS["bg"])
    colors_ = ["#E74C3C", "#E67E22", "#9B59B6"]
    # 净值对比
    ax = axes[0]
    bench_n = bench / bench.iloc[0]
    ax.plot(bench_n.index, bench_n.values, "b--", lw=1.5, alpha=0.7, label="沪深300")
    for (label, nav), col in zip(all_navs.items(), colors_):
        nav_n = nav / nav.iloc[0]
        ax.plot(nav_n.index, nav_n.values, color=col, lw=2, label=label)
    ax.set_title("多策略净值对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax.tick_params(axis="x", rotation=30)
    # 绩效对比柱状图
    ax2 = axes[1]
    metrics_compare = pd.DataFrame({
        k: {"年化收益": v["年化收益率"]*100,
            "夏普比率": v["夏普比率"],
            "最大回撤": abs(v["最大回撤"])*100}
        for k, v in all_metrics.items()
    }).T
    x = np.arange(len(metrics_compare))
    w = 0.25
    ax2.bar(x - w, metrics_compare["年化收益"], w, label="年化收益(%)", color="#E74C3C", alpha=0.8)
    ax2.bar(x,     metrics_compare["夏普比率"],  w, label="夏普比率",   color="#3498DB", alpha=0.8)
    ax2.bar(x + w, metrics_compare["最大回撤"],  w, label="最大回撤(%)", color="#95A5A6", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics_compare.index, fontsize=9, rotation=10)
    ax2.set_title("策略绩效对比", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25, axis="y")
    fig2.suptitle("多策略对比分析", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  对比图已保存: {comp_path}")

    # 保存 CSV
    nav_df = pd.DataFrame(all_navs)
    nav_df.to_csv(os.path.join(OUTPUT_DIR, "nav_all.csv"), encoding="utf-8-sig")
    pd.DataFrame(all_metrics).T.to_csv(
        os.path.join(OUTPUT_DIR, "metrics_all.csv"), encoding="utf-8-sig")

    # ── 打印最终汇总 ──────────────────────────────────────
    print("\n" + "="*55)
    print(f"  [BEST] [{best_label}] 完整绩效")
    print("="*55)
    fmt = {
        "总收益率": ".2%", "年化收益率": ".2%", "基准年化": ".2%",
        "超额年化": "+.2%", "年化波动率": ".2%", "夏普比率": ".4f",
        "最大回撤": ".2%", "卡玛比率": ".4f", "日胜率": ".2%",
        "盈亏比": ".4f", "Alpha": "+.4f", "Beta": ".4f",
        "信息比率": ".4f", "回测年数": ".2f",
    }
    for k, f in fmt.items():
        v = best_m.get(k, 0)
        print(f"  {k:<14}: {v:{f}}")

    print("\n  输出文件:")
    print(f"  [chart] {chart_path}")
    print(f"  [comp]  {comp_path}")
    print(f"  [csv]   {os.path.join(OUTPUT_DIR, 'nav_all.csv')}")
    print(f"  [csv]   {os.path.join(OUTPUT_DIR, 'metrics_all.csv')}")
    print("\n[DONE] 回测完成！")
    return chart_path, comp_path


if __name__ == "__main__":
    main()
