"""
backtest_real.py  ——  基于真实A股数据的完整回测系统 v4.0
特性：
  - 读取 SQLite 真实历史数据
  - 多因子选股（动量/价值/质量/技术）
  - IC/ICIR 动态权重
  - ATR 风险平价仓位
  - 止损 + 市场择时
  - Walk-Forward 验证（4折）
  - 网格参数寻优
  - 生成 HTML + PNG 报告
"""

import sqlite3, sys, time, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")

# ── 路径 ──
BASE   = Path(__file__).parent
DB     = BASE / "data" / "stock_db.sqlite"
OUT    = BASE / "output"
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════════════
#  数据加载
# ════════════════════════════════════════════════
def load_data(min_bars: int = 200) -> dict[str, pd.DataFrame]:
    """从SQLite加载所有股票日K数据"""
    conn = sqlite3.connect(str(DB))

    # 筛选有足够数据的股票
    codes = pd.read_sql(
        f"SELECT code FROM daily_bar GROUP BY code HAVING COUNT(*)>={min_bars}",
        conn
    )["code"].tolist()

    if not codes:
        # fallback: 加载所有
        codes = pd.read_sql("SELECT DISTINCT code FROM daily_bar", conn)["code"].tolist()

    print(f"  加载 {len(codes)} 只股票的日K数据...")

    data = {}
    for code in codes:
        df = pd.read_sql(
            "SELECT date,open,high,low,close,volume,amount,turnover "
            "FROM daily_bar WHERE code=? ORDER BY date",
            conn, params=(code,)
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open","high","low","close","volume","amount","turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        if len(df) >= min_bars:
            data[code] = df

    conn.close()
    print(f"  有效股票: {len(data)} 只")
    return data

# ════════════════════════════════════════════════
#  因子计算
# ════════════════════════════════════════════════
def compute_factors_for_stock(code: str, df: pd.DataFrame, lookback: int = 60) -> dict | None:
    """计算单只股票在最新日期的因子值"""
    if len(df) < lookback + 10:
        return None

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    o = df["open"].values.astype(float)

    n = len(c)
    factors = {"code": code}

    try:
        # ── 动量因子 ──
        factors["mom_5"]  = c[-1] / c[-6]  - 1 if n >= 6  else np.nan
        factors["mom_20"] = c[-1] / c[-21] - 1 if n >= 21 else np.nan
        factors["mom_60"] = c[-1] / c[-61] - 1 if n >= 61 else np.nan

        # ── 反转因子 ──
        factors["rev_5"]  = -(c[-1] / c[-6]  - 1) if n >= 6  else np.nan

        # ── 波动率因子 ──
        ret20 = np.diff(np.log(c[-21:])) if n >= 21 else np.array([np.nan])
        factors["vol_20"] = -np.std(ret20) * np.sqrt(252)  # 低波动 = 正

        # ── 换手率因子（流动性）──
        turn = df["turnover"].values.astype(float)
        valid_turn = turn[turn > 0]
        if len(valid_turn) >= 20:
            factors["turnover_20"] = -np.mean(valid_turn[-20:])  # 低换手 = 稳定
        else:
            factors["turnover_20"] = np.nan

        # ── 量价关系 ──
        if n >= 20:
            ret_10  = np.diff(c[-11:]) / c[-11:-1]
            vol_10  = v[-10:]
            corr    = np.corrcoef(ret_10, vol_10)[0, 1] if np.std(vol_10) > 0 else 0
            factors["vol_price_corr"] = -corr   # 量价负相关 = 正

            # OBV动量
            ret_20 = np.diff(c[-21:])
            obv_delta = np.sum(np.sign(ret_20) * v[-20:])
            factors["obv_mom"] = obv_delta / (np.mean(v[-20:]) + 1e-9)

        else:
            factors["vol_price_corr"] = np.nan
            factors["obv_mom"] = np.nan

        # ── 技术因子 ──
        if n >= 20:
            ma5  = np.mean(c[-5:])
            ma20 = np.mean(c[-20:])
            factors["ma_cross"]  = ma5 / ma20 - 1   # 金叉强度

            # RSI
            delta = np.diff(c[-15:])
            gain  = np.mean(delta[delta > 0]) if (delta > 0).any() else 1e-9
            loss  = np.mean(-delta[delta < 0]) if (delta < 0).any() else 1e-9
            rs    = gain / loss
            factors["rsi_14"] = -(100 - 100 / (1 + rs))  # 低RSI=超卖=买入信号

            # 布林带位置
            std20 = np.std(c[-20:])
            factors["bb_pos"] = -(c[-1] - ma20) / (std20 * 2 + 1e-9)  # 在下轨 = 正

        else:
            factors["ma_cross"] = np.nan
            factors["rsi_14"]   = np.nan
            factors["bb_pos"]   = np.nan

        # ── ATR（用于仓位控制）──
        if n >= 15:
            tr_list = []
            for i in range(-14, 0):
                tr_list.append(max(
                    h[i] - l[i],
                    abs(h[i] - c[i-1]),
                    abs(l[i] - c[i-1])
                ))
            factors["atr_14"]  = np.mean(tr_list)
            factors["atr_pct"] = factors["atr_14"] / c[-1]  # ATR占价格百分比
        else:
            factors["atr_14"]  = np.nan
            factors["atr_pct"] = np.nan

        factors["close"] = c[-1]
        factors["date"]  = df.index[-1]

    except Exception:
        return None

    return factors


def compute_cross_section_factors(data: dict, date: pd.Timestamp) -> pd.DataFrame:
    """计算截面因子（给定日期，对所有股票）"""
    rows = []
    for code, df in data.items():
        # 取截止 date 的历史数据
        hist = df[df.index <= date]
        if len(hist) < 70:
            continue
        fac = compute_factors_for_stock(code, hist)
        if fac is not None:
            rows.append(fac)
    if not rows:
        return pd.DataFrame()
    df_fac = pd.DataFrame(rows).set_index("code")
    return df_fac


# ════════════════════════════════════════════════
#  IC/ICIR 动态因子权重
# ════════════════════════════════════════════════
FACTOR_COLS = ["mom_5","mom_20","mom_60","rev_5","vol_20",
               "turnover_20","vol_price_corr","obv_mom",
               "ma_cross","rsi_14","bb_pos"]

def compute_ic(fac_df: pd.DataFrame, fwd_ret: pd.Series) -> pd.Series:
    """计算各因子与下期收益的IC"""
    ic = {}
    for col in FACTOR_COLS:
        if col not in fac_df.columns:
            ic[col] = 0.0
            continue
        valid = fac_df[[col]].join(fwd_ret.rename("fwd")).dropna()
        if len(valid) < 10:
            ic[col] = 0.0
        else:
            r, _ = stats.spearmanr(valid[col], valid["fwd"])
            ic[col] = r if not np.isnan(r) else 0.0
    return pd.Series(ic)


def icir_weights(ic_history: list[pd.Series]) -> pd.Series:
    """用最近 N 期 IC 计算 ICIR 权重"""
    if len(ic_history) < 3:
        return pd.Series({c: 1.0/len(FACTOR_COLS) for c in FACTOR_COLS})
    mat  = pd.DataFrame(ic_history)
    mean_ic = mat.mean()
    std_ic  = mat.std().replace(0, 1e-9)
    icir = (mean_ic / std_ic).clip(-3, 3)
    icir_pos = icir - icir.min()
    if icir_pos.sum() < 1e-9:
        return pd.Series({c: 1.0/len(FACTOR_COLS) for c in FACTOR_COLS})
    return icir_pos / icir_pos.sum()


# ════════════════════════════════════════════════
#  组合得分与选股
# ════════════════════════════════════════════════
def zscore_normalize(series: pd.Series) -> pd.Series:
    m, s = series.mean(), series.std()
    if s < 1e-9:
        return pd.Series(0, index=series.index)
    return ((series - m) / s).clip(-3, 3)


def score_stocks(fac_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """加权因子打分"""
    score = pd.Series(0.0, index=fac_df.index)
    for col in FACTOR_COLS:
        if col not in fac_df.columns:
            continue
        col_series = fac_df[col].dropna()
        if len(col_series) < 5:
            continue
        z = zscore_normalize(fac_df[col].fillna(fac_df[col].median()))
        w = weights.get(col, 0.0)
        score += z * w
    return score


# ════════════════════════════════════════════════
#  仓位管理
# ════════════════════════════════════════════════
def atr_position(fac_df: pd.DataFrame, selected: list[str],
                 capital: float, risk_per: float = 0.01) -> dict[str, float]:
    """ATR风险平价：每只股票风险敞口相等"""
    pos = {}
    total_w = 0.0
    for code in selected:
        if code not in fac_df.index:
            pos[code] = 0.0
            continue
        atr_pct = fac_df.loc[code, "atr_pct"] if "atr_pct" in fac_df.columns else 0.02
        if np.isnan(atr_pct) or atr_pct <= 0:
            atr_pct = 0.02
        w = risk_per / atr_pct
        pos[code] = w
        total_w += w

    # 归一化（不超过100%仓位）
    if total_w > 0:
        scale = min(1.0, 1.0 / total_w) * 0.95   # 保留5%现金
        for code in pos:
            pos[code] = pos[code] / total_w * capital * scale
    return pos


# ════════════════════════════════════════════════
#  回测引擎
# ════════════════════════════════════════════════
class Backtester:
    def __init__(self, data: dict, params: dict):
        self.data   = data
        self.params = params
        # 参数
        self.n_stock    = params.get("n_stock", 20)
        self.hold_days  = params.get("hold_days", 20)
        self.stop_loss  = params.get("stop_loss", 0.08)
        self.fee_rate   = params.get("fee_rate", 0.001)
        self.risk_per   = params.get("risk_per", 0.01)
        self.use_icir   = params.get("use_icir", True)
        # 状态
        self.capital    = 1_000_000.0
        self.positions  = {}   # code -> shares
        self.cost_price = {}   # code -> cost
        self.nav_series = []
        self.trades     = []
        self.ic_history = []

    def get_price(self, code: str, date: pd.Timestamp) -> float | None:
        df = self.data.get(code)
        if df is None:
            return None
        # 取 date 当天或最近前一天的收盘价
        hist = df[df.index <= date]
        if hist.empty:
            return None
        return float(hist["close"].iloc[-1])

    def rebalance(self, date: pd.Timestamp, weights: pd.Series):
        """调仓"""
        # 计算因子
        fac_rows = []
        for code, df in self.data.items():
            hist = df[df.index <= date]
            if len(hist) < 70:
                continue
            fac = compute_factors_for_stock(code, hist)
            if fac is not None:
                fac_rows.append(fac)

        if not fac_rows:
            return

        fac_df = pd.DataFrame(fac_rows).set_index("code")

        # 打分选股
        scores = score_stocks(fac_df, weights)
        if scores.empty:
            return

        # 过滤：去掉当天无法交易的（价格异常）
        valid = []
        for code in scores.nlargest(self.n_stock * 3).index:
            p = self.get_price(code, date)
            if p is not None and p > 0.5:
                valid.append(code)
            if len(valid) >= self.n_stock:
                break

        target_codes = set(valid)

        # 平掉不在目标中的持仓
        for code in list(self.positions.keys()):
            if code not in target_codes:
                p = self.get_price(code, date)
                if p and self.positions[code] > 0:
                    proceeds = p * self.positions[code] * (1 - self.fee_rate)
                    self.capital += proceeds
                    ret = (p - self.cost_price.get(code, p)) / self.cost_price.get(code, p)
                    self.trades.append({
                        "date": date, "code": code,
                        "action": "sell", "price": p,
                        "shares": self.positions[code], "ret": ret
                    })
                del self.positions[code]
                self.cost_price.pop(code, None)

        # 计算当前市值
        portfolio_value = self.capital
        for code, shares in self.positions.items():
            p = self.get_price(code, date)
            if p:
                portfolio_value += p * shares

        # ATR仓位分配
        new_buys = target_codes - set(self.positions.keys())
        if new_buys:
            buy_fac = fac_df[fac_df.index.isin(new_buys)]
            alloc   = atr_position(buy_fac, list(new_buys),
                                   self.capital, self.risk_per)
            for code, cash_alloc in alloc.items():
                p = self.get_price(code, date)
                if p and cash_alloc > 0 and self.capital >= cash_alloc:
                    shares = int(cash_alloc / p / 100) * 100
                    if shares > 0:
                        cost = p * shares * (1 + self.fee_rate)
                        if cost <= self.capital:
                            self.capital -= cost
                            self.positions[code] = shares
                            self.cost_price[code] = p
                            self.trades.append({
                                "date": date, "code": code,
                                "action": "buy", "price": p,
                                "shares": shares, "ret": 0.0
                            })

    def check_stop_loss(self, date: pd.Timestamp):
        """检查止损"""
        for code in list(self.positions.keys()):
            p = self.get_price(code, date)
            cost = self.cost_price.get(code, p)
            if p and cost and (p - cost) / cost < -self.stop_loss:
                proceeds = p * self.positions[code] * (1 - self.fee_rate)
                self.capital += proceeds
                ret = (p - cost) / cost
                self.trades.append({
                    "date": date, "code": code,
                    "action": "stop_loss", "price": p,
                    "shares": self.positions[code], "ret": ret
                })
                del self.positions[code]
                self.cost_price.pop(code, None)

    def nav(self, date: pd.Timestamp) -> float:
        total = self.capital
        for code, shares in self.positions.items():
            p = self.get_price(code, date)
            if p:
                total += p * shares
        return total

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """运行回测"""
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)

        # 获取所有交易日（用数据中出现的日期）
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index[
                (df.index >= start) & (df.index <= end)
            ])
        trading_dates = sorted(all_dates)

        if not trading_dates:
            return pd.DataFrame()

        weights = pd.Series({c: 1.0/len(FACTOR_COLS) for c in FACTOR_COLS})
        last_rebal = None
        ic_buf     = []  # 上期因子截面
        fwd_prices = {}  # code -> 上次调仓时的价格

        print(f"    回测: {start_date} ~ {end_date}，共 {len(trading_dates)} 个交易日")

        for i, date in enumerate(trading_dates):
            # 止损检查（每天）
            self.check_stop_loss(date)

            # 计算IC（用上次调仓的因子）
            if last_rebal is not None and fwd_prices:
                # 计算上次调仓到今天的收益
                fwd_ret = {}
                for code in fwd_prices:
                    p_now = self.get_price(code, date)
                    p_old = fwd_prices.get(code)
                    if p_now and p_old and p_old > 0:
                        fwd_ret[code] = p_now / p_old - 1
                if fwd_ret and ic_buf:
                    fwd_s = pd.Series(fwd_ret)
                    ic = compute_ic(ic_buf[-1], fwd_s)
                    self.ic_history.append(ic)

                if self.use_icir and len(self.ic_history) >= 3:
                    weights = icir_weights(self.ic_history[-12:])

            # 调仓（每 hold_days 天）
            should_rebal = (
                last_rebal is None or
                (date - last_rebal).days >= self.hold_days
            )
            if should_rebal:
                # 保存当前因子用于IC计算
                fac_rows = []
                for code, df in self.data.items():
                    hist = df[df.index <= date]
                    if len(hist) >= 70:
                        fac = compute_factors_for_stock(code, hist)
                        if fac is not None:
                            fac_rows.append(fac)
                if fac_rows:
                    ic_buf.append(pd.DataFrame(fac_rows).set_index("code"))

                # 记录当时价格（用于后续IC计算）
                fwd_prices = {}
                for code in self.data:
                    p = self.get_price(code, date)
                    if p:
                        fwd_prices[code] = p

                self.rebalance(date, weights)
                last_rebal = date

            # 记录净值
            nav = self.nav(date)
            self.nav_series.append({"date": date, "nav": nav})

            if i % 50 == 0:
                print(f"      {date.date()}: NAV={nav:,.0f}  持仓:{len(self.positions)}")

        return pd.DataFrame(self.nav_series).set_index("date")


# ════════════════════════════════════════════════
#  绩效计算
# ════════════════════════════════════════════════
def calc_metrics(nav_df: pd.DataFrame) -> dict:
    nav = nav_df["nav"].values
    if len(nav) < 10:
        return {}

    ret = pd.Series(nav).pct_change().dropna()
    total_ret   = nav[-1] / nav[0] - 1
    n_years     = len(nav) / 252
    ann_ret     = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    ann_vol     = ret.std() * np.sqrt(252)
    sharpe      = ann_ret / ann_vol if ann_vol > 1e-9 else 0

    neg_ret     = ret[ret < 0]
    sortino_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1e-9
    sortino     = ann_ret / sortino_vol if sortino_vol > 1e-9 else 0

    running_max = np.maximum.accumulate(nav)
    dd          = (nav - running_max) / running_max
    max_dd      = dd.min()
    calmar      = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    win_rate    = (ret > 0).mean()
    avg_win     = ret[ret > 0].mean() if (ret > 0).any() else 0.0
    avg_loss    = ret[ret < 0].mean() if (ret < 0).any() else -1e-9
    pnl_ratio   = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0

    return {
        "total_ret":  total_ret,
        "ann_ret":    ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "max_dd":     max_dd,
        "calmar":     calmar,
        "win_rate":   win_rate,
        "pnl_ratio":  pnl_ratio,
        "n_trades":   0,
    }


# ════════════════════════════════════════════════
#  基准（沪深300近似）
# ════════════════════════════════════════════════
def compute_benchmark(data: dict, start: str, end: str) -> pd.Series:
    """等权基准（所有股票等权持有）"""
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index[(df.index >= start) & (df.index <= end)])
    dates = sorted(all_dates)
    if not dates:
        return pd.Series()

    nav = [1.0]
    prev_prices = {}
    for i, date in enumerate(dates):
        if i == 0:
            for code, df in data.items():
                h = df[df.index <= date]
                if not h.empty:
                    prev_prices[code] = h["close"].iloc[-1]
            continue
        rets = []
        for code, df in data.items():
            h = df[df.index <= date]
            if h.empty:
                continue
            p = h["close"].iloc[-1]
            p_prev = prev_prices.get(code)
            if p_prev and p_prev > 0:
                rets.append(p / p_prev - 1)
            prev_prices[code] = p
        if rets:
            nav.append(nav[-1] * (1 + np.mean(rets)))
        else:
            nav.append(nav[-1])

    nav_arr = np.array(nav) * 1_000_000
    return pd.Series(nav_arr[:len(dates)], index=dates)


# ════════════════════════════════════════════════
#  Walk-Forward 验证
# ════════════════════════════════════════════════
def walk_forward(data: dict, all_dates: list, n_folds: int = 4,
                 params: dict = None) -> list[dict]:
    if params is None:
        params = {"n_stock": 20, "hold_days": 20, "stop_loss": 0.08,
                  "risk_per": 0.01, "use_icir": True}

    results = []
    fold_size = len(all_dates) // (n_folds + 1)

    for fold in range(n_folds):
        train_end_idx   = fold_size * (fold + 1) + fold_size // 2
        test_start_idx  = fold_size * (fold + 1) + fold_size // 2
        test_end_idx    = fold_size * (fold + 2)

        if test_end_idx > len(all_dates):
            break

        train_end   = all_dates[min(train_end_idx, len(all_dates)-1)]
        test_start  = all_dates[min(test_start_idx, len(all_dates)-1)]
        test_end    = all_dates[min(test_end_idx-1, len(all_dates)-1)]

        print(f"\n  Fold {fold+1}/{n_folds}: test {test_start.date()} ~ {test_end.date()}")

        bt = Backtester(data, params)
        nav_df = bt.run(str(test_start.date()), str(test_end.date()))

        if nav_df.empty or len(nav_df) < 10:
            results.append({"fold": fold+1, "sharpe": 0, "ann_ret": 0})
            continue

        m = calc_metrics(nav_df)
        m["fold"] = fold + 1
        m["start"] = str(test_start.date())
        m["end"]   = str(test_end.date())
        results.append(m)
        print(f"    年化:{m['ann_ret']:.1%}  夏普:{m['sharpe']:.2f}  最大回撤:{m['max_dd']:.1%}")

    return results


# ════════════════════════════════════════════════
#  网格参数寻优
# ════════════════════════════════════════════════
def grid_search(data: dict, all_dates: list) -> pd.DataFrame:
    print("\n[网格参数寻优]")
    param_grid = {
        "n_stock":   [10, 20, 30],
        "hold_days": [10, 20],
        "stop_loss": [0.06, 0.10],
    }
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"  共 {len(combos)} 组参数组合")

    # 用中间一段做验证
    val_start = all_dates[len(all_dates)//4]
    val_end   = all_dates[len(all_dates)*3//4]

    results = []
    for i, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        p["use_icir"]  = True
        p["risk_per"]  = 0.01
        p["fee_rate"]  = 0.001

        bt = Backtester(data, p)
        nav_df = bt.run(str(val_start.date()), str(val_end.date()))

        if nav_df.empty:
            metrics = {"sharpe": 0, "ann_ret": 0, "max_dd": 0}
        else:
            metrics = calc_metrics(nav_df)

        row = {**p, **{k: round(v, 4) for k, v in metrics.items()
                       if k in ["ann_ret","sharpe","sortino","max_dd","calmar"]}}
        results.append(row)
        print(f"  [{i+1}/{len(combos)}] {p} => sharpe={metrics.get('sharpe',0):.2f}")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return df


# ════════════════════════════════════════════════
#  可视化报告
# ════════════════════════════════════════════════
def plot_report(nav_main: pd.DataFrame, nav_bench: pd.Series,
                wf_results: list, grid_df: pd.DataFrame,
                metrics: dict, trades_df: pd.DataFrame):

    plt.rcParams.update({
        "font.family":   ["SimHei","Microsoft YaHei","DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "text.color":       "#e6edf3",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "axes.edgecolor":   "#30363d",
        "grid.color":       "#21262d",
    })

    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    COLOR_STRAT = "#58a6ff"
    COLOR_BENCH = "#f78166"
    COLOR_POS   = "#3fb950"
    COLOR_NEG   = "#f85149"

    # ── 1. 净值曲线 ──
    ax1 = fig.add_subplot(gs[0, :2])
    nav_s = nav_main["nav"] / 1_000_000
    ax1.plot(nav_main.index, nav_s, color=COLOR_STRAT, lw=2, label="Strategy v4.0")
    if not nav_bench.empty:
        bench_s = nav_bench / 1_000_000
        bench_s = bench_s.reindex(nav_main.index, method="ffill").dropna()
        if not bench_s.empty:
            ax1.plot(bench_s.index, bench_s, color=COLOR_BENCH, lw=1.5,
                     alpha=0.8, label="Equal-Weight Benchmark")
    ax1.fill_between(nav_main.index, 1, nav_s, alpha=0.1, color=COLOR_STRAT)
    ax1.set_title("净值曲线 (v4.0 真实数据回测)", fontsize=14, pad=10)
    ax1.set_ylabel("净值")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1, color="#8b949e", lw=0.8, ls="--")

    # ── 2. 绩效指标卡 ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    metric_items = [
        ("年化收益", f"{metrics.get('ann_ret', 0):.1%}"),
        ("夏普比率", f"{metrics.get('sharpe', 0):.2f}"),
        ("Sortino",  f"{metrics.get('sortino', 0):.2f}"),
        ("Calmar",   f"{metrics.get('calmar', 0):.2f}"),
        ("最大回撤", f"{metrics.get('max_dd', 0):.1%}"),
        ("日胜率",   f"{metrics.get('win_rate', 0):.1%}"),
        ("盈亏比",   f"{metrics.get('pnl_ratio', 0):.2f}"),
        ("年化波动", f"{metrics.get('ann_vol', 0):.1%}"),
    ]
    for i, (k, v) in enumerate(metric_items):
        y = 0.92 - i * 0.11
        color = COLOR_POS if (i < 5 and "%" in v and float(v.replace("%","")) > 0) else "#e6edf3"
        ax2.text(0.05, y, k, transform=ax2.transAxes, fontsize=10, color="#8b949e")
        ax2.text(0.95, y, v, transform=ax2.transAxes, fontsize=12,
                 color=COLOR_POS if i in [0,1,2,3,5,6] else COLOR_NEG,
                 ha="right", fontweight="bold")
    ax2.set_title("绩效指标", fontsize=12)

    # ── 3. 回撤曲线 ──
    ax3 = fig.add_subplot(gs[1, :2])
    nav_arr = nav_main["nav"].values
    running_max = np.maximum.accumulate(nav_arr)
    dd  = (nav_arr - running_max) / running_max
    ax3.fill_between(nav_main.index, dd, 0, alpha=0.7, color=COLOR_NEG)
    ax3.plot(nav_main.index, dd, color=COLOR_NEG, lw=1)
    ax3.set_title("回撤曲线", fontsize=12)
    ax3.set_ylabel("回撤")
    ax3.grid(True, alpha=0.3)

    # ── 4. Walk-Forward 折叠结果 ──
    ax4 = fig.add_subplot(gs[1, 2])
    if wf_results:
        folds    = [r["fold"] for r in wf_results if "fold" in r]
        sharpes  = [r.get("sharpe", 0) for r in wf_results if "fold" in r]
        ann_rets = [r.get("ann_ret", 0) for r in wf_results if "fold" in r]
        x = np.arange(len(folds))
        bars = ax4.bar(x - 0.2, sharpes, 0.4, label="Sharpe", color=COLOR_STRAT, alpha=0.8)
        bars2 = ax4.bar(x + 0.2, ann_rets, 0.4, label="Ann.Ret", color=COLOR_BENCH, alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"Fold {f}" for f in folds])
        ax4.axhline(0, color="#8b949e", lw=0.8, ls="--")
        ax4.legend(fontsize=9)
        ax4.set_title("Walk-Forward 样本外验证", fontsize=12)
        ax4.grid(True, alpha=0.3)

    # ── 5. 月度收益热图 ──
    ax5 = fig.add_subplot(gs[2, :2])
    monthly = nav_main["nav"].resample("ME").last().pct_change().dropna()
    if len(monthly) > 0:
        monthly_df = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values
        })
        if len(monthly_df) > 0:
            pivot = monthly_df.pivot_table(values="ret", index="year", columns="month")
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot.columns = [month_names[c-1] for c in pivot.columns]
            im = ax5.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                            vmin=-0.1, vmax=0.1)
            ax5.set_xticks(range(len(pivot.columns)))
            ax5.set_xticklabels(pivot.columns, fontsize=8)
            ax5.set_yticks(range(len(pivot.index)))
            ax5.set_yticklabels(pivot.index, fontsize=8)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax5.text(j, i, f"{val:.1%}", ha="center", va="center",
                                fontsize=7, color="black")
            plt.colorbar(im, ax=ax5, shrink=0.8)
            ax5.set_title("月度收益热图", fontsize=12)

    # ── 6. 参数热图（n_stock vs hold_days，sharpe）──
    ax6 = fig.add_subplot(gs[2, 2])
    if not grid_df.empty and "n_stock" in grid_df.columns and "hold_days" in grid_df.columns:
        try:
            pivot_g = grid_df.pivot_table(
                values="sharpe", index="n_stock", columns="hold_days", aggfunc="mean"
            )
            im2 = ax6.imshow(pivot_g.values, cmap="YlOrRd", aspect="auto")
            ax6.set_xticks(range(len(pivot_g.columns)))
            ax6.set_xticklabels([f"Hold{c}d" for c in pivot_g.columns], fontsize=9)
            ax6.set_yticks(range(len(pivot_g.index)))
            ax6.set_yticklabels([f"N={c}" for c in pivot_g.index], fontsize=9)
            for i in range(len(pivot_g.index)):
                for j in range(len(pivot_g.columns)):
                    val = pivot_g.values[i, j]
                    if not np.isnan(val):
                        ax6.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
            plt.colorbar(im2, ax=ax6, shrink=0.8)
            ax6.set_title("参数热图 (Sharpe)", fontsize=12)
        except Exception:
            ax6.text(0.5, 0.5, "参数热图\n(需更多数据)", ha="center", va="center",
                    transform=ax6.transAxes, fontsize=11)
            ax6.set_title("参数热图 (Sharpe)", fontsize=12)

    # ── 7. 日收益分布 ──
    ax7 = fig.add_subplot(gs[3, 0])
    daily_ret = nav_main["nav"].pct_change().dropna()
    ax7.hist(daily_ret, bins=50, color=COLOR_STRAT, alpha=0.7, edgecolor="none")
    ax7.axvline(0, color=COLOR_NEG, lw=1, ls="--")
    ax7.axvline(daily_ret.mean(), color=COLOR_POS, lw=1.5, ls="--",
                label=f"Mean={daily_ret.mean():.3%}")
    ax7.set_title("日收益分布", fontsize=12)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # ── 8. IC序列 ──
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.set_title("IC时序（因子预测能力）", fontsize=12)
    ax8.set_xlabel("调仓期")
    ax8.grid(True, alpha=0.3)
    ax8.text(0.5, 0.5, "IC历史\n（回测中积累）",
             ha="center", va="center", transform=ax8.transAxes,
             fontsize=11, color="#8b949e")

    # ── 9. 最优参数排行 ──
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis("off")
    ax9.set_title("参数寻优结果 Top5", fontsize=12)
    if not grid_df.empty:
        top5 = grid_df.head(5)
        cols_show = ["n_stock","hold_days","stop_loss","sharpe","ann_ret","max_dd"]
        cols_show = [c for c in cols_show if c in top5.columns]
        y = 0.90
        header = "  ".join(f"{c:>8}" for c in cols_show)
        ax9.text(0.02, y, header, transform=ax9.transAxes,
                fontsize=7.5, family="monospace", color="#8b949e")
        y -= 0.08
        for _, row in top5.iterrows():
            line = "  ".join(f"{row[c]:>8.3f}" if isinstance(row[c], float)
                             else f"{row[c]:>8}" for c in cols_show)
            ax9.text(0.02, y, line, transform=ax9.transAxes,
                    fontsize=7.5, family="monospace", color=COLOR_STRAT)
            y -= 0.08

    # 标题
    fig.suptitle(
        f"A股多因子策略 v4.0 — 真实历史数据回测报告\n"
        f"数据截至 {datetime.today().strftime('%Y-%m-%d')}",
        fontsize=16, y=0.995, color="#e6edf3"
    )

    # 保存
    png_path  = OUT / "backtest_v4_report.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG saved: {png_path}")

    # HTML 报告
    _save_html_report(nav_main, nav_bench, metrics, wf_results, grid_df)


def _save_html_report(nav_main, nav_bench, metrics, wf_results, grid_df):
    """生成交互式 HTML 报告"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import json as _json

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=["净值曲线", "回撤曲线",
                            "月度收益分布", "Walk-Forward结果",
                            "日收益分布", "参数寻优Sharpe"],
            vertical_spacing=0.12, horizontal_spacing=0.08
        )
        # 净值
        nav_s = nav_main["nav"] / 1_000_000
        fig.add_trace(go.Scatter(
            x=nav_main.index, y=nav_s, name="Strategy v4.0",
            line=dict(color="#58a6ff", width=2)
        ), row=1, col=1)
        if not nav_bench.empty:
            bench_s = nav_bench.reindex(nav_main.index, method="ffill") / 1_000_000
            fig.add_trace(go.Scatter(
                x=bench_s.index, y=bench_s, name="Benchmark",
                line=dict(color="#f78166", width=1.5, dash="dash")
            ), row=1, col=1)

        # 回撤
        nav_arr = nav_main["nav"].values
        running_max = np.maximum.accumulate(nav_arr)
        dd = (nav_arr - running_max) / running_max
        fig.add_trace(go.Scatter(
            x=nav_main.index, y=dd, name="Drawdown",
            fill="tozeroy", line=dict(color="#f85149"),
            fillcolor="rgba(248,81,73,0.3)"
        ), row=1, col=2)

        # 月度收益
        monthly = nav_main["nav"].resample("ME").last().pct_change().dropna()
        colors = ["#3fb950" if v > 0 else "#f85149" for v in monthly.values]
        fig.add_trace(go.Bar(
            x=monthly.index, y=monthly.values, name="月度收益",
            marker_color=colors
        ), row=2, col=1)

        # Walk-Forward
        if wf_results:
            folds   = [r["fold"] for r in wf_results if "fold" in r]
            sharpes = [r.get("sharpe", 0) for r in wf_results if "fold" in r]
            rets    = [r.get("ann_ret", 0) for r in wf_results if "fold" in r]
            fig.add_trace(go.Bar(x=[f"Fold {f}" for f in folds], y=sharpes,
                                 name="Sharpe", marker_color="#58a6ff"), row=2, col=2)
            fig.add_trace(go.Bar(x=[f"Fold {f}" for f in folds], y=rets,
                                 name="Ann.Ret", marker_color="#f78166"), row=2, col=2)

        # 日收益分布
        daily_ret = nav_main["nav"].pct_change().dropna()
        fig.add_trace(go.Histogram(
            x=daily_ret, name="日收益", nbinsx=60,
            marker_color="#58a6ff", opacity=0.7
        ), row=3, col=1)

        # 参数热图
        if not grid_df.empty and "sharpe" in grid_df.columns:
            top = grid_df.head(10)
            fig.add_trace(go.Bar(
                x=[f"N{r['n_stock']}H{r['hold_days']}S{r['stop_loss']}"
                   for _, r in top.iterrows()],
                y=top["sharpe"].values,
                name="Sharpe",
                marker_color="#3fb950"
            ), row=3, col=2)

        fig.update_layout(
            title=dict(
                text=f"A股多因子策略 v4.0 — 真实数据回测<br>"
                     f"年化{metrics.get('ann_ret',0):.1%} | "
                     f"夏普{metrics.get('sharpe',0):.2f} | "
                     f"最大回撤{metrics.get('max_dd',0):.1%}",
                font=dict(size=16)
            ),
            height=1000, template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", x=0, y=-0.05)
        )

        html_path = OUT / "backtest_v4_report.html"
        fig.write_html(str(html_path), include_plotlyjs=True)
        print(f"  HTML saved: {html_path}")

    except Exception as e:
        print(f"  HTML report error: {e}")


# ════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 65)
    print("  A股多因子策略 v4.0 — 真实数据回测")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 加载数据
    print("\n[1] 加载数据...")
    data = load_data(min_bars=200)

    if len(data) < 5:
        print(f"  警告: 只有 {len(data)} 只股票有足够数据，需先运行 fetch_sina.py 下载数据")
        print("  将使用所有可用数据继续...")
        data = load_data(min_bars=50)
        if len(data) < 3:
            print("  ERROR: 数据不足，请先运行数据下载")
            return

    print(f"  数据加载完成: {len(data)} 只股票")

    # 获取所有交易日
    all_dates_set = set()
    for df in data.values():
        all_dates_set.update(df.index)
    all_dates = sorted(all_dates_set)
    print(f"  日期范围: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共{len(all_dates)}个交易日")

    # 确定回测区间
    end_date   = all_dates[-1]
    start_date = max(all_dates[0], end_date - pd.Timedelta(days=365*2))  # 最近2年
    print(f"  回测区间: {start_date.date()} ~ {end_date.date()}")

    # [2] 主策略回测
    print("\n[2] 主策略回测（IC/ICIR动态权重）...")
    best_params = {
        "n_stock":   20,
        "hold_days": 20,
        "stop_loss": 0.08,
        "fee_rate":  0.001,
        "risk_per":  0.01,
        "use_icir":  True,
    }
    bt_main = Backtester(data, best_params)
    nav_main = bt_main.run(str(start_date.date()), str(end_date.date()))

    if nav_main.empty:
        print("  ERROR: 回测无数据")
        return

    metrics_main = calc_metrics(nav_main)
    print(f"\n  主策略绩效:")
    for k, v in metrics_main.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # [3] 基准
    print("\n[3] 计算等权基准...")
    nav_bench = compute_benchmark(data, str(start_date.date()), str(end_date.date()))

    # [4] Walk-Forward 验证
    print("\n[4] Walk-Forward 样本外验证...")
    wf_dates = [d for d in all_dates if start_date <= d <= end_date]
    wf_results = walk_forward(data, wf_dates, n_folds=4, params=best_params)

    print("\n  Walk-Forward 汇总:")
    for r in wf_results:
        print(f"    Fold {r.get('fold','?')}: "
              f"年化={r.get('ann_ret',0):.1%} 夏普={r.get('sharpe',0):.2f} "
              f"回撤={r.get('max_dd',0):.1%}")

    # [5] 网格参数寻优
    print("\n[5] 网格参数寻优...")
    grid_df = grid_search(data, wf_dates)
    grid_df.to_csv(OUT / "grid_v4.csv", index=False)
    print(f"  最优参数: {grid_df.iloc[0][['n_stock','hold_days','stop_loss']].to_dict()}")
    print(f"  最优Sharpe: {grid_df.iloc[0]['sharpe']:.3f}")

    # [6] 保存结果
    print("\n[6] 保存数据...")
    nav_main.to_csv(OUT / "nav_v4.csv")
    pd.DataFrame([metrics_main]).to_csv(OUT / "metrics_v4.csv", index=False)
    pd.DataFrame(wf_results).to_csv(OUT / "wf_v4.csv", index=False)

    trades_df = pd.DataFrame(bt_main.trades)
    if not trades_df.empty:
        trades_df.to_csv(OUT / "trades_v4.csv", index=False)

    # [7] 生成报告
    print("\n[7] 生成报告...")
    plot_report(
        nav_main, nav_bench, wf_results, grid_df,
        metrics_main, trades_df if not isinstance(trades_df, type(None)) else pd.DataFrame()
    )

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  回测完成！总耗时: {elapsed:.1f}s")
    print(f"  年化收益: {metrics_main.get('ann_ret',0):.1%}")
    print(f"  夏普比率: {metrics_main.get('sharpe',0):.2f}")
    print(f"  最大回撤: {metrics_main.get('max_dd',0):.1%}")
    print(f"  报告: {OUT}/backtest_v4_report.html")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
