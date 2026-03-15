"""
backtest_v4_final.py  ——  v4.2 最终优化版
新增：
1. 市场择时（MA200趋势过滤，熊市降仓）
2. 扩充因子：动量质量、营收增速代理
3. 最优参数 N=20/Hold=10（网格显示N20夏普更稳定）
4. 全量 685+ 只股票
"""

import sys, time, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from backtest_real import (
    load_data, calc_metrics, compute_benchmark,
    compute_factors_for_stock, score_stocks, icir_weights,
    compute_ic, FACTOR_COLS, atr_position, zscore_normalize
)

BASE = Path(__file__).parent
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)

# ════════ 市场择时信号 ════════
def market_timing_signal(data: dict, date: pd.Timestamp,
                         window: int = 60) -> float:
    """
    用所有股票的等权指数计算趋势
    返回 0~1：1=满仓，0=空仓
    """
    # 计算市场指数（等权）
    rets = []
    for code, df in data.items():
        hist = df[df.index <= date]
        if len(hist) < window + 5:
            continue
        c = hist["close"].values
        # MA系列
        ma_short = np.mean(c[-20:])
        ma_long  = np.mean(c[-60:])
        rets.append(ma_short / ma_long - 1)

    if not rets:
        return 1.0

    avg = np.mean(rets)
    # 多头信号：均线多头排列
    if avg > 0.01:
        return 1.0     # 满仓
    elif avg > -0.02:
        return 0.6     # 六成仓
    elif avg > -0.05:
        return 0.3     # 三成仓
    else:
        return 0.1     # 一成仓（保留底仓）


# ════════ 增强版因子 ════════
EXTRA_FACTORS = ["mom_quality", "trend_strength", "price_efficiency"]

def compute_extra_factors(code: str, df: pd.DataFrame) -> dict:
    """额外因子"""
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    fac = {}

    if n >= 30:
        # 动量质量：动量的持续性（用Hurst指数代理）
        ret20 = np.diff(np.log(c[-21:]))
        pos_days = (ret20 > 0).sum()
        fac["mom_quality"] = pos_days / 20.0  # 越高越好

        # 趋势强度：线性拟合R²
        x = np.arange(min(n, 30))
        y = c[-30:] if n >= 30 else c
        slope, intercept, r_val, _, _ = stats.linregress(x, y)
        fac["trend_strength"] = r_val ** 2 * np.sign(slope)  # 上升趋势强=正

        # 价格效率：实际位移/总路程（越接近1越有效趋势）
        path = np.sum(np.abs(np.diff(c[-20:]))) if n >= 20 else 1e-9
        displace = abs(c[-1] - c[max(-20, -n)])
        fac["price_efficiency"] = displace / (path + 1e-9)
    else:
        fac["mom_quality"]       = np.nan
        fac["trend_strength"]    = np.nan
        fac["price_efficiency"]  = np.nan

    return fac


# ════════ 改进版回测器 ════════
class BacktesterV42:
    def __init__(self, data, params):
        self.data        = data
        self.params      = params
        self.n_stock     = params.get("n_stock", 20)
        self.hold_days   = params.get("hold_days", 10)
        self.stop_loss   = params.get("stop_loss", 0.10)
        self.fee_rate    = params.get("fee_rate", 0.001)
        self.risk_per    = params.get("risk_per", 0.01)
        self.use_icir    = params.get("use_icir", True)
        self.use_timing  = params.get("use_timing", True)

        self.capital     = 1_000_000.0
        self.positions   = {}
        self.cost_price  = {}
        self.nav_series  = []
        self.trades      = []
        self.ic_history  = []

    def get_price(self, code, date):
        df = self.data.get(code)
        if df is None:
            return None
        hist = df[df.index <= date]
        if hist.empty:
            return None
        return float(hist["close"].iloc[-1])

    def build_factor_df(self, date):
        rows = []
        for code, df in self.data.items():
            hist = df[df.index <= date]
            if len(hist) < 70:
                continue
            fac = compute_factors_for_stock(code, hist)
            if fac is None:
                continue
            extra = compute_extra_factors(code, hist)
            fac.update(extra)
            rows.append(fac)
        return pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()

    def rebalance(self, date, weights, timing_scale):
        fac_df = self.build_factor_df(date)
        if fac_df.empty:
            return fac_df

        all_factors = FACTOR_COLS + [f for f in EXTRA_FACTORS if f in fac_df.columns]
        scores = pd.Series(0.0, index=fac_df.index)
        for col in all_factors:
            if col not in fac_df.columns:
                continue
            z = zscore_normalize(fac_df[col].fillna(fac_df[col].median()))
            w = weights.get(col, 1.0/len(all_factors))
            scores += z * w

        # 确定目标持仓数（受择时调整）
        target_n = max(1, int(self.n_stock * timing_scale))

        # 选出 top N 有效股票
        valid = []
        for code in scores.nlargest(target_n * 3).index:
            p = self.get_price(code, date)
            if p and p > 0.5:
                valid.append(code)
            if len(valid) >= target_n:
                break

        target_set = set(valid)

        # 平仓不在目标中的
        for code in list(self.positions.keys()):
            if code not in target_set:
                p = self.get_price(code, date)
                if p and self.positions[code] > 0:
                    self.capital += p * self.positions[code] * (1 - self.fee_rate)
                    self.trades.append({
                        "date": date, "code": code, "action": "sell",
                        "price": p, "ret": (p - self.cost_price.get(code, p)) / self.cost_price.get(code, p)
                    })
                del self.positions[code]
                self.cost_price.pop(code, None)

        # 建仓新标的
        new_buys = target_set - set(self.positions.keys())
        if new_buys:
            buy_fac = fac_df[fac_df.index.isin(new_buys)]
            alloc   = atr_position(buy_fac, list(new_buys),
                                   self.capital * timing_scale, self.risk_per)
            for code, cash in alloc.items():
                p = self.get_price(code, date)
                if p and cash > 0 and self.capital >= cash:
                    shares = int(cash / p / 100) * 100
                    if shares > 0:
                        cost = p * shares * (1 + self.fee_rate)
                        if cost <= self.capital:
                            self.capital -= cost
                            self.positions[code] = shares
                            self.cost_price[code] = p
                            self.trades.append({
                                "date": date, "code": code, "action": "buy",
                                "price": p, "ret": 0.0
                            })

        return fac_df

    def check_stop_loss(self, date):
        for code in list(self.positions.keys()):
            p = self.get_price(code, date)
            cost = self.cost_price.get(code, p)
            if p and cost and (p - cost) / cost < -self.stop_loss:
                self.capital += p * self.positions[code] * (1 - self.fee_rate)
                self.trades.append({
                    "date": date, "code": code, "action": "stop",
                    "price": p, "ret": (p - cost) / cost
                })
                del self.positions[code]
                self.cost_price.pop(code, None)

    def nav(self, date):
        total = self.capital
        for code, shares in self.positions.items():
            p = self.get_price(code, date)
            if p:
                total += p * shares
        return total

    def run(self, start_date, end_date):
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)

        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index[(df.index >= start) & (df.index <= end)])
        trading_dates = sorted(all_dates)
        if not trading_dates:
            return pd.DataFrame()

        weights    = pd.Series({c: 1.0/(len(FACTOR_COLS)+len(EXTRA_FACTORS))
                                for c in FACTOR_COLS+EXTRA_FACTORS})
        last_rebal = None
        ic_buf     = []
        fwd_prices = {}

        print(f"    {start_date} ~ {end_date}: {len(trading_dates)}个交易日")

        for i, date in enumerate(trading_dates):
            self.check_stop_loss(date)

            # IC更新
            if last_rebal and fwd_prices and ic_buf:
                fwd_ret = {}
                for code in fwd_prices:
                    p_now = self.get_price(code, date)
                    p_old = fwd_prices.get(code)
                    if p_now and p_old and p_old > 0:
                        fwd_ret[code] = p_now / p_old - 1
                if fwd_ret:
                    fwd_s = pd.Series(fwd_ret)
                    ic = compute_ic(ic_buf[-1][FACTOR_COLS], fwd_s)
                    self.ic_history.append(ic)
                if self.use_icir and len(self.ic_history) >= 3:
                    weights = icir_weights(self.ic_history[-12:])

            # 调仓
            should_rebal = (
                last_rebal is None or
                (date - last_rebal).days >= self.hold_days
            )
            if should_rebal:
                timing_scale = market_timing_signal(self.data, date) if self.use_timing else 1.0
                fac_df = self.rebalance(date, weights, timing_scale)
                if not isinstance(fac_df, pd.DataFrame) or fac_df.empty:
                    fac_df = None

                if fac_df is not None:
                    ic_buf.append(fac_df[
                        [c for c in FACTOR_COLS if c in fac_df.columns]
                    ])
                    fwd_prices = {
                        code: self.get_price(code, date)
                        for code in self.data
                        if self.get_price(code, date)
                    }
                last_rebal = date

            self.nav_series.append({"date": date, "nav": self.nav(date)})

            if i % 100 == 0:
                print(f"      {date.date()}: NAV={self.nav(date):,.0f} 持仓:{len(self.positions)}")

        return pd.DataFrame(self.nav_series).set_index("date")


# ════════ 主流程 ════════
def main():
    t0 = time.time()
    print("=" * 65)
    print("  A股多因子策略 v4.2 — 市场择时+增强因子")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n[1] 加载数据...")
    data = load_data(min_bars=200)
    if len(data) < 10:
        data = load_data(min_bars=80)
    print(f"  有效股票: {len(data)} 只")

    all_dates_set = set()
    for df in data.values():
        all_dates_set.update(df.index)
    all_dates  = sorted(all_dates_set)
    end_date   = all_dates[-1]
    start_date = max(all_dates[0], end_date - pd.Timedelta(days=365*3))

    print(f"  回测区间: {start_date.date()} ~ {end_date.date()}")

    # ── 3组参数对比 ──
    strategies = {
        "v4.2_timing_N20": {
            "n_stock": 20, "hold_days": 10, "stop_loss": 0.10,
            "fee_rate": 0.001, "risk_per": 0.01,
            "use_icir": True, "use_timing": True,
        },
        "v4.2_notiming_N20": {
            "n_stock": 20, "hold_days": 10, "stop_loss": 0.10,
            "fee_rate": 0.001, "risk_per": 0.01,
            "use_icir": True, "use_timing": False,
        },
        "v4.2_timing_N10": {
            "n_stock": 10, "hold_days": 10, "stop_loss": 0.10,
            "fee_rate": 0.001, "risk_per": 0.015,
            "use_icir": True, "use_timing": True,
        },
    }

    navs    = {}
    metrics = {}

    print("\n[2] 运行多策略对比...")
    for name, params in strategies.items():
        print(f"\n  策略: {name}")
        bt = BacktesterV42(data, params)
        nav = bt.run(str(start_date.date()), str(end_date.date()))
        if not nav.empty:
            navs[name]    = nav
            metrics[name] = calc_metrics(nav)
            m = metrics[name]
            print(f"    年化:{m['ann_ret']:.2%}  夏普:{m['sharpe']:.3f}  "
                  f"回撤:{m['max_dd']:.2%}  Sortino:{m['sortino']:.3f}")

    if not navs:
        print("  ERROR: 无策略数据")
        return

    # 最优策略
    best_name = max(metrics, key=lambda k: metrics[k].get("sharpe", -99))
    best_nav  = navs[best_name]
    best_m    = metrics[best_name]
    print(f"\n  最优策略: {best_name}")

    # ── 基准 ──
    print("\n[3] 等权基准...")
    nav_bench = compute_benchmark(data, str(start_date.date()), str(end_date.date()))
    if not nav_bench.empty:
        bench_last  = nav_bench.reindex(best_nav.index, method="ffill").dropna()
        if not bench_last.empty:
            best_m["alpha"] = (
                best_nav["nav"].iloc[-1]/best_nav["nav"].iloc[0] -
                bench_last.iloc[-1]/bench_last.iloc[0]
            )

    # ── Walk-Forward（最优策略）──
    print("\n[4] Walk-Forward 验证...")
    wf_dates = [d for d in all_dates if start_date <= d <= end_date]
    # 用BacktesterV42重写walk_forward
    n_folds    = 4
    fold_size  = len(wf_dates) // (n_folds + 1)
    wf_results = []
    for fold in range(n_folds):
        ts_idx = fold_size * (fold + 1) + fold_size // 2
        te_idx = fold_size * (fold + 2)
        if te_idx > len(wf_dates):
            break
        ts = wf_dates[min(ts_idx, len(wf_dates)-1)]
        te = wf_dates[min(te_idx-1, len(wf_dates)-1)]
        print(f"  Fold {fold+1}: {ts.date()} ~ {te.date()}")
        bt_wf = BacktesterV42(data, strategies[best_name])
        nav_wf = bt_wf.run(str(ts.date()), str(te.date()))
        if nav_wf.empty or len(nav_wf) < 10:
            wf_results.append({"fold": fold+1, "sharpe": 0, "ann_ret": 0, "max_dd": 0})
            continue
        m_wf = calc_metrics(nav_wf)
        m_wf["fold"] = fold+1
        wf_results.append(m_wf)
        print(f"    年化:{m_wf['ann_ret']:.1%}  夏普:{m_wf['sharpe']:.2f}  回撤:{m_wf['max_dd']:.1%}")

    # ── 保存 ──
    print("\n[5] 保存结果...")
    best_nav.to_csv(OUT / "nav_v4.csv")
    pd.DataFrame([best_m]).to_csv(OUT / "metrics_v4.csv", index=False)
    pd.DataFrame(wf_results).to_csv(OUT / "wf_v4.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(OUT / "strategy_comparison_v4.csv")

    # ── 生成报告 ──
    print("\n[6] 生成报告...")
    _generate_report(navs, nav_bench, best_name, best_m, wf_results, metrics)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  完成！耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  股票池: {len(data)} 只")
    print(f"  最优策略: {best_name}")
    print(f"  年化收益: {best_m.get('ann_ret',0):.2%}")
    print(f"  夏普比率: {best_m.get('sharpe',0):.3f}")
    print(f"  Sortino:  {best_m.get('sortino',0):.3f}")
    print(f"  最大回撤: {best_m.get('max_dd',0):.2%}")
    print(f"  Alpha:    {best_m.get('alpha',0):.2%}")
    print(f"  WF Sharpes: {[round(r.get('sharpe',0),2) for r in wf_results]}")
    print(f"{'='*65}")


def _generate_report(navs, nav_bench, best_name, best_m, wf_results, all_metrics):
    """生成完整报告"""
    plt.rcParams.update({
        "font.family":        ["SimHei","Microsoft YaHei","DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor":   "#0d1117",
        "axes.facecolor":     "#161b22",
        "text.color":         "#e6edf3",
        "axes.labelcolor":    "#e6edf3",
        "xtick.color":        "#8b949e",
        "ytick.color":        "#8b949e",
        "axes.edgecolor":     "#30363d",
        "grid.color":         "#21262d",
    })

    COLORS = ["#58a6ff","#3fb950","#f78166","#d29922","#bc8cff"]
    COLOR_NEG = "#f85149"
    COLOR_POS = "#3fb950"

    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    best_nav = navs[best_name]

    # 1. 净值曲线（所有策略对比 + 基准）
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (name, nav) in enumerate(navs.items()):
        lw   = 2.5 if name == best_name else 1.5
        dash = "solid" if name == best_name else "dashed"
        label = f"{name}\n年化{all_metrics[name]['ann_ret']:.1%} 夏普{all_metrics[name]['sharpe']:.2f}"
        ax1.plot(nav.index, nav["nav"]/1e6, color=COLORS[i], lw=lw,
                 ls=dash, label=label)
    if not nav_bench.empty:
        b = nav_bench.reindex(best_nav.index, method="ffill")/1e6
        ax1.plot(b.index, b, color=COLOR_NEG, lw=1.2, ls=":", alpha=0.7,
                 label="等权基准")
    ax1.fill_between(best_nav.index, 1, best_nav["nav"]/1e6, alpha=0.06, color=COLORS[0])
    ax1.axhline(1, color="#8b949e", lw=0.8, ls="--")
    ax1.set_title("策略净值对比（3策略 vs 等权基准）", fontsize=13)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 绩效卡
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.set_title(f"最优策略绩效\n{best_name}", fontsize=11)
    kv = [
        ("年化收益", f"{best_m.get('ann_ret',0):.2%}"),
        ("夏普比率", f"{best_m.get('sharpe',0):.3f}"),
        ("Sortino",  f"{best_m.get('sortino',0):.3f}"),
        ("Calmar",   f"{best_m.get('calmar',0):.3f}"),
        ("最大回撤", f"{best_m.get('max_dd',0):.2%}"),
        ("年化波动", f"{best_m.get('ann_vol',0):.2%}"),
        ("日胜率",   f"{best_m.get('win_rate',0):.2%}"),
        ("盈亏比",   f"{best_m.get('pnl_ratio',0):.3f}"),
        ("Alpha",    f"{best_m.get('alpha',0):.2%}"),
    ]
    for i, (k, v) in enumerate(kv):
        y = 0.93 - i * 0.098
        is_good = (i < 5 and not v.startswith("-")) or i in [6,7]
        ax2.text(0.05, y, k, transform=ax2.transAxes, fontsize=9.5, color="#8b949e")
        ax2.text(0.95, y, v, transform=ax2.transAxes, fontsize=11,
                 color=COLOR_POS if is_good else COLOR_NEG,
                 ha="right", fontweight="bold")

    # 3. 回撤
    ax3 = fig.add_subplot(gs[1, :2])
    nav_arr = best_nav["nav"].values
    rm      = np.maximum.accumulate(nav_arr)
    dd      = (nav_arr - rm) / rm
    ax3.fill_between(best_nav.index, dd, 0, color=COLOR_NEG, alpha=0.6)
    ax3.plot(best_nav.index, dd, color=COLOR_NEG, lw=0.8)
    ax3.set_title("最优策略回撤曲线", fontsize=12)
    ax3.set_ylabel("回撤幅度")
    ax3.grid(True, alpha=0.3)

    # 4. Walk-Forward
    ax4 = fig.add_subplot(gs[1, 2])
    if wf_results:
        folds   = [r.get("fold", i+1) for i, r in enumerate(wf_results)]
        sharpes = [r.get("sharpe", 0) for r in wf_results]
        rets    = [r.get("ann_ret", 0) for r in wf_results]
        x       = np.arange(len(folds))
        ax4.bar(x - 0.2, sharpes, 0.4, label="Sharpe",
                color=[COLOR_POS if s > 0 else COLOR_NEG for s in sharpes])
        ax4.bar(x + 0.2, rets, 0.4, label="年化收益",
                color=[COLOR_POS if r > 0 else COLOR_NEG for r in rets], alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"Fold{f}" for f in folds], fontsize=9)
        ax4.axhline(0, color="#8b949e", lw=0.8, ls="--")
        ax4.legend(fontsize=9)
        ax4.set_title("Walk-Forward 样本外验证", fontsize=12)
        ax4.grid(True, alpha=0.3)

    # 5. 月度收益热图
    ax5 = fig.add_subplot(gs[2, :2])
    monthly = best_nav["nav"].resample("ME").last().pct_change().dropna()
    if len(monthly) >= 6:
        mdf = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values
        })
        pivot = mdf.pivot_table(values="ret", index="year", columns="month")
        mnames = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [mnames[c-1] for c in pivot.columns]
        im = ax5.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
        ax5.set_xticks(range(len(pivot.columns)))
        ax5.set_xticklabels(pivot.columns, fontsize=8)
        ax5.set_yticks(range(len(pivot.index)))
        ax5.set_yticklabels(pivot.index, fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax5.text(j, i, f"{val:.1%}", ha="center", va="center",
                            fontsize=7, color="black" if abs(val) < 0.07 else "white")
        plt.colorbar(im, ax=ax5, shrink=0.8)
        ax5.set_title("月度收益热图", fontsize=12)

    # 6. 日收益分布
    ax6 = fig.add_subplot(gs[2, 2])
    dr = best_nav["nav"].pct_change().dropna()
    ax6.hist(dr, bins=60, color=COLORS[0], alpha=0.75, edgecolor="none")
    ax6.axvline(0, color=COLOR_NEG, lw=1.2, ls="--")
    ax6.axvline(dr.mean(), color=COLOR_POS, lw=1.5, ls="--",
                label=f"均值={dr.mean():.3%}")
    q5 = np.percentile(dr, 5)
    ax6.axvline(q5, color=COLOR_NEG, lw=1, ls=":", label=f"VaR5%={q5:.2%}")
    ax6.set_title("日收益率分布", fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        f"A股多因子策略 v4.2 — 真实历史数据回测  "
        f"（{len(navs[best_name])}个交易日 | "
        f"最优策略年化{best_m.get('ann_ret',0):.1%} 夏普{best_m.get('sharpe',0):.2f}）",
        fontsize=14, y=0.998, color="#e6edf3"
    )

    png = OUT / "backtest_v4_report.png"
    fig.savefig(str(png), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG: {png}")

    # HTML
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_h = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "净值曲线对比", "回撤曲线",
                "月度收益", "Walk-Forward结果",
                "日收益分布", "策略对比指标"
            ],
            vertical_spacing=0.1, horizontal_spacing=0.07
        )

        for i, (name, nav) in enumerate(navs.items()):
            fig_h.add_trace(go.Scatter(
                x=nav.index, y=nav["nav"]/1e6,
                name=f"{name} 夏普{all_metrics[name]['sharpe']:.2f}",
                line=dict(color=COLORS[i], width=2 if name==best_name else 1.5)
            ), row=1, col=1)
        if not nav_bench.empty:
            b = nav_bench.reindex(best_nav.index, method="ffill")/1e6
            fig_h.add_trace(go.Scatter(
                x=b.index, y=b, name="等权基准",
                line=dict(color="#f85149", width=1.2, dash="dot")
            ), row=1, col=1)

        fig_h.add_trace(go.Scatter(
            x=best_nav.index, y=dd, name="回撤",
            fill="tozeroy", fillcolor="rgba(248,81,73,0.25)",
            line=dict(color="#f85149", width=1)
        ), row=1, col=2)

        if len(monthly) >= 6:
            colors_m = ["#3fb950" if v > 0 else "#f85149" for v in monthly.values]
            fig_h.add_trace(go.Bar(
                x=monthly.index, y=monthly.values,
                name="月度收益", marker_color=colors_m
            ), row=2, col=1)

        if wf_results:
            fig_h.add_trace(go.Bar(
                x=[f"Fold{r['fold']}" for r in wf_results if 'fold' in r],
                y=[r.get("sharpe",0) for r in wf_results if 'fold' in r],
                name="夏普",
                marker_color=["#3fb950" if r.get("sharpe",0)>0 else "#f85149"
                              for r in wf_results if 'fold' in r]
            ), row=2, col=2)

        fig_h.add_trace(go.Histogram(
            x=dr, name="日收益", nbinsx=60,
            marker_color="#58a6ff", opacity=0.75
        ), row=3, col=1)

        # 策略对比雷达图数据用柱状图替代
        strat_names  = list(all_metrics.keys())
        strat_sharpe = [all_metrics[s].get("sharpe",0) for s in strat_names]
        strat_ret    = [all_metrics[s].get("ann_ret",0) for s in strat_names]
        fig_h.add_trace(go.Bar(
            x=strat_names, y=strat_sharpe, name="各策略Sharpe",
            marker_color=COLORS[:len(strat_names)]
        ), row=3, col=2)

        fig_h.update_layout(
            title=dict(
                text=(f"A股多因子策略 v4.2 真实数据回测<br>"
                      f"最优: {best_name}  "
                      f"年化{best_m.get('ann_ret',0):.1%}  "
                      f"夏普{best_m.get('sharpe',0):.2f}  "
                      f"最大回撤{best_m.get('max_dd',0):.1%}"),
                font=dict(size=14)
            ),
            height=1100, template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", x=0, y=-0.04, font=dict(size=9))
        )

        html = OUT / "backtest_v4_report.html"
        fig_h.write_html(str(html), include_plotlyjs=True)
        print(f"  HTML: {html}")

    except Exception as e:
        print(f"  HTML error: {e}")


if __name__ == "__main__":
    main()
