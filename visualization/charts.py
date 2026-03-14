"""
visualization/charts.py - 可视化模块
生成净值曲线、回撤图、因子重要性等图表
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

# 设置中文字体
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
except Exception:
    pass

COLORS = {
    "strategy": "#E74C3C",
    "benchmark": "#3498DB",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "bg": "#F8F9FA",
}


def plot_performance_dashboard(nav: pd.Series,
                                benchmark: pd.Series,
                                metrics: dict,
                                feature_importance: pd.Series = None,
                                save_path: str = "output/reports/dashboard.png"):
    """生成完整的回测看板"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(18, 14), facecolor=COLORS["bg"])
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── 1. 净值曲线 ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    nav_norm = nav / nav.iloc[0]
    bench_norm = benchmark / benchmark.iloc[0]

    ax1.plot(nav_norm.index, nav_norm.values,
             color=COLORS["strategy"], linewidth=2, label="AI策略")
    ax1.plot(bench_norm.index, bench_norm.values,
             color=COLORS["benchmark"], linewidth=1.5,
             linestyle="--", alpha=0.8, label="沪深300基准")
    ax1.fill_between(nav_norm.index, nav_norm.values, bench_norm.values,
                     where=nav_norm.values >= bench_norm.values,
                     alpha=0.1, color=COLORS["positive"])
    ax1.fill_between(nav_norm.index, nav_norm.values, bench_norm.values,
                     where=nav_norm.values < bench_norm.values,
                     alpha=0.1, color=COLORS["negative"])
    ax1.set_title("策略净值曲线 vs 基准", fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # ── 2. 绩效指标表格 ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    key_metrics = [
        ("年化收益率", metrics.get("年化收益率", "-")),
        ("基准年化收益", metrics.get("基准年化收益", "-")),
        ("夏普比率", metrics.get("夏普比率", "-")),
        ("最大回撤", metrics.get("最大回撤", "-")),
        ("卡玛比率", metrics.get("卡玛比率", "-")),
        ("日胜率", metrics.get("日胜率", "-")),
        ("Alpha", metrics.get("Alpha", "-")),
        ("Beta", metrics.get("Beta", "-")),
        ("信息比率", metrics.get("信息比率", "-")),
    ]
    table_data = [[k, v] for k, v in key_metrics]
    table = ax2.table(cellText=table_data, colLabels=["指标", "数值"],
                      cellLoc="center", loc="center",
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECF0F1")
        cell.set_edgecolor("#BDC3C7")
    ax2.set_title("回测绩效摘要", fontsize=13, fontweight="bold", pad=10)

    # ── 3. 回撤曲线 ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    cummax = nav_norm.cummax()
    drawdown = (nav_norm - cummax) / cummax * 100
    ax3.fill_between(drawdown.index, drawdown.values, 0,
                     color=COLORS["negative"], alpha=0.6, label="策略回撤")
    ax3.set_title("历史回撤曲线", fontsize=14, fontweight="bold", pad=10)
    ax3.set_ylabel("回撤 (%)")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)

    # ── 4. 月度收益热力图 ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    monthly_returns = nav.resample("M").last().pct_change().dropna()
    monthly_df = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values
    })
    if not monthly_df.empty:
        pivot = monthly_df.pivot_table(values="return", index="year", columns="month")
        months = ["1月","2月","3月","4月","5月","6月",
                  "7月","8月","9月","10月","11月","12月"]
        pivot.columns = [months[c-1] for c in pivot.columns if c <= 12]
        im = ax4.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                        vmin=-0.1, vmax=0.1)
        ax4.set_xticks(range(len(pivot.columns)))
        ax4.set_xticklabels(pivot.columns, fontsize=8, rotation=45)
        ax4.set_yticks(range(len(pivot.index)))
        ax4.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax4, label="月度收益率")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax4.text(j, i, f"{val:.1%}", ha="center", va="center",
                             fontsize=7, color="black")
    ax4.set_title("月度收益热力图", fontsize=13, fontweight="bold", pad=10)

    # ── 5. 日收益分布 ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    daily_ret = nav.pct_change().dropna() * 100
    ax5.hist(daily_ret, bins=50, color=COLORS["strategy"], alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax5.axvline(daily_ret.mean(), color="black", linestyle="--",
                linewidth=1.5, label=f"均值 {daily_ret.mean():.2f}%")
    ax5.set_title("日收益率分布", fontsize=13, fontweight="bold", pad=10)
    ax5.set_xlabel("日收益率 (%)")
    ax5.set_ylabel("频次")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ── 6. 因子重要性 ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1:])
    if feature_importance is not None and not feature_importance.empty:
        top_factors = feature_importance.head(15)
        colors = [COLORS["positive"] if i < 5 else
                  COLORS["strategy"] if i < 10 else
                  COLORS["neutral"] for i in range(len(top_factors))]
        bars = ax6.barh(range(len(top_factors)), top_factors.values[::-1],
                        color=colors[::-1], alpha=0.85, edgecolor="white")
        ax6.set_yticks(range(len(top_factors)))
        ax6.set_yticklabels(top_factors.index.tolist()[::-1], fontsize=9)
        ax6.set_title("Top 15 因子重要性", fontsize=13, fontweight="bold", pad=10)
        ax6.set_xlabel("重要性得分")
        ax6.grid(True, alpha=0.3, axis="x")
    else:
        ax6.text(0.5, 0.5, "暂无因子重要性数据",
                 ha="center", va="center", transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Top 15 因子重要性", fontsize=13, fontweight="bold", pad=10)

    # 标题
    fig.suptitle("AI 智能A股选股回测报告", fontsize=18, fontweight="bold",
                 y=0.98, color="#2C3E50")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  Dashboard saved: {save_path}")
    return save_path
