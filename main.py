"""
main.py - AI 智能A股选股与回测系统主入口
用法:
  python main.py --mode select   # 仅选股
  python main.py --mode backtest # 仅回测
  python main.py --mode full     # 完整流程（默认）
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from config import BACKTEST_CONFIG, MODEL_CONFIG, FACTOR_LIST, OUTPUT_DIR
from data.fetcher import get_stock_list, get_daily_data, get_index_data
from factors.technical import compute_all_factors
from models.selector import AIStockSelector
from backtest.engine import BacktestEngine, PerformanceMetrics

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)


def banner():
    print("\n" + "=" * 60)
    print("    🤖 AI 智能A股选股与回测系统 v1.0")
    print("    Powered by Machine Learning + 量化回测")
    print("=" * 60 + "\n")


def build_feature_matrix(stock_list, start_date, end_date, verbose=True):
    """构建全股票因子矩阵和标签"""
    all_factors = []
    all_labels = []
    price_dict = {}

    total = len(stock_list)
    for i, (_, row) in enumerate(stock_list.iterrows()):
        code = row["code"]
        if verbose and i % 10 == 0:
            print(f"  Loading [{i+1}/{total}] {code} {row.get('name','')}")

        df = get_daily_data(code, start_date, end_date)
        if df is None or len(df) < 60:
            continue

        price_dict[code] = df

        # 计算技术因子
        factors = compute_all_factors(df)

        # 构建标签：未来 N 日涨幅 > threshold 为正样本
        horizon = MODEL_CONFIG.get("predict_horizon", 20)
        threshold = MODEL_CONFIG.get("label_threshold", 0.05)
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        label = (future_return > threshold).astype(int)

        # 合并
        combined = factors.copy()
        combined["label"] = label
        combined["code"] = code
        combined = combined.dropna()

        if len(combined) > 0:
            all_factors.append(combined)

    if not all_factors:
        print("[Error] 无法获取足够数据")
        return None, None, None

    full_df = pd.concat(all_factors, ignore_index=False)
    feature_cols = [c for c in full_df.columns if c not in ("label", "code")]
    X = full_df[feature_cols]
    y = full_df["label"]

    print(f"\n  数据集大小: {len(X)} 样本, {len(feature_cols)} 特征")
    print(f"  正样本比例: {y.mean():.2%}")

    return X, y, price_dict


def run_full_pipeline(args):
    """完整流程：数据 → 因子 → 训练 → 选股 → 回测 → 报告"""
    banner()
    cfg = BACKTEST_CONFIG.copy()
    if args.start:
        cfg["start_date"] = args.start
    if args.end:
        cfg["end_date"] = args.end
    if args.capital:
        cfg["initial_capital"] = args.capital
    top_n = args.top or cfg["top_n"]

    print(f"[1/6] 获取股票池...")
    stock_list = get_stock_list("hs300")
    # 限制数量以加快演示（实际可去掉）
    if len(stock_list) > 50:
        stock_list = stock_list.head(50)
    print(f"  股票池大小: {len(stock_list)} 只")

    print(f"\n[2/6] 获取数据并计算因子...")
    # 训练集用回测开始前1年
    train_end = cfg["start_date"]
    from dateutil.relativedelta import relativedelta
    try:
        from dateutil.relativedelta import relativedelta
        train_start = (pd.Timestamp(cfg["start_date"]) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    except Exception:
        train_start = cfg["start_date"][:4 - 1] + str(int(cfg["start_date"][:4]) - 1) + cfg["start_date"][4:]

    X_train, y_train, price_dict_train = build_feature_matrix(
        stock_list, train_start, cfg["start_date"])

    if X_train is None:
        print("[Error] 训练数据不足，退出")
        return

    print(f"\n[3/6] 训练 AI 选股模型 ({MODEL_CONFIG['model_type']})...")
    selector = AIStockSelector(
        model_type=MODEL_CONFIG["model_type"],
        config=MODEL_CONFIG
    )
    X_prep, y_prep = selector.prepare_data(X_train, y_train)
    auc = selector.train(X_prep, y_prep)
    selector.save("output/models/selector.pkl")
    print(f"  训练 AUC: {auc:.4f}")

    print(f"\n[4/6] 获取回测期数据并生成选股信号...")
    _, _, price_dict_bt = build_feature_matrix(
        stock_list, cfg["start_date"], cfg["end_date"], verbose=False)

    if price_dict_bt is None:
        print("[Error] 回测数据不足")
        return

    # 构建价格矩阵
    price_matrix = pd.DataFrame({
        code: df["close"] for code, df in price_dict_bt.items()
    }).sort_index()

    # 按月生成信号
    dates = price_matrix.index
    monthly_dates = [g.iloc[0] for _, g in
                     pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M"))]

    signal_dict = {}
    for rebal_date in monthly_dates:
        factor_snapshot = {}
        for code, df in price_dict_bt.items():
            try:
                factors = compute_all_factors(df)
                if rebal_date in factors.index:
                    factor_snapshot[code] = factors.loc[rebal_date]
            except Exception:
                continue
        if factor_snapshot:
            snapshot_df = pd.DataFrame(factor_snapshot).T
            try:
                snapshot_clean = snapshot_df[selector.feature_names].fillna(
                    snapshot_df[selector.feature_names].median())
                scores = selector.predict_scores(snapshot_clean)
                signal_dict[rebal_date] = scores
            except Exception:
                pass

    print(f"  生成调仓信号: {len(signal_dict)} 个月")

    print(f"\n[5/6] 执行回测...")
    engine = BacktestEngine(cfg)
    nav = engine.run(price_matrix, signal_dict, weight_mode="score")
    nav_normalized = nav / nav.iloc[0] * 1.0  # 已经是绝对值，不需要归一

    # 基准数据
    bench_data = get_index_data(cfg["benchmark"], cfg["start_date"], cfg["end_date"])
    bench_nav = bench_data["close"]

    metrics = PerformanceMetrics.compute(nav, bench_nav)

    print("\n" + "=" * 50)
    print("  📊 回测绩效指标")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<16}: {v}")

    print(f"\n[6/6] 生成可视化报告...")
    feature_imp = selector.get_feature_importance()

    try:
        from visualization.charts import plot_performance_dashboard
        chart_path = plot_performance_dashboard(
            nav=nav,
            benchmark=bench_nav,
            metrics=metrics,
            feature_importance=feature_imp,
            save_path="output/reports/dashboard.png"
        )
        print(f"  报告已保存: {chart_path}")
    except Exception as e:
        print(f"  [Warning] 图表生成失败: {e}")

    # 保存指标 CSV
    pd.DataFrame([metrics]).to_csv("output/reports/metrics.csv", index=False, encoding="utf-8-sig")
    nav.to_csv("output/reports/nav.csv", header=True, encoding="utf-8-sig")
    print("\n✅ 全部完成！结果保存在 output/reports/ 目录")
    print(f"   年化收益: {metrics.get('年化收益率','N/A')}")
    print(f"   夏普比率: {metrics.get('夏普比率','N/A')}")
    print(f"   最大回撤: {metrics.get('最大回撤','N/A')}")


def main():
    parser = argparse.ArgumentParser(description="AI 智能A股选股与回测系统")
    parser.add_argument("--mode", default="full",
                        choices=["select", "backtest", "full"],
                        help="运行模式")
    parser.add_argument("--start", default=None, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None, help="初始资金（元）")
    parser.add_argument("--top", type=int, default=None, help="选股数量")
    args = parser.parse_args()

    if args.mode in ("full", "backtest", "select"):
        run_full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
