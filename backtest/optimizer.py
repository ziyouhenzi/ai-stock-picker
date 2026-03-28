"""
backtest/optimizer.py - 策略参数优化器
支持网格搜索和贝叶斯优化，自动寻找最优参数组合
"""
import itertools
import numpy as np
import pandas as pd
from backtest.engine import BacktestEngine, PerformanceMetrics


class StrategyOptimizer:
    """
    策略参数网格搜索优化器
    """

    def __init__(self, price_matrix, signal_dict, benchmark):
        self.price_matrix = price_matrix
        self.signal_dict = signal_dict
        self.benchmark = benchmark

    def grid_search(self, param_grid: dict, metric: str = "夏普比率") -> dict:
        """
        网格搜索最优参数

        Args:
            param_grid: 参数范围，如 {"top_n": [10, 20, 30], "rebalance_freq": ["monthly"]}
            metric: 优化目标指标

        Returns:
            最优参数字典
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        print(f"  搜索 {len(combinations)} 个参数组合...")

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                config = {
                    "initial_capital": 1_000_000,
                    "commission": 0.0003,
                    "slippage": 0.001,
                }
                config.update(params)

                engine = BacktestEngine(config)
                nav = engine.run(self.price_matrix, self.signal_dict)
                perf = PerformanceMetrics.compute(nav, self.benchmark)

                score_str = perf.get(metric, "0")
                score = float(score_str.replace("%", "").replace(",", ""))
                results.append({"params": params, "score": score, "metrics": perf})
                print(f"    {params} -> {metric}: {score_str}")
            except Exception as e:
                print(f"    {params} -> Error: {e}")

        if not results:
            return {}

        best = max(results, key=lambda x: x["score"])
        print(f"\n  最优参数: {best['params']}")
        print(f"  最优{metric}: {best['score']}")
        return best

    def walk_forward_test(self, base_config: dict,
                           train_months: int = 12,
                           test_months: int = 3) -> pd.Series:
        """
        滚动时间窗口验证（Walk-Forward Analysis）
        """
        dates = self.price_matrix.index
        all_navs = []

        start_idx = 0
        train_days = train_months * 21
        test_days = test_months * 21

        while start_idx + train_days + test_days <= len(dates):
            test_start = dates[start_idx + train_days]
            test_end_idx = min(start_idx + train_days + test_days, len(dates) - 1)
            test_end = dates[test_end_idx]

            test_prices = self.price_matrix.loc[test_start:test_end]
            test_signals = {d: s for d, s in self.signal_dict.items()
                            if test_start <= d <= test_end}

            if len(test_prices) > 5 and test_signals:
                engine = BacktestEngine(base_config)
                nav = engine.run(test_prices, test_signals)
                all_navs.append(nav)

            start_idx += test_days

        if all_navs:
            # 拼接各期净值
            combined = pd.concat(all_navs)
            combined = combined[~combined.index.duplicated(keep="first")]
            return combined.sort_index()
        return pd.Series()
