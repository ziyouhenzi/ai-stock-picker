"""
backtest/engine.py - 回测引擎
支持等权/评分加权持仓，月度调仓，计算完整绩效指标
"""
import numpy as np
import pandas as pd
from datetime import datetime


class BacktestEngine:
    """
    向量化回测引擎
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get("initial_capital", 1_000_000)
        self.commission = config.get("commission", 0.0003)
        self.slippage = config.get("slippage", 0.001)
        self.top_n = config.get("top_n", 20)
        self.rebalance_freq = config.get("rebalance_freq", "monthly")

        # 结果存储
        self.portfolio_value = None
        self.trades = []
        self.holdings = {}

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list:
        """按调仓频率获取调仓日期"""
        if self.rebalance_freq == "weekly":
            # 每周第一个交易日
            return [g.iloc[0] for _, g in
                    pd.Series(dates).groupby(pd.Series(dates).dt.isocalendar().week)]
        elif self.rebalance_freq == "monthly":
            return [g.iloc[0] for _, g in
                    pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M"))]
        elif self.rebalance_freq == "quarterly":
            return [g.iloc[0] for _, g in
                    pd.Series(dates).groupby(pd.Series(dates).dt.to_period("Q"))]
        else:
            return list(dates)

    def run(self, price_matrix: pd.DataFrame,
            signal_dict: dict,
            weight_mode: str = "equal") -> pd.DataFrame:
        """
        执行回测

        Args:
            price_matrix: 价格矩阵，index=日期，columns=股票代码
            signal_dict: 每个调仓日对应的选股列表或评分 Series
            weight_mode: equal（等权）或 score（按评分加权）

        Returns:
            portfolio_value: 每日组合净值 Series
        """
        dates = price_matrix.index
        rebalance_dates = self._get_rebalance_dates(dates)

        capital = self.initial_capital
        cash = capital
        positions = {}   # {code: shares}
        portfolio_values = []

        for date in dates:
            # 调仓
            if date in rebalance_dates and date in signal_dict:
                signals = signal_dict[date]

                # 计算当前持仓市值
                current_value = cash
                for code, shares in positions.items():
                    if code in price_matrix.columns and date in price_matrix.index:
                        price = price_matrix.loc[date, code]
                        if not np.isnan(price):
                            current_value += shares * price

                # 确定目标股票和权重
                if isinstance(signals, pd.Series):
                    top_stocks = signals.nlargest(self.top_n).index.tolist()
                    scores = signals[top_stocks]
                else:
                    top_stocks = signals[:self.top_n]
                    scores = None

                if weight_mode == "score" and scores is not None:
                    total_score = scores.sum()
                    weights = {s: scores[s] / total_score for s in top_stocks}
                else:
                    weights = {s: 1.0 / len(top_stocks) for s in top_stocks}

                # 平仓旧持仓
                for code in list(positions.keys()):
                    if code not in top_stocks:
                        if code in price_matrix.columns:
                            sell_price = price_matrix.loc[date, code]
                            if not np.isnan(sell_price):
                                proceeds = positions[code] * sell_price
                                cost = proceeds * (self.commission + self.slippage)
                                cash += proceeds - cost
                                self.trades.append({
                                    "date": date, "code": code,
                                    "action": "sell", "price": sell_price,
                                    "shares": positions[code]
                                })
                        del positions[code]

                # 建仓新持仓
                investable = current_value * 0.98  # 保留2%现金
                for code in top_stocks:
                    if code not in price_matrix.columns:
                        continue
                    buy_price = price_matrix.loc[date, code]
                    if np.isnan(buy_price) or buy_price <= 0:
                        continue
                    target_value = investable * weights.get(code, 0)
                    buy_price_adj = buy_price * (1 + self.slippage)
                    shares = int(target_value / buy_price_adj / 100) * 100  # 整手
                    if shares > 0:
                        cost = shares * buy_price_adj * (1 + self.commission)
                        if cost <= cash:
                            cash -= cost
                            positions[code] = positions.get(code, 0) + shares
                            self.trades.append({
                                "date": date, "code": code,
                                "action": "buy", "price": buy_price,
                                "shares": shares
                            })

            # 计算组合净值
            nav = cash
            for code, shares in positions.items():
                if code in price_matrix.columns and date in price_matrix.index:
                    price = price_matrix.loc[date, code]
                    if not np.isnan(price):
                        nav += shares * price
            portfolio_values.append(nav)

        self.portfolio_value = pd.Series(
            portfolio_values, index=dates, name="portfolio"
        )
        return self.portfolio_value


class PerformanceMetrics:
    """回测绩效指标计算"""

    @staticmethod
    def compute(nav: pd.Series, benchmark: pd.Series = None,
                rf_rate: float = 0.02) -> dict:
        """
        计算完整绩效指标

        Args:
            nav: 组合净值序列
            benchmark: 基准净值序列
            rf_rate: 无风险利率（年化）
        """
        nav = nav.dropna()
        daily_returns = nav.pct_change().dropna()
        n_days = len(daily_returns)
        n_years = n_days / 252

        # 年化收益
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 年化波动率
        annual_vol = daily_returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = (annual_return - rf_rate) / (annual_vol + 1e-9)

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡玛比率
        calmar = annual_return / abs(max_drawdown + 1e-9)

        # 胜率
        win_rate = (daily_returns > 0).mean()

        # 盈亏比
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = daily_returns[daily_returns < 0].mean()
        profit_loss_ratio = abs(avg_win / (avg_loss + 1e-9))

        metrics = {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.4f}",
            "最大回撤": f"{max_drawdown:.2%}",
            "卡玛比率": f"{calmar:.4f}",
            "日胜率": f"{win_rate:.2%}",
            "盈亏比": f"{profit_loss_ratio:.4f}",
            "回测天数": n_days,
            "回测年数": f"{n_years:.2f}",
        }

        # 相对基准指标
        if benchmark is not None:
            bench_returns = benchmark.pct_change().dropna()
            bench_returns, daily_returns_aligned = bench_returns.align(
                daily_returns, join="inner")

            # Alpha & Beta
            cov = np.cov(daily_returns_aligned, bench_returns)
            beta = cov[0, 1] / (cov[1, 1] + 1e-9)
            bench_annual = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1/n_years) - 1
            alpha = annual_return - (rf_rate + beta * (bench_annual - rf_rate))

            # 信息比率
            active_returns = daily_returns_aligned - bench_returns
            ir = active_returns.mean() / (active_returns.std() + 1e-9) * np.sqrt(252)

            metrics.update({
                "Alpha": f"{alpha:.4f}",
                "Beta": f"{beta:.4f}",
                "信息比率": f"{ir:.4f}",
                "基准年化收益": f"{bench_annual:.2%}",
            })

        return metrics
