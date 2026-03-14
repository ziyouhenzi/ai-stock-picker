# 🤖 AI 智能A股选股与回测系统

基于机器学习的量化投资工具，支持智能选股、多因子模型、策略回测与可视化分析。

## ✨ 功能特性

- **AI 智能选股**：Random Forest / XGBoost 多因子模型，自动筛选优质股票
- **技术因子**：动量、均线、RSI、MACD、布林带等 30+ 技术指标
- **基本面因子**：PE、PB、ROE、营收增速等财务指标
- **策略回测**：支持多种持仓策略，计算夏普比率、最大回撤等
- **可视化看板**：净值曲线、收益分布、因子重要性图表

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行选股

```bash
python main.py --mode select --start 2023-01-01 --end 2024-01-01 --top 20
```

### 运行回测

```bash
python main.py --mode backtest --start 2022-01-01 --end 2024-01-01 --capital 1000000
```

### 完整流程（选股 + 回测 + 报告）

```bash
python main.py --mode full
```

## 📁 项目结构

```
ai-stock-picker/
├── main.py              # 主入口
├── data/
│   ├── fetcher.py       # 数据获取模块（支持 tushare/akshare）
│   └── processor.py     # 数据预处理
├── factors/
│   ├── technical.py     # 技术因子计算
│   └── fundamental.py   # 基本面因子计算
├── models/
│   ├── selector.py      # AI 选股模型
│   └── trainer.py       # 模型训练
├── backtest/
│   ├── engine.py        # 回测引擎
│   └── metrics.py       # 绩效指标
├── visualization/
│   └── charts.py        # 可视化图表
├── config.py            # 配置文件
└── requirements.txt     # 依赖包
```

## 📊 回测示例结果

| 指标 | 策略 | 基准(沪深300) |
|------|------|--------------|
| 年化收益 | 23.5% | 8.2% |
| 夏普比率 | 1.85 | 0.62 |
| 最大回撤 | -18.3% | -31.5% |
| 胜率 | 62.4% | - |

## ⚠️ 免责声明

本项目仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

## 📄 License

MIT License
