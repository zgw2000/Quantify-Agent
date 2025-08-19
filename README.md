# Quantify-Agent
AI quantitative trading strategies for US stocks
## 项目简介
本项目 `Quantify-Agent` 是一个基于AI的美股量化交易策略系统，核心策略为**混合自适应机器学习驱动的TQQQ趋势增强策略**。系统集成了机器学习、动量、趋势跟踪、波动率控制等多因子信号，自动化回测与绩效分析，适合量化爱好者和研究者参考。

## 主要特性

- **机器学习信号**：采用`RandomForestRegressor`对TQQQ未来收益概率建模，动态生成买卖信号。
- **趋势跟踪**：结合超短/短期均线（MA3/MA10）判断市场趋势，灵敏捕捉行情变化。
- **动量与RSI因子**：引入动量回看期与RSI超买超卖阈值，辅助信号过滤。
- **波动率目标**：动态调整持仓，控制投资组合波动率。
- **多重风控**：最大回撤、成交量阈值等多维度风险管理。
- **自动回测与分析**：一键输出策略绩效、超额收益、信号分布、市场状态分布等图表和数据摘要。

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   主要依赖：`yfinance`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pandas_datareader`

2. **运行主程序**
   ```bash
   python main.py
   ```

3. **查看结果**
   - 绩效摘要（JSON/CSV）：`results/hybrid_adaptive_ml_tqqq_summary.json`
   - 分析图表：`results/hybrid_adaptive_ml_tqqq_analysis.png`
   - 控制台输出：策略收益、买入持有对比、信号统计等

## 策略参数（main.py可调）

- 初始资金：`INITIAL_CAPITAL = 50000.0`
- 机器学习买/卖阈值：`ML_BUY_THRESHOLD = 0.52`, `ML_SELL_THRESHOLD = 0.48`
- 均线参数：`TREND_MA_SHORT = 3`, `TREND_MA_LONG = 10`
- 波动率目标：`VOLATILITY_TARGET = 0.35`
- 动量回看期：`MOMENTUM_LOOKBACK = 5`
- RSI阈值：`RSI_OVERSOLD = 30`, `RSI_OVERBOUGHT = 70`
- 成交量阈值：`VOLUME_THRESHOLD = 1.5`

## 结果示例

- **策略收益率**、**CAGR**、**夏普比率**、**最大回撤**等关键指标
- **信号分布**、**市场状态分布**、**ML概率分布**等可视化图表
- **与买入持有TQQQ的超额收益对比**

## 免责声明

本项目仅供学术研究与技术交流，非投资建议。历史回测不代表未来表现，投资有风险，盈亏自负。

## 联系

如有建议或交流，欢迎提交Issue或PR。

---
