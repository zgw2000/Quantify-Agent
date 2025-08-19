# Quantify-Agent
AI quantitative trading strategies for US stocks
## 项目简介

本项目 `Quantify-Agent` 提供了一个基于AI的美股量化交易策略示例，核心策略文件为 `tqqq.py`，主要针对TQQQ（ProShares UltraPro QQQ）进行分层买卖操作，结合标普500指数（SPX）的回撤情况动态调整仓位。

## 主要特性

- **分层买入/卖出策略**：根据SPX从历史高点的回撤幅度，采用1:2:4:8:16的分层买入比例，卖出时采用16:8:4:2:1的分层卖出比例，并在每层卖出时设置基准价+5%的触发条件。
- **自动数据下载**：优先使用Yahoo Finance下载SPX和TQQQ的历史数据，支持多次重试，失败时可切换数据源或模拟数据。
- **动态回测周期**：默认回测近10年数据，自动计算起始日期。
- **可视化与结果分析**：集成matplotlib用于结果可视化，便于分析策略表现。

## 快速开始

1. **安装依赖**

   ```bash
   pip install yfinance pandas numpy matplotlib pandas_datareader
   ```

2. **运行策略**

   ```bash
   python tqqq.py
   ```

3. **参数说明**

   - `INITIAL_CAPITAL`：初始资金（默认50000美元）
   - `LAYER_THRESHOLDS`：分层买入的SPX回撤阈值（如2%, 4%, 6%, 8%, 10%）
   - `LAYER_WEIGHTS`/`SELL_WEIGHTS`：买入/卖出各层的资金分配比例
   - `SELL_STEP`：每层卖出触发价相对基准价的增幅（默认5%）

## 策略逻辑简述

- 当SPX从历史高点回撤达到设定阈值时，按比例分层买入TQQQ。
- 卖出时，分层设置不同的目标价，逐步止盈。
- 全过程自动化，支持多次数据下载重试，确保数据完整性。

## 参考

- [TQQQ 官方信息](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)
- [yfinance 文档](https://github.com/ranaroussi/yfinance)
- [pandas 文档](https://pandas.pydata.org/)

## 注意事项

- 本策略仅供学习与研究，实际投资请谨慎评估风险。
- 需科学上网以保证数据下载顺畅。

