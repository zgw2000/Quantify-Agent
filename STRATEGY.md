### Hybrid Adaptive ML 策略说明（当前实现）

本策略在 `main.py` 中实现，为杠杆 ETF（默认 `TQQQ`）提供基于机器学习的自适应策略。训练阶段使用时间序列交叉验证并在多模型之间自动择优，交易阶段采用目标敞口再平衡、交易容差与单日换手限制，以在收益与交易成本/噪音之间取得平衡。配套绘图脚本 `plot_cash_position_curve.py` 支持生成现金/持仓/净值、敞口、收益、以及“比例+净值”叠加图。

---

### 数据与范围
- **标的**:
  - 训练与信号：使用目标标的（默认 `TQQQ`）的行情数据
  - 基准指数：下载 `SPX`（`^GSPC`）以备扩展分析（当前回测逻辑未直接用作信号）
- **数据源**：优先 Yahoo Finance；失败回退至 Stooq（`SPY`/`TQQQ` 或 `SPY.US`/`TQQQ.US`）；均失败则模拟数据（用于离线调试）
- **起始日期**：由 `main.py` 中的 `START_DATE` 控制（默认 `2011-01-01`）

---

### 特征工程（`create_features`）
对价格序列构建多类特征（以 `Close` 为核心）：
- 基础：价格、普通/对数收益
- 均线与相对价位：MA 3/5/10/20/50、价/均线比
- RSI：14、短期 5、长期 21
- 波动率：5/10/20/50 日滚动标准差
- 动量：1/3/5/10/20 日
- 布林带：上/下轨、带宽、带内相对位置
- 成交量：20 日均量、量比、量均短/长比、`(Volume*Price)` 的10日均
- 趋势强度与方向：`abs(MA5-MA20)/MA20`、`MA5>MA20`
- 价格区间位置：相对过去50日高低点的位置
- 波动率比率：短/中、 中/长 比
- 动量一致与背离：`momentum_5>0 && momentum_10>0`、`momentum_5 - momentum_20`

---

### 目标变量
- 未来 5 个交易日的累计收益率：`pct_change(5).shift(-5)`
- 训练前进行对齐与缺失值清洗

---

### 模型训练（`train_ml_model`）
- 标准化：`StandardScaler`
- 仅使用训练集做调参，避免窥视：`TimeSeriesSplit(n_splits=5)`
- 候选模型与随机搜索：
  - RandomForestRegressor（含 n_estimators/max_depth 等搜索）
  - ExtraTreesRegressor（更强的随机特征/阈值）
  - GradientBoostingRegressor（弱学习器提升）
- 自定义评分（更贴近收益目标）：相关性 × 0.7 + 方向一致性 × 0.3
  - 相关性：`corr(y_true, y_pred)`
  - 方向一致性：`mean(sign(y_true)==sign(y_pred))` 映射到 [-1,1]
- 选择 CV 得分最佳的模型，并报告测试集上的综合评分

---

### 信号与目标敞口（Target Exposure）
对每个交易日：
1. 使用模型预测未来 5 日收益 `ml_pred5`（裁剪至 [-0.30, 0.30]）
2. 风险归一：`risk_adj = ml_pred5 / (vol20 * sqrt(5))`（为防缺失/极值设定下限）
3. Sigmoid 映射到 [0,1]：`ml_prob = 1 / (1 + exp(-risk_adj / SIGMOID_TEMP))`
4. 趋势微调（参考 MA10/MA50 与现价）：
   - 趋势下行：目标敞口 × 0.75
   - 趋势上行：目标敞口 × 1.10
5. 约束：`target_exposure ∈ [0, MAX_EXPOSURE]`

注：文件顶部存在 `ML_BUY_THRESHOLD / ML_SELL_THRESHOLD` 用于分析图上的参考线，当前执行层未采用硬阈值开关仓位（避免过度离散化）。

---

### 交易执行与风控（再平衡）
- 再平衡依据：将当前仓位价值向 `target_exposure × portfolio_value` 靠拢
- 仅在偏离超过 `REBALANCE_TOLERANCE` 时执行，降低噪音交易
- 单日最大换手：`MAX_TRADE_PORTFOLIO_PCT × portfolio_value`
- 交易成本：`TRANSACTION_COST_BPS`（单边基点）
- 关键参数（`main.py` 顶部可配置）：
  - `INITIAL_CAPITAL = 50000`
  - `MAX_EXPOSURE = 1.0`
  - `REBALANCE_TOLERANCE = 0.03`
  - `MAX_TRADE_PORTFOLIO_PCT = 0.25`
  - `TRANSACTION_COST_BPS = 5`
  - `SIGMOID_TEMP = 0.75`

---

### 回测输出（默认输出目录：`~/strategy_results`）
- 交易明细：`hybrid_adaptive_ml_tqqq_trades.csv`
- 净值曲线与序列：`hybrid_adaptive_ml_tqqq_equity.csv`
  - 包含：`date, tqqq_price, cash, shares, position_value, portfolio_value, exposure, ml_probability, target_exposure, market_state, signal`
- 摘要：`hybrid_adaptive_ml_tqqq_summary.(json|csv)`
  - 指标：起止日期、终值、总收益、CAGR、年化波动、Sharpe、最大回撤、买/卖次数
- 分析图：`hybrid_adaptive_ml_tqqq_analysis.png`

---

### 绘图与可视化（`plot_cash_position_curve.py`）
- 读取上面的 `equity.csv`，生成：
  - 现金/持仓/净值曲线：`cash_position_portfolio_curve_{ticker}.png`
  - 月度堆叠 + 净值线：`cash_position_monthly_curve_{ticker}.png`
  - 年度现金/持仓柱 + 净值线：`cash_position_yearly_bars_{ticker}.png`
  - 实际/目标敞口：`exposure_curve_{ticker}.png`
  - 价格-敞口叠加：`price_exposure_overlay_{ticker}.png`
  - 累计/日收益曲线：`returns_curve_{ticker}.png`
  - 现金/持仓比例 + 组合净值（双轴叠加）：`cash_position_ratio_with_portfolio_{ticker}.png`

#### 绘图用法
```bash
source .venv/bin/activate
python plot_cash_position_curve.py --ticker TQQQ \
  --resample W \
  --smooth 10 \
  --log-portfolio
```
- `--ticker`：标的（默认 TQQQ）
- `--resample`：D/W/M 重采样以降噪（默认 W）
- `--smooth`：滑动均值平滑窗口（默认 10，0 关闭）
- `--log-portfolio`：右轴净值使用对数刻度（便于长周期观察）

---

### 运行与环境
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 回测（main.py 当前不接受命令行参数，直接运行）
python main.py

# 生成 TQQQ 图表
python plot_cash_position_curve.py --ticker TQQQ
```

---

### 参数调优建议
- 降低交易频率：增大 `REBALANCE_TOLERANCE`、减小 `MAX_TRADE_PORTFOLIO_PCT`、在绘图层增加 `--resample`/`--smooth` 观察信号节奏
- 信号稳定性：增大 `SIGMOID_TEMP`（降低灵敏度）
- 训练稳健性：保留时间序列交叉验证，避免随机 K 折导致时间泄漏
- 稳健性检验：
  - 参数扰动（±10%~20%）后绩效不应大幅恶化
  - 标签置换测试应显著劣化
  - 交易成本/滑点敏感性分析

---

### 扩展与注意事项
- 支持更换目标标的：修改 `main.py` 顶部 `TARGET_TICKER` 常量即可（或在后续版本加入命令行参数）
- 若要以 `SPX` 作为训练与信号、但执行在 `TQQQ` 上，可在特征与目标计算处改为 `spx_df`/`spx_close`（此前我们做过探索版）
- 重要风险提示：回测为历史模拟，不构成投资建议；杠杆 ETF 具有路径依赖与波动衰减风险

---

### 文件关系与产出一览
- 回测：`main.py` → 生成 `~\strategy_results` 下的 CSV/JSON/PNG
- 可视化：`plot_cash_position_curve.py` → 读取 `equity.csv` 输出多张图
- 依赖：见 `requirements.txt`

---

### 常见问题
- 为什么回测收益很高？
  - 杠杆产品在长牛阶段的复利效应显著；且本策略为“回测阶段的最优模型”并使用 TSCV 与综合评分做了抑制过拟合的处理，但仍需样本外检验与成本敏感性评估。
- 为什么 `main.py` 忽略命令行 `--ticker`？
  - 当前版本未启用该参数。可直接修改文件顶部的 `TARGET_TICKER`。


