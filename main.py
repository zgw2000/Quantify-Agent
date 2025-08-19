import yfinance as yf
import numpy as np
import pandas as pd
import time
from pandas_datareader import data as pdr
import os
import matplotlib.pyplot as plt
import json

# -----------------------------
# Config
# -----------------------------
INITIAL_CAPITAL = 50000.0
LAYER_THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10]  # SPX从ATH回撤阈值（2%,4%,6%,8%,10%）
# 买入分层：1:2:4:8:16（归一化）
_BUY_RATIOS = [1, 2, 4, 8, 16]
_BUY_SUM = sum(_BUY_RATIOS)
LAYER_WEIGHTS = [r / _BUY_SUM for r in _BUY_RATIOS]
# 卖出分层：16:8:4:2:1（归一化），每层触发价较基准价+5%
_SELL_RATIOS = [16, 8, 4, 2, 1]
_SELL_SUM = sum(_SELL_RATIOS)
SELL_WEIGHTS = [r / _SELL_SUM for r in _SELL_RATIOS]
SELL_STEP = 0.05
# 动态近10年
START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
SPX_TICKER = "^GSPC"
TARGET_TICKER = "TQQQ"
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 10

# -----------------------------
# Helpers
# -----------------------------

def _select_close_column(df: pd.DataFrame) -> pd.Series:
    if 'Adj Close' in df.columns:
        return df['Adj Close']
    if 'Close' in df.columns:
        return df['Close']
    raise ValueError("DataFrame missing Close/Adj Close column")


def download_market_data(start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    下载SPX与TQQQ的日线数据，优先Yahoo，失败则Stooq，最后模拟数据。
    返回: (spx_df, tqqq_df)
    """
    for attempt in range(MAX_RETRIES):
        try:
            print(f"正在下载 {SPX_TICKER}, {TARGET_TICKER} 数据... (尝试 {attempt + 1}/{MAX_RETRIES})")
            data = yf.download([SPX_TICKER, TARGET_TICKER], start=start_date, group_by='ticker')
            if isinstance(data.columns, pd.MultiIndex):
                spx_df = data[SPX_TICKER].copy()
                tqqq_df = data[TARGET_TICKER].copy()
            else:
                spx_df = yf.download(SPX_TICKER, start=start_date)
                tqqq_df = yf.download(TARGET_TICKER, start=start_date)

            if spx_df.empty or tqqq_df.empty:
                raise ValueError("下载结果为空")

            return spx_df, tqqq_df
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < MAX_RETRIES - 1:
                print(f"请求被限制，等待 {RETRY_SLEEP_SEC} 秒后重试...")
                time.sleep(RETRY_SLEEP_SEC)
                continue
            # Yahoo失败后尝试Stooq
            print(f"Yahoo失败: {e}. 尝试使用Stooq...")
            try:
                # 先尝试SPY/TQQQ
                spx_df = pdr.DataReader('SPY', 'stooq', start=start_date)
                tqqq_df = pdr.DataReader('TQQQ', 'stooq', start=start_date)
                spx_df = spx_df.sort_index()
                tqqq_df = tqqq_df.sort_index()
                print("使用Stooq数据源成功(SPY/TQQQ)。")
                return spx_df, tqqq_df
            except Exception:
                try:
                    # 再尝试SPY.US/TQQQ.US
                    spx_df = pdr.DataReader('SPY.US', 'stooq', start=start_date)
                    tqqq_df = pdr.DataReader('TQQQ.US', 'stooq', start=start_date)
                    spx_df = spx_df.sort_index()
                    tqqq_df = tqqq_df.sort_index()
                    print("使用Stooq数据源成功(SPY.US/TQQQ.US)。")
                    return spx_df, tqqq_df
                except Exception as e2:
                    print(f"Stooq失败: {e2}")
                    print("使用模拟数据进行测试...")
                    dates = pd.date_range(start_date, pd.Timestamp.today().strftime('%Y-%m-%d'), freq='B')
                    np.random.seed(42)
                    # 模拟SPX指数
                    spx_returns = np.random.normal(0.0003, 0.01, len(dates))
                    spx_prices = [2000.0]
                    for r in spx_returns[1:]:
                        spx_prices.append(max(spx_prices[-1] * (1 + r), 500))
                    spx_df = pd.DataFrame({
                        'Open': spx_prices,
                        'High': np.array(spx_prices) * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))) ,
                        'Low':  np.array(spx_prices) * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))) ,
                        'Close': spx_prices,
                        'Adj Close': spx_prices,
                        'Volume': np.random.randint(1_000_000_000, 2_000_000_000, len(dates))
                    }, index=dates)
                    # 模拟TQQQ与SPX相关（约3倍波动+噪声）
                    tqqq_returns = 3.0 * spx_returns + np.random.normal(0, 0.01, len(dates))
                    tqqq_prices = [50.0]
                    for r in tqqq_returns[1:]:
                        tqqq_prices.append(max(tqqq_prices[-1] * (1 + r), 5))
                    tqqq_df = pd.DataFrame({
                        'Open': tqqq_prices,
                        'High': np.array(tqqq_prices) * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))) ,
                        'Low':  np.array(tqqq_prices) * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))) ,
                        'Close': tqqq_prices,
                        'Adj Close': tqqq_prices,
                        'Volume': np.random.randint(10_000_000, 50_000_000, len(dates))
                    }, index=dates)
                    return spx_df, tqqq_df


def compute_drawdown_from_ath(close_series: pd.Series) -> pd.Series:
    rolling_ath = close_series.cummax()
    drawdown = (close_series / rolling_ath) - 1.0
    return drawdown


# -----------------------------
# Main backtest: 连续加仓 + 分层卖出
# -----------------------------

spx_df, tqqq_df = download_market_data(START_DATE)

spx_close = _select_close_column(spx_df)
tqqq_close = _select_close_column(tqqq_df)

# 对齐日期（交集）
common_index = spx_close.index.intersection(tqqq_close.index)
spx_close = spx_close.loc[common_index]
tqqq_close = tqqq_close.loc[common_index]

# 计算SPX从ATH的回撤
spx_drawdown = compute_drawdown_from_ath(spx_close)

# 回测状态
cash = INITIAL_CAPITAL
position_shares = 0.0
avg_cost = 0.0
used_layers = 0            # 当前一轮回撤中已使用的买入层数（0~5）
portfolio_values = []
entry_log = []

# 卖出状态（反向马丁）
sell_base_cost = None      # 进入卖出序列时的基准均价
sell_initial_shares = 0.0
sell_level_idx = 0         # 已经触发到第几层卖出（0..4）

# 序列保存
dates_seq = []
price_seq = []
spx_dd_seq = []
cash_seq = []
shares_seq = []

print(f"开始回测（连续加仓 + 16:8:4:2:1 分层卖出） 初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"回测区间: {common_index[0].strftime('%Y-%m-%d')} 到 {common_index[-1].strftime('%Y-%m-%d')} (约10年)")
print(f"买入阈值: {LAYER_THRESHOLDS}，权重: {LAYER_WEIGHTS}")
print(f"卖出步长: +{int(SELL_STEP*100)}% × 5 层，权重: {SELL_WEIGHTS}")

for date in common_index:
    price = float(tqqq_close.loc[date])
    dd = float(spx_drawdown.loc[date])

    # 1) 连续加仓逻辑：当回撤加深触发更多层；回撤恢复到>-2%后重置used_layers允许下一轮再次分层买入
    if dd > -LAYER_THRESHOLDS[0]:
        used_layers = 0  # 解除上一轮，准备下一轮

    should_layers = sum(1 for th in LAYER_THRESHOLDS if dd <= -th)
    while used_layers < should_layers and used_layers < len(LAYER_THRESHOLDS):
        # 新买入将重置卖出序列
        sell_base_cost = None
        sell_initial_shares = 0.0
        sell_level_idx = 0

        layer_idx = used_layers
        layer_cash = min(cash, cash * LAYER_WEIGHTS[layer_idx])
        if layer_cash <= 0:
            break
        shares = layer_cash / price
        # 更新均价
        new_total_cost = avg_cost * position_shares + layer_cash
        position_shares += shares
        avg_cost = new_total_cost / position_shares if position_shares > 0 else 0.0
        cash -= layer_cash
        used_layers += 1
        entry_log.append({
            'date': date,
            'type': 'buy',
            'layer': layer_idx + 1,
            'dd': dd,
            'price': price,
            'amount': layer_cash,
            'shares': shares,
        })
        print(f"买入 L{layer_idx + 1}: {date.strftime('%Y-%m-%d')} @ ${price:.2f}, 回撤{dd*100:.2f}%, 金额${layer_cash:,.2f}, 持仓{position_shares:.2f}股, 均价${avg_cost:.2f}")

    # 2) 进入/执行分层卖出：当价格高于均价5%开始，按16:8:4:2:1分5层卖出
    if position_shares > 0:
        if sell_base_cost is None and price >= avg_cost * (1 + SELL_STEP):
            sell_base_cost = avg_cost
            sell_initial_shares = position_shares
            sell_level_idx = 0
        # 逐层检查触发
        while sell_base_cost is not None and sell_level_idx < 5 and price >= sell_base_cost * (1 + SELL_STEP * (sell_level_idx + 1)) and position_shares > 0:
            planned_shares = sell_initial_shares * SELL_WEIGHTS[sell_level_idx]
            sell_shares = min(planned_shares, position_shares)
            proceeds = sell_shares * price
            position_shares -= sell_shares
            cash += proceeds
            entry_log.append({
                'date': date,
                'type': 'sell',
                'layer': sell_level_idx + 1,
                'dd': dd,
                'price': price,
                'amount': proceeds,
                'shares': -sell_shares,
            })
            print(f"卖出 S{sell_level_idx + 1}: {date.strftime('%Y-%m-%d')} @ ${price:.2f}, +{SELL_STEP*100*(sell_level_idx+1):.0f}%, 卖出{sell_shares:.2f}股, 回笼${proceeds:,.2f}")
            sell_level_idx += 1
        # 卖完5层或仓位为0，结束卖出序列
        if sell_level_idx >= 5 or position_shares <= 1e-8:
            sell_base_cost = None
            sell_initial_shares = 0.0
            sell_level_idx = 0
            # 卖完后均价不可用，重置
            if position_shares <= 1e-8:
                position_shares = 0.0
                avg_cost = 0.0

    # 记录序列
    dates_seq.append(date)
    price_seq.append(price)
    spx_dd_seq.append(dd)
    cash_seq.append(cash)
    shares_seq.append(position_shares)
    portfolio_values.append(cash + position_shares * price)

# 结果统计
final_value = portfolio_values[-1] if portfolio_values else INITIAL_CAPITAL
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

print("\n=== 回测结果（连续加仓 + 分层卖出） ===")
print(f"初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"最终资产: ${final_value:,.2f}")
print(f"总收益率: {total_return*100:.2f}%")
print(f"剩余现金: ${cash:,.2f}")
print(f"持仓股数: {position_shares:.2f}")

# 导出CSV与图
results_dir = os.path.expanduser("~/strategy_results")
os.makedirs(results_dir, exist_ok=True)

if entry_log:
    trades_df = pd.DataFrame(entry_log)
    trades_df.sort_values('date', inplace=True)
    trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
    trades_csv = os.path.join(results_dir, 'spx_dd_tqqq_trades.csv')
    trades_df.to_csv(trades_csv, index=False)
    print(f"交易明细已导出: {trades_csv}")

if portfolio_values:
    equity_df = pd.DataFrame({
        'date': pd.to_datetime(dates_seq),
        'tqqq_price': price_seq,
        'spx_drawdown': spx_dd_seq,
        'cash': cash_seq,
        'shares': shares_seq,
        'portfolio_value': portfolio_values,
    })
    equity_df.sort_values('date', inplace=True)
    equity_csv = os.path.join(results_dir, 'spx_dd_tqqq_equity.csv')
    equity_df.to_csv(equity_csv, index=False, date_format='%Y-%m-%d')
    print(f"净值曲线已导出: {equity_csv}")

    # 绩效指标
    equity_series = equity_df.set_index('date')['portfolio_value']
    daily_returns = equity_series.pct_change().dropna()
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else 0
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    vol_annual = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    rolling_max = equity_series.cummax()
    portfolio_dd = equity_series / rolling_max - 1.0
    max_drawdown = portfolio_dd.min()
    num_buys = sum(1 for t in entry_log if t['type'] == 'buy')
    num_sells = sum(1 for t in entry_log if t['type'] == 'sell')

    summary = {
        'start_date': equity_series.index[0].strftime('%Y-%m-%d'),
        'end_date': equity_series.index[-1].strftime('%Y-%m-%d'),
        'initial_capital': INITIAL_CAPITAL,
        'final_value': float(equity_series.iloc[-1]),
        'total_return_pct': total_return * 100,
        'CAGR_pct': cagr * 100,
        'annual_volatility_pct': vol_annual * 100,
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': max_drawdown * 100,
        'num_buys': num_buys,
        'num_sells': num_sells,
    }
    # 保存JSON & CSV
    with open(os.path.join(results_dir, 'spx_dd_tqqq_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, 'spx_dd_tqqq_summary.csv'), index=False)
    print("绩效摘要已导出: ")
    print(os.path.join(results_dir, 'spx_dd_tqqq_summary.json'))

    spx_dd_series = pd.Series(spx_dd_seq, index=equity_series.index)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    equity_norm = equity_series / equity_series.iloc[0]
    ax1.plot(equity_norm.index, equity_norm.values, label='Portfolio (norm)', color='tab:blue')
    ax1.set_ylabel('Equity (normalized)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2.plot(portfolio_dd.index, portfolio_dd.values, label='Portfolio DD', color='tab:red')
    ax2.plot(spx_dd_series.index, spx_dd_series.values, label='SPX DD', color='tab:orange', alpha=0.7)
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'spx_dd_tqqq_equity_drawdown.png')
    plt.savefig(plot_path, dpi=150)
    print(f"图表已保存: {plot_path}")

# 简单对比买入持有TQQQ（同期间起点）
if len(dates_seq) > 0:
    start_price = float(tqqq_close.loc[dates_seq[0]])
    end_price = float(tqqq_close.loc[dates_seq[-1]])
    buy_hold_return = (end_price / start_price) - 1.0
    print(f"\n买入持有TQQQ收益率: {buy_hold_return*100:.2f}%")
