import yfinance as yf
import numpy as np
import pandas as pd
import time
from pandas_datareader import data as pdr
import os
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Config - Hybrid Adaptive ML Strategy
# -----------------------------
INITIAL_CAPITAL = 50000.0
# 机器学习阈值
ML_BUY_THRESHOLD = 0.6
ML_SELL_THRESHOLD = 0.4
# 趋势跟踪参数
TREND_MA_SHORT = 10
TREND_MA_LONG = 50
# 波动率目标
VOLATILITY_TARGET = 0.15
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
    # 兼容多种数据源列名/类型
    if isinstance(df, pd.Series):
        return df
    # 尝试常见列名（大小写不敏感）
    column_map = {str(c).lower(): c for c in df.columns}
    for key in [
        'adj close', 'adj_close', 'adjusted close', 'adjusted_close',
        'close', 'price'
    ]:
        if key in column_map:
            return df[column_map[key]]
    # 单列DataFrame时默认取该列
    if df.shape[1] == 1:
        return df.iloc[:, 0]
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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建机器学习特征"""
    features = pd.DataFrame(index=df.index)
    
    # 价格特征
    features['price'] = df['Close']
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 技术指标
    features['ma_5'] = df['Close'].rolling(5).mean()
    features['ma_10'] = df['Close'].rolling(10).mean()
    features['ma_20'] = df['Close'].rolling(20).mean()
    features['ma_50'] = df['Close'].rolling(50).mean()
    
    # 相对强弱指标
    features['rsi'] = calculate_rsi(df['Close'], 14)
    
    # 波动率
    features['volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # 动量指标
    features['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    features['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # 布林带
    bb_upper, bb_lower = calculate_bollinger_bands(df['Close'], 20, 2)
    features['bb_upper'] = bb_upper
    features['bb_lower'] = bb_lower
    features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # 成交量特征
    if 'Volume' in df.columns:
        features['volume_ma'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_ma']
    
    # 趋势特征
    features['trend_short'] = df['Close'] > features['ma_10']
    features['trend_medium'] = df['Close'] > features['ma_20']
    features['trend_long'] = df['Close'] > features['ma_50']
    
    return features


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple[pd.Series, pd.Series]:
    """计算布林带"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower


def train_ml_model(features: pd.DataFrame, target: pd.Series, train_size: float = 0.7) -> tuple[RandomForestRegressor, StandardScaler]:
    """训练机器学习模型"""
    # 准备数据
    features_clean = features.dropna()
    target_clean = target.loc[features_clean.index]
    
    # 确保数据对齐
    common_index = features_clean.index.intersection(target_clean.index)
    features_clean = features_clean.loc[common_index]
    target_clean = target_clean.loc[common_index]
    
    # 再次清理NaN值
    valid_mask = ~(features_clean.isna().any(axis=1) | target_clean.isna())
    features_clean = features_clean[valid_mask]
    target_clean = target_clean[valid_mask]
    
    if len(features_clean) < 100:
        print("警告：数据不足，使用默认模型")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        scaler = StandardScaler()
        return model, scaler
    
    # 分割训练测试集
    split_idx = int(len(features_clean) * train_size)
    X_train = features_clean.iloc[:split_idx]
    y_train = target_clean.iloc[:split_idx]
    X_test = features_clean.iloc[split_idx:]
    y_test = target_clean.iloc[split_idx:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    print(f"ML模型训练完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"训练集R²: {model.score(X_train_scaled, y_train):.3f}")
    
    # 安全地计算测试集R²
    try:
        test_score = model.score(X_test_scaled, y_test)
        print(f"测试集R²: {test_score:.3f}")
    except Exception as e:
        print(f"测试集R²计算失败: {e}")
    
    return model, scaler


def get_market_state(features: pd.DataFrame) -> str:
    """判断市场状态"""
    if len(features) < 20:
        return "Sideways"
    
    volatility = features['volatility'].iloc[-1]
    trend_strength = abs(features['momentum_20'].iloc[-1])
    
    if volatility > 0.03:  # 高波动率
        return "Volatile"
    elif trend_strength > 0.1:  # 强趋势
        return "Trending"
    else:
        return "Sideways"


# -----------------------------
# Main backtest: Hybrid Adaptive ML Strategy
# -----------------------------

spx_df, tqqq_df = download_market_data(START_DATE)

spx_close = _select_close_column(spx_df)
tqqq_close = _select_close_column(tqqq_df)

# 对齐日期（交集）
common_index = spx_close.index.intersection(tqqq_close.index)
spx_close = spx_close.loc[common_index]
tqqq_close = tqqq_close.loc[common_index]

# 创建特征
print("创建机器学习特征...")
features = create_features(tqqq_df)

# 创建目标变量（未来5日收益率）
target = tqqq_close.pct_change(5).shift(-5)

# 训练ML模型
print("训练机器学习模型...")
ml_model, scaler = train_ml_model(features, target)

# 回测状态
cash = INITIAL_CAPITAL
position_shares = 0.0
portfolio_values = []
entry_log = []
ml_predictions = []
market_states = []
signals = []

# 序列保存
dates_seq = []
price_seq = []
cash_seq = []
shares_seq = []

print(f"开始回测Hybrid Adaptive ML策略 初始资金: ${INITIAL_CAPITAL:,.2f}")
print(f"回测区间: {common_index[0].strftime('%Y-%m-%d')} 到 {common_index[-1].strftime('%Y-%m-%d')} (约10年)")
print(f"ML买入阈值: {ML_BUY_THRESHOLD}, ML卖出阈值: {ML_SELL_THRESHOLD}")

for i, date in enumerate(common_index):
    price = float(tqqq_close.loc[date])
    
    # 获取当前特征
    if i < 50:  # 需要足够的历史数据
        ml_prob = 0.5
        market_state = "Sideways"
        signal = 0
    else:
        try:
            current_features = features.loc[:date].iloc[-1:].dropna()
            if len(current_features) == 0:
                ml_prob = 0.5
                market_state = "Sideways"
                signal = 0
            else:
                # ML预测
                features_scaled = scaler.transform(current_features)
                ml_prob = ml_model.predict(features_scaled)[0]
                ml_prob = max(0.1, min(0.9, ml_prob + 0.5))  # 转换为0-1概率
                
                # 市场状态
                market_state = get_market_state(features.loc[:date])
                
                # 趋势信号
                try:
                    ma_short = features.loc[date, 'ma_10']
                    ma_long = features.loc[date, 'ma_50']
                    trend_signal = 1 if price > ma_short > ma_long else (-1 if price < ma_short < ma_long else 0)
                except:
                    trend_signal = 0
                
                # 综合信号
                if ml_prob > ML_BUY_THRESHOLD and trend_signal >= 0:
                    signal = 1  # 买入
                elif ml_prob < ML_SELL_THRESHOLD and trend_signal <= 0:
                    signal = -1  # 卖出
                else:
                    signal = 0  # 持有
        except Exception as e:
            print(f"特征处理错误: {e}")
            ml_prob = 0.5
            market_state = "Sideways"
            signal = 0
    
    # 执行交易
    if signal == 1 and cash > 0:  # 买入信号
        # 计算仓位大小（基于波动率目标）
        try:
            volatility = features.loc[date, 'volatility'] if date in features.index and 'volatility' in features.columns else 0.02
        except:
            volatility = 0.02
        position_size = min(cash * 0.95, cash * (VOLATILITY_TARGET / max(volatility, 0.01)))
        shares_to_buy = position_size / price
        position_shares += shares_to_buy
        cash -= position_size
        
        entry_log.append({
            'date': date,
            'type': 'buy',
            'price': price,
            'amount': position_size,
            'shares': shares_to_buy,
            'ml_prob': ml_prob,
            'market_state': market_state,
        })
        print(f"买入: {date.strftime('%Y-%m-%d')} @ ${price:.2f}, ML概率{ml_prob:.3f}, 市场状态{market_state}, 金额${position_size:,.2f}")
        
    elif signal == -1 and position_shares > 0:  # 卖出信号
        # 计算卖出数量（基于ML概率强度）
        sell_ratio = min(1.0, (ML_SELL_THRESHOLD - ml_prob) / ML_SELL_THRESHOLD + 0.5)
        shares_to_sell = position_shares * sell_ratio
        proceeds = shares_to_sell * price
        position_shares -= shares_to_sell
        cash += proceeds
        
        entry_log.append({
            'date': date,
            'type': 'sell',
            'price': price,
            'amount': proceeds,
            'shares': -shares_to_sell,
            'ml_prob': ml_prob,
            'market_state': market_state,
        })
        print(f"卖出: {date.strftime('%Y-%m-%d')} @ ${price:.2f}, ML概率{ml_prob:.3f}, 市场状态{market_state}, 回笼${proceeds:,.2f}")

    # 记录序列
    dates_seq.append(date)
    price_seq.append(price)
    cash_seq.append(cash)
    shares_seq.append(position_shares)
    portfolio_values.append(cash + position_shares * price)
    ml_predictions.append(ml_prob)
    market_states.append(market_state)
    signals.append(signal)

# 结果统计
final_value = portfolio_values[-1] if portfolio_values else INITIAL_CAPITAL
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

print("\n=== 回测结果（Hybrid Adaptive ML策略） ===")
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
    trades_csv = os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_trades.csv')
    trades_df.to_csv(trades_csv, index=False)
    print(f"交易明细已导出: {trades_csv}")

if portfolio_values:
    equity_df = pd.DataFrame({
        'date': pd.to_datetime(dates_seq),
        'tqqq_price': price_seq,
        'cash': cash_seq,
        'shares': shares_seq,
        'portfolio_value': portfolio_values,
        'ml_probability': ml_predictions,
        'market_state': market_states,
        'signal': signals,
    })
    equity_df.sort_values('date', inplace=True)
    equity_csv = os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_equity.csv')
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
    with open(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_summary.csv'), index=False)
    print("绩效摘要已导出: ")
    print(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_summary.json'))

    # 创建分析图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 投资组合价值
    ax1.plot(equity_series.index, equity_series.values, label='Portfolio Value', color='green', linewidth=2)
    ax1.set_title('Portfolio Value - Best Profitable Strategy')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 市场状态分布
    market_state_counts = pd.Series(market_states).value_counts()
    colors = ['red', 'teal', 'lightblue']
    ax2.pie(market_state_counts.values, labels=market_state_counts.index, autopct='%1.1f%%', colors=colors)
    ax2.set_title('Market State Distribution')
    
    # 3. 信号分布
    signal_counts = pd.Series(signals).value_counts().sort_index()
    signal_labels = {1: 'Buy', 0: 'Hold', -1: 'Sell'}
    signal_names = [signal_labels.get(s, str(s)) for s in signal_counts.index]
    colors = ['green', 'grey', 'red']
    ax3.bar(signal_names, signal_counts.values, color=colors)
    ax3.set_title('Signal Distribution')
    ax3.set_ylabel('Count')
    
    # 4. ML概率分布
    ml_probs = [p for p in ml_predictions if p > 0]
    ax4.hist(ml_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=ML_BUY_THRESHOLD, color='green', linestyle='--', label='Buy Threshold')
    ax4.axvline(x=ML_SELL_THRESHOLD, color='red', linestyle='--', label='Sell Threshold')
    ax4.set_title('ML Probability Distribution')
    ax4.set_xlabel('Probability')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"分析图表已保存: {plot_path}")

# 简单对比买入持有TQQQ（同期间起点）
if len(dates_seq) > 0:
    start_price = float(tqqq_close.loc[dates_seq[0]])
    end_price = float(tqqq_close.loc[dates_seq[-1]])
    buy_hold_return = (end_price / start_price) - 1.0
    print(f"\n买入持有TQQQ收益率: {buy_hold_return*100:.2f}%")
    print(f"策略超额收益: {(total_return - buy_hold_return)*100:.2f}%")
