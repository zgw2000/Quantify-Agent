import yfinance as yf
import numpy as np
import pandas as pd
import time
from pandas_datareader import data as pdr
import os
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer

# -----------------------------
# Config - Hybrid Adaptive ML Strategy (Optimized)
# -----------------------------
INITIAL_CAPITAL = 50000.0
# 机器学习阈值 - 超激进优化
ML_BUY_THRESHOLD = 0.52  # 进一步降低买入阈值
ML_SELL_THRESHOLD = 0.48  # 进一步提高卖出阈值
# 趋势跟踪参数 - 超敏感
TREND_MA_SHORT = 3  # 极短MA，超敏感
TREND_MA_LONG = 10  # 短MA，快速响应
# 波动率目标 - 超激进
VOLATILITY_TARGET = 0.35  # 大幅提高波动率目标
# 从2011年开始
START_DATE = "2011-01-01"
SPX_TICKER = "^GSPC"
TARGET_TICKER = "TQQQ"
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 10

# 风险与再平衡约束（避免无止境加仓）
MAX_EXPOSURE = 1.0            # 组合最大目标敞口（仓位/净值）
REBALANCE_TOLERANCE = 0.03    # 目标敞口偏离超过此阈值才行动（减少过度交易）
MAX_TRADE_PORTFOLIO_PCT = 0.25 # 单日最大交易额不超过净值的比例
TRANSACTION_COST_BPS = 5      # 单边交易成本（基点）
SIGMOID_TEMP = 0.75           # 信号映射温度参数，越小越激进

# 新增优化参数
MOMENTUM_LOOKBACK = 5  # 动量回看期
RSI_OVERSOLD = 30  # RSI超卖阈值
RSI_OVERBOUGHT = 70  # RSI超买阈值
VOLUME_THRESHOLD = 1.5  # 成交量阈值

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
    """创建机器学习特征 - 优化版本"""
    features = pd.DataFrame(index=df.index)
    
    # 基础价格特征
    features['price'] = df['Close']
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 多周期移动平均
    for window in [3, 5, 10, 20, 50]:
        features[f'ma_{window}'] = df['Close'].rolling(window).mean()
        features[f'ma_ratio_{window}'] = df['Close'] / features[f'ma_{window}']
    
    # 高级技术指标
    features['rsi'] = calculate_rsi(df['Close'], 14)
    features['rsi_5'] = calculate_rsi(df['Close'], 5)  # 短期RSI
    features['rsi_21'] = calculate_rsi(df['Close'], 21)  # 长期RSI
    
    # 多周期波动率
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = df['Close'].pct_change().rolling(window).std()
    
    # 多周期动量
    for window in [1, 3, 5, 10, 20]:
        features[f'momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
    
    # 布林带
    bb_upper, bb_lower = calculate_bollinger_bands(df['Close'], 20, 2)
    features['bb_upper'] = bb_upper
    features['bb_lower'] = bb_lower
    features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    features['bb_width'] = (bb_upper - bb_lower) / features['ma_20']
    
    # 成交量特征
    if 'Volume' in df.columns:
        features['volume_ma'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_ma']
        features['volume_sma'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    
    # 趋势强度指标
    features['trend_strength'] = abs(features['ma_5'] - features['ma_20']) / features['ma_20']
    features['trend_direction'] = np.where(features['ma_5'] > features['ma_20'], 1, -1)
    
    # 价格位置指标
    features['price_position'] = (df['Close'] - df['Close'].rolling(50).min()) / (df['Close'].rolling(50).max() - df['Close'].rolling(50).min())
    
    # 波动率比率
    features['vol_ratio_short'] = features['volatility_5'] / features['volatility_20']
    features['vol_ratio_long'] = features['volatility_20'] / features['volatility_50']
    
    # 动量确认
    features['momentum_confirmation'] = (features['momentum_5'] > 0) & (features['momentum_10'] > 0)
    features['momentum_divergence'] = features['momentum_5'] - features['momentum_20']
    
    # 价格加速度
    features['price_acceleration'] = features['returns'].diff()
    
    # 成交量价格关系
    if 'Volume' in df.columns:
        features['volume_price_trend'] = (df['Volume'] * df['Close']).rolling(10).mean()
    
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


def train_ml_model(features: pd.DataFrame, target: pd.Series, train_size: float = 0.7) -> tuple[object, StandardScaler]:
    """训练机器学习模型（时间序列CV + 多模型择优）"""
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
    
    # 自定义评分：结合相关性与方向一致性，更贴近收益目标
    def corr_sign_score(y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        corr = 0.0 if np.isnan(corr) else corr
        sign_acc = np.mean((y_true > 0) == (y_pred > 0))
        # 将方向准确率映射到 [-1,1] 再加权
        signed_acc = 2.0 * sign_acc - 1.0
        return 0.7 * corr + 0.3 * signed_acc

    scorer = make_scorer(corr_sign_score, greater_is_better=True)

    # 时间序列交叉验证（仅在训练集上调参，避免窥视测试集）
    tscv = TimeSeriesSplit(n_splits=5)

    candidates = [
        (
            'RandomForest',
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': [200, 400, 600],
                'max_depth': [None, 8, 12, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', None]
            },
        ),
        (
            'ExtraTrees',
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                'n_estimators': [300, 600, 900],
                'max_depth': [None, 8, 12, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', None]
            },
        ),
        (
            'GBRT',
            GradientBoostingRegressor(random_state=42),
            {
                'n_estimators': [200, 400, 800],
                'learning_rate': [0.01, 0.03, 0.1],
                'max_depth': [2, 3, 4],
                'subsample': [0.6, 0.8, 1.0]
            },
        ),
    ]

    best_model = None
    best_name = ''
    best_cv_score = -1e9

    for name, est, grid in candidates:
        try:
            search = RandomizedSearchCV(
                estimator=est,
                param_distributions=grid,
                n_iter=10,
                cv=tscv,
                random_state=42,
                n_jobs=-1,
                verbose=0,
                scoring=scorer,
                refit=True,
            )
            search.fit(X_train_scaled, y_train)
            if search.best_score_ > best_cv_score:
                best_cv_score = search.best_score_
                best_model = search.best_estimator_
                best_name = name
        except Exception as e:
            print(f"模型 {name} 搜索失败: {e}")

    if best_model is None:
        print("模型搜索失败，回退到默认RandomForest。")
        best_model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        best_model.fit(X_train_scaled, y_train)

    print(f"ML模型训练完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"CV最佳模型: {best_name}, CV得分: {best_cv_score:.4f}")

    # 计算测试集上的综合评分
    try:
        y_pred_test = best_model.predict(X_test_scaled)
        test_score = corr_sign_score(y_test, y_pred_test)
        print(f"测试集综合评分(相关性+方向): {test_score:.4f}")
    except Exception as e:
        print(f"测试集评分计算失败: {e}")
    
    return best_model, scaler


def get_market_state(features: pd.DataFrame) -> str:
    """判断市场状态"""
    if len(features) < 20:
        return "Sideways"
    
    try:
        volatility = features['volatility_20'].iloc[-1]
        trend_strength = abs(features['momentum_20'].iloc[-1])
        
        if volatility > 0.03:  # 高波动率
            return "Volatile"
        elif trend_strength > 0.1:  # 强趋势
            return "Trending"
        else:
            return "Sideways"
    except:
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
target_exposures = []
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
    # 每次迭代初始化目标敞口，避免变量遗留
    target_exposure = 0.0
    
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
                # ML预测（预测未来5日收益率）
                features_scaled = scaler.transform(current_features)
                ml_pred5 = float(ml_model.predict(features_scaled)[0])  # 预测的5日收益
                ml_pred5 = float(np.clip(ml_pred5, -0.30, 0.30))  # 裁剪极端值

                # 市场状态
                market_state = get_market_state(features.loc[:date])
                
                # 趋势信号
                try:
                    ma_short = features.loc[date, 'ma_10']
                    ma_long = features.loc[date, 'ma_50']
                    trend_signal = 1 if price > ma_short > ma_long else (-1 if price < ma_short < ma_long else 0)
                except:
                    trend_signal = 0

                # 风险调整信号→目标敞口（0-1）
                try:
                    vol20 = float(features.loc[:date]['volatility_20'].iloc[-1])
                except Exception:
                    vol20 = 0.02
                vol20 = 0.02 if np.isnan(vol20) or vol20 <= 0 else vol20
                risk_adj_signal = ml_pred5 / (vol20 * np.sqrt(5))
                risk_adj_signal = float(np.clip(risk_adj_signal, -5.0, 5.0))

                # 概率映射到[0,1]，再转换目标敞口
                ml_prob = 1.0 / (1.0 + np.exp(-risk_adj_signal / SIGMOID_TEMP))

                # 目标敞口基础值
                target_exposure = ml_prob
                # 趋势加减成分
                if trend_signal < 0:
                    target_exposure *= 0.75
                elif trend_signal > 0:
                    target_exposure *= 1.10
                # 全局上限
                target_exposure = float(np.clip(target_exposure, 0.0, MAX_EXPOSURE))
                # 记录一个简化信号用于统计
                if target_exposure - 0.5 > REBALANCE_TOLERANCE:
                    signal = 1
                elif 0.5 - target_exposure > REBALANCE_TOLERANCE:
                    signal = -1
                else:
                    signal = 0
        except Exception as e:
            print(f"特征处理错误: {e}")
            ml_prob = 0.5
            market_state = "Sideways"
            signal = 0
    
    # 执行交易（基于目标敞口的再平衡，避免无止境加仓）
    portfolio_value_now = cash + position_shares * price
    position_value_now = position_shares * price
    current_exposure = (position_value_now / portfolio_value_now) if portfolio_value_now > 0 else 0.0

    # 起始阶段已在迭代开头初始化

    desired_position_value = target_exposure * portfolio_value_now
    delta_value = desired_position_value - position_value_now

    # 单日最大交易额限制
    max_trade_value_today = MAX_TRADE_PORTFOLIO_PCT * portfolio_value_now

    # 只有当偏离超过容差才调整
    if delta_value > portfolio_value_now * REBALANCE_TOLERANCE and cash > 0:
        trade_value = min(delta_value, cash, max_trade_value_today)
        if trade_value > 0:
            shares_to_buy = trade_value / price
            position_shares += shares_to_buy
            # 交易成本
            fee = trade_value * TRANSACTION_COST_BPS / 10000.0
            cash -= (trade_value + fee)
            entry_log.append({
                'date': date,
                'type': 'buy',
                'price': price,
                'amount': -trade_value,
                'shares': shares_to_buy,
                'ml_prob': ml_prob,
                'market_state': market_state,
                'cash_after': cash,
                'shares_after': position_shares,
                'portfolio_after': cash + position_shares * price,
                'target_exposure': target_exposure,
                'fee': fee,
            })
            print(f"买入(再平衡): {date.strftime('%Y-%m-%d')} @ ${price:.2f}, 目标敞口{target_exposure:.2f}, 金额${trade_value:,.2f}")
    elif delta_value < -portfolio_value_now * REBALANCE_TOLERANCE and position_shares > 0:
        trade_value = min(-delta_value, position_value_now, max_trade_value_today)
        if trade_value > 0:
            shares_to_sell = trade_value / price
            proceeds = trade_value
            fee = proceeds * TRANSACTION_COST_BPS / 10000.0
            position_shares -= shares_to_sell
            cash += (proceeds - fee)
            entry_log.append({
                'date': date,
                'type': 'sell',
                'price': price,
                'amount': proceeds,
                'shares': -shares_to_sell,
                'ml_prob': ml_prob,
                'market_state': market_state,
                'cash_after': cash,
                'shares_after': position_shares,
                'portfolio_after': cash + position_shares * price,
                'target_exposure': target_exposure,
                'fee': fee,
            })
            print(f"卖出(再平衡): {date.strftime('%Y-%m-%d')} @ ${price:.2f}, 目标敞口{target_exposure:.2f}, 回笼${proceeds:,.2f}")

    # 记录序列
    dates_seq.append(date)
    price_seq.append(price)
    cash_seq.append(cash)
    shares_seq.append(position_shares)
    portfolio_values.append(cash + position_shares * price)
    ml_predictions.append(ml_prob)
    target_exposures.append(target_exposure)
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
        'position_value': np.array(shares_seq) * np.array(price_seq),
        'portfolio_value': portfolio_values,
        'exposure': np.divide(np.array(shares_seq) * np.array(price_seq), np.array(portfolio_values), out=np.zeros_like(np.array(portfolio_values), dtype=float), where=(np.array(portfolio_values) != 0)),
        'ml_probability': ml_predictions,
        'target_exposure': target_exposures,
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
