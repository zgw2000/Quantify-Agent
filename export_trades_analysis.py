import pandas as pd
import numpy as np
from datetime import datetime
import json

def load_data():
    """加载策略数据"""
    try:
        # 加载净值曲线
        equity_df = pd.read_csv('/home/ubuntu/strategy_results/hybrid_adaptive_ml_tqqq_equity.csv')
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # 加载交易记录
        trades_df = pd.read_csv('/home/ubuntu/strategy_results/hybrid_adaptive_ml_tqqq_trades.csv')
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # 加载摘要信息
        with open('/home/ubuntu/strategy_results/hybrid_adaptive_ml_tqqq_summary.json', 'r') as f:
            summary = json.load(f)
            
        return equity_df, trades_df, summary
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None

def create_detailed_trades_analysis(trades_df, equity_df):
    """创建详细的交易分析"""
    
    # 复制交易数据
    analysis_df = trades_df.copy()
    
    # 添加交易序号
    analysis_df['trade_id'] = range(1, len(analysis_df) + 1)
    
    # 计算每笔交易的收益率
    analysis_df['trade_return'] = 0.0
    analysis_df['cumulative_return'] = 0.0
    
    # 计算买入后的收益率
    buy_trades = analysis_df[analysis_df['type'] == 'buy'].copy()
    sell_trades = analysis_df[analysis_df['type'] == 'sell'].copy()
    
    # 为每笔买入找到对应的卖出
    for i, buy_trade in buy_trades.iterrows():
        # 找到该买入后的第一笔卖出
        subsequent_sells = sell_trades[sell_trades['date'] > buy_trade['date']]
        if not subsequent_sells.empty:
            sell_trade = subsequent_sells.iloc[0]
            buy_price = buy_trade['price']
            sell_price = sell_trade['price']
            trade_return = (sell_price - buy_price) / buy_price
            analysis_df.loc[i, 'trade_return'] = trade_return
    
    # 计算累计收益率
    analysis_df['cumulative_return'] = analysis_df['trade_return'].cumsum()
    
    # 添加交易金额统计
    analysis_df['trade_value'] = analysis_df['amount']
    analysis_df['cumulative_trade_value'] = analysis_df['trade_value'].cumsum()
    
    # 添加市场状态统计
    analysis_df['market_state_count'] = analysis_df.groupby('market_state').cumcount() + 1
    
    # 添加ML概率区间
    analysis_df['ml_prob_range'] = pd.cut(analysis_df['ml_prob'], 
                                         bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                                         labels=['极低(0-0.3)', '低(0.3-0.4)', '中低(0.4-0.5)', 
                                                '中高(0.5-0.6)', '高(0.6-0.7)', '极高(0.7-1.0)'])
    
    # 重新排列列顺序
    column_order = ['trade_id', 'date', 'type', 'price', 'amount', 'shares', 
                   'ml_prob', 'ml_prob_range', 'market_state', 'market_state_count',
                   'trade_return', 'cumulative_return', 'trade_value', 'cumulative_trade_value']
    
    analysis_df = analysis_df[column_order]
    
    return analysis_df

def create_trade_summary(trades_df, analysis_df):
    """创建交易汇总统计"""
    
    # 基础统计
    total_trades = len(trades_df)
    buy_trades = len(trades_df[trades_df['type'] == 'buy'])
    sell_trades = len(trades_df[trades_df['type'] == 'sell'])
    
    # 交易金额统计
    total_buy_amount = trades_df[trades_df['type'] == 'buy']['amount'].sum()
    total_sell_amount = trades_df[trades_df['type'] == 'sell']['amount'].sum()
    
    # 平均交易金额
    avg_buy_amount = trades_df[trades_df['type'] == 'buy']['amount'].mean()
    avg_sell_amount = trades_df[trades_df['type'] == 'sell']['amount'].mean()
    
    # ML概率统计
    avg_ml_prob_buy = trades_df[trades_df['type'] == 'buy']['ml_prob'].mean()
    avg_ml_prob_sell = trades_df[trades_df['type'] == 'sell']['ml_prob'].mean()
    
    # 市场状态统计
    market_state_stats = trades_df['market_state'].value_counts()
    
    # 收益率统计
    profitable_trades = len(analysis_df[analysis_df['trade_return'] > 0])
    loss_trades = len(analysis_df[analysis_df['trade_return'] < 0])
    win_rate = profitable_trades / len(analysis_df[analysis_df['trade_return'] != 0]) * 100 if len(analysis_df[analysis_df['trade_return'] != 0]) > 0 else 0
    
    # 创建汇总DataFrame
    summary_data = {
        '指标': [
            '总交易次数', '买入次数', '卖出次数', '买入/卖出比例',
            '总买入金额', '总卖出金额', '净资金流',
            '平均买入金额', '平均卖出金额',
            '平均买入ML概率', '平均卖出ML概率',
            '盈利交易次数', '亏损交易次数', '胜率(%)',
            '最大单笔收益', '最大单笔亏损', '平均交易收益'
        ],
        '数值': [
            total_trades, buy_trades, sell_trades, f"{buy_trades/sell_trades:.2f}" if sell_trades > 0 else "N/A",
            f"${total_buy_amount:,.2f}", f"${total_sell_amount:,.2f}", f"${total_sell_amount - total_buy_amount:,.2f}",
            f"${avg_buy_amount:,.2f}", f"${avg_sell_amount:,.2f}",
            f"{avg_ml_prob_buy:.3f}", f"{avg_ml_prob_sell:.3f}",
            profitable_trades, loss_trades, f"{win_rate:.2f}",
            f"{analysis_df['trade_return'].max()*100:.2f}%" if len(analysis_df) > 0 else "N/A",
            f"{analysis_df['trade_return'].min()*100:.2f}%" if len(analysis_df) > 0 else "N/A",
            f"{analysis_df['trade_return'].mean()*100:.2f}%" if len(analysis_df) > 0 else "N/A"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # 市场状态统计
    market_state_df = market_state_stats.reset_index()
    market_state_df.columns = ['市场状态', '交易次数']
    market_state_df['占比(%)'] = (market_state_df['交易次数'] / total_trades * 100).round(2)
    
    return summary_df, market_state_df

def create_monthly_analysis(trades_df):
    """创建月度交易分析"""
    
    # 按月份统计
    trades_df['year_month'] = trades_df['date'].dt.to_period('M')
    monthly_stats = trades_df.groupby('year_month').agg({
        'type': 'count',
        'amount': 'sum',
        'ml_prob': 'mean',
        'market_state': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    monthly_stats.columns = ['年月', '交易次数', '交易金额', '平均ML概率', '主要市场状态']
    monthly_stats['年月'] = monthly_stats['年月'].astype(str)
    
    # 分离买入和卖出
    monthly_buy = trades_df[trades_df['type'] == 'buy'].groupby('year_month').agg({
        'type': 'count',
        'amount': 'sum'
    }).reset_index()
    monthly_buy.columns = ['年月', '买入次数', '买入金额']
    
    monthly_sell = trades_df[trades_df['type'] == 'sell'].groupby('year_month').agg({
        'type': 'count',
        'amount': 'sum'
    }).reset_index()
    monthly_sell.columns = ['年月', '卖出次数', '卖出金额']
    
    # 确保年月列的数据类型一致
    monthly_stats['年月'] = monthly_stats['年月'].astype(str)
    monthly_buy['年月'] = monthly_buy['年月'].astype(str)
    monthly_sell['年月'] = monthly_sell['年月'].astype(str)
    
    # 合并数据
    monthly_analysis = monthly_stats.merge(monthly_buy, on='年月', how='left')
    monthly_analysis = monthly_analysis.merge(monthly_sell, on='年月', how='left')
    
    # 填充NaN值
    monthly_analysis = monthly_analysis.fillna(0)
    
    return monthly_analysis

def export_all_analysis():
    """导出所有分析结果"""
    
    print("=== 导出交易分析CSV文件 ===")
    
    # 加载数据
    equity_df, trades_df, summary = load_data()
    
    if equity_df is None or trades_df is None or summary is None:
        print("数据加载失败")
        return
    
    # 创建详细交易分析
    detailed_analysis = create_detailed_trades_analysis(trades_df, equity_df)
    
    # 创建交易汇总
    trade_summary, market_state_summary = create_trade_summary(trades_df, detailed_analysis)
    
    # 创建月度分析
    monthly_analysis = create_monthly_analysis(trades_df)
    
    # 导出文件
    base_path = '/home/ubuntu/strategy_results/'
    
    # 1. 详细交易记录
    detailed_analysis.to_csv(f'{base_path}detailed_trades_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 详细交易分析已导出: {base_path}detailed_trades_analysis.csv")
    
    # 2. 交易汇总统计
    trade_summary.to_csv(f'{base_path}trade_summary_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 交易汇总统计已导出: {base_path}trade_summary_statistics.csv")
    
    # 3. 市场状态统计
    market_state_summary.to_csv(f'{base_path}market_state_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 市场状态统计已导出: {base_path}market_state_statistics.csv")
    
    # 4. 月度交易分析
    monthly_analysis.to_csv(f'{base_path}monthly_trade_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 月度交易分析已导出: {base_path}monthly_trade_analysis.csv")
    
    # 5. 原始交易记录（重新格式化）
    formatted_trades = trades_df.copy()
    formatted_trades['date'] = formatted_trades['date'].dt.strftime('%Y-%m-%d')
    formatted_trades['amount'] = formatted_trades['amount'].round(2)
    formatted_trades['shares'] = formatted_trades['shares'].round(4)
    formatted_trades['ml_prob'] = formatted_trades['ml_prob'].round(4)
    formatted_trades.to_csv(f'{base_path}formatted_trades.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 格式化交易记录已导出: {base_path}formatted_trades.csv")
    
    # 6. 策略性能摘要
    performance_summary = pd.DataFrame({
        '指标': ['初始资金', '最终资产', '总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数', '胜率'],
        '数值': [
            f"${summary['initial_capital']:,.2f}",
            f"${summary['final_value']:,.2f}",
            f"{summary['total_return_pct']:.2f}%",
            f"{summary['CAGR_pct']:.2f}%",
            f"{summary['max_drawdown_pct']:.2f}%",
            f"{summary['sharpe_ratio']:.2f}",
            len(trades_df),
            f"{len(detailed_analysis[detailed_analysis['trade_return'] > 0]) / len(detailed_analysis[detailed_analysis['trade_return'] != 0]) * 100:.2f}%" if len(detailed_analysis[detailed_analysis['trade_return'] != 0]) > 0 else "N/A"
        ]
    })
    performance_summary.to_csv(f'{base_path}strategy_performance_summary.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 策略性能摘要已导出: {base_path}strategy_performance_summary.csv")
    
    print("\n=== 所有CSV文件导出完成 ===")
    print(f"共导出6个CSV文件到: {base_path}")
    
    # 显示文件大小
    import os
    for filename in ['detailed_trades_analysis.csv', 'trade_summary_statistics.csv', 
                    'market_state_statistics.csv', 'monthly_trade_analysis.csv', 
                    'formatted_trades.csv', 'strategy_performance_summary.csv']:
        filepath = f'{base_path}{filename}'
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  {filename}: {size:,} bytes")

if __name__ == "__main__":
    export_all_analysis()
