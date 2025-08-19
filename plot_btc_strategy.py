import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
results_dir = os.path.expanduser("~/strategy_results")

# 读取净值曲线数据
equity_df = pd.read_csv(os.path.join(results_dir, 'hybrid_adaptive_ml_btc_equity.csv'))
equity_df['date'] = pd.to_datetime(equity_df['date'])
equity_df.set_index('date', inplace=True)

# 读取交易记录
trades_df = pd.read_csv(os.path.join(results_dir, 'hybrid_adaptive_ml_btc_trades.csv'))
trades_df['date'] = pd.to_datetime(trades_df['date'])

# 读取绩效摘要
with open(os.path.join(results_dir, 'hybrid_adaptive_ml_btc_summary.json'), 'r') as f:
    summary = json.load(f)

print("数据加载完成")
print(f"净值曲线数据: {len(equity_df)} 条记录")
print(f"交易记录: {len(trades_df)} 条记录")
print(f"策略总收益率: {summary['total_return_pct']:.2f}%")

# 创建图表
fig = plt.figure(figsize=(20, 16))

# 1. 投资组合价值 + 买卖点位标注
ax1 = plt.subplot(3, 2, 1)
equity_series = equity_df['portfolio_value']
ax1.plot(equity_series.index, equity_series.values, label='Portfolio Value', color='green', linewidth=2)

# 标注买卖点位
if len(trades_df) > 0:
    # 买入点位
    buy_trades = trades_df[trades_df['type'] == 'buy']
    if len(buy_trades) > 0:
        buy_dates = buy_trades['date']
        buy_values = [equity_series.loc[date] if date in equity_series.index else equity_series.iloc[-1] for date in buy_dates]
        ax1.scatter(buy_dates, buy_values, color='red', marker='^', s=50, alpha=0.7, label='Buy Signals', zorder=5)
        
        # 在买入点位旁边添加小饼图
        for i, (date, value, shares, amount) in enumerate(zip(buy_dates, buy_values, buy_trades['shares'], buy_trades['amount'])):
            if i % 20 == 0:  # 每20个点标注一次，避免过于密集
                # 计算现金和持仓比例
                cash_after = equity_df.loc[date, 'cash'] if date in equity_df.index else 0
                position_value = shares * equity_df.loc[date, 'price'] if date in equity_df.index else 0
                total = cash_after + position_value
                
                if total > 0:
                    cash_pct = cash_after / total
                    position_pct = position_value / total
                    
                    # 创建小饼图
                    pie_ax = fig.add_axes([0.1 + (i * 0.01) % 0.3, 0.7 + (i * 0.005) % 0.2, 0.06, 0.06])
                    pie_ax.pie([cash_pct, position_pct], colors=['lightgreen', 'red'], 
                              labels=[f'Cash\n{cash_pct:.1%}', f'Position\n{position_pct:.1%}'],
                              textprops={'fontsize': 4})
                    pie_ax.set_title(f'Buy\n{shares:.0f} shares\n${amount:,.0f}', fontsize=5)
    
    # 卖出点位
    sell_trades = trades_df[trades_df['type'] == 'sell']
    if len(sell_trades) > 0:
        sell_dates = sell_trades['date']
        sell_values = [equity_series.loc[date] if date in equity_series.index else equity_series.iloc[-1] for date in sell_dates]
        ax1.scatter(sell_dates, sell_values, color='blue', marker='v', s=50, alpha=0.7, label='Sell Signals', zorder=5)
        
        # 在卖出点位旁边添加小饼图
        for i, (date, value, shares, amount) in enumerate(zip(sell_dates, sell_values, sell_trades['shares'], sell_trades['amount'])):
            if i % 20 == 0:  # 每20个点标注一次
                # 计算现金和持仓比例
                cash_after = equity_df.loc[date, 'cash'] if date in equity_df.index else 0
                position_value = abs(shares) * equity_df.loc[date, 'price'] if date in equity_df.index else 0
                total = cash_after + position_value
                
                if total > 0:
                    cash_pct = cash_after / total
                    position_pct = position_value / total
                    
                    # 创建小饼图
                    pie_ax = fig.add_axes([0.1 + (i * 0.01) % 0.3, 0.1 + (i * 0.005) % 0.2, 0.06, 0.06])
                    pie_ax.pie([cash_pct, position_pct], colors=['lightgreen', 'red'], 
                              labels=[f'Cash\n{cash_pct:.1%}', f'Position\n{position_pct:.1%}'],
                              textprops={'fontsize': 4})
                    pie_ax.set_title(f'Sell\n{abs(shares):.0f} shares\n${amount:,.0f}', fontsize=5)

ax1.set_title('Bitcoin Strategy Portfolio Value with Allocation Pie Charts', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value ($)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2. 市场状态分布
ax2 = plt.subplot(3, 2, 2)
market_state_counts = equity_df['market_state'].value_counts()
colors = ['red', 'teal', 'lightblue']
ax2.pie(market_state_counts.values, labels=market_state_counts.index, autopct='%1.1f%%', colors=colors)
ax2.set_title('Market State Distribution', fontsize=14, fontweight='bold')

# 3. 信号分布
ax3 = plt.subplot(3, 2, 3)
signal_counts = equity_df['signal'].value_counts().sort_index()
signal_labels = {1: 'Buy', 0: 'Hold', -1: 'Sell'}
signal_names = [signal_labels.get(s, str(s)) for s in signal_counts.index]
colors = ['green', 'grey', 'red']
bars = ax3.bar(signal_names, signal_counts.values, color=colors)
ax3.set_title('Signal Distribution', fontsize=14, fontweight='bold')
ax3.set_ylabel('Count')

# 添加数值标签
for bar, count in zip(bars, signal_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{count:,}', ha='center', va='bottom', fontweight='bold')

# 4. ML概率分布 + 买卖阈值线
ax4 = plt.subplot(3, 2, 4)
ml_probs = equity_df['ml_prob'].values
ax4.hist(ml_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax4.axvline(x=0.51, color='green', linestyle='--', label='Buy Threshold (0.51)', linewidth=2)
ax4.axvline(x=0.49, color='red', linestyle='--', label='Sell Threshold (0.49)', linewidth=2)
ax4.set_title('ML Probability Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Probability')
ax4.set_ylabel('Frequency')
ax4.legend()

# 5. 比特币价格走势 + 买卖点位
ax5 = plt.subplot(3, 2, 5)
price_series = equity_df['price']
ax5.plot(price_series.index, price_series.values, label='Bitcoin Price', color='black', linewidth=1, alpha=0.7)

# 标注买卖点位
if len(trades_df) > 0:
    # 买入点位
    if len(buy_trades) > 0:
        buy_prices = buy_trades['price']
        ax5.scatter(buy_dates, buy_prices, color='red', marker='^', s=80, alpha=0.8, label='Buy Signals', zorder=5)
        
        # 在买入点位旁边添加小饼图
        for i, (date, price, shares, amount, ml_prob) in enumerate(zip(buy_dates, buy_prices, buy_trades['shares'], buy_trades['amount'], buy_trades['ml_prob'])):
            if i % 15 == 0:  # 每15个点标注一次
                # 计算现金和持仓比例
                cash_after = equity_df.loc[date, 'cash'] if date in equity_df.index else 0
                position_value = shares * price
                total = cash_after + position_value
                
                if total > 0:
                    cash_pct = cash_after / total
                    position_pct = position_value / total
                    
                    # 创建小饼图
                    pie_ax = fig.add_axes([0.6 + (i * 0.01) % 0.3, 0.7 + (i * 0.005) % 0.2, 0.06, 0.06])
                    pie_ax.pie([cash_pct, position_pct], colors=['lightgreen', 'red'], 
                              labels=[f'Cash\n{cash_pct:.1%}', f'Position\n{position_pct:.1%}'],
                              textprops={'fontsize': 4})
                    pie_ax.set_title(f'Buy @ ${price:.0f}\n{shares:.0f} shares\nML: {ml_prob:.3f}', fontsize=5)
    
    # 卖出点位
    if len(sell_trades) > 0:
        sell_prices = sell_trades['price']
        ax5.scatter(sell_dates, sell_prices, color='blue', marker='v', s=80, alpha=0.8, label='Sell Signals', zorder=5)
        
        # 在卖出点位旁边添加小饼图
        for i, (date, price, shares, amount, ml_prob) in enumerate(zip(sell_dates, sell_prices, sell_trades['shares'], sell_trades['amount'], sell_trades['ml_prob'])):
            if i % 15 == 0:  # 每15个点标注一次
                # 计算现金和持仓比例
                cash_after = equity_df.loc[date, 'cash'] if date in equity_df.index else 0
                position_value = abs(shares) * price
                total = cash_after + position_value
                
                if total > 0:
                    cash_pct = cash_after / total
                    position_pct = position_value / total
                    
                    # 创建小饼图
                    pie_ax = fig.add_axes([0.6 + (i * 0.01) % 0.3, 0.1 + (i * 0.005) % 0.2, 0.06, 0.06])
                    pie_ax.pie([cash_pct, position_pct], colors=['lightgreen', 'red'], 
                              labels=[f'Cash\n{cash_pct:.1%}', f'Position\n{position_pct:.1%}'],
                              textprops={'fontsize': 4})
                    pie_ax.set_title(f'Sell @ ${price:.0f}\n{abs(shares):.0f} shares\nML: {ml_prob:.3f}', fontsize=5)

ax5.set_title('Bitcoin Price with Trading Signals and Allocation Pie Charts', fontsize=14, fontweight='bold')
ax5.set_xlabel('Date')
ax5.set_ylabel('Bitcoin Price ($)')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.tick_params(axis='x', rotation=45)

# 6. 持仓变化
ax6 = plt.subplot(3, 2, 6)
shares_series = equity_df['shares']
ax6.plot(shares_series.index, shares_series.values, label='Position Shares', color='orange', linewidth=2)

# 标注买卖点位在持仓图上
if len(trades_df) > 0:
    # 买入点位
    if len(buy_trades) > 0:
        buy_shares = [shares_series.loc[date] if date in shares_series.index else shares_series.iloc[-1] for date in buy_dates]
        ax6.scatter(buy_dates, buy_shares, color='red', marker='^', s=50, alpha=0.8, label='Buy Signals', zorder=5)
    
    # 卖出点位
    if len(sell_trades) > 0:
        sell_shares = [shares_series.loc[date] if date in shares_series.index else shares_series.iloc[-1] for date in sell_dates]
        ax6.scatter(sell_dates, sell_shares, color='blue', marker='v', s=50, alpha=0.8, label='Sell Signals', zorder=5)

ax6.set_title('Position Shares Over Time', fontsize=14, fontweight='bold')
ax6.set_xlabel('Date')
ax6.set_ylabel('Number of Shares')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()

# 保存图表
plot_path = os.path.join(results_dir, 'bitcoin_strategy_comprehensive_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"综合分析图表已保存: {plot_path}")

# 创建第二个图表：ML概率和交易信号
fig2, (ax7, ax8) = plt.subplots(2, 1, figsize=(20, 12))

# 7. ML概率 + 买卖阈值
ml_prob_series = equity_df['ml_prob']
ax7.plot(ml_prob_series.index, ml_prob_series.values, label='ML Probability', color='purple', linewidth=1, alpha=0.7)
ax7.axhline(y=0.51, color='green', linestyle='--', label='Buy Threshold (0.51)', linewidth=2)
ax7.axhline(y=0.49, color='red', linestyle='--', label='Sell Threshold (0.49)', linewidth=2)

# 标注买卖点位在概率图上
if len(trades_df) > 0:
    # 买入点位
    if len(buy_trades) > 0:
        buy_probs = buy_trades['ml_prob']
        ax7.scatter(buy_dates, buy_probs, color='red', marker='^', s=60, alpha=0.8, label='Buy Signals', zorder=5)
    
    # 卖出点位
    if len(sell_trades) > 0:
        sell_probs = sell_trades['ml_prob']
        ax7.scatter(sell_dates, sell_probs, color='blue', marker='v', s=60, alpha=0.8, label='Sell Signals', zorder=5)

ax7.set_title('ML Probability with Trading Signals', fontsize=16, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('ML Probability')
ax7.grid(True, alpha=0.3)
ax7.legend()
ax7.tick_params(axis='x', rotation=45)

# 8. 持仓价值变化
position_values = shares_series * price_series
ax8.plot(position_values.index, position_values.values, label='Position Value', color='brown', linewidth=2)

# 标注买卖点位在持仓价值图上
if len(trades_df) > 0:
    # 买入点位
    if len(buy_trades) > 0:
        buy_pos_values = [position_values.loc[date] if date in position_values.index else position_values.iloc[-1] for date in buy_dates]
        ax8.scatter(buy_dates, buy_pos_values, color='red', marker='^', s=50, alpha=0.8, label='Buy Signals', zorder=5)
    
    # 卖出点位
    if len(sell_trades) > 0:
        sell_pos_values = [position_values.loc[date] if date in position_values.index else position_values.iloc[-1] for date in sell_dates]
        ax8.scatter(sell_dates, sell_pos_values, color='blue', marker='v', s=50, alpha=0.8, label='Sell Signals', zorder=5)

ax8.set_title('Position Value Over Time', fontsize=16, fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_ylabel('Position Value ($)')
ax8.grid(True, alpha=0.3)
ax8.legend()
ax8.tick_params(axis='x', rotation=45)

plt.tight_layout()

# 保存第二个图表
plot_path2 = os.path.join(results_dir, 'bitcoin_strategy_ml_and_position.png')
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"ML概率和持仓图表已保存: {plot_path2}")

# 创建第三个图表：绩效统计
fig3, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2, figsize=(16, 12))

# 9. 月度收益率分布
monthly_returns = equity_series.resample('M').last().pct_change().dropna()
ax9.hist(monthly_returns.values * 100, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax9.axvline(x=monthly_returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {monthly_returns.mean()*100:.2f}%', linewidth=2)
ax9.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
ax9.set_xlabel('Monthly Return (%)')
ax9.set_ylabel('Frequency')
ax9.legend()

# 10. 累积收益率
cumulative_returns = (equity_series / equity_series.iloc[0] - 1) * 100
ax10.plot(cumulative_returns.index, cumulative_returns.values, label='Strategy Returns', color='green', linewidth=2)
ax10.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
ax10.set_xlabel('Date')
ax10.set_ylabel('Cumulative Return (%)')
ax10.grid(True, alpha=0.3)
ax10.legend()
ax10.tick_params(axis='x', rotation=45)

# 11. 交易金额分布
if len(trades_df) > 0:
    trade_amounts = trades_df['amount'].abs()
    ax11.hist(trade_amounts.values, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax11.axvline(x=trade_amounts.mean(), color='red', linestyle='--', label=f'Mean: ${trade_amounts.mean():,.0f}', linewidth=2)
    ax11.set_title('Trade Amount Distribution', fontsize=14, fontweight='bold')
    ax11.set_xlabel('Trade Amount ($)')
    ax11.set_ylabel('Frequency')
    ax11.legend()

# 12. 策略表现总结
ax12.axis('off')
summary_text = f"""
Bitcoin Strategy Performance Summary

Initial Capital: ${summary['initial_capital']:,.2f}
Final Value: ${summary['final_value']:,.2f}
Total Return: {summary['total_return_pct']:.2f}%

Total Trades: {summary['total_trades']:,}
Buy Trades: {summary['buy_trades']:,}
Sell Trades: {summary['sell_trades']:,}

Remaining Cash: ${summary['remaining_cash']:,.2f}
Final Shares: {summary['final_shares']:.0f}

Strategy: {summary['strategy']}
Target Asset: {summary['target']}
"""
ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=12, 
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.tight_layout()

# 保存第三个图表
plot_path3 = os.path.join(results_dir, 'bitcoin_strategy_performance_stats.png')
plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
print(f"绩效统计图表已保存: {plot_path3}")

print("\n所有图表已生成完成！")
print(f"1. 综合分析图表: {plot_path}")
print(f"2. ML概率和持仓图表: {plot_path2}")
print(f"3. 绩效统计图表: {plot_path3}")

