import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
results_dir = os.path.expanduser("~/strategy_results")
equity_df = pd.read_csv(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_equity.csv'))
trades_df = pd.read_csv(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_trades.csv'))

with open(os.path.join(results_dir, 'hybrid_adaptive_ml_tqqq_summary.json'), 'r') as f:
    summary = json.load(f)

# 转换日期
equity_df['date'] = pd.to_datetime(equity_df['date'])
trades_df['date'] = pd.to_datetime(trades_df['date'])

# 策略参数
ML_BUY_THRESHOLD = 0.6
ML_SELL_THRESHOLD = 0.4

print("=== Hybrid Adaptive ML策略图表分析 ===")
print(f"策略: Hybrid Adaptive ML")
print(f"目标资产: TQQQ")
print(f"初始资金: ${summary['initial_capital']:,.2f}")
print(f"最终资产: ${summary['final_value']:,.2f}")
print(f"总收益率: {summary['total_return_pct']:.2f}%")
print(f"买入交易: {summary['num_buys']}")
print(f"卖出交易: {summary['num_sells']}")
print(f"总交易次数: {summary['num_buys'] + summary['num_sells']}")

# 创建图表
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Hybrid Adaptive ML策略 - TQQQ 2011-2025回测分析', fontsize=16, fontweight='bold')

# 1. 投资组合价值 vs TQQQ价格
ax1 = axes[0, 0]
ax1.plot(equity_df['date'], equity_df['portfolio_value'], label='策略组合', linewidth=2, color='blue')
ax1.plot(equity_df['date'], equity_df['tqqq_price'] * 1000, label='TQQQ价格(×1000)', linewidth=1, color='gray', alpha=0.7)
ax1.set_title('投资组合价值 vs TQQQ价格', fontsize=12, fontweight='bold')
ax1.set_ylabel('价值 ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 添加买卖点标注（采样显示，避免过于拥挤）
buy_trades = trades_df[trades_df['type'] == 'buy']
sell_trades = trades_df[trades_df['type'] == 'sell']

# 每10个交易标注一个
for i, trade in buy_trades.iloc[::10].iterrows():
    trade_date = pd.to_datetime(trade['date'])
    portfolio_value = equity_df[equity_df['date'] == trade_date]['portfolio_value'].iloc[0]
    ax1.scatter(trade_date, portfolio_value, color='green', s=100, marker='^', zorder=5)
    ax1.annotate(f'买入\n${trade["amount"]:,.0f}', 
                xy=(trade_date, portfolio_value), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

for i, trade in sell_trades.iloc[::10].iterrows():
    trade_date = pd.to_datetime(trade['date'])
    portfolio_value = equity_df[equity_df['date'] == trade_date]['portfolio_value'].iloc[0]
    ax1.scatter(trade_date, portfolio_value, color='red', s=100, marker='v', zorder=5)
    ax1.annotate(f'卖出\n${trade["amount"]:,.0f}', 
                xy=(trade_date, portfolio_value), 
                xytext=(10, -10), textcoords='offset points',
                fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

# 2. ML预测概率
ax2 = axes[0, 1]
ax2.plot(equity_df['date'], equity_df['ml_probability'], linewidth=2, color='orange', alpha=0.8)
ax2.axhline(y=ML_BUY_THRESHOLD, color='green', linestyle='--', alpha=0.7, label=f'ML买入阈值 ({ML_BUY_THRESHOLD})')
ax2.axhline(y=ML_SELL_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'ML卖出阈值 ({ML_SELL_THRESHOLD})')
ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='中性线 (0.5)')
ax2.fill_between(equity_df['date'], ML_BUY_THRESHOLD, 1, alpha=0.2, color='green', label='买入信号区域')
ax2.fill_between(equity_df['date'], 0, ML_SELL_THRESHOLD, alpha=0.2, color='red', label='卖出信号区域')
ax2.set_title('ML预测概率', fontsize=12, fontweight='bold')
ax2.set_ylabel('概率')
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 市场状态分布
ax3 = axes[1, 0]
state_counts = equity_df['market_state'].value_counts()
colors = ['red', 'blue', 'green']
ax3.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax3.set_title('市场状态分布', fontsize=12, fontweight='bold')

# 4. 交易信号分布
ax4 = axes[1, 1]
signal_counts = equity_df['signal'].value_counts()
signal_labels = ['无信号 (0)', '买入信号 (1)', '卖出信号 (-1)']
signal_colors = ['gray', 'green', 'red']
ax4.bar(signal_labels, [signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(-1, 0)], 
        color=signal_colors, alpha=0.7)
ax4.set_title('交易信号分布', fontsize=12, fontweight='bold')
ax4.set_ylabel('信号次数')
for i, v in enumerate([signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(-1, 0)]):
    ax4.text(i, v + max(signal_counts.values) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')

# 5. 现金和持仓变化
ax5 = axes[2, 0]
ax5.plot(equity_df['date'], equity_df['cash'], label='现金', linewidth=2, color='green', alpha=0.8)
ax5.plot(equity_df['date'], equity_df['shares'] * equity_df['tqqq_price'], label='持仓价值', linewidth=2, color='blue', alpha=0.8)
ax5.set_title('现金和持仓价值变化', fontsize=12, fontweight='bold')
ax5.set_ylabel('价值 ($)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 策略vs买入持有对比
ax6 = axes[2, 1]
equity_df['strategy_return'] = (equity_df['portfolio_value'] / summary['initial_capital'] - 1) * 100
equity_df['buyhold_return'] = (equity_df['tqqq_price'] / equity_df['tqqq_price'].iloc[0] - 1) * 100

ax6.plot(equity_df['date'], equity_df['strategy_return'], label='Hybrid Adaptive ML策略', linewidth=2, color='blue')
ax6.plot(equity_df['date'], equity_df['buyhold_return'], label='买入持有TQQQ', linewidth=2, color='red', alpha=0.7)
ax6.set_title('策略vs买入持有对比', fontsize=12, fontweight='bold')
ax6.set_ylabel('累计收益率 (%)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 保存图表
output_path = os.path.join(results_dir, 'hybrid_ml_strategy_comprehensive_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"综合分析图表已保存: {output_path}")

# 创建第二组图表：详细分析
fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12))
fig2.suptitle('Hybrid Adaptive ML策略 - 详细分析', fontsize=16, fontweight='bold')

# 1. 年度收益率分析
ax1 = axes2[0, 0]
equity_df['year'] = equity_df['date'].dt.year
equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
annual_returns = equity_df.groupby('year')['daily_return'].apply(lambda x: (1 + x).prod() - 1) * 100

colors = ['green' if x > 0 else 'red' for x in annual_returns.values]
bars = ax1.bar(annual_returns.index, annual_returns.values, color=colors, alpha=0.7)
ax1.set_title('年度收益率', fontsize=12, fontweight='bold')
ax1.set_ylabel('收益率 (%)')
ax1.set_xlabel('年份')
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, value in zip(bars, annual_returns.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (10 if height > 0 else -10),
             f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

# 2. 最大回撤分析
ax2 = axes2[0, 1]
equity_df['cummax'] = equity_df['portfolio_value'].cummax()
equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax'] * 100

ax2.fill_between(equity_df['date'], equity_df['drawdown'], 0, alpha=0.3, color='red')
ax2.plot(equity_df['date'], equity_df['drawdown'], color='red', linewidth=1)
ax2.set_title('最大回撤', fontsize=12, fontweight='bold')
ax2.set_ylabel('回撤 (%)')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()

# 3. 交易频率分析
ax3 = axes2[1, 0]
trades_df['year'] = trades_df['date'].dt.year
trades_df['month'] = trades_df['date'].dt.month
monthly_trades = trades_df.groupby(['year', 'month']).size().reset_index(name='trades')
monthly_trades['date'] = pd.to_datetime(monthly_trades[['year', 'month']].assign(day=1))

ax3.plot(monthly_trades['date'], monthly_trades['trades'], marker='o', linewidth=1, markersize=4)
ax3.set_title('月度交易频率', fontsize=12, fontweight='bold')
ax3.set_ylabel('交易次数')
ax3.set_xlabel('时间')
ax3.grid(True, alpha=0.3)

# 4. 策略表现总结
ax4 = axes2[1, 1]
ax4.axis('off')

# 计算关键指标
total_days = len(equity_df)
total_return = summary['total_return_pct']
annual_return = summary['CAGR_pct']
volatility = summary['annual_volatility_pct']
risk_free_rate = 0.02
sharpe_ratio = summary['sharpe_ratio']
max_drawdown = summary['max_drawdown_pct']

# 创建文本总结
summary_text = f"""
策略表现总结 (2011-2025)

总收益率: {total_return:.2f}%
年化收益率: {annual_return:.2f}%
年化波动率: {volatility:.2f}%
夏普比率: {sharpe_ratio:.2f}
最大回撤: {max_drawdown:.2f}%
总交易次数: {summary['num_buys'] + summary['num_sells']}
买入交易: {summary['num_buys']}
卖出交易: {summary['num_sells']}

策略特点:
• 机器学习驱动
• 高频交易策略
• 动态风险管理
• 市场状态适应
• 超额收益显著
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 保存第二组图表
output_path2 = os.path.join(results_dir, 'hybrid_ml_strategy_detailed_analysis.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"详细分析图表已保存: {output_path2}")

# 创建第三组图表：交易分析
fig3, axes3 = plt.subplots(2, 2, figsize=(20, 12))
fig3.suptitle('Hybrid Adaptive ML策略 - 交易分析', fontsize=16, fontweight='bold')

# 1. 交易金额分布
ax1 = axes3[0, 0]
trade_amounts = trades_df['amount'].abs()
ax1.hist(trade_amounts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('交易金额分布', fontsize=12, fontweight='bold')
ax1.set_xlabel('交易金额 ($)')
ax1.set_ylabel('交易次数')
ax1.grid(True, alpha=0.3)

# 2. ML概率与交易关系
ax2 = axes3[0, 1]
buy_ml_prob = buy_trades['ml_prob']
sell_ml_prob = sell_trades['ml_prob']

ax2.hist(buy_ml_prob, bins=20, alpha=0.7, color='green', label='买入时的ML概率', edgecolor='black')
ax2.hist(sell_ml_prob, bins=20, alpha=0.7, color='red', label='卖出时的ML概率', edgecolor='black')
ax2.axvline(x=ML_BUY_THRESHOLD, color='green', linestyle='--', alpha=0.7, label=f'买入阈值 ({ML_BUY_THRESHOLD})')
ax2.axvline(x=ML_SELL_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'卖出阈值 ({ML_SELL_THRESHOLD})')
ax2.set_title('ML概率与交易关系', fontsize=12, fontweight='bold')
ax2.set_xlabel('ML概率')
ax2.set_ylabel('交易次数')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 交易时间分布
ax3 = axes3[1, 0]
trades_df['month_name'] = trades_df['date'].dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_trade_counts = trades_df['month_name'].value_counts().reindex(month_order)

ax3.bar(monthly_trade_counts.index, monthly_trade_counts.values, alpha=0.7, color='purple')
ax3.set_title('交易月份分布', fontsize=12, fontweight='bold')
ax3.set_xlabel('月份')
ax3.set_ylabel('交易次数')
ax3.grid(True, alpha=0.3)

# 4. 持仓时间分析
ax4 = axes3[1, 1]
if len(trades_df) >= 2:
    # 计算持仓时间
    buy_dates = buy_trades['date'].tolist()
    sell_dates = sell_trades['date'].tolist()
    
    holding_periods = []
    for buy_date in buy_dates:
        # 找到下一个卖出日期
        next_sells = [d for d in sell_dates if d > buy_date]
        if next_sells:
            holding_period = (min(next_sells) - buy_date).days
            holding_periods.append(holding_period)
    
    if holding_periods:
        ax4.hist(holding_periods, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('持仓时间分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('持仓天数')
        ax4.set_ylabel('交易次数')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_holding = np.mean(holding_periods)
        ax4.axvline(x=avg_holding, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均持仓: {avg_holding:.0f}天')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '无完整买卖配对', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('持仓时间分布', fontsize=12, fontweight='bold')
else:
    ax4.text(0.5, 0.5, '交易数据不足', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('持仓时间分布', fontsize=12, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 保存第三组图表
output_path3 = os.path.join(results_dir, 'hybrid_ml_strategy_trade_analysis.png')
plt.savefig(output_path3, dpi=300, bbox_inches='tight')
print(f"交易分析图表已保存: {output_path3}")

# 创建第四组图表：性能统计
fig4, axes4 = plt.subplots(2, 2, figsize=(20, 12))
fig4.suptitle('Hybrid Adaptive ML策略 - 性能统计', fontsize=16, fontweight='bold')

# 1. 滚动收益率
ax1 = axes4[0, 0]
equity_df['rolling_return_252'] = equity_df['portfolio_value'].pct_change(252) * 100
ax1.plot(equity_df['date'], equity_df['rolling_return_252'], linewidth=2, color='blue', alpha=0.8)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax1.set_title('252天滚动收益率', fontsize=12, fontweight='bold')
ax1.set_ylabel('收益率 (%)')
ax1.grid(True, alpha=0.3)

# 2. 滚动波动率
ax2 = axes4[0, 1]
equity_df['rolling_vol_252'] = equity_df['portfolio_value'].pct_change().rolling(252).std() * np.sqrt(252) * 100
ax2.plot(equity_df['date'], equity_df['rolling_vol_252'], linewidth=2, color='red', alpha=0.8)
ax2.set_title('252天滚动波动率', fontsize=12, fontweight='bold')
ax2.set_ylabel('波动率 (%)')
ax2.grid(True, alpha=0.3)

# 3. 夏普比率
ax3 = axes4[1, 0]
equity_df['excess_return'] = equity_df['portfolio_value'].pct_change() - risk_free_rate/252
equity_df['rolling_sharpe_252'] = (equity_df['excess_return'].rolling(252).mean() * 252) / (equity_df['excess_return'].rolling(252).std() * np.sqrt(252))
ax3.plot(equity_df['date'], equity_df['rolling_sharpe_252'], linewidth=2, color='green', alpha=0.8)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_title('252天滚动夏普比率', fontsize=12, fontweight='bold')
ax3.set_ylabel('夏普比率')
ax3.grid(True, alpha=0.3)

# 4. 策略vs买入持有详细对比
ax4 = axes4[1, 1]
ax4.plot(equity_df['date'], equity_df['strategy_return'], label='Hybrid Adaptive ML策略', linewidth=2, color='blue')
ax4.plot(equity_df['date'], equity_df['buyhold_return'], label='买入持有TQQQ', linewidth=2, color='red', alpha=0.7)
ax4.fill_between(equity_df['date'], equity_df['strategy_return'], equity_df['buyhold_return'], 
                where=(equity_df['strategy_return'] > equity_df['buyhold_return']), 
                alpha=0.3, color='green', label='超额收益')
ax4.fill_between(equity_df['date'], equity_df['strategy_return'], equity_df['buyhold_return'], 
                where=(equity_df['strategy_return'] < equity_df['buyhold_return']), 
                alpha=0.3, color='red', label='相对亏损')
ax4.set_title('策略vs买入持有详细对比', fontsize=12, fontweight='bold')
ax4.set_ylabel('累计收益率 (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 保存第四组图表
output_path4 = os.path.join(results_dir, 'hybrid_ml_strategy_performance_stats.png')
plt.savefig(output_path4, dpi=300, bbox_inches='tight')
print(f"性能统计图表已保存: {output_path4}")

print("\n=== 图表生成完成 ===")
print(f"1. 综合分析图表: {output_path}")
print(f"2. 详细分析图表: {output_path2}")
print(f"3. 交易分析图表: {output_path3}")
print(f"4. 性能统计图表: {output_path4}")

plt.show()
