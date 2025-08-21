import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def create_main_chart(equity_df, trades_df, summary):
    """创建主图 - 展示进出点位"""
    
    # 创建大尺寸图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
    
    # 获取数据范围
    start_date = equity_df.index[0]
    end_date = equity_df.index[-1]
    
    # 主图：价格和策略净值
    ax1.plot(equity_df.index, equity_df['tqqq_price'], 
             color='#1f77b4', alpha=0.7, linewidth=1, label='TQQQ价格')
    
    # 策略净值（对数刻度）
    ax1_twin = ax1.twinx()
    ax1_twin.plot(equity_df.index, equity_df['portfolio_value'], 
                  color='#ff7f0e', linewidth=2, label='策略净值')
    ax1_twin.set_yscale('log')
    
    # 分离买入和卖出交易
    buy_trades = trades_df[trades_df['type'] == 'buy']
    sell_trades = trades_df[trades_df['type'] == 'sell']
    
    # 绘制买入点
    if not buy_trades.empty:
        ax1.scatter(buy_trades['date'], buy_trades['price'], 
                   color='red', s=30, alpha=0.8, marker='^', 
                   label=f'买入 ({len(buy_trades)}次)', zorder=5)
    
    # 绘制卖出点
    if not sell_trades.empty:
        ax1.scatter(sell_trades['date'], sell_trades['price'], 
                   color='green', s=30, alpha=0.8, marker='v', 
                   label=f'卖出 ({len(sell_trades)}次)', zorder=5)
    
    # 设置主图格式
    ax1.set_title('Hybrid Adaptive ML策略 - 主图分析\nTQQQ价格走势与交易点位', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('TQQQ价格 ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 设置双Y轴
    ax1_twin.set_ylabel('策略净值 ($)', fontsize=12, color='#ff7f0e')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # 格式化X轴日期
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(3))
    
    # 子图：交易频率
    if not trades_df.empty:
        # 按月份统计交易次数
        trades_df['year_month'] = trades_df['date'].dt.to_period('M')
        monthly_trades = trades_df.groupby('year_month').size()
        monthly_trades.index = monthly_trades.index.astype(str).map(pd.to_datetime)
        
        ax2.bar(monthly_trades.index, monthly_trades.values, 
                color='#2ca02c', alpha=0.7, width=20)
        ax2.set_ylabel('月度交易次数', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加交易统计
        total_trades = len(trades_df)
        avg_monthly_trades = total_trades / len(monthly_trades)
        ax2.axhline(y=avg_monthly_trades, color='red', linestyle='--', alpha=0.7,
                   label=f'平均月交易: {avg_monthly_trades:.1f}次')
        ax2.legend(fontsize=10)
    
    # 格式化子图X轴
    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 添加策略信息
    info_text = f"""
策略信息:
• 初始资金: ${summary['initial_capital']:,.0f}
• 最终资产: ${summary['final_value']:,.0f}
• 总收益率: {summary['total_return_pct']:.2f}%
• 买入交易: {len(buy_trades)}次
• 卖出交易: {len(sell_trades)}次
• 总交易: {len(trades_df)}次
• 回测期间: {start_date.strftime('%Y-%m')} 至 {end_date.strftime('%Y-%m')}
    """
    
    # 在图表右上角添加信息框
    ax1.text(0.98, 0.98, info_text.strip(), transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = '/home/ubuntu/strategy_results/hybrid_ml_strategy_main_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"主图已保存: {output_path}")
    
    # 显示图表
    plt.show()

def main():
    """主函数"""
    print("=== 生成Hybrid Adaptive ML策略主图 ===")
    
    # 加载数据
    equity_df, trades_df, summary = load_data()
    
    if equity_df is None or trades_df is None or summary is None:
        print("数据加载失败，请检查文件路径")
        return
    
    # 创建主图
    create_main_chart(equity_df, trades_df, summary)
    
    print("=== 主图生成完成 ===")

if __name__ == "__main__":
    main()
