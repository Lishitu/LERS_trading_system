"""策略全流程执行与基准对比主入口。

运行：
D:\\anaconda3\\python.exe src/run_momentum_backtest.py
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_moomoo import MoomooDataLoader
from strategy_momentum import MomentumRegimeStrategy
from portfolio_allocator import PortfolioAllocator
from trading_costs import TradingCostCalculator
from backtest_vwap import DailyVWAPBacktester
from feature_engineering_ts import TimeSeriesFeatureExtractor
from meta_labeling import TripleBarrierLabeler
from ml_winrate_model import WalkForwardWinRatePipeline


def run_pipeline(
    start_date: str = "2016-01-01",
    top_n: int = 3,
    exit_rank: int = 4,
    max_cap: float = 0.35,
    buffer: float = 0.05,
    initial_capital: float = 100000.0,
    save_plot: bool = True,
    use_ml: bool = False,
    min_win_rate: float = 0.50,
):
    print("=" * 60)
    mode_str = f" [STRATEGY] 结构性行业/主题 ETF 截面动量 + 进攻杠杆轮动策略 ({'ML Meta-Labeling 胜率优化版' if use_ml else '基准规则版'}) "
    print(mode_str)
    print("=" * 60)

    # 1. 加载数据
    loader = MoomooDataLoader()
    print("\n[Step 1/5] 正在加载 Moomoo ETF 行情数据（优先读取本地 Parquet 缓存）...")
    data_dict = loader.load_all_universe(start_date=start_date)
    print(f"-> 成功加载 {len(data_dict)} 只标的，时间跨度: {start_date} ~ 最新")

    # 2. ML 胜率预测 (如果启用)
    win_rates_df = None
    if use_ml:
        print("\n[Step 2/5] [ML 胜率引擎] 提取时序量价特征 (动量加速度/量能异动/VWAP基差/RSI/ATR比率)...")
        extractor = TimeSeriesFeatureExtractor()
        features_dict = extractor.extract_universe_features(data_dict)

        print("[ML 胜率引擎] 依据 14日动态 ATR 构建三重屏障盈亏比标签 (上屏障 +1.5 ATR / 下屏障 -1.0 ATR)...")
        labeler = TripleBarrierLabeler(atr_window=14, upper_atr_mult=1.5, lower_atr_mult=1.0, max_holding_days=10)
        labels_df = labeler.generate_universe_labels(data_dict)

        print("[ML 胜率引擎] 执行 Walk-Forward 滚动前向训练与 Isotonic 概率校准 (每半年滚动重训)...")
        pipeline = WalkForwardWinRatePipeline(warmup_days=400, refit_interval=126, barrier_embargo_days=10)
        candidate_tickers = [t for t in data_dict.keys() if t not in ['US.SPY', 'US.BIL']]
        win_rates_df = pipeline.run_walkforward_predictions(features_dict, labels_df, candidate_tickers)
        print(f"-> Walk-Forward 预测胜率生成完毕 (均值: {win_rates_df.mean().mean():.1%})")
    else:
        print("\n[Step 2/5] 未启用 ML 模块，采用纯截面动量规则模式。")

    # 3. 初始化策略与分配器
    print(f"\n[Step 3/5] 初始化截面动量与波动率倒数加权引擎 (Top {top_n}, Exit Rank {exit_rank}, Buffer {int(buffer*100)}%)...")
    strategy = MomentumRegimeStrategy(
        top_n=top_n,
        exit_rank=exit_rank,
        ma_filter_window=60,
        vol_window=20,
        vol_threshold=0.25,
    )
    allocator = PortfolioAllocator(
        max_asset_weight=max_cap,
        rebalance_buffer=buffer,
        cash_ticker='US.BIL',
    )
    cost_calc = TradingCostCalculator()

    # 4. 运行回测
    print("\n[Step 4/5] 开始执行日频 VWAP 高保真回测模拟（含5%换手缓冲区与真实佣金费率）...")
    backtester = DailyVWAPBacktester(
        strategy=strategy,
        allocator=allocator,
        cost_calculator=cost_calc,
        initial_capital=initial_capital,
        warmup_period=200,
    )
    df_results = backtester.run(
        data_dict,
        win_rates_df=win_rates_df,
        min_win_rate=min_win_rate,
    )

    # 4. 统计指标与多基准对比
    print("\n[Step 4/4] 绩效计算与基准收益分析...")
    strat_metrics = backtester.calculate_performance_metrics(df_results)

    # 计算基准指标
    def calc_series_metrics(equity_series, name):
        ret = equity_series.pct_change(fill_method=None).fillna(0.0)
        years = len(equity_series) / 252.0
        tot_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ((ret - 0.02 / 252).mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
        cum_max = equity_series.cummax()
        dd = (equity_series - cum_max) / cum_max
        max_dd = dd.min()
        calmar = (cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0
        return {
            'Years': round(years, 2),
            'Total Return': f"{round(tot_ret * 100, 2)}%",
            'CAGR': f"{round(cagr * 100, 2)}%",
            'Annual Volatility': f"{round(ann_vol * 100, 2)}%",
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown': f"{round(max_dd * 100, 2)}%",
            'Calmar Ratio': round(calmar, 2),
            'Final Equity': f"${round(equity_series.iloc[-1], 2)}",
        }

    spy_metrics = calc_series_metrics(df_results['spy_equity'], 'SPY')
    qqq_metrics = calc_series_metrics(df_results['qqq_equity'], 'QQQ')
    tqqq_metrics = calc_series_metrics(df_results['tqqq_equity'], 'TQQQ')

    comparison_df = pd.DataFrame({
        '截面动量+杠杆策略(本策略)': strat_metrics,
        'SPY (标普500基准)': spy_metrics,
        'QQQ (纳斯达克100基准)': qqq_metrics,
        'TQQQ (3倍杠杆纳指)': tqqq_metrics,
    }).fillna('-')

    print("\n" + "=" * 80)
    print("                    [PERFORMANCE] 策略与基准多维度绩效对比表")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)

    # 保存结果
    os.makedirs('data/results', exist_ok=True)
    csv_path = 'data/results/momentum_backtest_results.csv'
    parquet_path = 'data/results/momentum_backtest_results.parquet'
    df_results.to_csv(csv_path)
    df_results.to_parquet(parquet_path)
    print(f"\n[OK] 每日详细净值与调仓流水已保存至: {csv_path}")

    # 生成净值曲线图
    if save_plot:
        plot_path = 'data/results/momentum_equity_curve.png'
        plt.figure(figsize=(12, 7))
        plt.plot(df_results.index, df_results['equity'], label='Strategy (Momentum + Leverage + InvVol)', color='#1f77b4', lw=2)
        plt.plot(df_results.index, df_results['spy_equity'], label='SPY Buy & Hold', color='#7f7f7f', lw=1.2, ls='--')
        plt.plot(df_results.index, df_results['qqq_equity'], label='QQQ Buy & Hold', color='#ff7f0e', lw=1.2, ls='--')
        plt.plot(df_results.index, df_results['tqqq_equity'], label='TQQQ Buy & Hold', color='#d62728', lw=1.0, alpha=0.7)

        plt.title('Cross-Sectional Momentum & Leveraged ETF Rotation Strategy (VWAP Execution)', fontsize=14, pad=12)
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Portfolio Equity ($)', fontsize=11)
        plt.yscale('log')  # 对数坐标展示复利效应
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[OK] 净值走势对比图 (对数坐标) 已保存至: {plot_path}")

    return df_results, comparison_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行结构性行业动量与杠杆轮动策略回测")
    parser.add_argument('--start', type=str, default='2016-01-01', help='回测起始日期')
    parser.add_argument('--top_n', type=int, default=3, help='Top N 标的数')
    parser.add_argument('--exit_rank', type=int, default=4, help='退出排名迟滞阈值 (若大于 top_n 则启用排名迟滞缓冲区)')
    parser.add_argument('--cap', type=float, default=0.35, help='单标的权重上限')
    parser.add_argument('--buffer', type=float, default=0.05, help='换手缓冲区阈值')
    parser.add_argument('--use_ml', action='store_true', help='启用 Meta-Labeling 时序胜率质检与凯利头寸缩放')
    parser.add_argument('--min_win_rate', type=float, default=0.50, help='入选最低胜率硬门槛 (默认 0.50)')
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start,
        top_n=args.top_n,
        exit_rank=args.exit_rank,
        max_cap=args.cap,
        buffer=args.buffer,
        use_ml=args.use_ml,
        min_win_rate=args.min_win_rate,
    )
