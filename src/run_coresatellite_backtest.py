"""核心底仓 + 卫星动量轮动 + 波动率目标化系统 (Core-Satellite) 全量回测运行脚本。

严格遵循实盘量化执行规范：
1. T-1 日收盘后生成宏观状态、因子得分、核心底仓、卫星标的与目标杠杆
2. T 日以全日前复权真实 VWAP (成交额/成交量折算) 成交
3. 严格计入 Moomoo 真实佣金与平台费结构 ($0.0049/股佣金 + $0.005/股平台费 + SEC规费)
4. 引入 5% 仓位偏离度缓冲带 (Rebalance Buffer)，杜绝低效无效调仓
5. 对标 QQQ (核心基准) 与 SPY，计算信息比率 (IR)、跟踪误差、夏普比率、卡玛比率与最大回撤
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from typing import Optional, Tuple, Dict, List, Any
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_moomoo import MoomooDataLoader, DEFAULT_DEFENSE
from strategy_coresatellite import CoreSatelliteStrategy
from portfolio_allocator import PortfolioAllocator
from trading_costs import TradingCostCalculator
from backtest_vwap import DailyVWAPBacktester


def run_coresatellite_pipeline(
    start_date: str = "2016-01-01",
    core_ratio: float = 0.50,
    satellite_ratio: float = 0.50,
    target_vol: float = 0.25,
    min_leverage: float = 0.60,
    max_leverage: float = 1.80,
    buffer: float = 0.06,
    initial_capital: float = 100000.0,
    benchmark: str = 'US.QQQ',
    save_plot: bool = True,
    top_n: int = 2,
    exit_rank: int = 3,
    max_satellite_cap: float = 0.30,
    max_asset_weight: float = 0.45,
    tiered_ratio: Optional[tuple] = None,
):
    print("=" * 80)
    print(" [SYSTEM] 核心底仓 + 卫星动量轮动 + 波动率目标化 (Core-Satellite) 策略回测 ")
    print(f" 配置: 核心底仓 {int(core_ratio*100)}% | 卫星轮动 {int(satellite_ratio*100)}% (Top {top_n}, 上限 {int(max_satellite_cap*100)}%) | 目标波动率 {int(target_vol*100)}% | 杠杆区间 [{min_leverage}x, {max_leverage}x]")
    print("=" * 80)

    # 1. 加载数据
    loader = MoomooDataLoader()
    print("\n[Step 1/4] 加载 Moomoo ETF 行情数据（全量命中本地 Parquet 缓存）...")
    data_dict = loader.load_all_universe(start_date=start_date)
    print(f"-> 成功加载 {len(data_dict)} 只标的，覆盖科技、半导体、大宗商品(铜/银/金/铀/油气)、QDII、2x/3x杠杆与防御短债。")

    # 2. 初始化核心-卫星策略引擎
    print("\n[Step 2/4] 初始化 Core-Satellite 策略引擎...")
    strategy = CoreSatelliteStrategy(
        benchmark=benchmark,
        core_ratio=core_ratio,
        satellite_ratio=satellite_ratio,
        target_vol=target_vol,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        top_n=top_n,
        exit_rank=exit_rank,
        max_satellite_cap=max_satellite_cap,
        ma_filter_window=60,
        tiered_ratio=tiered_ratio,
    )
    allocator = PortfolioAllocator(
        max_asset_weight=max_asset_weight,
        rebalance_buffer=buffer,
        cash_ticker=DEFAULT_DEFENSE,
    )
    cost_calc = TradingCostCalculator()

    # 3. 预计算特征、大盘均线、波动率与宏观状态
    print("\n[Step 3/4] 计算多因子矩阵、大宗商品动量相对强度与宏观状态切换序列...")
    factors_dict, vols, ma_trend, df_close, regimes = strategy.compute_factors_and_indicators(data_dict)

    # 提取成交额折算 VWAP 矩阵
    vwap_dict = {ticker: df.get('vwap', df['close']) for ticker, df in data_dict.items()}
    df_vwap = pd.DataFrame(vwap_dict).reindex(df_close.index)
    for col in df_vwap.columns:
        df_vwap[col] = df_vwap[col].fillna(df_close[col])

    dates = df_close.index
    n_days = len(dates)
    warmup_period = 200

    if n_days <= warmup_period:
        raise ValueError(f"数据总长度 {n_days} 天不足以支持预热期 {warmup_period} 天")

    cash = initial_capital
    holdings: dict[str, float] = {}
    records = []

    first_sim_idx = warmup_period
    spy_init_price = df_close['US.SPY'].iloc[first_sim_idx]
    qqq_init_price = df_close['US.QQQ'].iloc[first_sim_idx]
    qld_init_price = df_close['US.QLD'].iloc[first_sim_idx]
    tqqq_init_price = df_close['US.TQQQ'].iloc[first_sim_idx]

    # 用于动态跟踪组合已实现波动率 (Rolling 20-day return series)
    portfolio_daily_returns: list[float] = []

    print("\n[Step 4/4] 启动日频 VWAP 高保真回测模拟（严格无未来函数，含真实交易滑点规费）...")
    for i in range(first_sim_idx, n_days):
        date_t = dates[i]
        date_prev = dates[i - 1]

        # 估算组合前 20 日已实现年化波动率
        if len(portfolio_daily_returns) >= 20:
            recent_ret = np.array(portfolio_daily_returns[-20:])
            port_vol = float(np.std(recent_ret, ddof=1) * np.sqrt(252))
        else:
            port_vol = target_vol

        # ---------------------------
        # (A) T-1 日收盘后生成目标配置
        # ---------------------------
        regime_prev = regimes.loc[date_prev]
        trend_prev = ma_trend.loc[date_prev]
        vols_prev = vols.loc[date_prev]

        target_weights = strategy.generate_portfolio_target(
            date=date_prev,
            factors_dict=factors_dict,
            current_trend=trend_prev,
            current_vols=vols_prev,
            regime=regime_prev,
            df_close=df_close,
            realized_portfolio_vol=port_vol,
        )

        vol_mult = strategy.calculate_volatility_multiplier(port_vol)

        # ---------------------------
        # (B) T 日盘中以 VWAP 执行调仓
        # ---------------------------
        prev_close_vals = {
            t: shares * df_close[t].loc[date_prev]
            for t, shares in holdings.items()
            if shares > 0 and t in df_close
        }
        total_val_prev = cash + sum(prev_close_vals.values())

        all_active_tickers = set(holdings.keys()) | set(target_weights.keys())
        target_dollars = {}

        for t in all_active_tickers:
            cur_shares = holdings.get(t, 0.0)
            cur_w = (cur_shares * df_close[t].loc[date_prev] / total_val_prev) if total_val_prev > 0 else 0.0
            targ_w = target_weights.get(t, 0.0)

            if targ_w <= 1e-4:
                target_dollars[t] = 0.0
            elif cur_shares <= 1e-4:
                target_dollars[t] = targ_w * total_val_prev
            else:
                # 5% 缓冲区过滤
                if abs(targ_w - cur_w) <= buffer:
                    target_dollars[t] = None
                else:
                    target_dollars[t] = targ_w * total_val_prev

        daily_cost = 0.0
        daily_turnover = 0.0

        # 先卖出
        for t, targ_d in target_dollars.items():
            if targ_d is None:
                continue
            cur_shares = holdings.get(t, 0.0)
            vwap_price = df_vwap[t].loc[date_t]
            if np.isnan(vwap_price) or vwap_price <= 0:
                vwap_price = df_close[t].loc[date_t]

            cur_dollars = cur_shares * vwap_price
            if targ_d < cur_dollars:
                dollar_delta = cur_dollars - targ_d
                shares_to_sell = min(cur_shares, dollar_delta / vwap_price)

                if shares_to_sell > 0.01:
                    cost_breakdown = cost_calc.calc_trade_cost(
                        shares=shares_to_sell, price=vwap_price, is_buy=False
                    )
                    daily_cost += cost_breakdown.total
                    daily_turnover += shares_to_sell * vwap_price

                    cash += (shares_to_sell * vwap_price - cost_breakdown.total)
                    holdings[t] = cur_shares - shares_to_sell
                    if holdings[t] <= 1e-4:
                        holdings[t] = 0.0

        # 后买入
        for t, targ_d in target_dollars.items():
            if targ_d is None:
                continue
            cur_shares = holdings.get(t, 0.0)
            vwap_price = df_vwap[t].loc[date_t]
            if np.isnan(vwap_price) or vwap_price <= 0:
                vwap_price = df_close[t].loc[date_t]

            cur_dollars = cur_shares * vwap_price
            if targ_d > cur_dollars:
                dollar_delta = targ_d - cur_dollars
                if dollar_delta > 50.0 and vwap_price > 0 and cash > 100.0:
                    max_shares = max(0.0, (cash - 100.0) / (vwap_price + 0.015))
                    shares_to_buy = min(dollar_delta / vwap_price, max_shares)
                    if shares_to_buy > 0.01:
                        cost_breakdown = cost_calc.calc_trade_cost(
                            shares=shares_to_buy, price=vwap_price, is_buy=True
                        )
                        daily_cost += cost_breakdown.total
                        daily_turnover += shares_to_buy * vwap_price

                        cash -= (shares_to_buy * vwap_price + cost_breakdown.total)
                        holdings[t] = cur_shares + shares_to_buy

        holdings = {t: s for t, s in holdings.items() if s > 1e-4}

        # ---------------------------
        # (C) 结算收盘净值
        # ---------------------------
        holdings_value = sum(
            s * df_close[t].loc[date_t] for t, s in holdings.items() if t in df_close
        )
        total_equity = cash + holdings_value

        if len(records) > 0:
            daily_pnl_ret = (total_equity / records[-1]['equity']) - 1.0
        else:
            daily_pnl_ret = 0.0
        portfolio_daily_returns.append(daily_pnl_ret)

        spy_equity = initial_capital * (df_close['US.SPY'].loc[date_t] / spy_init_price)
        qqq_equity = initial_capital * (df_close['US.QQQ'].loc[date_t] / qqq_init_price)
        qld_equity = initial_capital * (df_close['US.QLD'].loc[date_t] / qld_init_price)
        tqqq_equity = initial_capital * (df_close['US.TQQQ'].loc[date_t] / tqqq_init_price)

        active_holdings_str = ", ".join(
            f"{t.replace('US.', '')}:{round(s * df_close[t].loc[date_t] / total_equity * 100, 1)}%"
            for t, s in holdings.items()
            if s * df_close[t].loc[date_t] > 10.0
        )

        records.append({
            'date': date_t,
            'equity': total_equity,
            'cash': cash,
            'holdings_value': holdings_value,
            'holdings_desc': active_holdings_str,
            'regime': regime_prev,
            'vol_multiplier': vol_mult,
            'port_vol': port_vol,
            'cost': daily_cost,
            'turnover': daily_turnover,
            'spy_equity': spy_equity,
            'qqq_equity': qqq_equity,
            'qld_equity': qld_equity,
            'tqqq_equity': tqqq_equity,
        })

    df_results = pd.DataFrame(records).set_index('date')
    df_results['daily_return'] = df_results['equity'].pct_change(fill_method=None).fillna(0.0)

    # 绩效评估
    strat_metrics = DailyVWAPBacktester.calculate_performance_metrics(df_results)

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
            'Sortino Ratio': '-',
            'Information Ratio (vs QQQ)': 0.0 if name == 'QQQ' else '-',
            'Tracking Error (vs QQQ)': '0.0%' if name == 'QQQ' else '-',
            'Max Drawdown': f"{round(max_dd * 100, 2)}%",
            'Calmar Ratio': round(calmar, 2),
            'Final Equity': f"${round(equity_series.iloc[-1], 2)}",
        }

    qqq_metrics = calc_series_metrics(df_results['qqq_equity'], 'QQQ')
    spy_metrics = calc_series_metrics(df_results['spy_equity'], 'SPY')
    qld_metrics = calc_series_metrics(df_results['qld_equity'], 'QLD')
    tqqq_metrics = calc_series_metrics(df_results['tqqq_equity'], 'TQQQ')

    comparison_df = pd.DataFrame({
        'Core-Satellite策略(本策略)': strat_metrics,
        'QQQ (核心对标基准)': qqq_metrics,
        'SPY (标普500基准)': spy_metrics,
        'QLD (2倍杠杆纳指)': qld_metrics,
        'TQQQ (3倍杠杆纳指)': tqqq_metrics,
    }).fillna('-')

    print("\n" + "=" * 95)
    print("         [PERFORMANCE] Core-Satellite + 宏观状态 + 波动率目标化 绩效全量对比表")
    print("=" * 95)
    print(comparison_df.to_string())
    print("=" * 95)

    # 年度收益分解表
    annual_records = []
    strat_daily = df_results['equity'].pct_change().fillna(0.0)
    qqq_daily = df_results['qqq_equity'].pct_change().fillna(0.0)
    spy_daily = df_results['spy_equity'].pct_change().fillna(0.0)

    for yr in sorted(list(set(df_results.index.year))):
        sub_idx = df_results.index[df_results.index.year == yr]
        if len(sub_idx) > 5:
            strat_yr = (strat_daily.loc[sub_idx].add(1.0).prod() - 1.0) * 100
            qqq_yr = (qqq_daily.loc[sub_idx].add(1.0).prod() - 1.0) * 100
            spy_yr = (spy_daily.loc[sub_idx].add(1.0).prod() - 1.0) * 100
            alpha = strat_yr - qqq_yr
            annual_records.append({
                'Year': yr,
                'Strategy': f"{strat_yr:+.2f}%",
                'QQQ': f"{qqq_yr:+.2f}%",
                'Alpha vs QQQ': f"{alpha:+.2f}%",
                'SPY': f"{spy_yr:+.2f}%",
                'Win?': 'WIN' if alpha > 0 else ('TIE' if abs(alpha) < 2.0 else 'LOSS'),
            })
    df_annual = pd.DataFrame(annual_records).set_index('Year')
    print("\n" + "=" * 80)
    print("                  [ANNUAL BREAKDOWN] 分历年超额收益与胜负统计")
    print("=" * 80)
    print(df_annual.to_string())
    print("=" * 80)

    # 保存明细数据
    os.makedirs('data/results', exist_ok=True)
    df_results.to_csv('data/results/coresatellite_backtest_results.csv')
    df_results.to_parquet('data/results/coresatellite_backtest_results.parquet')

    if save_plot:
        plot_path = 'data/results/coresatellite_equity_curve.png'
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw={'height_ratios': [3, 1.2, 1.2]})

        # 子图 1: 对数净值走势
        ax1.plot(df_results.index, df_results['equity'], label='Core-Satellite Strategy (VWAP)', color='#1f77b4', lw=2.2)
        ax1.plot(df_results.index, df_results['qqq_equity'], label='QQQ Benchmark (1x)', color='#ff7f0e', lw=1.5, ls='--')
        ax1.plot(df_results.index, df_results['qld_equity'], label='QLD Buy & Hold (2x)', color='#2ca02c', lw=1.2, ls='-.', alpha=0.7)
        ax1.plot(df_results.index, df_results['spy_equity'], label='SPY Benchmark', color='#7f7f7f', lw=1.0, ls=':')
        ax1.plot(df_results.index, df_results['tqqq_equity'], label='TQQQ Buy & Hold (3x)', color='#d62728', lw=1.0, alpha=0.5)

        ax1.set_title('Core-Satellite Dynamic Allocation (Core 40% + Satellite 60% + Vol Targeting) vs Benchmarks', fontsize=13, pad=10)
        ax1.set_ylabel('Portfolio Equity ($)', fontsize=11)
        ax1.set_yscale('log')
        ax1.grid(True, which='both', ls=':', alpha=0.5)
        ax1.legend(loc='upper left', fontsize=9.5)

        # 子图 2: 动态回撤对比
        cum_strat = df_results['equity'].cummax()
        dd_strat = (df_results['equity'] - cum_strat) / cum_strat
        cum_qqq = df_results['qqq_equity'].cummax()
        dd_qqq = (df_results['qqq_equity'] - cum_qqq) / cum_qqq
        cum_qld = df_results['qld_equity'].cummax()
        dd_qld = (df_results['qld_equity'] - cum_qld) / cum_qld

        ax2.fill_between(df_results.index, dd_strat * 100, 0, color='#1f77b4', alpha=0.35, label='Strategy Drawdown')
        ax2.plot(df_results.index, dd_qqq * 100, color='#ff7f0e', lw=1.2, ls='--', label='QQQ Drawdown')
        ax2.plot(df_results.index, dd_qld * 100, color='#2ca02c', lw=1.0, ls='-.', alpha=0.7, label='QLD Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.grid(True, ls=':', alpha=0.5)
        ax2.legend(loc='lower left', fontsize=9)

        # 子图 3: 波动率目标化杠杆倍数 & 宏观状态
        ax3.plot(df_results.index, df_results['vol_multiplier'], color='#9467bd', lw=1.5, label='Vol Target Multiplier (Lt)')
        ax3.axhline(1.0, color='gray', ls='--', alpha=0.6)
        ax3.set_ylabel('Leverage Scale', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, ls=':', alpha=0.5)
        ax3.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] 核心-卫星多维度净值与回撤对比图已保存至: {plot_path}")

    return df_results, comparison_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Core-Satellite ETF 轮动回测")
    parser.add_argument('--start', type=str, default='2016-01-01', help='回测起始日期')
    parser.add_argument('--core', type=float, default=0.50, help='核心底仓基准比例')
    parser.add_argument('--satellite', type=float, default=0.50, help='卫星轮动基准比例')
    parser.add_argument('--target_vol', type=float, default=0.25, help='目标年化波动率')
    parser.add_argument('--buffer', type=float, default=0.06, help='换手偏离缓冲区')
    args = parser.parse_args()

    run_coresatellite_pipeline(
        start_date=args.start,
        core_ratio=args.core,
        satellite_ratio=args.satellite,
        target_vol=args.target_vol,
        buffer=args.buffer,
    )
