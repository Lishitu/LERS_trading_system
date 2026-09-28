"""多因子选基与组合优化轮动回测主入口。

功能：
1. 策略基准对齐为 QQQ，并增加信息比率 (Information Ratio, IR) 与跟踪误差
2. 扩充标的池：美股主线行业 + QDII (DXJ 日股) + 期货/大宗商品 (COPX 铜矿, SLV 白银, GLD 黄金)
3. 动量、加速度、低波动、量能爆发与 VWAP 基差合成为多因子评分
4. 因子倾斜的波动率倒数加权 (Factor-Tilted Inverse Volatility Optimization) + 35% Cap + 5% 缓冲区
5. QQQ 大盘环境滤网与进攻性杠杆 ETF 升级
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

from data_moomoo import MoomooDataLoader, DEFAULT_SECTORS, DEFAULT_DEFENSE
from strategy_multifactor import MultiFactorETFStrategy
from portfolio_allocator import PortfolioAllocator
from trading_costs import TradingCostCalculator
from backtest_vwap import DailyVWAPBacktester


def run_multifactor_pipeline(
    start_date: str = "2016-01-01",
    top_n: int = 3,
    exit_rank: int = 4,
    max_cap: float = 0.35,
    buffer: float = 0.05,
    initial_capital: float = 100000.0,
    benchmark: str = 'US.QQQ',
    save_plot: bool = True,
):
    print("=" * 70)
    print(" [STRATEGY] 多因子 ETF 选基与组合优化轮动系统 (基准: QQQ | 含 QDII 与大宗商品) ")
    print("=" * 70)

    # 1. 加载数据
    loader = MoomooDataLoader()
    print("\n[Step 1/4] 正在加载 Moomoo ETF 行情数据（优先读取本地 Parquet 缓存）...")
    data_dict = loader.load_all_universe(start_date=start_date)
    print(f"-> 成功加载 {len(data_dict)} 只标的，覆盖核心行业、QDII (DXJ)、大宗商品 (COPX/SLV/GLD)")

    # 2. 初始化策略与分配器
    print(f"\n[Step 2/4] 初始化多因子选基引擎 (基准: {benchmark}, Top {top_n}, Exit Rank {exit_rank}, Cap {int(max_cap*100)}%)...")
    strategy = MultiFactorETFStrategy(
        benchmark=benchmark,
        top_n=top_n,
        exit_rank=exit_rank,
        ma_filter_window=60,
        vol_threshold=0.28,
    )
    allocator = PortfolioAllocator(
        max_asset_weight=max_cap,
        rebalance_buffer=buffer,
        cash_ticker=DEFAULT_DEFENSE,
    )
    cost_calc = TradingCostCalculator()

    # 3. 预先计算因子与大盘环境
    print("\n[Step 3/4] 计算多因子矩阵 (动量、加速度、低波动、量能爆发、VWAP防御) 与大盘状态机...")
    factors_dict, vols, ma_trend, df_close = strategy.compute_factors_for_all(data_dict)
    regimes = strategy.get_market_regimes(df_close[benchmark])

    # 提取收盘价与 VWAP 矩阵
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
    tqqq_init_price = df_close['US.TQQQ'].iloc[first_sim_idx]

    print("\n[Step 4/4] 执行日频 VWAP 高保真回测模拟（严格无未来函数，含真实佣金与5%缓冲区）...")
    for i in range(first_sim_idx, n_days):
        date_t = dates[i]
        date_prev = dates[i - 1]

        # ---------------------------
        # (A) T-1 日收盘后生成目标配置 (多因子选基 + 因子倾斜加权)
        # ---------------------------
        regime_prev = regimes.loc[date_prev]
        trend_prev = ma_trend.loc[date_prev]
        vols_prev = vols.loc[date_prev]

        selected_assets = strategy.generate_daily_signals(
            date=date_prev,
            factors_dict=factors_dict,
            current_trend=trend_prev,
            current_vols=vols_prev,
            regime=regime_prev,
        )
        # 因子倾斜的波动率倒数加权
        raw_target_weights = allocator.compute_target_weights(selected_assets)

        # ---------------------------
        # (B) T 日盘中以 VWAP 执行调仓
        # ---------------------------
        prev_close_vals = {
            t: shares * df_close[t].loc[date_prev]
            for t, shares in holdings.items()
            if shares > 0 and t in df_close
        }
        total_val_prev = cash + sum(prev_close_vals.values())

        all_active_tickers = set(holdings.keys()) | set(raw_target_weights.keys())
        target_dollars = {}

        for t in all_active_tickers:
            cur_shares = holdings.get(t, 0.0)
            cur_w = (cur_shares * df_close[t].loc[date_prev] / total_val_prev) if total_val_prev > 0 else 0.0
            targ_w = raw_target_weights.get(t, 0.0)

            if targ_w <= 1e-4:
                target_dollars[t] = 0.0
            elif cur_shares <= 1e-4:
                target_dollars[t] = targ_w * total_val_prev
            else:
                if abs(targ_w - cur_w) <= allocator.rebalance_buffer:
                    target_dollars[t] = None
                else:
                    target_dollars[t] = targ_w * total_val_prev

        daily_cost = 0.0
        daily_turnover = 0.0

        # 卖出
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
                shares_to_sell = dollar_delta / vwap_price
                shares_to_sell = min(shares_to_sell, cur_shares)

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

        # 买入
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
                # 动态规费摩擦预留，防止低价股大股数交易产生高额 SEC/FINRA/平台费时透支现金
                max_allocatable_cash = max(0.0, cash - 100.0)
                effective_price = vwap_price + 0.015
                max_affordable_shares = max_allocatable_cash / effective_price
                target_shares_delta = dollar_delta / vwap_price
                shares_to_buy = min(target_shares_delta, max_affordable_shares)

                if shares_to_buy > 0.01 and vwap_price > 0:
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

        spy_equity = initial_capital * (df_close['US.SPY'].loc[date_t] / spy_init_price)
        qqq_equity = initial_capital * (df_close['US.QQQ'].loc[date_t] / qqq_init_price)
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
            'cost': daily_cost,
            'turnover': daily_turnover,
            'spy_equity': spy_equity,
            'qqq_equity': qqq_equity,
            'tqqq_equity': tqqq_equity,
        })

    df_results = pd.DataFrame(records).set_index('date')
    df_results['daily_return'] = df_results['equity'].pct_change(fill_method=None).fillna(0.0)

    # 绩效计算
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
    tqqq_metrics = calc_series_metrics(df_results['tqqq_equity'], 'TQQQ')

    comparison_df = pd.DataFrame({
        '多因子优化策略(本策略)': strat_metrics,
        'QQQ (核心基准)': qqq_metrics,
        'SPY (标普500基准)': spy_metrics,
        'TQQQ (3倍杠杆纳指)': tqqq_metrics,
    }).fillna('-')

    print("\n" + "=" * 85)
    print("               [PERFORMANCE] 多因子选基与组合优化策略绩效对比表 (基准: QQQ)")
    print("=" * 85)
    print(comparison_df.to_string())
    print("=" * 85)

    # 保存结果与图像
    os.makedirs('data/results', exist_ok=True)
    df_results.to_csv('data/results/multifactor_backtest_results.csv')
    df_results.to_parquet('data/results/multifactor_backtest_results.parquet')

    if save_plot:
        plot_path = 'data/results/multifactor_equity_curve.png'
        plt.figure(figsize=(12, 7))
        plt.plot(df_results.index, df_results['equity'], label='Multi-Factor ETF Strategy (VWAP)', color='#1f77b4', lw=2)
        plt.plot(df_results.index, df_results['qqq_equity'], label='QQQ Benchmark', color='#ff7f0e', lw=1.5, ls='--')
        plt.plot(df_results.index, df_results['spy_equity'], label='SPY Benchmark', color='#7f7f7f', lw=1.2, ls=':')
        plt.plot(df_results.index, df_results['tqqq_equity'], label='TQQQ Buy & Hold', color='#d62728', lw=1.0, alpha=0.6)

        plt.title('Multi-Factor ETF Strategy (Sectors + QDII + Commodities) vs QQQ Benchmark', fontsize=14, pad=12)
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Portfolio Equity ($)', fontsize=11)
        plt.yscale('log')
        plt.grid(True, which='both', ls=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] 净值走势对比图已保存至: {plot_path}")

    return df_results, comparison_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多因子ETF选基与组合优化回测")
    parser.add_argument('--start', type=str, default='2016-01-01', help='回测起始日期')
    parser.add_argument('--top_n', type=int, default=3, help='Top N 席位')
    parser.add_argument('--exit_rank', type=int, default=4, help='退出排名迟滞阈值')
    parser.add_argument('--cap', type=float, default=0.35, help='单标的权重上限')
    parser.add_argument('--buffer', type=float, default=0.05, help='换手偏离缓冲区')
    args = parser.parse_args()

    run_multifactor_pipeline(
        start_date=args.start,
        top_n=args.top_n,
        exit_rank=args.exit_rank,
        max_cap=args.cap,
        buffer=args.buffer,
    )
