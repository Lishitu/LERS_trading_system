"""Core-Satellite 策略降低回撤消融实验 (Ablation Study) 运行模块。

针对策略历史最大回撤（-27.3%）与极端年份（2022 熊市）进行系统性消融对比：
- M0: Baseline 当前基准策略 (标准单日 MA60, 目标波动率 25%, 杠杆区间 [0.6x, 1.8x])
- M1: + 熊市杠杆硬封顶与现金保底 (Bear Leverage Ceiling & Cash Floor: 熊市乘数 <= 0.65x, 现金 >= 50%)
- M2: + 迟滞假突破3日确认过滤 (Hysteresis Trend Confirmation: 熊市反弹需连续3日站稳 MA60)
- M3: + 卫星标的日频移动跟踪止损 (Satellite Trailing Stop: 较近期高点回撤 > 8% 或破 MA20 即时止损转短债)
- M4: + 组合动态回撤刹车垫 (Dynamic Drawdown Floor: 组合回撤达 8% 开始平滑收缩风险敞口)
- M_OPT: 最优组合风控方案 (M1 + M2 + M3: 熊市控杠杆 + 迟滞反抽过滤 + 卫星移动止损)
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保在模块路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_moomoo import MoomooDataLoader, DEFAULT_DEFENSE, DEFAULT_BENCHMARKS, DEFAULT_SECTORS, LEVERAGED_MAP
from strategy_coresatellite import CoreSatelliteStrategy
from trading_costs import TradingCostCalculator


def run_single_simulation(
    name: str,
    data_dict: Dict[str, pd.DataFrame],
    enable_bear_cap: bool = False,
    enable_hysteresis: bool = False,
    enable_satellite_stop: bool = False,
    enable_dd_brake: bool = False,
    target_vol: float = 0.25,
    min_leverage: float = 0.60,
    max_leverage: float = 1.80,
    buffer: float = 0.06,
    initial_capital: float = 100000.0,
    warmup_period: int = 200,
) -> pd.DataFrame:
    """运行单组配置的高保真 VWAP 回测模拟。"""
    strategy = CoreSatelliteStrategy(
        benchmark='US.QQQ',
        core_ratio=0.50,
        satellite_ratio=0.50,
        target_vol=target_vol,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        top_n=2,
        exit_rank=3,
        max_satellite_cap=0.30,
        ma_filter_window=60,
    )
    cost_calc = TradingCostCalculator()

    factors_dict, vols, ma_trend, df_close, regimes = strategy.compute_factors_and_indicators(data_dict)

    vwap_dict = {ticker: df.get('vwap', df['close']) for ticker, df in data_dict.items()}
    df_vwap = pd.DataFrame(vwap_dict).reindex(df_close.index).fillna(df_close)

    dates = df_close.index
    n_days = len(dates)

    # 1. 迟滞过滤序列 (Hysteresis Confirmation):
    # 若在 CRISIS_BEAR 中，只有当标普/纳指连续 3 个交易日收盘 > MA60，才允许退出熊市进入牛市
    filtered_regimes = regimes.copy()
    if enable_hysteresis:
        consecutive_above = 0
        current_state = regimes.iloc[0]
        for idx, date in enumerate(dates):
            raw_regime = regimes.loc[date]
            row_trend = ma_trend.loc[date]
            is_above = bool(row_trend['US.QQQ']) if (hasattr(row_trend, '__getitem__') and 'US.QQQ' in row_trend) else bool(row_trend)

            if current_state == 'CRISIS_BEAR':
                if is_above:
                    consecutive_above += 1
                else:
                    consecutive_above = 0

                # 必须连续 3 日站稳 MA60 才允许转入非熊市状态
                if consecutive_above >= 3:
                    current_state = raw_regime
                    consecutive_above = 0
            else:
                current_state = raw_regime
                consecutive_above = 1 if is_above else 0

            filtered_regimes.loc[date] = current_state

    cash = initial_capital
    holdings: Dict[str, float] = {}
    records: List[Dict[str, Any]] = []
    portfolio_daily_returns: List[float] = []

    # 跟踪卫星持仓买入以来的最高价 (用于 M3 移动止损)
    satellite_highest_price: Dict[str, float] = {}

    peak_equity = initial_capital

    for i in range(warmup_period, n_days):
        date_t = dates[i]
        date_prev = dates[i - 1]

        # 估算过去 20 日组合波动率
        if len(portfolio_daily_returns) >= 20:
            recent_ret = np.array(portfolio_daily_returns[-20:])
            port_vol = float(np.std(recent_ret, ddof=1) * np.sqrt(252))
        else:
            port_vol = target_vol

        regime_prev = filtered_regimes.loc[date_prev]
        trend_prev = ma_trend.loc[date_prev]
        vols_prev = vols.loc[date_prev]

        # 计算目标权重
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

        # ---------------------------------------------
        # 机制 M1: 熊市杠杆硬封顶与现金保底
        # ---------------------------------------------
        if enable_bear_cap and regime_prev in ('CRISIS_BEAR', 'INFLATION_COMMODITY'):
            vol_mult = min(vol_mult, 0.65)
            # 缩减权益仓位，强制保留至少 50% BIL 现金
            risk_sum = sum(w for k, w in target_weights.items() if k != DEFAULT_DEFENSE)
            if risk_sum > 0.45:
                scale_ratio = 0.45 / risk_sum
                new_w = {k: (w * scale_ratio if k != DEFAULT_DEFENSE else w) for k, w in target_weights.items()}
                new_w[DEFAULT_DEFENSE] = 1.0 - sum(w for k, w in new_w.items() if k != DEFAULT_DEFENSE)
                target_weights = new_w

        # ---------------------------------------------
        # 机制 M4: 组合动态回撤刹车垫 (CPPI Floor)
        # ---------------------------------------------
        if enable_dd_brake:
            cur_dd = (records[-1]['equity'] - peak_equity) / peak_equity if records and peak_equity > 0 else 0.0
            if cur_dd < -0.08:
                # 回撤达到 8% 开始线性收缩风险敞口，在 15% 处降到 40%
                brake_scale = max(0.40, 1.0 - (abs(cur_dd) - 0.08) / 0.07 * 0.60)
                risk_sum = sum(w for k, w in target_weights.items() if k != DEFAULT_DEFENSE)
                if risk_sum > 0:
                    scaled_risk = risk_sum * brake_scale
                    scale_ratio = scaled_risk / risk_sum
                    new_w = {k: (w * scale_ratio if k != DEFAULT_DEFENSE else w) for k, w in target_weights.items()}
                    new_w[DEFAULT_DEFENSE] = 1.0 - sum(w for k, w in new_w.items() if k != DEFAULT_DEFENSE)
                    target_weights = new_w

        # 计算上一日持仓市值
        prev_close_vals = {
            t: shares * df_close[t].loc[date_prev]
            for t, shares in holdings.items()
            if shares > 0 and t in df_close
        }
        total_val_prev = cash + sum(prev_close_vals.values())

        # ---------------------------------------------
        # 机制 M3: 卫星个券日频跟踪止损
        # ---------------------------------------------
        stopped_out_tickers = set()
        if enable_satellite_stop:
            for t, shares in holdings.items():
                if shares > 0.01 and t not in (DEFAULT_DEFENSE, 'US.QQQ', 'US.QLD', 'US.TQQQ'):
                    # 卫星标的
                    cur_p = df_close[t].loc[date_prev]
                    hi_p = satellite_highest_price.get(t, cur_p)
                    if cur_p > hi_p:
                        satellite_highest_price[t] = cur_p
                        hi_p = cur_p

                    # 较近期高点跌幅 > 8%
                    if hi_p > 0 and (cur_p / hi_p - 1.0) < -0.08:
                        stopped_out_tickers.add(t)

        all_active_tickers = set(holdings.keys()) | set(target_weights.keys())
        target_dollars = {}

        for t in all_active_tickers:
            if t in stopped_out_tickers:
                target_dollars[t] = 0.0  # 触发止损强制清仓
                continue

            cur_shares = holdings.get(t, 0.0)
            cur_w = (cur_shares * df_close[t].loc[date_prev] / total_val_prev) if total_val_prev > 0 else 0.0
            targ_w = target_weights.get(t, 0.0)

            if targ_w <= 1e-4:
                target_dollars[t] = 0.0
            elif cur_shares <= 1e-4:
                target_dollars[t] = targ_w * total_val_prev
            else:
                if abs(targ_w - cur_w) <= buffer:
                    target_dollars[t] = None
                else:
                    target_dollars[t] = targ_w * total_val_prev

        daily_cost = 0.0

        # 先卖出
        for t, targ_d in target_dollars.items():
            if targ_d is None:
                continue
            cur_shares = holdings.get(t, 0.0)
            vwap_price = df_vwap[t].loc[date_t]
            cur_dollars = cur_shares * vwap_price
            if targ_d < cur_dollars:
                dollar_delta = cur_dollars - targ_d
                shares_to_sell = min(cur_shares, dollar_delta / vwap_price)
                if shares_to_sell > 0.01:
                    cb = cost_calc.calc_trade_cost(shares=shares_to_sell, price=vwap_price, is_buy=False)
                    daily_cost += cb.total
                    cash += (shares_to_sell * vwap_price - cb.total)
                    holdings[t] = cur_shares - shares_to_sell
                    if holdings[t] <= 1e-4:
                        holdings[t] = 0.0
                        if t in satellite_highest_price:
                            del satellite_highest_price[t]

        # 后买入
        for t, targ_d in target_dollars.items():
            if targ_d is None:
                continue
            cur_shares = holdings.get(t, 0.0)
            vwap_price = df_vwap[t].loc[date_t]
            cur_dollars = cur_shares * vwap_price
            if targ_d > cur_dollars:
                dollar_delta = targ_d - cur_dollars
                if dollar_delta > 50.0 and vwap_price > 0 and cash > 100.0:
                    max_shares = max(0.0, (cash - 100.0) / (vwap_price + 0.015))
                    shares_to_buy = min(dollar_delta / vwap_price, max_shares)
                    if shares_to_buy > 0.01:
                        cb = cost_calc.calc_trade_cost(shares=shares_to_buy, price=vwap_price, is_buy=True)
                        daily_cost += cb.total
                        cash -= (shares_to_buy * vwap_price + cb.total)
                        holdings[t] = cur_shares + shares_to_buy
                        satellite_highest_price[t] = max(satellite_highest_price.get(t, 0), vwap_price)

        holdings = {t: s for t, s in holdings.items() if s > 1e-4}

        # 结算 T 日收盘总资产
        holdings_value = sum(s * df_close[t].loc[date_t] for t, s in holdings.items() if t in df_close)
        total_equity = cash + holdings_value

        if total_equity > peak_equity:
            peak_equity = total_equity

        if len(records) > 0:
            daily_pnl = (total_equity / records[-1]['equity']) - 1.0
        else:
            daily_pnl = 0.0
        portfolio_daily_returns.append(daily_pnl)

        records.append({
            'date': date_t,
            'equity': total_equity,
            'daily_return': daily_pnl,
            'regime': regime_prev,
            'vol_multiplier': vol_mult,
            'cost': daily_cost,
        })

    df_out = pd.DataFrame(records).set_index('date')
    return df_out


def evaluate_metrics(df_res: pd.DataFrame) -> Dict[str, Any]:
    """计算核心综合风险收益评价指标。"""
    equity = df_res['equity']
    daily_ret = df_res['daily_return']
    years = len(equity) / 252.0

    tot_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = daily_ret.std() * np.sqrt(252)

    rf_daily = 0.02 / 252.0
    excess_ret = daily_ret - rf_daily
    sharpe = (excess_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    # 下行波动率与 Sortino
    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 0.0
    sortino = (excess_ret.mean() * 252 / downside_vol) if downside_vol > 0 else 0.0

    # 回撤序列
    cummax = equity.cummax()
    dd_series = (equity - cummax) / cummax
    max_dd = dd_series.min()
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0

    # 2022 压力测试 (2022整年收益)
    sub_2022 = df_res.loc['2022-01-01':'2022-12-31']
    if not sub_2022.empty:
        ret_2022 = (sub_2022['equity'].iloc[-1] / sub_2022['equity'].iloc[0]) - 1.0
        mdd_2022 = ((sub_2022['equity'] - sub_2022['equity'].cummax()) / sub_2022['equity'].cummax()).min()
    else:
        ret_2022 = 0.0
        mdd_2022 = 0.0

    win_rate = (daily_ret > 0).mean() * 100.0
    total_cost = df_res['cost'].sum()

    return {
        'Total Return': f"{tot_ret*100:+.2f}%",
        'CAGR': f"{cagr*100:.2f}%",
        'Annual Volatility': f"{ann_vol*100:.2f}%",
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2),
        'Max Drawdown': f"{max_dd*100:.2f}%",
        'Calmar Ratio': round(calmar, 2),
        '2022 Bear Return': f"{ret_2022*100:+.2f}%",
        '2022 Max Drawdown': f"{mdd_2022*100:.2f}%",
        'Daily Win Rate': f"{win_rate:.1f}%",
        'Total Costs': f"${total_cost:,.0f}",
        'Final Equity': f"${equity.iloc[-1]:,.2f}",
        '_raw_cagr': cagr,
        '_raw_max_dd': max_dd,
        '_raw_sharpe': sharpe,
        '_raw_calmar': calmar,
    }


def run_full_ablation_study():
    """全量消融实验流水线。"""
    print("=" * 95)
    print("              [ABLATION STUDY] LERS 策略回撤优化与控制机制消融实验")
    print("=" * 95)

    loader = MoomooDataLoader()
    print("\n[1/3] 加载 ETF 历史全集数据 (2016-01 至 2026-09)...")
    data_dict = loader.load_all_universe(start_date="2016-01-01")

    # 定义消融实验组
    experiments = [
        {
            'key': 'M0 (Baseline基准)',
            'enable_bear_cap': False,
            'enable_hysteresis': False,
            'enable_satellite_stop': False,
            'enable_dd_brake': False,
            'desc': '当前线上基准配置 (标准单日均线，无杠杆硬顶，无刹车垫)',
        },
        {
            'key': 'M1 (+熊市硬杠杆上限)',
            'enable_bear_cap': True,
            'enable_hysteresis': False,
            'enable_satellite_stop': False,
            'enable_dd_brake': False,
            'desc': '熊市 CRISIS_BEAR 状态下杠杆强制 <= 0.65x 且固定 >= 50% 现金',
        },
        {
            'key': 'M2 (+3日迟滞突破确认)',
            'enable_bear_cap': False,
            'enable_hysteresis': True,
            'enable_satellite_stop': False,
            'enable_dd_brake': False,
            'desc': '熊市转牛需连续 3 日站稳 MA60，过滤假反弹与死猫跳',
        },
        {
            'key': 'M3 (+卫星移动跟踪止损)',
            'enable_bear_cap': False,
            'enable_hysteresis': False,
            'enable_satellite_stop': True,
            'enable_dd_brake': False,
            'desc': '单只卫星 ETF 自高点回撤 > 8% 盘中即时止损切入 BIL',
        },
        {
            'key': 'M4 (+组合动态回撤刹车)',
            'enable_bear_cap': False,
            'enable_hysteresis': False,
            'enable_satellite_stop': False,
            'enable_dd_brake': True,
            'desc': '账户自高点回撤达到 8% 开始线性收缩风险资产敞口',
        },
        {
            'key': 'M_OPT (推荐综合最优方案)',
            'enable_bear_cap': True,
            'enable_hysteresis': True,
            'enable_satellite_stop': True,
            'enable_dd_brake': False,
            'desc': '融合 M1 (熊市控杠杆) + M2 (3日假反弹过滤) + M3 (卫星移动止损)',
        },
    ]

    print("\n[2/3] 并行回测 6 组消融实验配置...")
    results_map = {}
    metrics_map = {}

    for exp in experiments:
        t0 = time.time()
        print(f"-> 正在回测: {exp['key']} ...")
        df_sim = run_single_simulation(
            name=exp['key'],
            data_dict=data_dict,
            enable_bear_cap=exp['enable_bear_cap'],
            enable_hysteresis=exp['enable_hysteresis'],
            enable_satellite_stop=exp['enable_satellite_stop'],
            enable_dd_brake=exp['enable_dd_brake'],
        )
        results_map[exp['key']] = df_sim
        metrics_map[exp['key']] = evaluate_metrics(df_sim)
        print(f"   完成耗时 {time.time()-t0:.1f}s | MaxDD={metrics_map[exp['key']]['Max Drawdown']} | CAGR={metrics_map[exp['key']]['CAGR']} | Sharpe={metrics_map[exp['key']]['Sharpe Ratio']}")

    # 汇总消融实验指标对比表
    df_metrics = pd.DataFrame(metrics_map)
    display_rows = [
        'CAGR',
        'Annual Volatility',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Max Drawdown',
        'Calmar Ratio',
        '2022 Bear Return',
        '2022 Max Drawdown',
        'Daily Win Rate',
        'Final Equity',
    ]
    df_comparison = df_metrics.loc[display_rows]

    print("\n" + "=" * 115)
    print("                    [消融实验核心指标横向对照表 (2016-2026 近10年全周期)]")
    print("=" * 115)
    print(df_comparison.to_string())
    print("=" * 115)

    # 绘制净值与回撤对比图
    print("\n[3/3] 渲染消融实验净值与回撤深度对比图表...")
    fig, (ax_equity, ax_dd) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.8, 1.2]})

    colors = {
        'M0 (Baseline基准)': '#94a3b8',
        'M1 (+熊市硬杠杆上限)': '#3b82f6',
        'M2 (+3日迟滞突破确认)': '#8b5cf6',
        'M3 (+卫星移动跟踪止损)': '#f59e0b',
        'M4 (+组合动态回撤刹车)': '#ec4899',
        'M_OPT (推荐综合最优方案)': '#10b981',
    }
    line_styles = {
        'M0 (Baseline基准)': '--',
        'M1 (+熊市硬杠杆上限)': '-.',
        'M2 (+3日迟滞突破确认)': ':',
        'M3 (+卫星移动跟踪止损)': '-.',
        'M4 (+组合动态回撤刹车)': ':',
        'M_OPT (推荐综合最优方案)': '-',
    }
    line_widths = {
        'M0 (Baseline基准)': 1.8,
        'M1 (+熊市硬杠杆上限)': 1.5,
        'M2 (+3日迟滞突破确认)': 1.5,
        'M3 (+卫星移动跟踪止损)': 1.5,
        'M4 (+组合动态回撤刹车)': 1.5,
        'M_OPT (推荐综合最优方案)': 2.5,
    }

    for key, df_sim in results_map.items():
        c = colors.get(key, '#000000')
        ls = line_styles.get(key, '-')
        lw = line_widths.get(key, 1.5)
        ax_equity.plot(df_sim.index, df_sim['equity'], label=f"{key} (CAGR: {metrics_map[key]['CAGR']}, MDD: {metrics_map[key]['Max Drawdown']})", color=c, linestyle=ls, linewidth=lw)

        # Drawdown curve
        eq = df_sim['equity']
        dd = (eq - eq.cummax()) / eq.cummax() * 100.0
        ax_dd.plot(df_sim.index, dd, label=key, color=c, linestyle=ls, linewidth=lw)

    ax_equity.set_yscale('log')
    ax_equity.set_title("LERS Core-Satellite Strategy Drawdown Reduction - Ablation Study (2016-2026)", fontsize=15, fontweight='bold', pad=15)
    ax_equity.set_ylabel("Portfolio Equity (USD, Log Scale)", fontsize=12)
    ax_equity.grid(True, linestyle="--", alpha=0.5)
    ax_equity.legend(loc="upper left", framealpha=0.9, fontsize=10)

    ax_dd.set_title("Historical Underwater Drawdown Comparison (%)", fontsize=12, pad=8)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=12)
    ax_dd.set_xlabel("Date", fontsize=12)
    ax_dd.grid(True, linestyle="--", alpha=0.5)
    ax_dd.axhline(-10, color='gray', linestyle=':', alpha=0.6)
    ax_dd.axhline(-20, color='red', linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs('data/results', exist_ok=True)
    plot_path = 'data/results/ablation_equity_curves.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> 净值与回撤对比图已保存至: {plot_path}")

    # 将消融实验结果明细导出为 CSV
    df_comparison.to_csv('data/results/ablation_study_summary.csv')
    print("-> 消融实验汇总已保存至: data/results/ablation_study_summary.csv")

    return df_comparison


if __name__ == "__main__":
    run_full_ablation_study()
