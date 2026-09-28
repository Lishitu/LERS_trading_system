"""日频 VWAP 高保真回测引擎。

特性：
1. 严格无未来函数：T 日收盘后计算截面动量与目标权重，T+1 日以真实全天 VWAP 执行
2. 5% 调仓缓冲区抑制非必要微调
3. 集成真实交易成本 (Moomoo 美股费率结构)
4. 多基准对比：SPY, QQQ, Buy&Hold TQQQ, 1x 行业等权
5. 完备绩效归因与评估指标 (CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_moomoo import DEFAULT_SECTORS, DEFAULT_DEFENSE, DEFAULT_BENCHMARKS
from strategy_momentum import MomentumRegimeStrategy
from portfolio_allocator import PortfolioAllocator
from trading_costs import TradingCostCalculator


class DailyVWAPBacktester:
    """日频 VWAP 回测器。"""

    def __init__(
        self,
        strategy: Optional[MomentumRegimeStrategy] = None,
        allocator: Optional[PortfolioAllocator] = None,
        cost_calculator: Optional[TradingCostCalculator] = None,
        initial_capital: float = 100000.0,
        warmup_period: int = 200,
    ):
        self.strategy = strategy or MomentumRegimeStrategy()
        self.allocator = allocator or PortfolioAllocator()
        self.cost_calculator = cost_calculator or TradingCostCalculator()
        self.initial_capital = initial_capital
        self.warmup_period = warmup_period

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        win_rates_df: Optional[pd.DataFrame] = None,
        min_win_rate: float = 0.50,
    ) -> pd.DataFrame:
        """执行完整回测模拟。

        Args:
            data_dict: 各标的代码对应的日K DataFrame，须包含 'close' 与 'vwap'
            win_rates_df: (可选) Meta-Labeling 各标的纯样本外预测胜率矩阵
            min_win_rate: (可选) 胜率质检准入门槛，默认 0.50

        Returns:
            回测每日绩效与持仓记录 DataFrame
        """
        # 1. 提取收盘价与 VWAP 矩阵
        close_dict = {ticker: df['close'] for ticker, df in data_dict.items()}
        vwap_dict = {ticker: df['vwap'] for ticker, df in data_dict.items()}

        df_close = pd.DataFrame(close_dict).dropna(subset=['US.SPY'])
        df_vwap = pd.DataFrame(vwap_dict).reindex(df_close.index)

        # 填补个别非交易日或缺失项为 close
        for col in df_vwap.columns:
            df_vwap[col] = df_vwap[col].fillna(df_close[col])

        # 2. 预先计算策略信号与指标（严格基于当时及历史收盘价）
        scores, vols, ma_trend = self.strategy.compute_asset_indicators(df_close)
        regimes = self.strategy.get_market_regimes(df_close['US.SPY'])

        dates = df_close.index
        n_days = len(dates)

        if n_days <= self.warmup_period:
            raise ValueError(f"数据总长度 {n_days} 天不足以支持预热期 {self.warmup_period} 天")

        # 3. 模拟账户状态初始化
        cash = self.initial_capital
        holdings: Dict[str, float] = {}  # ticker -> shares
        current_weights: Dict[str, float] = {}

        # 跟踪记录
        records = []
        
        # 预计算基准的初始价格
        first_sim_idx = self.warmup_period
        spy_init_price = df_close['US.SPY'].iloc[first_sim_idx]
        qqq_init_price = df_close['US.QQQ'].iloc[first_sim_idx]
        tqqq_init_price = df_close['US.TQQQ'].iloc[first_sim_idx]

        # 4. 逐日模拟撮合循环
        # 注意：在第 t 天 (date_t)，根据第 t-1 天收盘后生成的信号，以第 t 天的 VWAP 进行买卖
        for i in range(first_sim_idx, n_days):
            date_t = dates[i]
            date_prev = dates[i - 1]

            # ---------------------------
            # (A) T-1 日收盘后生成目标配置
            # ---------------------------
            regime_prev = regimes.loc[date_prev]
            scores_prev = scores.loc[date_prev]
            trend_prev = ma_trend.loc[date_prev]
            vols_prev = vols.loc[date_prev]

            selected_assets = self.strategy.generate_daily_signals(
                date=date_prev,
                current_scores=scores_prev,
                current_trend=trend_prev,
                current_vols=vols_prev,
                regime=regime_prev,
            )

            # 获取 T-1 日各标的预测胜率
            win_rates_prev = None
            if win_rates_df is not None and date_prev in win_rates_df.index:
                win_rates_prev = win_rates_df.loc[date_prev].to_dict()

            raw_target_weights = self.allocator.compute_target_weights(
                selected_assets,
                win_rates=win_rates_prev,
                min_win_rate=min_win_rate,
            )

            # ---------------------------
            # (B) T 日盘中以 VWAP 执行调仓
            # ---------------------------
            # 计算盘前估算的当前权重
            prev_close_vals = {
                t: shares * df_close[t].loc[date_prev]
                for t, shares in holdings.items()
                if shares > 0 and t in df_close
            }
            total_val_prev = cash + sum(prev_close_vals.values())

            # 决定各标的是否需要调仓
            all_active_tickers = set(holdings.keys()) | set(raw_target_weights.keys())
            target_dollars = {}

            for t in all_active_tickers:
                cur_shares = holdings.get(t, 0.0)
                cur_w = (cur_shares * df_close[t].loc[date_prev] / total_val_prev) if total_val_prev > 0 else 0.0
                targ_w = raw_target_weights.get(t, 0.0)

                # 情况 1: 退出标的 -> 彻底清仓
                if targ_w <= 1e-4:
                    target_dollars[t] = 0.0
                # 情况 2: 新入选标的 -> 买入至目标金额
                elif cur_shares <= 1e-4:
                    target_dollars[t] = targ_w * total_val_prev
                # 情况 3: 持续持仓标的 -> 检查是否超出 5% 缓冲区
                else:
                    if abs(targ_w - cur_w) <= self.allocator.rebalance_buffer:
                        # 在容忍度以内，完全不产生订单（锁定当前股数）
                        target_dollars[t] = None
                    else:
                        target_dollars[t] = targ_w * total_val_prev

            daily_cost = 0.0
            daily_turnover = 0.0

            # 优先执行卖出，回收现金
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
                        cost_breakdown = self.cost_calculator.calc_trade_cost(
                            shares=shares_to_sell, price=vwap_price, is_buy=False
                        )
                        daily_cost += cost_breakdown.total
                        daily_turnover += shares_to_sell * vwap_price

                        cash += (shares_to_sell * vwap_price - cost_breakdown.total)
                        holdings[t] = cur_shares - shares_to_sell
                        if holdings[t] <= 1e-4:
                            holdings[t] = 0.0

            # 随后执行买入
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
                            cost_breakdown = self.cost_calculator.calc_trade_cost(
                                shares=shares_to_buy, price=vwap_price, is_buy=True
                            )
                            daily_cost += cost_breakdown.total
                            daily_turnover += shares_to_buy * vwap_price

                            cash -= (shares_to_buy * vwap_price + cost_breakdown.total)
                            holdings[t] = cur_shares + shares_to_buy

            # 清理为 0 的仓位
            holdings = {t: s for t, s in holdings.items() if s > 1e-4}

            # ---------------------------
            # (C) T 日收盘结算账户总资产
            # ---------------------------
            holdings_value = sum(
                s * df_close[t].loc[date_t] for t, s in holdings.items() if t in df_close
            )
            total_equity = cash + holdings_value

            # 基准表现跟踪
            spy_equity = self.initial_capital * (df_close['US.SPY'].loc[date_t] / spy_init_price)
            qqq_equity = self.initial_capital * (df_close['US.QQQ'].loc[date_t] / qqq_init_price)
            tqqq_equity = self.initial_capital * (df_close['US.TQQQ'].loc[date_t] / tqqq_init_price)

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
        df_results['cum_return'] = df_results['equity'] / self.initial_capital - 1.0
        df_results['spy_cum_return'] = df_results['spy_equity'] / self.initial_capital - 1.0
        df_results['qqq_cum_return'] = df_results['qqq_equity'] / self.initial_capital - 1.0
        df_results['tqqq_cum_return'] = df_results['tqqq_equity'] / self.initial_capital - 1.0

        return df_results

    @staticmethod
    def calculate_performance_metrics(df_results: pd.DataFrame, rf: float = 0.02) -> Dict[str, Any]:
        """计算核心回测绩效指标。"""
        equity = df_results['equity']
        daily_ret = df_results['daily_return']
        total_days = len(df_results)
        years = total_days / 252.0

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        ann_vol = daily_ret.std() * np.sqrt(252)
        excess_ret = daily_ret - (rf / 252.0)
        sharpe = (excess_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

        annual_excess_ret = excess_ret.mean() * 252.0
        rf_daily = rf / 252.0
        downside_diff = np.minimum(daily_ret - rf_daily, 0.0)
        downside_std_ann = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)) if len(daily_ret) > 0 else 0.0
        sortino = (annual_excess_ret / downside_std_ann) if downside_std_ann > 0 else 0.0

        # 回撤
        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max
        max_drawdown = drawdown.min()

        calmar = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0
        win_rate = (daily_ret > 0).mean()

        # 信息比率 (Information Ratio vs QQQ)
        if 'qqq_equity' in df_results.columns:
            qqq_ret = df_results['qqq_equity'].pct_change(fill_method=None).fillna(0.0)
            active_ret = daily_ret - qqq_ret
            tracking_error = active_ret.std() * np.sqrt(252)
            ir = (active_ret.mean() * 252.0) / tracking_error if tracking_error > 0 else 0.0
        else:
            ir = 0.0
            tracking_error = 0.0

        total_turnover = df_results['turnover'].sum()
        avg_annual_turnover = (total_turnover / equity.mean()) / years if years > 0 else 0.0
        total_costs = df_results['cost'].sum()

        return {
            'Years': round(years, 2),
            'Total Return': f"{round(total_return * 100, 2)}%",
            'CAGR': f"{round(cagr * 100, 2)}%",
            'Annual Volatility': f"{round(ann_vol * 100, 2)}%",
            'Sharpe Ratio': round(sharpe, 2),
            'Sortino Ratio': round(sortino, 2),
            'Information Ratio (vs QQQ)': round(ir, 2),
            'Tracking Error (vs QQQ)': f"{round(tracking_error * 100, 2)}%",
            'Max Drawdown': f"{round(max_drawdown * 100, 2)}%",
            'Calmar Ratio': round(calmar, 2),
            'Daily Win Rate': f"{round(win_rate * 100, 2)}%",
            'Annual Turnover': f"{round(avg_annual_turnover * 100, 1)}%",
            'Total Trading Costs': f"${round(total_costs, 2)}",
            'Final Equity': f"${round(equity.iloc[-1], 2)}",
        }
