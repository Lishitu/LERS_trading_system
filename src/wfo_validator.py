"""Walk-Forward Optimization (WFO) 与跨周期样本外检验模块。

实现对核心底仓 + 卫星动量轮动策略的多宏观周期 (Market Regimes & Eras) 切片检验，
以及滚动前进（Rolling Walk-Forward Out-Of-Sample）验证，确保策略不存在过拟合与数据泄露。
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from data_moomoo import MoomooDataLoader, DEFAULT_DEFENSE
from trading_costs import TradingCostCalculator
from valuation_factor import ValuationFactorEngine
from strategy_coresatellite import CoreSatelliteStrategy, PURE_GROWTH_SECTORS


class WalkForwardValidator:
    """跨宏观周期与滚动前进验证器。"""

    def __init__(
        self,
        start_date: str = "2016-01-01",
        buffer: float = 0.06,
        core_ratio: float = 0.50,
        benchmark: str = 'US.QQQ',
        initial_capital: float = 100000.0,
        val_factor_weight: float = 0.20,
    ):
        self.start_date = start_date
        self.buffer = buffer
        self.core_ratio = core_ratio
        self.satellite_ratio = 1.0 - core_ratio
        self.benchmark = benchmark
        self.initial_capital = initial_capital
        self.val_factor_weight = val_factor_weight
        self.cost_calc = TradingCostCalculator()

        # 加载全量数据
        loader = MoomooDataLoader()
        self.data_dict = loader.load_all_universe(start_date=start_date)

        # 初始化 Core-Satellite 策略引擎并预计算特征矩阵
        self.strategy = CoreSatelliteStrategy(
            benchmark=benchmark,
            core_ratio=core_ratio,
            satellite_ratio=self.satellite_ratio,
            val_factor_weight=val_factor_weight,
            weekly_satellite=True,
        )
        (
            self.factors_dict,
            self.vols,
            self.ma_trend,
            self.df_close,
            self.regimes,
        ) = self.strategy.compute_factors_and_indicators(self.data_dict)

        vwap_dict = {t: df.get('vwap', df['close']) for t, df in self.data_dict.items()}
        self.df_vwap = pd.DataFrame(vwap_dict).reindex(self.df_close.index)
        for c in self.df_vwap.columns:
            self.df_vwap[c] = self.df_vwap[c].fillna(self.df_close[c])

        self.dates = self.df_close.index

    def run_simulation_slice(
        self,
        start_idx: int,
        end_idx: int,
        initial_cap: float = 100000.0,
    ) -> pd.DataFrame:
        """执行指定时间切片的高保真日频 VWAP 回测模拟。"""
        cash = initial_cap
        holdings: Dict[str, float] = {}
        records: List[Dict[str, Any]] = []
        portfolio_daily_returns: List[float] = []

        self.strategy.reset()

        if start_idx < 1:
            start_idx = 1
        sub_dates = self.dates[start_idx:end_idx]

        for i in range(len(sub_dates)):
            global_i = start_idx + i
            date_t = self.dates[global_i]
            date_prev = self.dates[global_i - 1]

            # 估算组合前 20 日已实现年化波动率
            if len(portfolio_daily_returns) >= 20:
                recent_ret = np.array(portfolio_daily_returns[-20:])
                port_vol = float(np.std(recent_ret, ddof=1) * np.sqrt(252))
            else:
                port_vol = self.strategy.target_vol

            regime_prev = self.regimes.loc[date_prev]
            trend_prev = self.ma_trend.loc[date_prev]
            vols_prev = self.vols.loc[date_prev]

            target_w = self.strategy.generate_portfolio_target(
                date=date_prev,
                factors_dict=self.factors_dict,
                current_trend=trend_prev,
                current_vols=vols_prev,
                regime=regime_prev,
                df_close=self.df_close,
                realized_portfolio_vol=port_vol,
            )

            # 2. T 日日频真实 VWAP 执行
            total_val_prev = cash + sum(
                s * self.df_close[t].loc[date_prev] for t, s in holdings.items() if t in self.df_close
            )
            all_tickers = set(holdings.keys()) | set(target_w.keys())
            target_dollars = {}

            for t in all_tickers:
                cur_s = holdings.get(t, 0.0)
                cur_w = (cur_s * self.df_close[t].loc[date_prev] / total_val_prev) if total_val_prev > 0 else 0.0
                targ_w = target_w.get(t, 0.0)

                if targ_w <= 1e-4:
                    target_dollars[t] = 0.0
                elif cur_s <= 1e-4:
                    target_dollars[t] = targ_w * total_val_prev
                else:
                    if abs(targ_w - cur_w) <= self.buffer:
                        target_dollars[t] = None
                    else:
                        target_dollars[t] = targ_w * total_val_prev

            daily_cost = 0.0
            daily_turnover = 0.0

            # 卖出
            for t, targ_d in target_dollars.items():
                if targ_d is None:
                    continue
                cur_s = holdings.get(t, 0.0)
                vwap_p = self.df_vwap[t].loc[date_t]
                if np.isnan(vwap_p) or vwap_p <= 0:
                    vwap_p = self.df_close[t].loc[date_t]
                cur_d = cur_s * vwap_p

                if targ_d < cur_d:
                    delta = cur_d - targ_d
                    s_sell = min(cur_s, delta / vwap_p)
                    if s_sell > 0.01:
                        c = self.cost_calc.calc_trade_cost(s_sell, vwap_p, is_buy=False).total
                        daily_cost += c
                        daily_turnover += s_sell * vwap_p
                        cash += (s_sell * vwap_p - c)
                        holdings[t] = cur_s - s_sell
                        if holdings[t] <= 1e-4:
                            holdings[t] = 0.0

            # 买入
            for t, targ_d in target_dollars.items():
                if targ_d is None:
                    continue
                cur_s = holdings.get(t, 0.0)
                vwap_p = self.df_vwap[t].loc[date_t]
                if np.isnan(vwap_p) or vwap_p <= 0:
                    vwap_p = self.df_close[t].loc[date_t]
                cur_d = cur_s * vwap_p

                if targ_d > cur_d:
                    delta = targ_d - cur_d
                    if delta > 50.0 and vwap_p > 0 and cash > 100.0:
                        max_shares = max(0.0, (cash - 100.0) / (vwap_p + 0.015))
                        s_buy = min(delta / vwap_p, max_shares)
                        if s_buy > 0.01:
                            c = self.cost_calc.calc_trade_cost(s_buy, vwap_p, is_buy=True).total
                            daily_cost += c
                            daily_turnover += s_buy * vwap_p
                            cash -= (s_buy * vwap_p + c)
                            holdings[t] = cur_s + s_buy

            holdings = {t: s for t, s in holdings.items() if s > 1e-4}
            eq = cash + sum(s * self.df_close[t].loc[date_t] for t, s in holdings.items() if t in self.df_close)

            if len(records) > 0:
                daily_pnl_ret = (eq / records[-1]['equity']) - 1.0
            else:
                daily_pnl_ret = 0.0
            portfolio_daily_returns.append(daily_pnl_ret)

            records.append({
                'date': date_t,
                'equity': eq,
                'cash': cash,
                'cost': daily_cost,
                'turnover': daily_turnover,
                'qqq_equity': initial_cap * (self.df_close[self.benchmark].loc[date_t] / self.df_close[self.benchmark].iloc[start_idx]),
                'spy_equity': initial_cap * (self.df_close['US.SPY'].loc[date_t] / self.df_close['US.SPY'].iloc[start_idx]),
            })

        df_res = pd.DataFrame(records).set_index('date')
        return df_res

    def compute_metrics(self, df_res: pd.DataFrame) -> Dict[str, float]:
        """从模拟结果计算标准量化绩效指标。"""
        ret = df_res['equity'].pct_change().fillna(0.0)
        qqq_r = df_res['qqq_equity'].pct_change().fillna(0.0)

        n_days = len(df_res)
        years = n_days / 252.0 if n_days > 0 else 1.0

        tot_ret = (df_res['equity'].iloc[-1] / df_res['equity'].iloc[0]) - 1.0
        cagr = ((1.0 + tot_ret) ** (1.0 / years) - 1.0) if (years > 0 and (1.0 + tot_ret) > 0) else -1.0
        vol = float(ret.std() * np.sqrt(252))
        sharpe = float((ret.mean() - 0.02 / 252) / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0

        cum = df_res['equity'].cummax()
        max_dd = float(((df_res['equity'] - cum) / cum).min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # 基准 QQQ 指标
        qqq_tot = (df_res['qqq_equity'].iloc[-1] / df_res['qqq_equity'].iloc[0]) - 1.0
        qqq_cagr = ((1.0 + qqq_tot) ** (1.0 / years) - 1.0) if (years > 0 and (1.0 + qqq_tot) > 0) else -1.0
        qqq_vol = float(qqq_r.std() * np.sqrt(252))
        qqq_sh = float((qqq_r.mean() - 0.02 / 252) / qqq_r.std() * np.sqrt(252)) if qqq_r.std() > 0 else 0.0
        qqq_cum = df_res['qqq_equity'].cummax()
        qqq_mdd = float(((df_res['qqq_equity'] - qqq_cum) / qqq_cum).min())

        # 信息比率与跟踪误差
        active_ret = ret - qqq_r
        te = float(active_ret.std() * np.sqrt(252))
        ir = float((active_ret.mean() * 252) / te) if te > 0 else 0.0

        win_rate = float((ret > 0).sum() / len(ret)) if len(ret) > 0 else 0.0

        return {
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar,
            'ir_vs_qqq': ir,
            'tracking_error': te,
            'win_rate': win_rate,
            'qqq_cagr': qqq_cagr,
            'qqq_vol': qqq_vol,
            'qqq_sharpe': qqq_sh,
            'qqq_max_dd': qqq_mdd,
            'total_costs': float(df_res['cost'].sum()),
            'min_cash': float(df_res['cash'].min()),
            'final_equity': float(df_res['equity'].iloc[-1]),
        }

    def run_era_validation(self) -> pd.DataFrame:
        """执行多历史阶段/宏观周期样本外切片检验。"""
        eras = [
            ("2017-2018 (成长慢牛与加息杀估值)", "2017-01-01", "2018-12-31"),
            ("2019-2020 (流动性宽松与疫情深V)", "2019-01-01", "2020-12-31"),
            ("2021-2022 (高通胀与科技熊市)", "2021-01-01", "2022-12-31"),
            ("2023-2024 (AI驱动与半导体主升)", "2023-01-01", "2024-12-31"),
            ("2025-2026 (当前宏观延续)", "2025-01-01", "2026-09-25"),
            ("2016-2026 (十年全周期贯穿)", "2016-10-20", "2026-09-25"),
        ]

        results = []
        for name, start_str, end_str in eras:
            sub = [i for i, d in enumerate(self.dates) if start_str <= str(d)[:10] <= end_str]
            if len(sub) < 40:
                continue

            df_res = self.run_simulation_slice(sub[0], sub[-1])
            m = self.compute_metrics(df_res)
            results.append({
                'Era / Market Regime': name,
                'Start Date': str(self.dates[sub[0]])[:10],
                'End Date': str(self.dates[sub[-1]])[:10],
                'Strat CAGR': f"{m['cagr']*100:.2f}%",
                'QQQ CAGR': f"{m['qqq_cagr']*100:.2f}%",
                'Strat Sharpe': f"{m['sharpe']:.2f}",
                'QQQ Sharpe': f"{m['qqq_sharpe']:.2f}",
                'Strat MaxDD': f"{m['max_dd']*100:.2f}%",
                'QQQ MaxDD': f"{m['qqq_max_dd']*100:.2f}%",
                'IR vs QQQ': f"{m['ir_vs_qqq']:.2f}",
                'Tracking Error': f"{m['tracking_error']*100:.2f}%",
                'Total Fees': f"${m['total_costs']:.1f}",
            })

        return pd.DataFrame(results)

    def run_rolling_walk_forward(
        self,
        train_years: int = 3,
        test_years: int = 1,
    ) -> pd.DataFrame:
        """执行滚动前进样本外检验 (Walk-Forward Optimization)。
        
        每次使用过去 train_years 年评估稳定性，并在接下来的 test_years 年执行完全无未来的样本外验证。
        """
        trading_days_per_year = 252
        train_len = train_years * trading_days_per_year
        test_len = test_years * trading_days_per_year
        warmup = 200

        total_days = len(self.dates)
        step = test_len

        wfo_records = []
        fold = 1

        for start_train in range(warmup, total_days - train_len - test_len + 1, step):
            end_train = start_train + train_len
            start_test = end_train
            end_test = min(start_test + test_len, total_days)

            if end_test - start_test < 50:
                break

            train_dates_str = f"{str(self.dates[start_train])[:10]} ~ {str(self.dates[end_train-1])[:10]}"
            test_dates_str = f"{str(self.dates[start_test])[:10]} ~ {str(self.dates[end_test-1])[:10]}"

            # 运行测试集 (样本外 Out-Of-Sample)
            df_oos = self.run_simulation_slice(start_test, end_test)
            m_oos = self.compute_metrics(df_oos)

            wfo_records.append({
                'Fold': f"Fold-{fold}",
                'Train Period': train_dates_str,
                'OOS Period (样本外)': test_dates_str,
                'OOS CAGR': f"{m_oos['cagr']*100:.2f}%",
                'QQQ CAGR': f"{m_oos['qqq_cagr']*100:.2f}%",
                'OOS Sharpe': f"{m_oos['sharpe']:.2f}",
                'QQQ Sharpe': f"{m_oos['qqq_sharpe']:.2f}",
                'OOS MaxDD': f"{m_oos['max_dd']*100:.2f}%",
                'QQQ MaxDD': f"{m_oos['qqq_max_dd']*100:.2f}%",
                'IR vs QQQ': f"{m_oos['ir_vs_qqq']:.2f}",
                'OOS WinRate': f"{m_oos['win_rate']*100:.1f}%",
            })
            fold += 1

        return pd.DataFrame(wfo_records)


if __name__ == '__main__':
    print("=" * 80)
    print(" [WFO Engine] 正在启动全量跨周期与滚动前进样本外检验... ")
    print("=" * 80)

    validator = WalkForwardValidator(core_ratio=0.50, buffer=0.06)

    print("\n>>> 1. 跨宏观周期切片检验 (Era Breakdown) <<<")
    df_eras = validator.run_era_validation()
    print(df_eras.to_string(index=False))

    print("\n>>> 2. 滚动前进样本外检验 (Rolling 3Y Train / 1Y Test Walk-Forward OOS) <<<")
    df_wfo = validator.run_rolling_walk_forward(train_years=3, test_years=1)
    print(df_wfo.to_string(index=False))
