import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from trading_costs import TradingCostCalculator, TradingCostBreakdown


class PortfolioEngine:
    """组合决策引擎，根据模型预测概率构建投资组合并计算绩效。

    支持三状态持仓（TQQQ / SQQQ / Cash），包含真实交易成本模拟。

    Attributes:
        risk_on_threshold: Risk-On 状态阈值，prob_riskon 超过则做多 TQQQ
        risk_off_threshold: Risk-Off 状态阈值，prob_riskoff 超过则做空 SQQQ
        shares_per_trade: 每笔交易的固定股数
        use_real_costs: 是否使用真实交易成本模型
        commission_bps: 简化模型佣金费率（基点）
        slippage_bps: 简化模型滑点成本（基点）
        include_costs: 是否计入交易成本
        cost_calculator: 真实成本计算器实例
    """

    VALID_POSITIONS = {'TQQQ', 'SQQQ', 'Cash', 'PSQ', 'QQQ'}

    def __init__(
        self,
        risk_on_threshold: float = 0.6,
        risk_off_threshold: float = 0.6,
        shares_per_trade: int = 100,
        use_real_costs: bool = True,
        commission_bps: float = 5.0,
        slippage_bps: float = 10.0,
        financing_cost_daily: float = 0.0,
        include_costs: bool = True,
        disable_short: bool = False,  # 是否禁用做空（SQQQ/PSQ → Cash）
    ):
        """初始化组合决策引擎。

        Args:
            risk_on_threshold: Risk-On 概率阈值，默认 0.6
            risk_off_threshold: Risk-Off 概率阈值，默认 0.6
            shares_per_trade: 每笔交易股数，默认 100
            use_real_costs: 是否使用真实成本模型，默认 True
            commission_bps: 简化模型佣金（基点），默认 5bp
            slippage_bps: 简化模型滑点（基点），默认 10bp
            financing_cost_daily: 融资成本/日，默认 0.0
            include_costs: 是否计入成本，默认 True
        """
        self.risk_on_threshold = risk_on_threshold
        self.risk_off_threshold = risk_off_threshold
        self.shares_per_trade = shares_per_trade
        self.use_real_costs = use_real_costs
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.financing_cost_daily = financing_cost_daily
        self.include_costs = include_costs
        self.disable_short = disable_short

        # 初始化真实成本计算器
        self.cost_calculator = TradingCostCalculator(shares_per_trade=shares_per_trade)

    def load_prices(self, data_dir: str = 'data/raw') -> pd.DataFrame:
        tqqq_path = os.path.join(data_dir, 'tqqq.csv')
        sqqq_path = os.path.join(data_dir, 'sqqq.csv')
        psq_path = os.path.join(data_dir, 'psq.csv')
        qqq_path = os.path.join(data_dir, 'qqq.csv')

        tqqq_df = pd.read_csv(tqqq_path, index_col=0, parse_dates=True)
        sqqq_df = pd.read_csv(sqqq_path, index_col=0, parse_dates=True)
        psq_df = pd.read_csv(psq_path, index_col=0, parse_dates=True)
        qqq_df = pd.read_csv(qqq_path, index_col=0, parse_dates=True)

        prices_df = pd.DataFrame({
            'TQQQ': tqqq_df['Close'],
            'SQQQ': sqqq_df['Close'],
            'PSQ': psq_df['Close'],
            'QQQ': qqq_df['Close'],
            'TQQQ_Open': tqqq_df['Open'],
            'SQQQ_Open': sqqq_df['Open'],
            'PSQ_Open': psq_df['Open'],
            'QQQ_Open': qqq_df['Open'],
        })

        prices_df = prices_df.dropna()
        print(f"已加载价格数据: {len(prices_df)} 行, "
              f"{prices_df.index[0].date()} ~ {prices_df.index[-1].date()}")
        return prices_df

    def generate_signals(self, prob_df: pd.DataFrame, regime_params=None) -> pd.Series:
        required_cols = {'prob_sqqq', 'prob_psq', 'prob_cash', 'prob_qqq', 'prob_tqqq'}
        if not required_cols.issubset(prob_df.columns):
            missing = required_cols - set(prob_df.columns)
            raise KeyError(f"prob_df 缺少必需列: {missing}")

        idx_to_pos = {0: 'SQQQ', 1: 'PSQ', 2: 'Cash', 3: 'QQQ', 4: 'TQQQ'}
        col_to_idx = {'prob_sqqq': 0, 'prob_psq': 1, 'prob_cash': 2, 'prob_qqq': 3, 'prob_tqqq': 4}
        
        prob_vals = prob_df[['prob_sqqq', 'prob_psq', 'prob_cash', 'prob_qqq', 'prob_tqqq']].values

        # Regime 概率调整
        if regime_params is not None:
            prob_vals = self._apply_regime_probability_adjustment(prob_vals, regime_params)

        max_idx = np.argmax(prob_vals, axis=1)
        
        target_position = pd.Series([idx_to_pos[i] for i in max_idx], index=prob_df.index, name='target_position')

        # 如果禁用做空，把 SQQQ/PSQ 映射到 Cash
        if self.disable_short:
            target_position = target_position.replace({'SQQQ': 'Cash', 'PSQ': 'Cash'})
        
        counts = target_position.value_counts()
        print(f"目标仓位分布: TQQQ={counts.get('TQQQ', 0)}, QQQ={counts.get('QQQ', 0)}, "
              f"SQQQ={counts.get('SQQQ', 0)}, PSQ={counts.get('PSQ', 0)}, Cash={counts.get('Cash', 0)}")
        return target_position

    def _apply_regime_probability_adjustment(
        self,
        prob_vals: np.ndarray,
        regime_params,
    ) -> np.ndarray:
        """
        应用 Regime 概率调整

        调整逻辑：
        - Risk-On Regime: 放大多头类别的概率（TQQQ, QQQ），缩小空头类别
        - Risk-Off Regime: 放大空头类别的概率（SQQQ, PSQ），缩小多头类别
        - 概率缩放: probability_scale 控制整体的"确定性"

        Args:
            prob_vals: 原始概率数组 (n_samples, 5)
            regime_params: Regime 参数

        Returns:
            调整后的概率数组
        """
        adjusted = prob_vals.copy()

        # 方向调整
        risk_on_adj = getattr(regime_params, 'risk_on_threshold_adjustment', 0.0)
        risk_off_adj = getattr(regime_params, 'risk_off_threshold_adjustment', 0.0)

        # 多头列: QQQ(3), TQQQ(4)
        # 空头列: SQQQ(0), PSQ(1)
        # 中性列: Cash(2)

        # 计算调整因子
        # risk_on_adj > 0: 更难做多 → 降低多头概率
        # risk_off_adj > 0: 更难做空 → 降低空头概率
        bull_factor = 1.0 - risk_on_adj * 2  # 调整因子
        bear_factor = 1.0 - risk_off_adj * 2

        # 应用方向调整
        adjusted[:, 3] *= bull_factor  # QQQ
        adjusted[:, 4] *= bull_factor  # TQQQ
        adjusted[:, 0] *= bear_factor  # SQQQ
        adjusted[:, 1] *= bear_factor  # PSQ

        # 概率缩放（整体确定性）
        prob_scale = getattr(regime_params, 'probability_scale', 1.0)
        if prob_scale != 1.0:
            eps = 1e-6
            clipped = np.clip(adjusted, eps, 1 - eps)
            logits = np.log(clipped / (1 - clipped))
            scaled_logits = logits * prob_scale
            adjusted = 1 / (1 + np.exp(-scaled_logits))

        # 归一化
        row_sums = adjusted.sum(axis=1, keepdims=True)
        adjusted = adjusted / row_sums

        return adjusted

    def apply_execution_lag(self, target_pos: pd.Series) -> pd.Series:
        """应用执行滞后，信号后移一天（次日开盘执行）。

        T 日收盘后产生信号，T+1 日开盘执行，因此 T+1 日的实际持仓
        等于 T 日的目标持仓。

        Args:
            target_pos: 目标仓位 Series

        Returns:
            实际持仓 Series（滞后一天）
        """
        actual_pos = target_pos.shift(1)
        actual_pos.name = 'actual_position'

        first_valid = target_pos.iloc[0] if len(target_pos) > 0 else 'Cash'
        if pd.isna(actual_pos.iloc[0]):
            actual_pos.iloc[0] = 'Cash'

        print(f"执行滞后已应用: 首日持仓 = {actual_pos.iloc[0]}")
        return actual_pos

    def calculate_daily_returns(
        self,
        actual_pos: pd.Series,
        prices_df: pd.DataFrame,
    ) -> pd.Series:
        """计算每日收益率（不考虑交易成本）。

        - 持有 TQQQ: TQQQ 当日收益率
        - 持有 SQQQ: SQQQ 当日收益率
        - 持有 Cash: 0

        Args:
            actual_pos: 实际持仓 Series
            prices_df: 价格 DataFrame，包含 TQQQ、SQQQ 列

        Returns:
            每日收益率 Series
        """
        common_index = actual_pos.index.intersection(prices_df.index)
        actual_pos = actual_pos.loc[common_index]
        prices = prices_df.loc[common_index]

        pos_prev = actual_pos.shift(1).fillna('Cash')
        pos_curr = actual_pos

        ret_overnight = pd.Series(0.0, index=common_index)
        ret_daytime = pd.Series(0.0, index=common_index)

        for ticker in ['TQQQ', 'SQQQ', 'QQQ', 'PSQ']:
            if ticker in prices.columns:
                open_col = f'{ticker}_Open'
                if open_col not in prices.columns:
                    prices[open_col] = prices[ticker]
                prev_close = prices[ticker].shift(1)
                overnight = (prices[open_col] / prev_close - 1.0).fillna(0.0)
                daytime = (prices[ticker] / prices[open_col] - 1.0).fillna(0.0)
                ret_overnight[pos_prev == ticker] = overnight[pos_prev == ticker]
                ret_daytime[pos_curr == ticker] = daytime[pos_curr == ticker]

        daily_returns = (1.0 + ret_overnight) * (1.0 + ret_daytime) - 1.0
        daily_returns.name = 'daily_return'
        return daily_returns

    def apply_transaction_costs(
        self,
        daily_returns: pd.Series,
        actual_pos: pd.Series,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """应用交易成本（真实成本或简化模型）。

        成本构成：
        1. 真实成本模式：
           - 买入：佣金 + 平台使用费 + 交收费
           - 卖出：佣金 + 平台使用费 + 交收费 + 证监会规费 + 交易活动费
           - 使用当日收盘价计算成交金额
        2. 简化成本模式：
           - 换手成本：(commission_bps + slippage_bps) / 10000

        此外，不论使用哪种模式，只要持有杠杆 ETF（TQQQ 或 SQQQ），每日都会扣除融资成本。

        Args:
            daily_returns: 原始日收益率 Series
            actual_pos: 实际持仓 Series
            prices_df: 价格 DataFrame，包含 TQQQ、SQQQ 列（简化成本且非必要时可为空）

        Returns:
            扣除成本后的日收益率 Series
        """
        if not self.include_costs:
            return daily_returns.copy()

        common_index = daily_returns.index.intersection(actual_pos.index)
        if prices_df is not None:
            common_index = common_index.intersection(prices_df.index)
            
        daily_returns = daily_returns.loc[common_index]
        actual_pos = actual_pos.loc[common_index]
        if prices_df is not None:
            prices = prices_df.loc[common_index]

        cost_returns = daily_returns.copy()
        cost_returns.name = 'daily_return_cost'

        prev_pos = actual_pos.shift(1)
        turnover_mask = (actual_pos != prev_pos) & prev_pos.notna()

        if self.use_real_costs and prices_df is not None:
            # 使用真实成本模型
            total_cost = 0.0
            cost_details: Dict[str, float] = {}

            for date in turnover_mask[turnover_mask].index:
                prev_position = prev_pos.loc[date]
                new_position = actual_pos.loc[date]

                def _get_trade_price(pos_name):
                    if pos_name in ['TQQQ', 'SQQQ', 'QQQ', 'PSQ'] and pos_name in prices.columns:
                        open_col = f'{pos_name}_Open'
                        return prices.loc[date, open_col] if open_col in prices.columns else prices.loc[date, pos_name]
                    return None

                sell_price = _get_trade_price(prev_position)
                buy_price = _get_trade_price(new_position)

                # 计算成本
                trade_cost = 0.0
                if sell_price is not None:
                    sell_cost = self.cost_calculator.calc_sell_cost(sell_price)
                    trade_cost += sell_cost.total

                if buy_price is not None:
                    buy_cost = self.cost_calculator.calc_buy_cost(buy_price)
                    trade_cost += buy_cost.total

                # 将成本转换为收益率扣除
                ref_price = sell_price if sell_price is not None else buy_price
                if ref_price is not None and ref_price > 0:
                    amount = self.shares_per_trade * ref_price
                    cost_pct = trade_cost / amount
                else:
                    cost_pct = 0.0

                cost_returns.loc[date] -= cost_pct
                total_cost += trade_cost

            n_turnover = int(turnover_mask.sum())
            print(f"交易成本统计 (真实成本): 换手次数={n_turnover}, "
                  f"总成本≈${total_cost:.2f}, "
                  f"平均单次≈${total_cost/n_turnover if n_turnover > 0 else 0:.2f}")
        else:
            # 使用简化成本模型
            turnover_cost = (self.commission_bps + self.slippage_bps) / 10000.0
            cost_returns[turnover_mask] -= turnover_cost

            n_turnover = int(turnover_mask.sum())
            total_turnover_cost = n_turnover * turnover_cost
            print(f"交易成本统计 (简化模型): 换手次数={n_turnover}, "
                  f"换手总成本≈{total_turnover_cost:.4f}")

        # 应用融资成本 (仅在持有杠杆标的时)
        leveraged_mask = actual_pos.isin(['TQQQ', 'SQQQ'])
        cost_returns[leveraged_mask] -= self.financing_cost_daily

        return cost_returns

    def calculate_nav(
        self,
        daily_returns: pd.Series,
        initial_value: float = 1.0,
    ) -> pd.Series:
        """计算净值曲线。

        Args:
            daily_returns: 日收益率 Series
            initial_value: 初始净值，默认 1.0

        Returns:
            净值 Series
        """
        nav = (1.0 + daily_returns).cumprod() * initial_value
        nav.name = 'nav'
        return nav

    def run(
        self,
        prob_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        regime_params=None,
    ) -> pd.DataFrame:
        """完整运行组合引擎，生成结果 DataFrame。

        执行流程：
        1. 生成目标仓位信号
        2. 应用执行滞后
        3. 计算日收益率（无成本）
        4. 应用交易成本
        5. 计算净值曲线

        Args:
            prob_df: 预测概率 DataFrame
            prices_df: 价格 DataFrame
            regime_params: Regime 参数（可选），用于调整概率分布

        Returns:
            结果 DataFrame，包含列：
            target_position, actual_position, daily_return,
            daily_return_cost, nav, nav_cost
        """
        print("\n" + "=" * 60)
        print("组合引擎开始运行")
        print("=" * 60)

        target_pos = self.generate_signals(prob_df, regime_params=regime_params)
        actual_pos = self.apply_execution_lag(target_pos)

        daily_return = self.calculate_daily_returns(actual_pos, prices_df)
        daily_return_cost = self.apply_transaction_costs(daily_return, actual_pos, prices_df)

        nav = self.calculate_nav(daily_return)
        nav_cost = self.calculate_nav(daily_return_cost)
        nav_cost.name = 'nav_cost'

        result = pd.DataFrame({
            'target_position': target_pos,
            'actual_position': actual_pos,
            'daily_return': daily_return,
            'daily_return_cost': daily_return_cost,
            'nav': nav,
            'nav_cost': nav_cost,
        })

        result = result.dropna(subset=['actual_position'])

        print("\n" + "=" * 60)
        print(f"组合引擎运行完成，共 {len(result)} 个交易日")
        print(f"最终净值 (无成本): {nav.iloc[-1]:.4f}")
        print(f"最终净值 (有成本): {nav_cost.iloc[-1]:.4f}")
        print("=" * 60)

        return result


def calculate_max_drawdown(nav: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """计算最大回撤。

    Args:
        nav: 净值 Series

    Returns:
        (最大回撤比例, 回撤起点日期, 回撤终点日期)
    """
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    end_date = drawdown.idxmin()
    max_dd = drawdown.loc[end_date]
    start_date = nav.loc[:end_date].idxmax()
    return max_dd, start_date, end_date


def print_performance_stats(
    result: pd.DataFrame,
    initial_value: float = 1.0,
) -> None:
    """打印绩效统计信息。

    Args:
        result: 组合引擎运行结果 DataFrame
        initial_value: 初始净值
    """
    nav = result['nav']
    nav_cost = result['nav_cost']

    total_days = len(result)
    years = total_days / 252.0

    final_nav = nav.iloc[-1]
    final_nav_cost = nav_cost.iloc[-1]

    total_return = final_nav / initial_value - 1.0
    total_return_cost = final_nav_cost / initial_value - 1.0

    annual_return = (final_nav / initial_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    annual_return_cost = (final_nav_cost / initial_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_ret = result['daily_return']
    daily_ret_cost = result['daily_return_cost']

    annual_vol = daily_ret.std() * np.sqrt(252)
    annual_vol_cost = daily_ret_cost.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    sharpe_cost = annual_return_cost / annual_vol_cost if annual_vol_cost > 0 else 0.0

    max_dd, dd_start, dd_end = calculate_max_drawdown(nav)
    max_dd_cost, dd_start_cost, dd_end_cost = calculate_max_drawdown(nav_cost)

    print("\n" + "=" * 60)
    print("绩效统计")
    print("=" * 60)
    print(f"交易天数: {total_days} ({years:.2f} 年)")
    print(f"回测区间: {result.index[0].date()} ~ {result.index[-1].date()}")
    print()
    print(f"{'指标':<20s} {'无成本':>12s} {'有成本':>12s}")
    print("-" * 50)
    print(f"{'最终净值':<20s} {final_nav:>12.4f} {final_nav_cost:>12.4f}")
    print(f"{'总收益率':<20s} {total_return:>12.2%} {total_return_cost:>12.2%}")
    print(f"{'年化收益率':<20s} {annual_return:>12.2%} {annual_return_cost:>12.2%}")
    print(f"{'年化波动率':<20s} {annual_vol:>12.2%} {annual_vol_cost:>12.2%}")
    print(f"{'夏普比率':<20s} {sharpe:>12.4f} {sharpe_cost:>12.4f}")
    print(f"{'最大回撤':<20s} {max_dd:>12.2%} {max_dd_cost:>12.2%}")
    print(f"{'回撤起点':<20s} {str(dd_start.date()):>12s} {str(dd_start_cost.date()):>12s}")
    print(f"{'回撤终点':<20s} {str(dd_end.date()):>12s} {str(dd_end_cost.date()):>12s}")
    print("=" * 60)

    cost_drag = final_nav - final_nav_cost
    cost_drag_pct = cost_drag / final_nav if final_nav > 0 else 0.0
    print(f"\n成本拖累: {cost_drag:.4f} ({cost_drag_pct:.2%})")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    legacy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'legacy'))
    if os.path.exists(legacy_dir) and legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)

    try:
        from train_model import ModelTrainer
        trainer = ModelTrainer(random_state=42)
    except ImportError:
        ModelTrainer = None
        trainer = None

    print("=" * 60)
    print("加载特征数据")
    print("=" * 60)
    df = trainer.load_data('data/processed/features.parquet')

    X, y, feature_cols = trainer.prepare_data(df)

    print("\n" + "=" * 60)
    print("加载已训练的 LightGBM 模型")
    print("=" * 60)
    model = trainer.load_model('models/lgb_model.pkl')

    print("\n" + "=" * 60)
    print("全样本预测概率")
    print("=" * 60)
    prob_df = trainer.predict_proba(model, X)
    print(f"预测完成: {len(prob_df)} 行")

    print("\n" + "=" * 60)
    print("加载价格数据")
    print("=" * 60)
    engine = PortfolioEngine(
        risk_on_threshold=0.6,
        risk_off_threshold=0.6,
        shares_per_trade=100,
        use_real_costs=True,
        include_costs=True,
    )
    prices_df = engine.load_prices('data/raw')

    result = engine.run(prob_df, prices_df)

    print_performance_stats(result)

    position_counts = result['actual_position'].value_counts()
    print(f"\n实际持仓分布:")
    for pos in ['TQQQ', 'SQQQ', 'Cash']:
        cnt = position_counts.get(pos, 0)
        pct = cnt / len(result) * 100
        print(f"  {pos}: {cnt} 天 ({pct:.1f}%)")

    prev_pos = result['actual_position'].shift(1)
    turnover = (result['actual_position'] != prev_pos) & prev_pos.notna()
    print(f"\n总换手次数: {int(turnover.sum())}")
