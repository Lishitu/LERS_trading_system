"""绩效评估模块，用于评估策略的绩效表现。

提供组合绩效指标、交易指标、分类指标计算，
以及基准策略对比和完整报告输出。
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_max_drawdown(nav: pd.Series) -> Tuple[float, int]:
    """计算最大回撤及持续天数。

    Args:
        nav: 净值序列，索引为日期

    Returns:
        (最大回撤比例, 最大回撤持续天数)
    """
    if len(nav) == 0:
        return 0.0, 0

    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()

    max_duration = 0
    current_duration = 0
    in_drawdown = False

    for i in range(len(nav)):
        if nav.iloc[i] < peak.iloc[i]:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1
        else:
            if in_drawdown:
                if current_duration > max_duration:
                    max_duration = current_duration
                in_drawdown = False
                current_duration = 0

    if in_drawdown and current_duration > max_duration:
        max_duration = current_duration

    return float(max_dd), max_duration


def calculate_cagr(nav: pd.Series, days: int) -> float:
    """计算复合年化增长率（CAGR）。

    Args:
        nav: 净值序列
        days: 总交易日数

    Returns:
        年化复合增长率
    """
    if len(nav) == 0 or days == 0:
        return 0.0

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    years = days / 252.0
    if years <= 0:
        return 0.0

    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    return float(cagr)


class PerformanceEvaluator:
    """绩效评估器，计算策略绩效指标并生成完整报告。

    Attributes:
        risk_free_rate: 无风险利率
        wf_results: Walk-Forward 结果 DataFrame
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """初始化绩效评估器。

        Args:
            risk_free_rate: 无风险利率，默认 0
        """
        self.risk_free_rate = risk_free_rate
        self.wf_results: Optional[pd.DataFrame] = None

    def load_walkforward_results(self, path: str) -> pd.DataFrame:
        """加载 Walk-Forward 结果。

        Args:
            path: Walk-Forward 结果文件路径（parquet 格式）

        Returns:
            加载的结果 DataFrame

        Raises:
            FileNotFoundError: 当文件不存在时
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Walk-Forward 结果文件不存在: {path}")

        self.wf_results = pd.read_parquet(path, engine='pyarrow')
        print(f"已加载 Walk-Forward 结果: {len(self.wf_results)} 行, "
              f"{len(self.wf_results.columns)} 列")
        print(f"时间区间: {self.wf_results.index[0].date()} ~ "
              f"{self.wf_results.index[-1].date()}")
        return self.wf_results

    def calc_portfolio_metrics(
        self,
        daily_returns: pd.Series,
        nav: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """计算组合绩效指标。

        Args:
            daily_returns: 日收益率序列
            nav: 净值序列，为 None 时根据 daily_returns 计算

        Returns:
            包含绩效指标的字典：
            - total_return: 总收益率
            - cagr: 复合年化增长率
            - annual_return: 年化收益率
            - annual_volatility: 年化波动率
            - sharpe_ratio: 夏普比率
            - sortino_ratio: 索提诺比率
            - calmar_ratio: 卡玛比率
            - max_drawdown: 最大回撤
            - max_drawdown_duration: 最大回撤持续天数
        """
        if len(daily_returns) == 0:
            return {
                'total_return': 0.0,
                'cagr': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
            }

        if nav is None:
            nav = (1.0 + daily_returns).cumprod()

        days = len(daily_returns)
        years = days / 252.0

        total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
        cagr = calculate_cagr(nav, days)
        annual_return = cagr

        daily_vol = daily_returns.std()
        annual_volatility = daily_vol * np.sqrt(252)

        excess_return = annual_return - self.risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0.0

        rf_daily = self.risk_free_rate / 252.0
        downside_diff = np.minimum(daily_returns - rf_daily, 0.0)
        downside_vol = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)) if len(daily_returns) > 0 else 0.0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0.0

        max_dd, max_dd_duration = calculate_max_drawdown(nav)
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_dd),
            'max_drawdown_duration': int(max_dd_duration),
        }

    def calc_trading_metrics(
        self,
        positions: pd.Series,
        daily_returns: pd.Series,
    ) -> Dict[str, float]:
        """计算交易指标。

        Args:
            positions: 持仓序列，值为 'TQQQ' / 'SQQQ' / 'Cash'
            daily_returns: 日收益率序列

        Returns:
            包含交易指标的字典：
            - total_trades: 总交易次数（换手次数）
            - turnover_rate: 年均换手率（次/年）
            - avg_holding_period: 平均持仓期（天）
            - win_rate: 胜率（盈利交易占比，按每次开仓到平仓算）
        """
        if len(positions) == 0:
            return {
                'total_trades': 0,
                'turnover_rate': 0.0,
                'avg_holding_period': 0.0,
                'win_rate': 0.0,
            }

        common_idx = positions.index.intersection(daily_returns.index)
        positions = positions.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]

        prev_pos = positions.shift(1)
        turnover_mask = (positions != prev_pos) & prev_pos.notna()
        total_trades = int(turnover_mask.sum())

        days = len(positions)
        years = days / 252.0
        turnover_rate = total_trades / years if years > 0 else 0.0

        trades = []
        current_pos = None
        entry_idx = 0

        for i in range(len(positions)):
            pos = positions.iloc[i]
            if current_pos is None:
                current_pos = pos
                entry_idx = i
            elif pos != current_pos:
                period_returns = daily_returns.iloc[entry_idx:i]
                trade_return = (1.0 + period_returns).prod() - 1.0
                trades.append({
                    'position': current_pos,
                    'return': trade_return,
                    'duration': i - entry_idx,
                })
                current_pos = pos
                entry_idx = i

        if current_pos is not None and entry_idx < len(positions):
            period_returns = daily_returns.iloc[entry_idx:]
            trade_return = (1.0 + period_returns).prod() - 1.0
            trades.append({
                'position': current_pos,
                'return': trade_return,
                'duration': len(positions) - entry_idx,
            })

        non_cash_trades = [
            t for t in trades
            if str(t['position']).replace('US.', '') not in ('Cash', 'BIL', 'BOXX')
        ]
        if len(non_cash_trades) > 0:
            avg_holding_period = np.mean([t['duration'] for t in non_cash_trades])
            winning_trades = [t for t in non_cash_trades if t['return'] > 0]
            win_rate = len(winning_trades) / len(non_cash_trades)
        else:
            avg_holding_period = 0.0
            win_rate = 0.0

        return {
            'total_trades': total_trades,
            'turnover_rate': float(turnover_rate),
            'avg_holding_period': float(avg_holding_period),
            'win_rate': float(win_rate),
        }

    def calc_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """计算分类指标。

        Args:
            y_true: 真实标签数组（0/1/2/3/4）
            y_pred: 预测标签数组（0/1/2/3/4）

        Returns:
            包含分类指标的字典：
            - accuracy: 准确率
            - precision_macro: 宏平均精确率
            - recall_macro: 宏平均召回率
            - f1_macro: 宏平均 F1
            - confusion_matrix: 5x5 混淆矩阵（numpy array）
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'confusion_matrix': np.zeros((5, 5), dtype=int),
            }

        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

        return {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'confusion_matrix': cm,
        }

    def run_benchmarks(
        self,
        prices_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> Dict[str, Dict[str, pd.Series]]:
        """运行三个基准策略。

        基准策略：
        - Benchmark A: Buy & Hold TQQQ
        - Benchmark B: MA200 择时（QQQ收盘价 > MA200 → TQQQ，否则 Cash）
        - Benchmark C: MA20/60 金叉（MA20 > MA60 → TQQQ，否则 Cash）

        Args:
            prices_df: 价格 DataFrame，需包含 QQQ 和 TQQQ 的 Close 列
            dates: 回测日期索引

        Returns:
            基准策略结果字典，每个基准包含 daily_returns 和 nav
        """
        common_dates = prices_df.index.intersection(dates)
        prices = prices_df.loc[common_dates].copy()

        qqq_close = prices['QQQ']

        ma200 = qqq_close.rolling(window=200).mean()
        ma20 = qqq_close.rolling(window=20).mean()
        ma60 = qqq_close.rolling(window=60).mean()

        target_b = pd.Series('Cash', index=common_dates)
        target_b[qqq_close > ma200] = 'TQQQ'

        target_c = pd.Series('Cash', index=common_dates)
        target_c[ma20 > ma60] = 'TQQQ'

        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from portfolio_engine import PortfolioEngine
        engine = PortfolioEngine(include_costs=True)

        # Benchmark A
        target_a = pd.Series('TQQQ', index=common_dates)
        actual_a = engine.apply_execution_lag(target_a)
        ret_a = engine.calculate_daily_returns(actual_a, prices)
        cost_ret_a = engine.apply_transaction_costs(ret_a, actual_a, prices)
        nav_a = (1.0 + cost_ret_a).cumprod()

        # Benchmark B
        actual_b = engine.apply_execution_lag(target_b)
        ret_b = engine.calculate_daily_returns(actual_b, prices)
        cost_ret_b = engine.apply_transaction_costs(ret_b, actual_b, prices)
        nav_b = (1.0 + cost_ret_b).cumprod()

        # Benchmark C
        actual_c = engine.apply_execution_lag(target_c)
        ret_c = engine.calculate_daily_returns(actual_c, prices)
        cost_ret_c = engine.apply_transaction_costs(ret_c, actual_c, prices)
        nav_c = (1.0 + cost_ret_c).cumprod()

        benchmarks = {
            'Benchmark_A_BuyHold_TQQQ': {
                'daily_returns': cost_ret_a,
                'nav': nav_a,
            },
            'Benchmark_B_MA200': {
                'daily_returns': cost_ret_b,
                'nav': nav_b,
            },
            'Benchmark_C_MA20_60': {
                'daily_returns': cost_ret_c,
                'nav': nav_c,
            },
        }

        return benchmarks

    def print_report(
        self,
        strategy_metrics: Dict[str, float],
        trading_metrics: Dict[str, float],
        classification_metrics: Dict,
        benchmark_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """打印完整绩效报告。

        Args:
            strategy_metrics: 策略组合绩效指标
            trading_metrics: 交易指标
            classification_metrics: 分类指标
            benchmark_metrics: 基准策略绩效指标字典
        """
        print("\n" + "=" * 80)
        print("绩效评估报告")
        print("=" * 80)

        print("\n" + "-" * 80)
        print("策略绩效 vs 基准对比")
        print("-" * 80)

        all_metrics = {'策略': strategy_metrics}
        all_metrics.update(benchmark_metrics)

        metric_names = [
            ('total_return', '总收益率'),
            ('cagr', '年化收益率(CAGR)'),
            ('annual_volatility', '年化波动率'),
            ('sharpe_ratio', '夏普比率'),
            ('sortino_ratio', '索提诺比率'),
            ('calmar_ratio', '卡玛比率'),
            ('max_drawdown', '最大回撤'),
            ('max_drawdown_duration', '最大回撤持续(天)'),
        ]

        header = f"{'指标':<22s}"
        for name in all_metrics.keys():
            header += f" {name:>18s}"
        print(header)
        print("-" * 80)

        for key, label in metric_names:
            line = f"{label:<22s}"
            for metrics in all_metrics.values():
                val = metrics.get(key, 0.0)
                if key in ('max_drawdown_duration',):
                    line += f" {val:>18d}"
                elif key in ('total_return', 'cagr', 'annual_volatility', 'max_drawdown'):
                    line += f" {val:>17.2%}"
                else:
                    line += f" {val:>18.4f}"
            print(line)

        print("\n" + "-" * 80)
        print("交易指标")
        print("-" * 80)
        print(f"  总交易次数:        {trading_metrics['total_trades']} 次")
        print(f"  年均换手率:        {trading_metrics['turnover_rate']:.2f} 次/年")
        print(f"  平均持仓期:        {trading_metrics['avg_holding_period']:.2f} 天")
        print(f"  胜率:              {trading_metrics['win_rate']:.2%}")

        print("\n" + "-" * 80)
        print("分类指标")
        print("-" * 80)
        print(f"  准确率:            {classification_metrics['accuracy']:.4f} "
              f"({classification_metrics['accuracy']*100:.2f}%)")
        print(f"  宏平均精确率:      {classification_metrics['precision_macro']:.4f}")
        print(f"  宏平均召回率:      {classification_metrics['recall_macro']:.4f}")
        print(f"  宏平均 F1:         {classification_metrics['f1_macro']:.4f}")

        print("\n混淆矩阵:")
        cm = classification_metrics['confusion_matrix']
        labels = ['SQQQ(0)', 'PSQ(1)', 'Cash(2)', 'QQQ(3)', 'TQQQ(4)']
        header = f"{'预测\\\\真实':<12s}"
        for l in labels:
            header += f" {l:>10s}"
        print(header)
        for i, label in enumerate(labels):
            row = f"{label:<12s}"
            for j in range(5):
                row += f" {cm[i, j]:>10d}"
            print(row)

        print("\n" + "=" * 80)


def _load_prices(data_dir: str = 'data/raw') -> pd.DataFrame:
    """加载价格数据。

    Args:
        data_dir: 原始数据目录

    Returns:
        包含 QQQ 和 TQQQ/SQQQ 收盘及开盘价的 DataFrame
    """
    qqq_path = os.path.join(data_dir, 'qqq.csv')
    tqqq_path = os.path.join(data_dir, 'tqqq.csv')
    sqqq_path = os.path.join(data_dir, 'sqqq.csv')
    psq_path = os.path.join(data_dir, 'psq.csv')

    qqq_df = pd.read_csv(qqq_path, index_col=0, parse_dates=True)
    tqqq_df = pd.read_csv(tqqq_path, index_col=0, parse_dates=True)
    sqqq_df = pd.read_csv(sqqq_path, index_col=0, parse_dates=True)
    psq_df = pd.read_csv(psq_path, index_col=0, parse_dates=True)

    prices_df = pd.DataFrame({
        'QQQ': qqq_df['Close'],
        'QQQ_Open': qqq_df['Open'],
        'TQQQ': tqqq_df['Close'],
        'TQQQ_Open': tqqq_df['Open'],
        'SQQQ': sqqq_df['Close'],
        'SQQQ_Open': sqqq_df['Open'],
        'PSQ': psq_df['Close'],
        'PSQ_Open': psq_df['Open'],
    })
    prices_df = prices_df.dropna()
    return prices_df


def main() -> None:
    """命令行入口函数。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("绩效评估 - 主程序")
    print("=" * 80)

    evaluator = PerformanceEvaluator(risk_free_rate=0.0)

    wf_path = 'data/processed/walkforward_results.parquet'
    try:
        evaluator.load_walkforward_results(wf_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请先运行 Walk-Forward 验证生成结果，例如:")
        print("  python src/walkforward.py")
        sys.exit(1)

    wf = evaluator.wf_results
    if wf is None:
        print("错误: Walk-Forward 结果为空")
        sys.exit(1)

    print("\n" + "-" * 80)
    print("加载价格数据")
    print("-" * 80)
    try:
        prices_df = _load_prices('data/raw')
        print(f"已加载价格数据: {len(prices_df)} 行")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    common_dates = wf.index.intersection(prices_df.index)
    print(f"共同交易日数: {len(common_dates)}")

    print("\n" + "-" * 80)
    print("运行基准策略")
    print("-" * 80)
    benchmarks = evaluator.run_benchmarks(prices_df, wf.index)
    for name, data in benchmarks.items():
        print(f"  {name}: 最终净值 = {data['nav'].iloc[-1]:.4f}")

    print("\n" + "-" * 80)
    print("计算策略组合绩效指标")
    print("-" * 80)
    daily_returns = wf['daily_return_cost'] if 'daily_return_cost' in wf.columns else wf['daily_return']
    nav_col = 'nav_cost_combined' if 'nav_cost_combined' in wf.columns else None
    nav = wf[nav_col] if nav_col else None
    strategy_metrics = evaluator.calc_portfolio_metrics(daily_returns, nav)
    print(f"  总收益率: {strategy_metrics['total_return']:.2%}")
    print(f"  年化收益率: {strategy_metrics['cagr']:.2%}")
    print(f"  夏普比率: {strategy_metrics['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {strategy_metrics['max_drawdown']:.2%}")

    benchmark_metrics = {}
    for name, data in benchmarks.items():
        benchmark_metrics[name] = evaluator.calc_portfolio_metrics(
            data['daily_returns'], data['nav']
        )

    print("\n" + "-" * 80)
    print("计算交易指标")
    print("-" * 80)
    positions = wf['actual_position']
    trading_metrics = evaluator.calc_trading_metrics(positions, daily_returns)
    print(f"  总交易次数: {trading_metrics['total_trades']}")
    print(f"  年均换手率: {trading_metrics['turnover_rate']:.2f} 次/年")
    print(f"  平均持仓期: {trading_metrics['avg_holding_period']:.2f} 天")
    print(f"  胜率: {trading_metrics['win_rate']:.2%}")

    print("\n" + "-" * 80)
    print("计算分类指标")
    print("-" * 80)

    feature_path = 'data/processed/features.parquet'
    y_true = None
    if os.path.exists(feature_path):
        features_df = pd.read_parquet(feature_path, engine='pyarrow')
        if 'label' in features_df.columns:
            common_cls_idx = wf.index.intersection(features_df.index)
            y_true = features_df.loc[common_cls_idx, 'label'].astype(int).values

            prob_cols = ['prob_sqqq', 'prob_psq', 'prob_cash', 'prob_qqq', 'prob_tqqq']
            prob_df = wf.loc[common_cls_idx, prob_cols]
            y_pred = prob_df.idxmax(axis=1).map({
                'prob_sqqq': 0,
                'prob_psq': 1,
                'prob_cash': 2,
                'prob_qqq': 3,
                'prob_tqqq': 4,
            }).astype(int).values

            classification_metrics = evaluator.calc_classification_metrics(y_true, y_pred)
            print(f"  准确率: {classification_metrics['accuracy']:.4f} "
                  f"({classification_metrics['accuracy']*100:.2f}%)")
            print(f"  宏平均 F1: {classification_metrics['f1_macro']:.4f}")
        else:
            print("  警告: 特征数据中无 label 列，跳过分类指标")
            classification_metrics = {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'confusion_matrix': np.zeros((5, 5), dtype=int),
            }
    else:
        print("  警告: 特征数据文件不存在，跳过分类指标")
        classification_metrics = {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'confusion_matrix': np.zeros((5, 5), dtype=int),
        }

    evaluator.print_report(strategy_metrics, trading_metrics, classification_metrics, benchmark_metrics)


if __name__ == '__main__':
    main()
