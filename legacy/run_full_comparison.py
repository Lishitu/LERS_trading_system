"""
完整对比测试脚本

对比 4 个策略：
1. Buy-Hold TQQQ（纯基准）
2. MA20-60 均线交叉（技术分析基准）
3. ML 模型（无 Regime）
4. ML + Regime（我们优化的目标）
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from walkforward import WalkForwardValidator
from benchmark_strategies import BenchmarkRunner
from logger import setup_logger


def load_prices_for_benchmarks(price_dir: str = 'data/raw') -> pd.DataFrame:
    """加载价格数据用于基准策略"""
    from data_loader import DataLoader

    loader = DataLoader(save_dir=price_dir)
    data = loader.load_all()

    # DataLoader 返回的 key 是大写的（如 'TQQQ'）
    tqqq = data.get('TQQQ')
    sqqq = data.get('SQQQ')

    if tqqq is None or sqqq is None:
        raise ValueError(f"需要 TQQQ 和 SQQQ 数据，可用 keys: {list(data.keys())}")

    # 检查列名
    if 'close' not in tqqq.columns:
        print(f"TQQQ 列名: {tqqq.columns.tolist()}")
        # 尝试使用第一个数值列
        price_col = tqqq.select_dtypes(include=[np.number]).columns[0]
    else:
        price_col = 'close'

    if 'close' not in sqqq.columns:
        sqqq_price_col = sqqq.select_dtypes(include=[np.number]).columns[0]
    else:
        sqqq_price_col = 'close'

    prices_df = pd.DataFrame({
        'tqqq_close': tqqq[price_col],
        'sqqq_close': sqqq[sqqq_price_col],
    })

    return prices_df


def calculate_metrics(nav_series: pd.Series) -> dict:
    """计算绩效指标"""
    nav = nav_series.dropna()
    if len(nav) == 0:
        return {}

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()

    daily_returns = nav.pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0

    # 交易次数
    if hasattr(nav_series, 'name') and 'position' in str(nav_series.name).lower():
        # 对于有 position 列的
        pass

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_nav': nav.iloc[-1],
        'start_date': str(nav.index[0].date()),
        'end_date': str(nav.index[-1].date()),
    }


def run_ml_strategy(logger, use_regime: bool = False, regime_type: str = 'rule_based', disable_short: bool = True) -> tuple:
    """运行 ML 策略"""
    validator = WalkForwardValidator(
        frequency='quarterly',
        window_mode='expanding',
        window_years=5,
        min_train_years=5,
        random_state=42,
        use_regime=use_regime,
        regime_agent_type=regime_type,
        portfolio_config={'disable_short': disable_short},
    )

    validator.load_data(
        feature_path='data/processed/features.parquet',
        price_dir='data/raw',
    )

    result = validator.run()

    # 提取净值
    nav_col = None
    if 'nav_cost_combined' in result.columns:
        nav_col = 'nav_cost_combined'
    elif 'nav_combined' in result.columns:
        nav_col = 'nav_combined'
    else:
        for col in result.columns:
            if 'nav' in col.lower() and 'cost' in col.lower():
                nav_col = col
                break
    if nav_col is None:
        for col in result.columns:
            if 'nav' in col.lower():
                nav_col = col
                break

    if nav_col is not None:
        nav = result[nav_col]
    else:
        # 使用 daily_return_cum
        ret_col = None
        for col in result.columns:
            if 'daily_return' in col.lower() and 'cost' in col.lower():
                ret_col = col
                break
        if ret_col is None:
            for col in result.columns:
                if 'return' in col.lower():
                    ret_col = col
                    break
        if ret_col is not None:
            nav = (1 + result[ret_col].fillna(0)).cumprod()
        else:
            nav = pd.Series(1.0, index=result.index)

    metrics = calculate_metrics(nav)

    return result, nav, metrics


def main():
    """主函数"""
    logger = setup_logger(
        name="full_comparison",
        log_dir="logs",
        level=20,
        console=True,
        file=True,
    )

    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + "  LERS 完整对比测试".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # 1. 加载数据
    logger.info("加载价格数据...")
    try:
        prices_df = load_prices_for_benchmarks()
        logger.info(f"  价格数据: {len(prices_df)} 行, {prices_df.index[0].date()} ~ {prices_df.index[-1].date()}")
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        return

    # 3. 运行 ML 策略（无 Regime）
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行 ML 策略（无 Regime）")
    logger.info("=" * 60)

    ml_nav = None
    ml_metrics = {}
    ml_result = None
    try:
        ml_result, ml_nav, ml_metrics = run_ml_strategy(logger, use_regime=False)
        logger.info(f"  ML (无 Regime): 总收益={ml_metrics['total_return']:.2%}")
    except Exception as e:
        logger.error(f"ML 策略运行失败: {e}")
        import traceback
        traceback.print_exc()
        ml_nav = None
        ml_metrics = {}

    # 获取 ML 策略的测试区间
    if ml_nav is not None and len(ml_nav) > 0:
        test_start = ml_nav.index[0]
        test_end = ml_nav.index[-1]
        logger.info(f"ML 测试区间: {test_start.date()} ~ {test_end.date()}")
    else:
        test_start = None
        test_end = None

    # 2. 运行基准策略（使用 ML 相同区间）
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行基准策略")
    logger.info("=" * 60)

    runner = BenchmarkRunner(initial_capital=10000.0)

    # 截取相同区间
    prices_for_benchmark = prices_df.copy()
    if test_start is not None and test_end is not None:
        prices_for_benchmark = prices_for_benchmark.loc[test_start:test_end]
        logger.info(f"  基准策略区间: {prices_for_benchmark.index[0].date()} ~ {prices_for_benchmark.index[-1].date()}")

    benchmark_results = runner.run_all(prices_for_benchmark)

    benchmark_metrics = {}
    for name, df in benchmark_results.items():
        logger.info(f"  {name}: 总收益={df['nav'].iloc[-1]/df['nav'].iloc[0]-1:.2%}")
        benchmark_metrics[name] = calculate_metrics(df['nav'])

    # 4. 运行 ML + Regime 策略
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行 ML + Regime 策略")
    logger.info("=" * 60)

    try:
        regime_result, regime_nav, regime_metrics = run_ml_strategy(
            logger, use_regime=True, regime_type='rule_based'
        )
        logger.info(f"  ML + Regime: 总收益={regime_metrics['total_return']:.2%}")
    except Exception as e:
        logger.error(f"ML + Regime 策略运行失败: {e}")
        import traceback
        traceback.print_exc()
        regime_nav = None
        regime_metrics = {}

    # 5. 汇总对比
    logger.info("")
    logger.info("=" * 60)
    logger.info("策略对比汇总")
    logger.info("=" * 60)

    all_metrics = {**benchmark_metrics}
    all_navs = {}

    if ml_nav is not None:
        all_metrics['ml_no_regime'] = ml_metrics
        all_navs['ml_no_regime'] = ml_nav

    if regime_nav is not None:
        all_metrics['ml_with_regime'] = regime_metrics
        all_navs['ml_with_regime'] = regime_nav

    # 打印对比表
    strategy_names = {
        'buyhold_tqqq': 'Buy-Hold TQQQ',
        'ma20_60': 'MA20-60 均线',
        'ml_no_regime': 'ML (无 Regime)',
        'ml_with_regime': 'ML + Regime',
    }

    logger.info("")
    logger.info(f"{'策略':<20} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8}")
    logger.info("-" * 60)

    for key, name in strategy_names.items():
        if key in all_metrics:
            m = all_metrics[key]
            logger.info(
                f"{name:<20} "
                f"{m['total_return']:>+10.2%} "
                f"{m['annualized_return']:>+10.2%} "
                f"{m['max_drawdown']:>+10.2%} "
                f"{m['sharpe_ratio']:>+8.2f}"
            )

    # 6. 保存结果
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"测试完成，总耗时: {elapsed:.1f} 秒")
    logger.info("=" * 60)

    # 保存结果
    os.makedirs('data/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for name, df in benchmark_results.items():
        df.to_parquet(f'data/results/benchmark_{name}_{timestamp}.parquet')

    if ml_nav is not None:
        ml_nav.to_frame(name='nav').to_parquet(f'data/results/ml_no_regime_{timestamp}.parquet')

    if regime_nav is not None:
        regime_nav.to_frame(name='nav').to_parquet(f'data/results/ml_with_regime_{timestamp}.parquet')

    # 保存汇总 CSV
    summary_data = []
    for key, name in strategy_names.items():
        if key in all_metrics:
            m = all_metrics[key]
            summary_data.append({
                '策略': name,
                '总收益': m['total_return'],
                '年化收益': m['annualized_return'],
                '最大回撤': m['max_drawdown'],
                '夏普比率': m['sharpe_ratio'],
                '最终净值': m['final_nav'],
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'data/results/comparison_full_{timestamp}.csv', index=False, encoding='utf-8-sig')
    logger.info(f"汇总结果已保存: data/results/comparison_full_{timestamp}.csv")


if __name__ == '__main__':
    main()
