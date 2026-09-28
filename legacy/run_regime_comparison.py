"""
基准 vs Regime 对比测试脚本

运行两个版本的 Walk-Forward 验证并对比结果：
1. 基准版本（无 Regime 调整）
2. 规则化 Regime 版本
"""

import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from walkforward import WalkForwardValidator, print_wf_performance
from logger import setup_logger, BacktestLogger


def calculate_key_metrics(result_df: pd.DataFrame) -> dict:
    """
    计算关键绩效指标

    Args:
        result_df: 回测结果 DataFrame

    Returns:
        指标字典
    """
    if result_df is None or result_df.empty:
        return {}

    # 确保日期索引
    if 'date' in result_df.columns:
        result_df = result_df.set_index('date')

    # 获取 nav 列
    nav_col = None
    for col in result_df.columns:
        if 'nav' in col.lower() and 'cost' in col.lower():
            nav_col = col
            break
    if nav_col is None:
        for col in result_df.columns:
            if 'nav' in col.lower():
                nav_col = col
                break

    if nav_col is None:
        return {}

    nav = result_df[nav_col].dropna()
    if len(nav) == 0:
        return {}

    # 总收益
    total_return = nav.iloc[-1] / nav.iloc[0] - 1

    # 年化收益
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 最大回撤
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()

    # 夏普比率
    daily_returns = nav.pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0

    # 卡玛比率
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 交易次数
    position_col = None
    for col in result_df.columns:
        if 'position' in col.lower():
            position_col = col
            break

    total_trades = 0
    if position_col is not None:
        positions = result_df[position_col].fillna('Cash')
        # 统计持仓变化次数（字符串比较）
        total_trades = int((positions != positions.shift(1).fillna('Cash')).sum())

    # 胜率
    trade_return_col = None
    for col in result_df.columns:
        if 'return' in col.lower() and 'cost' in col.lower():
            trade_return_col = col
            break
    if trade_return_col is None:
        for col in result_df.columns:
            if 'daily_return' in col.lower() or col == 'return':
                trade_return_col = col
                break

    win_rate = 0.0
    if trade_return_col is not None and position_col is not None:
        position = result_df[position_col].fillna(0)
        daily_ret = result_df[trade_return_col].fillna(0)
        # 只算有持仓的日子
        holding_days = daily_ret[position != 0]
        if len(holding_days) > 0:
            win_rate = (holding_days > 0).mean()

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'final_nav': nav.iloc[-1],
        'start_date': str(nav.index[0].date()),
        'end_date': str(nav.index[-1].date()),
        'num_days': len(nav),
    }


def run_benchmark(logger) -> tuple[pd.DataFrame, dict]:
    """运行基准版本"""
    logger.info("=" * 60)
    logger.info("运行基准版本（无 Regime 调整）")
    logger.info("=" * 60)

    validator = WalkForwardValidator(
        frequency='quarterly',
        window_mode='expanding',
        window_years=5,
        min_train_years=5,
        random_state=42,
        use_regime=False,
    )

    validator.load_data(
        feature_path='data/processed/features.parquet',
        price_dir='data/raw',
    )

    result = validator.run()
    metrics = calculate_key_metrics(result)

    logger.info("")
    logger.info("基准版本结果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return result, metrics


def run_regime_rule_based(logger) -> tuple[pd.DataFrame, dict]:
    """运行规则化 Regime 版本"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行规则化 Regime 版本")
    logger.info("=" * 60)

    validator = WalkForwardValidator(
        frequency='quarterly',
        window_mode='expanding',
        window_years=5,
        min_train_years=5,
        random_state=42,
        use_regime=True,
        regime_agent_type='rule_based',
    )

    validator.load_data(
        feature_path='data/processed/features.parquet',
        price_dir='data/raw',
    )

    result = validator.run()
    metrics = calculate_key_metrics(result)

    logger.info("")
    logger.info("规则化 Regime 版本结果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return result, metrics


def compare_results(
    logger,
    benchmark_metrics: dict,
    regime_metrics: dict,
) -> pd.DataFrame:
    """对比两个版本的结果"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("对比结果")
    logger.info("=" * 60)

    comparison_data = []
    for key in [
        'total_return', 'annualized_return', 'max_drawdown',
        'sharpe_ratio', 'calmar_ratio', 'total_trades',
        'win_rate', 'final_nav'
    ]:
        bm_val = benchmark_metrics.get(key, 0)
        rg_val = regime_metrics.get(key, 0)
        diff = rg_val - bm_val

        # 格式
        if key in ['max_drawdown']:
            # 回撤越小越好，diff < 0 表示 Regime 更好
            improvement = "✅ Regime 更优" if diff < 0 else "❌ 基准更优" if diff > 0 else "➖ 持平"
        elif key in ['total_trades']:
            improvement = "➖ 中性"
        else:
            improvement = "✅ Regime 更优" if diff > 0 else "❌ 基准更优" if diff < 0 else "➖ 持平"

        comparison_data.append({
            '指标': key,
            '基准': bm_val,
            'Regime': rg_val,
            '差异': diff,
            '结论': improvement,
        })

    df = pd.DataFrame(comparison_data)

    # 格式化输出
    logger.info(f"{'指标':<20} {'基准':>12} {'Regime':>12} {'差异':>12} {'结论':>12}")
    logger.info("-" * 70)
    for _, row in df.iterrows():
        key = row['指标']
        bm = row['基准']
        rg = row['Regime']
        diff = row['差异']
        conclusion = row['结论']

        if key in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
            logger.info(f"{key:<20} {bm:>11.2%} {rg:>11.2%} {diff:>+11.2%} {conclusion:>12}")
        elif key in ['sharpe_ratio', 'calmar_ratio']:
            logger.info(f"{key:<20} {bm:>12.4f} {rg:>12.4f} {diff:>+12.4f} {conclusion:>12}")
        elif key in ['total_trades']:
            logger.info(f"{key:<20} {bm:>12.0f} {rg:>12.0f} {diff:>+12.0f} {conclusion:>12}")
        elif key in ['final_nav']:
            logger.info(f"{key:<20} {bm:>12.4f} {rg:>12.4f} {diff:>+12.4f} {conclusion:>12}")
        else:
            logger.info(f"{key:<20} {bm:>12.4f} {rg:>12.4f} {diff:>+12.4f} {conclusion:>12}")

    return df


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="regime_comparison",
        log_dir="logs",
        level=20,  # INFO
        console=True,
        file=True,
    )

    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + "  LERS Regime 对比测试".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"数据目录: data/raw, data/processed")
    logger.info("")

    start_time = time.time()

    # 1. 运行基准版本
    try:
        benchmark_result, benchmark_metrics = run_benchmark(logger)
    except Exception as e:
        logger.error(f"基准版本运行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 运行 Regime 版本
    try:
        regime_result, regime_metrics = run_regime_rule_based(logger)
    except Exception as e:
        logger.error(f"Regime 版本运行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 对比结果
    comparison_df = compare_results(logger, benchmark_metrics, regime_metrics)

    # 4. 保存结果
    os.makedirs('data/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    benchmark_result.to_parquet(f'data/results/benchmark_{timestamp}.parquet')
    regime_result.to_parquet(f'data/results/regime_rule_based_{timestamp}.parquet')
    comparison_df.to_csv(f'data/results/comparison_{timestamp}.csv', index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"测试完成，总耗时: {elapsed:.1f} 秒")
    logger.info(f"结果已保存到 data/results/")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
