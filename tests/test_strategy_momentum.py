"""截面动量策略与分配器单元测试。"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategy_momentum import MomentumRegimeStrategy
from portfolio_allocator import PortfolioAllocator
from trading_costs import TradingCostCalculator


def test_inverse_vol_allocation_cap():
    allocator = PortfolioAllocator(max_asset_weight=0.35, cash_ticker='US.BIL')

    # 给定 3 个标的，波动率分别为 0.10, 0.20, 0.30
    assets = [('A', 0.10), ('B', 0.20), ('C', 0.30)]
    weights = allocator.compute_target_weights(assets)

    # 1. 验证没有任何单一风险资产权重超过 35%
    for t in ['A', 'B', 'C']:
        assert weights[t] <= 0.35 + 1e-6, f"资产 {t} 权重 {weights[t]} 超过上限 0.35"

    # 2. 验证总权重合理接近 1.0 (含BIL兜底)
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-4


def test_bear_regime_defensive_switch():
    strategy = MomentumRegimeStrategy(top_n=3)

    date = pd.Timestamp('2022-06-01')
    scores = pd.Series({'US.XLK': 1.5, 'US.SMH': 2.0, 'US.XLF': 1.0})
    trend = pd.Series({'US.XLK': True, 'US.SMH': True, 'US.XLF': True})
    vols = pd.Series({'US.XLK': 0.25, 'US.SMH': 0.35, 'US.XLF': 0.20, 'US.BIL': 0.01})

    # 当大盘处于 BEAR 状态
    signals = strategy.generate_daily_signals(
        date=date,
        current_scores=scores,
        current_trend=trend,
        current_vols=vols,
        regime='BEAR',
    )

    # 应该全部退守防御性资产 US.BIL
    assert len(signals) == 1
    assert signals[0][0] == 'US.BIL'


def test_bull_regime_leverage_upgrade():
    strategy = MomentumRegimeStrategy(top_n=2)

    date = pd.Timestamp('2023-06-01')
    scores = pd.Series({'US.SMH': 3.5, 'US.XLK': 2.5, 'US.XLE': 0.5})
    trend = pd.Series({'US.SMH': True, 'US.XLK': True, 'US.XLE': True})
    vols = pd.Series({'US.SMH': 0.25, 'US.XLK': 0.20, 'US.SOXL': 0.70, 'US.TQQQ': 0.55})

    # 当大盘处于 BULL 状态
    signals = strategy.generate_daily_signals(
        date=date,
        current_scores=scores,
        current_trend=trend,
        current_vols=vols,
        regime='BULL',
    )

    # SMH 应升级为 SOXL，XLK 应升级为 TQQQ
    tickers = [s[0] for s in signals]
    assert 'US.SOXL' in tickers
    assert 'US.TQQQ' in tickers


def test_turnover_buffer_tolerates_small_drift():
    allocator = PortfolioAllocator(rebalance_buffer=0.05, cash_ticker='US.BIL')

    curr = {'US.XLK': 0.33, 'US.SMH': 0.33, 'US.BIL': 0.34}
    # 目标权重发生小幅微变 (<5%)
    target = {'US.XLK': 0.35, 'US.SMH': 0.31, 'US.BIL': 0.34}

    buffered = allocator.apply_turnover_buffer(curr, target)

    # 在 5% 缓冲区内，应维持原有权重不变
    assert abs(buffered['US.XLK'] - 0.33) < 1e-4
    assert abs(buffered['US.SMH'] - 0.33) < 1e-4


def test_moomoo_cost_calculator():
    calc = TradingCostCalculator()

    # 买入 100 股 $100
    buy_cost = calc.calc_trade_cost(shares=100, price=100.0, is_buy=True)
    # 佣金 0.99 min, 平台费 1.0 min, 交收费 0.003*100 = 0.3 -> 合计 2.29
    assert buy_cost.commission == 0.99
    assert buy_cost.platform_fee == 1.0
    assert buy_cost.clearing_fee == 0.3
    assert abs(buy_cost.total - 2.29) < 1e-3

    # 卖出 100 股 $100 包含规费
    sell_cost = calc.calc_trade_cost(shares=100, price=100.0, is_buy=False)
    assert sell_cost.total > buy_cost.total  # 卖出多了 SEC 与 TA 规费


def test_calc_trading_metrics_excludes_defense():
    from evaluate import PerformanceEvaluator
    evaluator = PerformanceEvaluator()
    dates = pd.date_range("2024-01-01", periods=10, freq='B')
    positions = pd.Series(['US.BIL'] * 5 + ['US.TQQQ'] * 5, index=dates)
    returns = pd.Series([0.0001] * 5 + [0.02] * 5, index=dates)

    metrics = evaluator.calc_trading_metrics(positions, returns)
    assert metrics['total_trades'] == 1
    # 纯风险交易只有 US.TQQQ 一笔，持有 5 天
    assert metrics['avg_holding_period'] == 5.0
    assert metrics['win_rate'] == 1.0

