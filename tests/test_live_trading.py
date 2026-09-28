"""Unit tests for live trading, mock execution, safety locks and rebalance management."""

import pytest
import os
import sys
import pandas as pd
import numpy as np

# Ensure src is in search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trade_moomoo import MoomooTrader, AccountBalance, PositionInfo, _normalize_ticker
from rebalance_manager import RebalanceManager
from notifier import TradeOrder, NotificationService
from feature_engineering import FeatureEngineer


def test_normalize_ticker():
    assert _normalize_ticker("QQQ") == "US.QQQ"
    assert _normalize_ticker("US.QQQ") == "US.QQQ"
    assert _normalize_ticker("SPY") == "US.SPY"
    assert _normalize_ticker("US.TQQQ") == "US.TQQQ"


def test_mock_trader_execution():
    trader = MoomooTrader(force_mock=True)
    bal_initial = trader.get_account_balance(env="SIMULATE")
    assert bal_initial.is_mock is True
    assert bal_initial.cash == 100000.0

    # 买入 100 股 QQQ @ $500.0
    res_buy = trader.place_order(code="US.QQQ", action="BUY", shares=100, price=500.0, env="SIMULATE")
    assert res_buy['success'] is True
    bal_after_buy = trader.get_account_balance(env="SIMULATE")
    assert bal_after_buy.cash == 50000.0
    assert bal_after_buy.market_val == 50000.0
    assert bal_after_buy.total_assets == 100000.0

    # 验证持仓记录
    positions = trader.get_positions(env="SIMULATE")
    assert "US.QQQ" in positions
    assert positions["US.QQQ"].shares == 100

    # 卖出 40 股 QQQ @ $520.0
    res_sell = trader.place_order(code="US.QQQ", action="SELL", shares=40, price=520.0, env="SIMULATE")
    assert res_sell['success'] is True
    bal_after_sell = trader.get_account_balance(env="SIMULATE")
    assert bal_after_sell.cash == 50000.0 + (40 * 520.0)
    assert trader.get_positions(env="SIMULATE")["US.QQQ"].shares == 60


def test_real_safety_lock_interception():
    trader = MoomooTrader(force_mock=True)
    # 试图向 REAL 实盘模式下单，必须被物理拦截
    res = trader.place_order(code="US.TQQQ", action="BUY", shares=10, price=80.0, env="REAL")
    assert res['success'] is False
    assert res.get('is_real_blocked') is True

    # 通过 RebalanceManager 试图报送实盘订单，也必须被安全拦截
    mgr = RebalanceManager(trader=trader)
    exec_res = mgr.execute_rebalance_plan(orders=[TradeOrder(ticker="US.QQQ", action="BUY", shares=10, price=500, est_amount=5000, reason="test")], env="REAL")
    assert len(exec_res) == 1
    assert exec_res[0]['success'] is False
    assert exec_res[0].get('is_real_blocked') is True


def test_rebalance_manager_budget_protection():
    trader = MoomooTrader(force_mock=True)
    # 模拟现金仅剩 $1000
    trader._mock_cash = 1000.0
    mgr = RebalanceManager(trader=trader)

    # 构造超出预算的买单计划
    oversized_orders = [
        TradeOrder(ticker="US.SMH", action="BUY", shares=10, price=600.0, est_amount=6000.0, reason="test"),
        TradeOrder(ticker="US.XLK", action="BUY", shares=20, price=200.0, est_amount=4000.0, reason="test"),
    ]
    # 执行前买单总预估为 $10,000 > $1000 可用现金
    # 运行 generate_rebalance_plan 时内部预算缩减机制会生效
    acc_bal, orders, target_w = mgr.generate_rebalance_plan(env="SIMULATE")
    buy_orders = [o for o in orders if o.action == "BUY"]
    total_buy = sum(o.est_amount for o in buy_orders)
    # 买单总额绝不能超过账户总资金加上合理缓冲区
    assert total_buy <= acc_bal.total_assets + 1.0


def test_feature_engineering_market_breadth_restored():
    # 验证修复后市场广度特征不再被截断且正确生成
    fe = FeatureEngineer()
    dates = pd.date_range("2024-01-01", periods=300, freq='B')
    df_dummy = pd.DataFrame({
        'qqq_close': 400.0 + np.cumsum(np.random.normal(0.1, 1.0, 300)),
        'spy_close': 500.0 + np.cumsum(np.random.normal(0.1, 1.0, 300)),
        'iwm_close': 200.0 + np.cumsum(np.random.normal(0.05, 0.8, 300)),
        'dia_close': 380.0 + np.cumsum(np.random.normal(0.08, 0.9, 300)),
    }, index=dates)

    df_out = fe._calc_market_breadth(df_dummy)
    assert 'spy_ret_20d' in df_out.columns
    assert 'spy_high_252d_ratio' in df_out.columns
    assert 'spy_low_252d_ratio' in df_out.columns
    assert not df_out['spy_ret_20d'].iloc[30:].isna().all()


def test_allocator_duplicate_ticker_accumulation():
    from portfolio_allocator import PortfolioAllocator
    allocator = PortfolioAllocator(max_asset_weight=0.60)
    # 两只标的映射为同一代码 US.TQQQ
    selected = [("US.TQQQ", 0.20), ("US.TQQQ", 0.20)]
    weights = allocator.compute_target_weights(selected)
    assert "US.TQQQ" in weights
    assert weights["US.TQQQ"] == pytest.approx(0.60, abs=1e-4)


def test_coresatellite_reset_cleans_caches():
    from strategy_coresatellite import CoreSatelliteStrategy
    strat = CoreSatelliteStrategy()
    strat._cached_df_id = 12345
    strat._cached_ma200 = pd.Series([1.0, 2.0])
    strat._cached_ma50 = pd.Series([1.0, 2.0])

    strat.reset()
    assert strat._cached_df_id is None
    assert strat._cached_ma200 is None
    assert strat._cached_ma50 is None

