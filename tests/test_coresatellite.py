"""Unit tests for Core-Satellite ETF quantitative system."""

import pytest
import numpy as np
import pandas as pd

from src.strategy_coresatellite import (
    MacroRegimeDetector,
    CoreSatelliteStrategy,
)
from src.data_moomoo import DEFAULT_DEFENSE


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2020-01-01", periods=300, freq='B')
    np.random.seed(42)

    tickers = [
        'US.QQQ', 'US.SPY', 'US.XLK', 'US.SMH', 'US.XLE', 'US.XOP',
        'US.COPX', 'US.SLV', 'US.GLD', 'US.URA', 'US.QLD', 'US.TQQQ',
        'US.USD', 'US.SOXL', 'US.BIL',
    ]

    data_dict = {}
    for t in tickers:
        # 构造带有漂移的几何布朗运动
        drift = 0.0005 if 'QQQ' in t or 'XLK' in t else 0.0002
        v = 0.015 if 'BIL' not in t else 0.0001
        rets = np.random.normal(drift, v, len(dates))
        prices = 100.0 * np.exp(np.cumsum(rets))
        vols = np.random.uniform(1e6, 5e6, len(dates))
        data_dict[t] = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': vols,
            'turnover': vols * prices,
            'vwap': prices * 1.0001,
        }, index=dates)

    return data_dict


def test_macro_regime_detector(sample_market_data):
    detector = MacroRegimeDetector(benchmark='US.QQQ')
    close_dict = {t: df['close'] for t, df in sample_market_data.items()}
    df_close = pd.DataFrame(close_dict)

    regimes = detector.detect_regimes(df_close)
    assert len(regimes) == len(df_close)
    valid_states = {'BULL_GROWTH', 'INFLATION_COMMODITY', 'CRISIS_BEAR', 'NEUTRAL_CHOP'}
    assert set(regimes.unique()).issubset(valid_states)


def test_core_satellite_components(sample_market_data):
    strat = CoreSatelliteStrategy(
        benchmark='US.QQQ',
        core_ratio=0.40,
        satellite_ratio=0.60,
        target_vol=0.22,
        min_leverage=0.60,
        max_leverage=1.80,
    )

    factors, vols, ma_trend, df_close, regimes = strat.compute_factors_and_indicators(sample_market_data)

    test_date = df_close.index[-1]
    regime = regimes.loc[test_date]
    trend = ma_trend.loc[test_date]
    vol_t = vols.loc[test_date]

    # Test Volatility Multiplier
    mult_low = strat.calculate_volatility_multiplier(0.10)
    assert mult_low == 1.80  # clipped at max_leverage
    mult_clipped_min = strat.calculate_volatility_multiplier(0.50)
    assert mult_clipped_min == 0.60  # clipped at min_leverage
    mult_unclipped = strat.calculate_volatility_multiplier(0.30)
    assert mult_unclipped == pytest.approx(0.22 / 0.30, rel=1e-2)

    # Test Core Allocation
    core_alloc = strat.determine_core_allocation(test_date, regime, df_close, vol_t)
    assert sum(core_alloc.values()) == pytest.approx(0.40, abs=1e-5)

    # Test Satellite Allocation
    sat_alloc = strat.determine_satellite_allocation(test_date, regime, factors, trend, vol_t)
    assert sum(sat_alloc.values()) == pytest.approx(0.60, abs=1e-5)
    for asset, w in sat_alloc.items():
        if asset != DEFAULT_DEFENSE:
            assert w <= strat.max_satellite_cap + 1e-4

    # Test Full Portfolio Target
    target = strat.generate_portfolio_target(
        date=test_date,
        factors_dict=factors,
        current_trend=trend,
        current_vols=vol_t,
        regime=regime,
        df_close=df_close,
        realized_portfolio_vol=0.20,
    )
    assert isinstance(target, dict)
    assert len(target) > 0
    # No negative weights
    for t, w in target.items():
        assert w >= 0.0


def test_core_satellite_reset():
    strat = CoreSatelliteStrategy()
    strat._current_satellite_held = ['US.SMH', 'US.XLK']
    strat._last_eval_date = pd.Timestamp("2024-01-05")

    strat.reset()
    assert strat._current_satellite_held == []
    assert strat._last_eval_date is None


def test_feature_engineering_ts_rsi_boundaries():
    from src.feature_engineering_ts import TimeSeriesFeatureExtractor
    # 纯单边上涨：15天每天+1%
    dates = pd.date_range("2024-01-01", periods=20, freq='D')
    prices_up = pd.Series([100.0 * (1.01 ** i) for i in range(20)], index=dates)
    rsi_up = TimeSeriesFeatureExtractor.compute_rsi(prices_up, window=14)
    # 第 15 天起 avg_loss 为 0，RSI 必须为 100.0
    assert rsi_up.iloc[-1] == 100.0

    # 纯单边下跌：15天每天-1%
    prices_down = pd.Series([100.0 * (0.99 ** i) for i in range(20)], index=dates)
    rsi_down = TimeSeriesFeatureExtractor.compute_rsi(prices_down, window=14)
    # 第 15 天起 avg_gain 为 0，RSI 必须为 0.0
    assert rsi_down.iloc[-1] == 0.0


def test_satellite_macro_tilt_discrimination(sample_market_data):
    strat = CoreSatelliteStrategy()
    strat.weekly_satellite = False  # 确保每日重新计算打分
    close_dict = {t: df['close'] for t, df in sample_market_data.items()}
    df_close = pd.DataFrame(close_dict)
    test_date = df_close.index[-1]

    # 构造同等动量的初始环境
    sec_list = ['US.XLK', 'US.XLE']
    mom_df = pd.DataFrame(
        {s: [0.10] * len(df_close) for s in sec_list},
        index=df_close.index
    )
    factors = {'momentum': mom_df}
    trend = pd.Series({s: True for s in sec_list})
    vols = pd.Series({s: 0.15 for s in sec_list})

    # 当 BULL_GROWTH 时，XLK(科技)应倾斜胜出
    strat.reset()
    bull_alloc = strat.determine_satellite_allocation(
        test_date, 'BULL_GROWTH', factors, trend, vols, candidate_sectors=sec_list
    )
    assert 'US.XLK' in bull_alloc
    assert bull_alloc['US.XLK'] > 0

    # 当 INFLATION_COMMODITY 时，XLE(能源)应倾斜胜出
    strat.reset()
    inf_alloc = strat.determine_satellite_allocation(
        test_date, 'INFLATION_COMMODITY', factors, trend, vols, candidate_sectors=sec_list
    )
    assert 'US.XLE' in inf_alloc
    assert inf_alloc['US.XLE'] > 0

