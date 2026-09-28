import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from feature_engineering import FeatureEngineer
from label_generator import LabelGenerator


def _create_mock_raw_data(data_dir: str, n_days: int = 300) -> None:
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)

    np.random.seed(42)

    base_price = 100.0
    returns = np.random.randn(n_days) * 0.01
    qqq_close = base_price * np.cumprod(1 + returns)

    tqqq_close = 50.0 * np.cumprod(1 + returns * 3)
    sqqq_close = 50.0 * np.cumprod(1 - returns * 3)
    spy_close = 300.0 * np.cumprod(1 + returns * 0.9)
    vix_close = 20.0 + np.random.randn(n_days).cumsum() * 0.5
    vix_close = np.clip(vix_close, 10, 80)
    tnx_close = 2.0 + np.random.randn(n_days).cumsum() * 0.05
    dxy_close = 100.0 + np.random.randn(n_days).cumsum() * 0.2

    def make_ohlcv(close: np.ndarray) -> pd.DataFrame:
        high = close * (1 + np.abs(np.random.randn(n_days)) * 0.01)
        low = close * (1 - np.abs(np.random.randn(n_days)) * 0.01)
        open_p = close * (1 + np.random.randn(n_days) * 0.005)
        volume = np.random.randint(1000000, 10000000, n_days)
        return pd.DataFrame({
            'Open': open_p,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

    tickers_data = {
        'qqq': make_ohlcv(qqq_close),
        'tqqq': make_ohlcv(tqqq_close),
        'sqqq': make_ohlcv(sqqq_close),
        'spy': make_ohlcv(spy_close),
        'vix': make_ohlcv(vix_close),
        'tnx': make_ohlcv(tnx_close),
        'dxy': make_ohlcv(dxy_close),
    }

    os.makedirs(data_dir, exist_ok=True)
    for name, df in tickers_data.items():
        df.to_csv(os.path.join(data_dir, f'{name}.csv'))


class TestFutureReturnNoLookahead:
    def test_future_return_no_lookahead(self):
        close = pd.Series([100, 105, 110, 108, 112, 115, 120],
                          index=pd.bdate_range('2020-01-02', periods=7))

        gen = LabelGenerator(horizon=3, threshold=0.02)
        future_ret = gen.generate_future_returns(close)

        assert pd.isna(future_ret.iloc[-1])
        assert pd.isna(future_ret.iloc[-2])
        assert pd.isna(future_ret.iloc[-3])

        expected_day0 = close.iloc[3] / close.iloc[0] - 1
        assert abs(future_ret.iloc[0] - expected_day0) < 1e-10

        expected_day1 = close.iloc[4] / close.iloc[1] - 1
        assert abs(future_ret.iloc[1] - expected_day1) < 1e-10

        assert future_ret.iloc[0] != (close.iloc[-1] / close.iloc[0] - 1)

    def test_future_return_horizon_1(self):
        close = pd.Series([100, 110, 120, 115],
                          index=pd.bdate_range('2020-01-02', periods=4))

        gen = LabelGenerator(horizon=1, threshold=0.02)
        future_ret = gen.generate_future_returns(close)

        assert pd.isna(future_ret.iloc[-1])
        assert abs(future_ret.iloc[0] - 0.10) < 1e-10
        assert abs(future_ret.iloc[1] - (120 / 110 - 1)) < 1e-10


class TestFeatureShape:
    def test_feature_shape_with_mock_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, 'raw')
            processed_dir = os.path.join(tmpdir, 'processed')
            _create_mock_raw_data(raw_dir, n_days=300)

            engineer = FeatureEngineer(raw_dir=raw_dir, processed_dir=processed_dir)
            df = engineer.build_features()

            feature_cols = [col for col in df.columns
                            if not col.startswith('qqq_')
                            and col not in ('tqqq_close', 'sqqq_close',
                                            'spy_close', 'vix_close',
                                            'tnx_close', 'dxy_close')]

            assert len(feature_cols) > 0
            assert len(df) > 0
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_feature_columns_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, 'raw')
            processed_dir = os.path.join(tmpdir, 'processed')
            _create_mock_raw_data(raw_dir, n_days=300)

            engineer = FeatureEngineer(raw_dir=raw_dir, processed_dir=processed_dir)
            df = engineer.build_features()

            expected_core_cols = ['ret_1d', 'ma5', 'vol_20', 'vix_level']
            for col in expected_core_cols:
                assert col in df.columns, f"缺少特征列: {col}"


class TestNoInfNan:
    def test_no_inf_nan_after_dropna(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, 'raw')
            processed_dir = os.path.join(tmpdir, 'processed')
            _create_mock_raw_data(raw_dir, n_days=300)

            engineer = FeatureEngineer(raw_dir=raw_dir, processed_dir=processed_dir)
            df = engineer.build_features()

            feature_cols = [col for col in df.columns
                            if not col.startswith('qqq_')
                            and col not in ('tqqq_close', 'sqqq_close',
                                            'spy_close', 'vix_close',
                                            'tnx_close', 'dxy_close')]

            numeric_df = df[feature_cols].select_dtypes(include=[np.number])

            assert not numeric_df.isna().any().any(), "特征中存在 NaN"
            assert not np.isinf(numeric_df).any().any(), "特征中存在 Inf"

    def test_remove_outliers_truncation(self):
        engineer = FeatureEngineer()

        df = pd.DataFrame({
            'ret_1d': [0.5, -0.5, 0.01, 0.15],
            'ma5': [100, 101, 102, 103],
        })

        result = engineer._remove_outliers(df)

        assert result['ret_1d'].iloc[0] <= 0.2
        assert result['ret_1d'].iloc[1] >= -0.2
        assert abs(result['ret_1d'].iloc[2] - 0.01) < 1e-10

    def test_remove_outliers_inf_to_nan(self):
        engineer = FeatureEngineer()

        df = pd.DataFrame({
            'ret_1d': [np.inf, -np.inf, 0.01],
            'ma5': [100, 101, 102],
        })

        result = engineer._remove_outliers(df)

        assert not np.isinf(result['ret_1d']).any()


class TestMomentumFeatures:
    def test_momentum_calculation(self):
        engineer = FeatureEngineer()

        dates = pd.bdate_range('2020-01-02', periods=30)
        close = 100.0 * np.cumprod(1 + np.random.RandomState(42).randn(30) * 0.01)
        df = pd.DataFrame({'qqq_close': close}, index=dates)

        result = engineer._calc_momentum(df)

        assert 'ret_1d' in result.columns
        assert 'ret_5d' in result.columns
        assert 'ret_20d' in result.columns
        assert 'mom_5_20' in result.columns
