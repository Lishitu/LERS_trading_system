import os
import tempfile
import pytest

from data_loader import DataLoader


class TestDataLoader:
    def test_data_loader_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, 'raw')
            loader = DataLoader(
                save_dir=save_dir,
                start_date='2020-01-01',
                end_date='2023-12-31'
            )

            assert loader.save_dir == save_dir
            assert loader.start_date == '2020-01-01'
            assert loader.end_date == '2023-12-31'
            assert os.path.exists(save_dir)

    def test_tickers_config(self):
        expected_keys = [
            'QQQ', 'TQQQ', 'SQQQ', 'PSQ', 'SPY', 'IWM', 'DIA',
            'HYG', 'IEF', 'TLT', '^VIX', '^TNX', 'DX-Y.NYB'
        ]

        assert len(DataLoader.TICKERS) == 13
        for k in expected_keys:
            assert k in DataLoader.TICKERS

    def test_data_loader_default_end_date_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(save_dir=tmpdir, start_date='2020-01-01')
            assert loader.end_date is None

    def test_validate_data_empty(self):
        import pandas as pd
        loader = DataLoader()
        df = pd.DataFrame()
        result = loader.validate_data(df, 'TEST')
        assert result['valid'] is False
        assert result['rows'] == 0

    def test_validate_data_with_data(self):
        import pandas as pd
        import numpy as np

        loader = DataLoader()
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(10).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 10)
        }, index=dates)

        result = loader.validate_data(df, 'TEST')
        assert result['valid'] is True
        assert result['rows'] == 10
        assert result['missing_total'] == 0
        assert len(result['columns']) == 2
