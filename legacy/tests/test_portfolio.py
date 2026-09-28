import pytest
import pandas as pd
import numpy as np

from portfolio_engine import PortfolioEngine


def _create_mock_prices(n_days: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)
    np.random.seed(42)
    returns = np.random.randn(n_days) * 0.01

    tqqq = 50.0 * np.cumprod(1 + returns * 3)
    sqqq = 50.0 * np.cumprod(1 - returns * 3)

    return pd.DataFrame({
        'TQQQ': tqqq,
        'SQQQ': sqqq,
    }, index=dates)


def _create_mock_prob_df(n_days: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)
    np.random.seed(123)
    probs = np.random.dirichlet([1, 1, 1], size=n_days)
    return pd.DataFrame({
        'prob_riskoff': probs[:, 0],
        'prob_neutral': probs[:, 1],
        'prob_riskon': probs[:, 2],
    }, index=dates)


class TestPositionStates:
    def test_valid_positions_constant(self):
        assert PortfolioEngine.VALID_POSITIONS == {'TQQQ', 'SQQQ', 'Cash'}
        assert len(PortfolioEngine.VALID_POSITIONS) == 3

    def test_generate_signals_only_three_states(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.5,
            risk_off_threshold=0.5,
            include_costs=False
        )

        prob_df = _create_mock_prob_df(n_days=50)
        target_pos = engine.generate_signals(prob_df)

        unique_positions = set(target_pos.unique())
        assert unique_positions.issubset({'TQQQ', 'SQQQ', 'Cash'})
        assert len(unique_positions) <= 3

    def test_run_result_positions_only_three_states(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.4,
            risk_off_threshold=0.4,
            include_costs=False
        )

        prob_df = _create_mock_prob_df(n_days=30)
        prices_df = _create_mock_prices(n_days=30)

        result = engine.run(prob_df, prices_df)

        target_positions = set(result['target_position'].unique())
        actual_positions = set(result['actual_position'].unique())

        assert target_positions.issubset({'TQQQ', 'SQQQ', 'Cash'})
        assert actual_positions.issubset({'TQQQ', 'SQQQ', 'Cash'})

    def test_cash_when_below_thresholds(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.8,
            risk_off_threshold=0.8,
            include_costs=False
        )

        dates = pd.bdate_range('2020-01-02', periods=5)
        prob_df = pd.DataFrame({
            'prob_riskoff': [0.1, 0.2, 0.3, 0.2, 0.1],
            'prob_neutral': [0.8, 0.6, 0.4, 0.6, 0.8],
            'prob_riskon': [0.1, 0.2, 0.3, 0.2, 0.1],
        }, index=dates)

        target_pos = engine.generate_signals(prob_df)
        assert (target_pos == 'Cash').all()

    def test_tqqq_when_riskon_above_threshold(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.5,
            risk_off_threshold=0.8,
            include_costs=False
        )

        dates = pd.bdate_range('2020-01-02', periods=3)
        prob_df = pd.DataFrame({
            'prob_riskoff': [0.1, 0.1, 0.1],
            'prob_neutral': [0.2, 0.2, 0.2],
            'prob_riskon': [0.7, 0.9, 0.6],
        }, index=dates)

        target_pos = engine.generate_signals(prob_df)
        assert (target_pos == 'TQQQ').all()

    def test_sqqq_when_riskoff_above_threshold(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.8,
            risk_off_threshold=0.5,
            include_costs=False
        )

        dates = pd.bdate_range('2020-01-02', periods=3)
        prob_df = pd.DataFrame({
            'prob_riskoff': [0.7, 0.9, 0.6],
            'prob_neutral': [0.2, 0.1, 0.3],
            'prob_riskon': [0.1, 0.0, 0.1],
        }, index=dates)

        target_pos = engine.generate_signals(prob_df)
        assert (target_pos == 'SQQQ').all()


class TestExecutionLag:
    def test_execution_lag_shifts_by_one(self):
        engine = PortfolioEngine(include_costs=False)

        dates = pd.bdate_range('2020-01-02', periods=5)
        target_pos = pd.Series(
            ['TQQQ', 'SQQQ', 'Cash', 'TQQQ', 'SQQQ'],
            index=dates,
            name='target_position'
        )

        actual_pos = engine.apply_execution_lag(target_pos)

        assert actual_pos.iloc[0] == 'Cash'

        for i in range(1, len(dates)):
            assert actual_pos.iloc[i] == target_pos.iloc[i - 1]

    def test_execution_lag_first_day_cash(self):
        engine = PortfolioEngine(include_costs=False)

        dates = pd.bdate_range('2020-01-02', periods=3)
        target_pos = pd.Series(
            ['TQQQ', 'TQQQ', 'TQQQ'],
            index=dates,
            name='target_position'
        )

        actual_pos = engine.apply_execution_lag(target_pos)

        assert actual_pos.iloc[0] == 'Cash'
        assert actual_pos.iloc[1] == 'TQQQ'
        assert actual_pos.iloc[2] == 'TQQQ'

    def test_run_includes_execution_lag(self):
        engine = PortfolioEngine(
            risk_on_threshold=0.0,
            risk_off_threshold=1.0,
            include_costs=False
        )

        n_days = 5
        prob_df = _create_mock_prob_df(n_days=n_days)
        prob_df['prob_riskon'] = 0.9
        prob_df['prob_riskoff'] = 0.05
        prob_df['prob_neutral'] = 0.05

        prices_df = _create_mock_prices(n_days=n_days)

        result = engine.run(prob_df, prices_df)

        assert result['actual_position'].iloc[0] == 'Cash'

        for i in range(1, len(result)):
            assert result['actual_position'].iloc[i] == result['target_position'].iloc[i - 1]


class TestTransactionCosts:
    def test_transaction_costs_on_turnover(self):
        engine = PortfolioEngine(
            commission_bps=5.0,
            slippage_bps=10.0,
            financing_cost_daily=0.0,
            include_costs=True
        )

        dates = pd.bdate_range('2020-01-02', periods=4)
        actual_pos = pd.Series(
            ['Cash', 'TQQQ', 'TQQQ', 'Cash'],
            index=dates,
            name='actual_position'
        )

        daily_returns = pd.Series(
            [0.0, 0.02, 0.01, -0.01],
            index=dates,
            name='daily_return'
        )

        cost_returns = engine.apply_transaction_costs(daily_returns, actual_pos)

        turnover_cost = (5.0 + 10.0) / 10000.0

        assert cost_returns.iloc[1] == daily_returns.iloc[1] - turnover_cost
        assert cost_returns.iloc[3] == daily_returns.iloc[3] - turnover_cost
        assert cost_returns.iloc[2] == daily_returns.iloc[2]

    def test_no_transaction_costs_when_include_costs_false(self):
        engine = PortfolioEngine(
            commission_bps=5.0,
            slippage_bps=10.0,
            financing_cost_daily=0.0002,
            include_costs=False
        )

        dates = pd.bdate_range('2020-01-02', periods=4)
        actual_pos = pd.Series(
            ['Cash', 'TQQQ', 'SQQQ', 'Cash'],
            index=dates,
            name='actual_position'
        )

        daily_returns = pd.Series(
            [0.0, 0.02, -0.03, 0.0],
            index=dates,
            name='daily_return'
        )

        cost_returns = engine.apply_transaction_costs(daily_returns, actual_pos)

        pd.testing.assert_series_equal(
            cost_returns.reset_index(drop=True),
            daily_returns.reset_index(drop=True),
            check_names=False
        )

    def test_transaction_costs_amount(self):
        engine = PortfolioEngine(
            commission_bps=5.0,
            slippage_bps=10.0,
            financing_cost_daily=0.0,
            include_costs=True
        )

        expected_turnover_cost = (5.0 + 10.0) / 10000.0
        assert abs(expected_turnover_cost - 0.0015) < 1e-10


class TestFinancingCostOnlyLeveraged:
    def test_financing_cost_only_tqqq_sqqq(self):
        engine = PortfolioEngine(
            commission_bps=0.0,
            slippage_bps=0.0,
            financing_cost_daily=0.0002,
            include_costs=True
        )

        dates = pd.bdate_range('2020-01-02', periods=6)
        actual_pos = pd.Series(
            ['Cash', 'TQQQ', 'TQQQ', 'SQQQ', 'Cash', 'Cash'],
            index=dates,
            name='actual_position'
        )

        daily_returns = pd.Series(
            [0.0, 0.01, 0.02, -0.03, 0.0, 0.0],
            index=dates,
            name='daily_return'
        )

        cost_returns = engine.apply_transaction_costs(daily_returns, actual_pos)

        financing = 0.0002

        assert cost_returns.iloc[0] == daily_returns.iloc[0]
        assert cost_returns.iloc[1] == daily_returns.iloc[1] - financing
        assert cost_returns.iloc[2] == daily_returns.iloc[2] - financing
        assert cost_returns.iloc[3] == daily_returns.iloc[3] - financing
        assert cost_returns.iloc[4] == daily_returns.iloc[4]
        assert cost_returns.iloc[5] == daily_returns.iloc[5]

    def test_cash_has_no_financing_cost(self):
        engine = PortfolioEngine(
            commission_bps=0.0,
            slippage_bps=0.0,
            financing_cost_daily=0.0002,
            include_costs=True
        )

        dates = pd.bdate_range('2020-01-02', periods=5)
        actual_pos = pd.Series(
            ['Cash', 'Cash', 'Cash', 'Cash', 'Cash'],
            index=dates,
            name='actual_position'
        )

        daily_returns = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            index=dates,
            name='daily_return'
        )

        cost_returns = engine.apply_transaction_costs(daily_returns, actual_pos)

        pd.testing.assert_series_equal(
            cost_returns.reset_index(drop=True),
            daily_returns.reset_index(drop=True),
            check_names=False
        )


class TestWalkForwardNoDataLeakage:
    def test_walkforward_train_test_no_overlap(self):
        from walkforward import WalkForwardValidator

        validator = WalkForwardValidator(
            frequency='yearly',
            window_mode='expanding',
            min_train_years=1,
            random_state=42,
        )

        dates = pd.bdate_range(start='2018-01-02', end='2022-12-30')

        splits = validator.generate_splits(dates)

        assert len(splits) > 0

        for train_start, train_end, test_start, test_end in splits:
            assert train_end < test_start, \
                f"训练集结束日期 {train_end} 应早于测试集开始日期 {test_start}"

            train_dates = dates[(dates >= train_start) & (dates <= train_end)]
            test_dates = dates[(dates >= test_start) & (dates <= test_end)]

            assert len(train_dates) > 0
            assert len(test_dates) > 0

            overlap = set(train_dates) & set(test_dates)
            assert len(overlap) == 0, \
                f"训练集和测试集有重叠: {overlap}"

    def test_walkforward_rolling_window_no_future_data(self):
        from walkforward import WalkForwardValidator

        validator = WalkForwardValidator(
            frequency='yearly',
            window_mode='rolling',
            window_years=2,
            min_train_years=1,
            random_state=42,
        )

        dates = pd.bdate_range(start='2017-01-02', end='2022-12-30')

        splits = validator.generate_splits(dates)

        assert len(splits) > 0

        for train_start, train_end, test_start, test_end in splits:
            assert train_end < test_start

            train_days = len(dates[(dates >= train_start) & (dates <= train_end)])
            max_days = int(252 * (validator.window_years + 0.5))
            assert train_days <= max_days, \
                f"训练集天数 {train_days} 超过预期最大值 {max_days}"

    def test_walkforward_expanding_train_grows(self):
        from walkforward import WalkForwardValidator

        validator = WalkForwardValidator(
            frequency='yearly',
            window_mode='expanding',
            min_train_years=1,
            random_state=42,
        )

        dates = pd.bdate_range(start='2018-01-02', end='2022-12-30')

        splits = validator.generate_splits(dates)

        if len(splits) >= 2:
            first_train_start = splits[0][0]
            for train_start, _, _, _ in splits:
                assert train_start == first_train_start, \
                    "expanding 模式下训练集起始日期应保持不变"


class TestDailyReturns:
    def test_daily_returns_tqqq(self):
        engine = PortfolioEngine(include_costs=False)

        dates = pd.bdate_range('2020-01-02', periods=5)
        prices_df = pd.DataFrame({
            'TQQQ': [50.0, 55.0, 60.0, 57.0, 60.0],
            'SQQQ': [50.0, 45.0, 40.0, 43.0, 40.0],
        }, index=dates)

        actual_pos = pd.Series(
            ['TQQQ', 'TQQQ', 'TQQQ', 'TQQQ', 'TQQQ'],
            index=dates,
            name='actual_position'
        )

        daily_ret = engine.calculate_daily_returns(actual_pos, prices_df)

        assert daily_ret.iloc[0] == 0.0
        assert abs(daily_ret.iloc[1] - 0.10) < 1e-10
        assert abs(daily_ret.iloc[2] - (60/55 - 1)) < 1e-10

    def test_daily_returns_cash_zero(self):
        engine = PortfolioEngine(include_costs=False)

        dates = pd.bdate_range('2020-01-02', periods=5)
        prices_df = _create_mock_prices(n_days=5)

        actual_pos = pd.Series(
            ['Cash', 'Cash', 'Cash', 'Cash', 'Cash'],
            index=dates,
            name='actual_position'
        )

        daily_ret = engine.calculate_daily_returns(actual_pos, prices_df)

        assert (daily_ret == 0.0).all()


class TestNavCalculation:
    def test_nav_starts_at_initial_value(self):
        engine = PortfolioEngine(include_costs=False)

        daily_returns = pd.Series(
            [0.0, 0.01, 0.02, -0.01],
            index=pd.bdate_range('2020-01-02', periods=4)
        )

        nav = engine.calculate_nav(daily_returns, initial_value=1.0)
        assert nav.iloc[0] == 1.0

    def test_nav_cumulative_product(self):
        engine = PortfolioEngine(include_costs=False)

        daily_returns = pd.Series(
            [0.0, 0.10, 0.0],
            index=pd.bdate_range('2020-01-02', periods=3)
        )

        nav = engine.calculate_nav(daily_returns, initial_value=100.0)

        assert abs(nav.iloc[0] - 100.0) < 1e-10
        assert abs(nav.iloc[1] - 100.0 * 1.10) < 1e-10
        assert abs(nav.iloc[2] - 100.0 * 1.10) < 1e-10
