import os
import sys
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from train_model import ModelTrainer
from portfolio_engine import PortfolioEngine, calculate_max_drawdown

# Regime Agent 相关导入
from regime import (
    RuleBasedRegimeAgent,
    MacroAgent,
    SentimentAgent,
    RegimeAggregator,
    RegimeAwareModelWrapper,
    MacroSnapshot,
    SentimentSnapshot,
)


class WalkForwardValidator:
    """Walk-Forward 验证器，执行滚动窗口样本外验证。

    支持多种滚动频率（年/季/月）和窗口模式（扩张/滚动），
    通过切分训练集和测试集，模拟真实交易场景下的模型表现。

    Attributes:
        frequency: 滚动频率，'yearly' | 'quarterly' | 'monthly'
        window_mode: 窗口模式，'expanding' | 'rolling'
        window_years: rolling 模式下的窗口年数
        min_train_years: 最小训练年数
        random_state: 随机种子
        feature_df: 特征数据 DataFrame
        prices_df: 价格数据 DataFrame
        feature_cols: 特征列名列表
        trainer: 模型训练器实例
    """

    VALID_FREQUENCIES = {'yearly', 'quarterly', 'monthly'}
    VALID_WINDOW_MODES = {'expanding', 'rolling'}

    def __init__(
        self,
        frequency: str = 'quarterly',
        window_mode: str = 'expanding',
        window_years: int = 5,
        min_train_years: int = 5,
        embargo_days: int = 5,
        random_state: int = 42,
        model_type: str = 'lightgbm',
        portfolio_config: Optional[Dict[str, Any]] = None,
        # Regime Agent 配置
        use_regime: bool = False,
        regime_agent_type: str = 'rule_based',  # 'rule_based', 'llm_macro', 'llm_both'
        regime_llm_provider: str = 'mock',       # 'mock', 'openai', 'anthropic'
        regime_llm_model: str = None,
        regime_macro_weight: float = 0.6,
        regime_sentiment_weight: float = 0.4,
    ):
        """初始化 Walk-Forward 验证器。

        Args:
            frequency: 滚动频率，默认 'quarterly'
            window_mode: 窗口模式，默认 'expanding'
            window_years: rolling 模式下的窗口年数，默认 5
            min_train_years: 最小训练年数，默认 5
            embargo_days: 数据隔离期天数，默认 5（防未来函数）
            random_state: 随机种子，默认 42
            model_type: 模型类型，默认 'lightgbm'
            portfolio_config: 组合配置字典
            use_regime: 是否使用 Regime Agent
            regime_agent_type: Regime Agent 类型 ('rule_based', 'llm_macro', 'llm_both')
            regime_llm_provider: LLM 提供商
            regime_llm_model: LLM 模型名称
            regime_macro_weight: 宏观 Agent 权重
            regime_sentiment_weight: 消息面 Agent 权重
        """
        if frequency not in self.VALID_FREQUENCIES:
            raise ValueError(
                f"无效的 frequency: {frequency}，"
                f"可选值: {self.VALID_FREQUENCIES}"
            )
        if window_mode not in self.VALID_WINDOW_MODES:
            raise ValueError(
                f"无效的 window_mode: {window_mode}，"
                f"可选值: {self.VALID_WINDOW_MODES}"
            )

        self.frequency = frequency
        self.window_mode = window_mode
        self.window_years = window_years
        self.min_train_years = min_train_years
        self.embargo_days = embargo_days
        self.random_state = random_state
        self.model_type = model_type
        self.portfolio_config = portfolio_config or {
            'risk_on_threshold': 0.6,
            'risk_off_threshold': 0.6,
            'shares_per_trade': 100,
            'use_real_costs': True,
            'commission_bps': 5.0,
            'slippage_bps': 10.0,
            'include_costs': True,
        }

        # Regime Agent 配置
        self.use_regime = use_regime
        self.regime_agent_type = regime_agent_type
        self.regime_llm_provider = regime_llm_provider
        self.regime_llm_model = regime_llm_model
        self.regime_macro_weight = regime_macro_weight
        self.regime_sentiment_weight = regime_sentiment_weight
        self.disable_short = portfolio_config.get('disable_short', False)

        self.feature_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None
        self.trainer = ModelTrainer(random_state=random_state)

        # Regime Agent 实例（延迟初始化）
        self._regime_agents = None

    def _init_regime_agents(self):
        """初始化 Regime Agent"""
        if self._regime_agents is not None:
            return

        if self.regime_agent_type == 'rule_based':
            self._regime_agents = {
                'rule': RuleBasedRegimeAgent(),
                'aggregator': None,
            }
        elif self.regime_agent_type == 'llm_macro':
            self._regime_agents = {
                'macro': MacroAgent(
                    provider=self.regime_llm_provider,
                    model=self.regime_llm_model,
                ),
                'aggregator': RegimeAggregator(
                    macro_weight=1.0,
                    sentiment_weight=0.0,
                ),
            }
        elif self.regime_agent_type == 'llm_both':
            self._regime_agents = {
                'macro': MacroAgent(
                    provider=self.regime_llm_provider,
                    model=self.regime_llm_model,
                ),
                'sentiment': SentimentAgent(
                    provider=self.regime_llm_provider,
                    model=self.regime_llm_model,
                ),
                'aggregator': RegimeAggregator(
                    macro_weight=self.regime_macro_weight,
                    sentiment_weight=self.regime_sentiment_weight,
                ),
            }
        else:
            raise ValueError(f"Unknown regime_agent_type: {self.regime_agent_type}")

    def _build_macro_snapshot(self, date: pd.Timestamp) -> Optional[MacroSnapshot]:
        """从特征数据构建宏观快照"""
        if self.feature_df is None:
            return None

        # 获取 date 之前的最近数据
        mask = self.feature_df.index <= date
        if not mask.any():
            return None

        row = self.feature_df.loc[mask].iloc[-1]

        return MacroSnapshot(
            date=str(date.date()),
            vix_level=row.get('vix_level', 20.0),
            vix_5d_change=row.get('vix_change_5d', 0.0) * 100,  # 转为百分比
            vix_20d_ma_ratio=row.get('vix_ma20_ratio', 1.0),
            vix_60d_percentile=row.get('vix_quantile_60d', 0.5),
            tnx_level=row.get('tnx_level', 4.0),
            tnx_5d_change=row.get('tnx_change_5d', 0.0) * 100,  # 转为 bps
            tnx_trend=self._get_tnx_trend(row),
            dxy_level=row.get('dxy_level', 100.0),
            dxy_20d_change=row.get('dxy_change_20d', 0.0) * 100,  # 转为百分比
            spy_20d_return=row.get('spy_ret_20d', 0.0),
            spy_60d_trend=self._get_spy_trend(row),
            qqq_spy_relative=self._get_qqq_spy_relative(row),
        )

    def _get_tnx_trend(self, row: pd.Series) -> str:
        """判断 TNX 趋势"""
        change = row.get('tnx_change_20d', 0.0)
        if change > 0.01:
            return "rising"
        elif change < -0.01:
            return "falling"
        return "flat"

    def _get_spy_trend(self, row: pd.Series) -> str:
        """判断 SPY 趋势"""
        ret = row.get('spy_ret_20d', 0.0)
        if ret > 0.03:
            return "uptrend"
        elif ret < -0.03:
            return "downtrend"
        return "sideways"

    def _get_qqq_spy_relative(self, row: pd.Series) -> str:
        """判断 QQQ vs SPY 相对表现"""
        rs_ret = row.get('rs_qv_spy_ret_20d', 0.0)
        if rs_ret > 0.02:
            return "outperforming"
        elif rs_ret < -0.02:
            return "underperforming"
        return "neutral"

    def _get_regime_for_date(
        self,
        date: pd.Timestamp,
    ) -> Optional[Any]:
        """获取特定日期的 Regime 参数"""
        self._init_regime_agents()

        macro_snapshot = self._build_macro_snapshot(date)
        if macro_snapshot is None:
            return None

        agents = self._regime_agents

        if self.regime_agent_type == 'rule_based':
            return agents['rule'].predict(macro_snapshot=macro_snapshot)

        elif self.regime_agent_type == 'llm_macro':
            macro_output = agents['macro'].predict(macro_snapshot)
            return agents['aggregator'].aggregate(macro_output=macro_output)

        elif self.regime_agent_type == 'llm_both':
            macro_output = agents['macro'].predict(macro_snapshot)
            sentiment_snapshot = self._build_sentiment_snapshot(date)
            sentiment_output = None
            if sentiment_snapshot:
                sentiment_output = agents['sentiment'].predict(sentiment_snapshot)
            return agents['aggregator'].aggregate(
                macro_output=macro_output,
                sentiment_output=sentiment_output,
            )

        return None

    def _build_sentiment_snapshot(self, date: pd.Timestamp) -> Optional[SentimentSnapshot]:
        """从特征数据构建消息面快照"""
        # 目前先用简单实现，后续可扩展
        return SentimentSnapshot(
            date=str(date.date()),
            news_count_7d=0,
            news_sentiment_score=0.0,
            put_call_ratio=1.0,
            wsb_sentiment="neutral",
            stocktwits_sentiment="neutral",
        )

    def load_data(
        self,
        feature_path: str = 'data/processed/features.parquet',
        price_dir: str = 'data/raw',
    ) -> None:
        """加载特征数据和价格数据。

        Args:
            feature_path: 特征数据文件路径
            price_dir: 价格数据目录路径

        Raises:
            FileNotFoundError: 当数据文件不存在时
        """
        print("=" * 60)
        print("Walk-Forward: 加载数据")
        print("=" * 60)

        self.feature_df = self.trainer.load_data(feature_path)
        _, _, self.feature_cols = self.trainer.prepare_data(self.feature_df)

        engine = PortfolioEngine()
        self.prices_df = engine.load_prices(price_dir)

        print(f"特征数据区间: {self.feature_df.index[0].date()} ~ "
              f"{self.feature_df.index[-1].date()}")
        print(f"价格数据区间: {self.prices_df.index[0].date()} ~ "
              f"{self.prices_df.index[-1].date()}")

    def generate_splits(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """生成训练/测试切分列表。

        根据配置的频率和窗口模式，生成多组 (train_start, train_end,
        test_start, test_end) 日期元组。

        Args:
            dates: 数据的日期索引

        Returns:
            切分元组列表，每个元组为 (train_start, train_end, test_start, test_end)

        Raises:
            ValueError: 当数据不足以满足最小训练年数时
        """
        if len(dates) < 252 * self.min_train_years:
            raise ValueError(
                f"数据量不足，需要至少 {self.min_train_years} 年数据 "
                f"(约 {252 * self.min_train_years} 个交易日)，"
                f"实际只有 {len(dates)} 个交易日"
            )

        if self.frequency == 'yearly':
            raw_splits = self._split_by_year(dates)
        elif self.frequency == 'quarterly':
            raw_splits = self._split_by_quarter(dates)
        else:
            raw_splits = self._split_by_month(dates)

        filtered_splits = []
        for test_start, test_end in raw_splits:
            train_candidates_end = dates[dates < test_start]
            if len(train_candidates_end) <= self.embargo_days:
                continue
            train_end = train_candidates_end[-(self.embargo_days + 1)]

            if self.window_mode == 'expanding':
                train_start = dates[0]
            else:
                cutoff = test_start - pd.DateOffset(years=self.window_years)
                train_candidates = dates[dates >= cutoff]
                train_candidates = train_candidates[train_candidates < test_start]
                if len(train_candidates) == 0:
                    continue
                train_start = train_candidates[0]

            train_days = len(dates[(dates >= train_start) & (dates <= train_end)])
            min_days = 252 * self.min_train_years
            if train_days < min_days:
                continue

            filtered_splits.append((train_start, train_end, test_start, test_end))

        print(f"生成 {len(filtered_splits)} 个切分 "
              f"(frequency={self.frequency}, mode={self.window_mode})")
        for i, (ts, te, tes, tee) in enumerate(filtered_splits):
            print(f"  Split {i + 1}: 训练 {ts.date()} ~ {te.date()} | "
                  f"测试 {tes.date()} ~ {tee.date()}")

        return filtered_splits

    def _split_by_year(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """按年频生成测试区间。

        每个测试区间为一个完整的日历年。

        Args:
            dates: 数据的日期索引

        Returns:
            (test_start, test_end) 元组列表
        """
        years = sorted(dates.year.unique())
        splits = []

        for year in years:
            year_dates = dates[dates.year == year]
            if len(year_dates) == 0:
                continue
            test_start = year_dates[0]
            test_end = year_dates[-1]
            splits.append((test_start, test_end))

        return splits

    def _split_by_quarter(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """按季频生成测试区间。

        每个测试区间为一个完整的季度。

        Args:
            dates: 数据的日期索引

        Returns:
            (test_start, test_end) 元组列表
        """
        df_temp = pd.DataFrame(index=dates)
        df_temp['year'] = df_temp.index.year
        df_temp['quarter'] = df_temp.index.quarter

        splits = []
        grouped = df_temp.groupby(['year', 'quarter'])

        for (year, quarter), group in grouped:
            if len(group) == 0:
                continue
            test_start = group.index[0]
            test_end = group.index[-1]
            splits.append((test_start, test_end))

        return splits

    def _split_by_month(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """按月频生成测试区间。

        每个测试区间为一个完整的月份。

        Args:
            dates: 数据的日期索引

        Returns:
            (test_start, test_end) 元组列表
        """
        df_temp = pd.DataFrame(index=dates)
        df_temp['year'] = df_temp.index.year
        df_temp['month'] = df_temp.index.month

        splits = []
        grouped = df_temp.groupby(['year', 'month'])

        for (year, month), group in grouped:
            if len(group) == 0:
                continue
            test_start = group.index[0]
            test_end = group.index[-1]
            splits.append((test_start, test_end))

        return splits

    def run_single_split(
        self,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """运行单个切分的训练和测试。

        在训练集上训练 LightGBM 模型，在测试集上预测概率，
        并通过组合引擎计算收益。

        Args:
            train_idx: 训练集日期索引
            test_idx: 测试集日期索引

        Returns:
            测试集结果 DataFrame，包含 prob、position、return、nav 等列
        """
        if self.feature_df is None or self.prices_df is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        train_df = self.feature_df.loc[train_idx].copy()
        test_df = self.feature_df.loc[test_idx].copy()

        X_train, y_train, _ = self.trainer.prepare_data(train_df)
        X_test, y_test, _ = self.trainer.prepare_data(test_df)

        model_type = self.model_type
        if model_type == 'lightgbm':
            model = self.trainer.train_lightgbm(X_train, y_train)
        elif model_type == 'random_forest':
            model = self.trainer.train_random_forest(X_train, y_train)
        elif model_type == 'logistic_regression':
            model = self.trainer.train_logistic_regression(X_train, y_train)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        prob_df = self.trainer.predict_proba(model, X_test)

        # 如果启用 Regime Agent，获取基础阈值调整
        base_risk_on_thresh = self.portfolio_config.get('risk_on_threshold', 0.6)
        base_risk_off_thresh = self.portfolio_config.get('risk_off_threshold', 0.6)
        risk_on_adj = 0.0
        risk_off_adj = 0.0

        if self.use_regime and len(test_idx) > 0:
            # 获取第一个测试日期的 Regime 参数
            first_test_date = test_idx[0]
            regime = self._get_regime_for_date(first_test_date)

            if regime is not None and hasattr(regime, 'params') and regime.params:
                risk_on_adj = regime.params.risk_on_threshold_adjustment
                risk_off_adj = regime.params.risk_off_threshold_adjustment
                print(f"  Regime: {regime.sentiment.value}, "
                      f"risk_on_adj={risk_on_adj:+.2f}, "
                      f"risk_off_adj={risk_off_adj:+.2f}")

        pc = self.portfolio_config.copy()
        pc['risk_on_threshold'] = base_risk_on_thresh + risk_on_adj
        pc['risk_off_threshold'] = base_risk_off_thresh + risk_off_adj

        engine = PortfolioEngine(
            risk_on_threshold=pc.get('risk_on_threshold', 0.6),
            risk_off_threshold=pc.get('risk_off_threshold', 0.6),
            shares_per_trade=pc.get('shares_per_trade', 100),
            use_real_costs=pc.get('use_real_costs', True),
            commission_bps=pc.get('commission_bps', 5.0),
            slippage_bps=pc.get('slippage_bps', 10.0),
            include_costs=pc.get('include_costs', True),
            disable_short=pc.get('disable_short', False),
        )

        # 获取 Regime 参数（如果启用）
        regime_params = None
        if self.use_regime and len(test_idx) > 0:
            first_test_date = test_idx[0]
            regime = self._get_regime_for_date(first_test_date)
            if regime is not None and hasattr(regime, 'params') and regime.params:
                regime_params = regime.params
                print(f"  Regime: {regime.sentiment.value}, "
                      f"risk_on_adj={regime_params.risk_on_threshold_adjustment:+.2f}, "
                      f"risk_off_adj={regime_params.risk_off_threshold_adjustment:+.2f}")

        portfolio_result = engine.run(prob_df, self.prices_df, regime_params=regime_params)

        result = prob_df.join(portfolio_result, how='inner')

        train_accuracy = self._calc_accuracy(model, X_train, y_train)
        test_accuracy = self._calc_accuracy(model, X_test, y_test)
        print(f"训练集准确率: {train_accuracy:.4f} | "
              f"测试集准确率: {test_accuracy:.4f}")

        return result

    def _calc_accuracy(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """计算模型准确率。

        Args:
            model: 训练好的模型
            X: 特征数据
            y: 真实标签

        Returns:
            准确率
        """
        prob_df = self.trainer.predict_proba(model, X)
        y_pred = prob_df.idxmax(axis=1).map({
            'prob_sqqq': 0,
            'prob_psq': 1,
            'prob_cash': 2,
            'prob_qqq': 3,
            'prob_tqqq': 4,
        })
        return (y_pred.values == y.values).mean()

    def run(self) -> pd.DataFrame:
        """运行完整 Walk-Forward 验证。

        遍历所有切分，分别训练模型并在测试集上预测，
        最后拼接所有样本外结果。

        Returns:
            完整的样本外结果 DataFrame，包含 prob、position、return、nav 等列

        Raises:
            ValueError: 当数据未加载时
        """
        if self.feature_df is None or self.prices_df is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        print("\n" + "=" * 60)
        print("Walk-Forward 验证开始")
        print("=" * 60)
        print(f"配置: frequency={self.frequency}, "
              f"window_mode={self.window_mode}, "
              f"window_years={self.window_years}, "
              f"min_train_years={self.min_train_years}")

        dates = self.feature_df.dropna(subset=['label']).index
        splits = self.generate_splits(dates)

        if len(splits) == 0:
            raise ValueError("没有生成有效的切分，请检查参数设置")

        all_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            print(f"\n{'=' * 60}")
            print(f"Split {i + 1}/{len(splits)}: "
                  f"{test_start.date()} ~ {test_end.date()}")
            print(f"{'=' * 60}")

            train_dates = dates[(dates >= train_start) & (dates <= train_end)]
            test_dates = dates[(dates >= test_start) & (dates <= test_end)]

            if len(train_dates) == 0 or len(test_dates) == 0:
                print(f"跳过 Split {i + 1}: 训练集或测试集为空")
                continue

            try:
                split_result = self.run_single_split(train_dates, test_dates)
                all_results.append(split_result)
                print(f"Split {i + 1} 完成，测试集 {len(split_result)} 行")
            except Exception as e:
                print(f"Split {i + 1} 失败: {e}")
                continue

        if len(all_results) == 0:
            raise RuntimeError("所有切分均失败")

        full_result = pd.concat(all_results, axis=0)
        full_result = full_result.sort_index()

        full_result = self._compute_combined_nav(full_result)

        print("\n" + "=" * 60)
        print(f"Walk-Forward 验证完成")
        print(f"样本外总交易日: {len(full_result)}")
        print(f"样本外区间: {full_result.index[0].date()} ~ "
              f"{full_result.index[-1].date()}")
        print("=" * 60)

        return full_result

    def _compute_combined_nav(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """重新计算整个样本外区间的累计净值。

        Args:
            result_df: 拼接后的结果 DataFrame

        Returns:
            添加了整体 nav 的结果 DataFrame
        """
        result_df = result_df.copy()

        if 'daily_return' in result_df.columns:
            result_df['nav_combined'] = (
                1.0 + result_df['daily_return']
            ).cumprod()

        if 'daily_return_cost' in result_df.columns:
            result_df['nav_cost_combined'] = (
                1.0 + result_df['daily_return_cost']
            ).cumprod()

        return result_df

    def compare_schemes(
        self,
        schemes: List[Dict],
    ) -> pd.DataFrame:
        """对比多种 Walk-Forward 方案的绩效。

        Args:
            schemes: 方案列表，每个 dict 包含 frequency, window_mode, window_years

        Returns:
            各方案的绩效对比 DataFrame，包含 CAGR、Sharpe、MaxDD 等指标
        """
        if self.feature_df is None or self.prices_df is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        results = {}

        for i, scheme in enumerate(schemes):
            scheme_name = scheme.get(
                'name',
                f"Scheme_{i + 1}_{scheme.get('frequency', '?')}_"
                f"{scheme.get('window_mode', '?')}"
            )
            print(f"\n{'#' * 60}")
            print(f"运行方案: {scheme_name}")
            print(f"{'#' * 60}")

            validator = WalkForwardValidator(
                frequency=scheme.get('frequency', 'quarterly'),
                window_mode=scheme.get('window_mode', 'expanding'),
                window_years=scheme.get('window_years', 5),
                min_train_years=scheme.get('min_train_years', self.min_train_years),
                random_state=self.random_state,
            )
            validator.feature_df = self.feature_df
            validator.prices_df = self.prices_df
            validator.feature_cols = self.feature_cols

            try:
                result = validator.run()
                stats = self._compute_performance_stats(result)
                results[scheme_name] = stats
                print(f"\n方案 {scheme_name} 完成")
            except Exception as e:
                print(f"\n方案 {scheme_name} 失败: {e}")
                continue

        comparison_df = pd.DataFrame(results).T
        print("\n" + "=" * 60)
        print("方案对比")
        print("=" * 60)
        print(comparison_df.to_string())

        return comparison_df

    def _compute_performance_stats(
        self,
        result_df: pd.DataFrame,
    ) -> Dict:
        """计算绩效统计指标。

        Args:
            result_df: 结果 DataFrame

        Returns:
            绩效指标字典
        """
        stats = {}

        total_days = len(result_df)
        years = total_days / 252.0
        stats['总交易日'] = total_days
        stats['年数'] = round(years, 2)

        if 'nav_combined' in result_df.columns:
            nav = result_df['nav_combined']
            final_nav = nav.iloc[-1]
            total_return = final_nav - 1.0
            cagr = final_nav ** (1.0 / years) - 1.0 if years > 0 else 0.0
            daily_ret = result_df['daily_return']
            annual_vol = daily_ret.std() * np.sqrt(252)
            sharpe = cagr / annual_vol if annual_vol > 0 else 0.0
            max_dd, dd_start, dd_end = calculate_max_drawdown(nav)

            stats['总收益(无成本)'] = f"{total_return:.2%}"
            stats['CAGR(无成本)'] = f"{cagr:.2%}"
            stats['夏普(无成本)'] = round(sharpe, 4)
            stats['最大回撤(无成本)'] = f"{max_dd:.2%}"

        if 'nav_cost_combined' in result_df.columns:
            nav_cost = result_df['nav_cost_combined']
            final_nav_cost = nav_cost.iloc[-1]
            total_return_cost = final_nav_cost - 1.0
            cagr_cost = final_nav_cost ** (1.0 / years) - 1.0 if years > 0 else 0.0
            daily_ret_cost = result_df['daily_return_cost']
            annual_vol_cost = daily_ret_cost.std() * np.sqrt(252)
            sharpe_cost = cagr_cost / annual_vol_cost if annual_vol_cost > 0 else 0.0
            max_dd_cost, _, _ = calculate_max_drawdown(nav_cost)

            stats['总收益(有成本)'] = f"{total_return_cost:.2%}"
            stats['CAGR(有成本)'] = f"{cagr_cost:.2%}"
            stats['夏普(有成本)'] = round(sharpe_cost, 4)
            stats['最大回撤(有成本)'] = f"{max_dd_cost:.2%}"

        return stats

    def save_results(
        self,
        results: pd.DataFrame,
        path: str = 'data/processed/walkforward_results.parquet',
    ) -> str:
        """保存结果为 Parquet 格式。

        Args:
            results: 结果 DataFrame
            path: 保存路径

        Returns:
            保存文件的完整路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results.to_parquet(path, engine='pyarrow')
        print(f"Walk-Forward 结果已保存到: {path}")
        print(f"共 {len(results)} 行, {len(results.columns)} 列")
        return path


def print_wf_performance(result: pd.DataFrame) -> None:
    """打印 Walk-Forward 样本外套绩统计。

    Args:
        result: Walk-Forward 结果 DataFrame
    """
    print("\n" + "=" * 60)
    print("Walk-Forward 样本外套绩统计")
    print("=" * 60)

    total_days = len(result)
    years = total_days / 252.0

    print(f"样本外交易天数: {total_days} ({years:.2f} 年)")
    print(f"样本外区间: {result.index[0].date()} ~ {result.index[-1].date()}")
    print()

    nav_col = 'nav_combined' if 'nav_combined' in result.columns else 'nav'
    nav_cost_col = ('nav_cost_combined'
                    if 'nav_cost_combined' in result.columns
                    else 'nav_cost')

    if nav_col in result.columns:
        nav = result[nav_col]
        final_nav = nav.iloc[-1]
        total_return = final_nav - 1.0
        cagr = final_nav ** (1.0 / years) - 1.0 if years > 0 else 0.0
        daily_ret = result['daily_return']
        annual_vol = daily_ret.std() * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0.0
        max_dd, dd_start, dd_end = calculate_max_drawdown(nav)

        print(f"{'指标':<20s} {'无成本':>12s}")
        print("-" * 35)
        print(f"{'最终净值':<20s} {final_nav:>12.4f}")
        print(f"{'总收益率':<20s} {total_return:>12.2%}")
        print(f"{'年化收益率(CAGR)':<20s} {cagr:>12.2%}")
        print(f"{'年化波动率':<20s} {annual_vol:>12.2%}")
        print(f"{'夏普比率':<20s} {sharpe:>12.4f}")
        print(f"{'最大回撤':<20s} {max_dd:>12.2%}")
        print(f"{'回撤起点':<20s} {str(dd_start.date()):>12s}")
        print(f"{'回撤终点':<20s} {str(dd_end.date()):>12s}")
        print()

    if nav_cost_col in result.columns:
        nav_cost = result[nav_cost_col]
        final_nav_cost = nav_cost.iloc[-1]
        total_return_cost = final_nav_cost - 1.0
        cagr_cost = final_nav_cost ** (1.0 / years) - 1.0 if years > 0 else 0.0
        daily_ret_cost = result['daily_return_cost']
        annual_vol_cost = daily_ret_cost.std() * np.sqrt(252)
        sharpe_cost = cagr_cost / annual_vol_cost if annual_vol_cost > 0 else 0.0
        max_dd_cost, dd_start_cost, dd_end_cost = calculate_max_drawdown(nav_cost)

        print(f"{'指标':<20s} {'有成本':>12s}")
        print("-" * 35)
        print(f"{'最终净值':<20s} {final_nav_cost:>12.4f}")
        print(f"{'总收益率':<20s} {total_return_cost:>12.2%}")
        print(f"{'年化收益率(CAGR)':<20s} {cagr_cost:>12.2%}")
        print(f"{'年化波动率':<20s} {annual_vol_cost:>12.2%}")
        print(f"{'夏普比率':<20s} {sharpe_cost:>12.4f}")
        print(f"{'最大回撤':<20s} {max_dd_cost:>12.2%}")
        print(f"{'回撤起点':<20s} {str(dd_start_cost.date()):>12s}")
        print(f"{'回撤终点':<20s} {str(dd_end_cost.date()):>12s}")

    if 'actual_position' in result.columns:
        position_counts = result['actual_position'].value_counts()
        print(f"\n实际持仓分布:")
        for pos in ['TQQQ', 'SQQQ', 'Cash']:
            cnt = position_counts.get(pos, 0)
            pct = cnt / len(result) * 100
            print(f"  {pos}: {cnt} 天 ({pct:.1f}%)")

    print("=" * 60)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Walk-Forward 验证 - 主程序")
    print("=" * 60)

    # 检查是否启用 Regime Agent
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward 验证')
    parser.add_argument('--use-regime', action='store_true',
                        help='启用 Regime Agent')
    parser.add_argument('--regime-type', type=str, default='rule_based',
                        choices=['rule_based', 'llm_macro', 'llm_both'],
                        help='Regime Agent 类型')
    parser.add_argument('--llm-provider', type=str, default='mock',
                        choices=['mock', 'openai', 'anthropic'],
                        help='LLM 提供商')
    args = parser.parse_args()

    validator = WalkForwardValidator(
        frequency='quarterly',
        window_mode='expanding',
        window_years=5,
        min_train_years=5,
        random_state=42,
        use_regime=args.use_regime,
        regime_agent_type=args.regime_type,
        regime_llm_provider=args.llm_provider,
    )

    if args.use_regime:
        print(f"\n[Regime Agent 已启用]")
        print(f"  Agent 类型: {args.regime_type}")
        if args.regime_type != 'rule_based':
            print(f"  LLM 提供商: {args.llm_provider}")

    try:
        validator.load_data(
            feature_path='data/processed/features.parquet',
            price_dir='data/raw',
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请先运行数据准备脚本生成特征数据，例如:")
        print("  python src/feature_engineering.py")
        sys.exit(1)

    result = validator.run()

    print_wf_performance(result)

    save_path = 'data/processed/walkforward_results.parquet'
    validator.save_results(result, save_path)

    print("\n" + "=" * 60)
    print("可选: 运行 3 种方案对比")
    print("=" * 60)
    run_comparison = input("是否运行方案对比? (y/n): ").strip().lower()

    if run_comparison == 'y':
        schemes = [
            {
                'name': '季频扩张',
                'frequency': 'quarterly',
                'window_mode': 'expanding',
                'window_years': 5,
            },
            {
                'name': '年频扩张',
                'frequency': 'yearly',
                'window_mode': 'expanding',
                'window_years': 5,
            },
            {
                'name': '季频滚动(5年)',
                'frequency': 'quarterly',
                'window_mode': 'rolling',
                'window_years': 5,
            },
        ]
        comparison = validator.compare_schemes(schemes)
        print("\n对比表已生成")
