import os
import sys
import warnings
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
legacy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'legacy'))
if os.path.exists(legacy_path) and legacy_path not in sys.path:
    sys.path.insert(0, legacy_path)

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from label_generator import LabelGenerator
from train_model import ModelTrainer
from walkforward import WalkForwardValidator, print_wf_performance
from evaluate import PerformanceEvaluator
from visualization import Visualizer


warnings.filterwarnings('ignore')


class BacktestEngine:
    """主回测引擎，串联完整量化策略流程。

    负责从数据下载、特征工程、标签生成、模型训练、
    Walk-Forward 验证、绩效评估到可视化的全流程管理。

    Attributes:
        config: 配置字典
        data_loader: 数据加载器实例
        feature_engineer: 特征工程实例
        label_generator: 标签生成器实例
        model_trainer: 模型训练器实例
        wf_validator: Walk-Forward 验证器实例
        evaluator: 绩效评估器实例
        visualizer: 可视化器实例
        results: 存储各阶段结果的字典
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化回测引擎。

        Args:
            config: 配置字典，为 None 时使用默认配置
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        self.data_loader = None
        self.feature_engineer = None
        self.label_generator = None
        self.model_trainer = None
        self.wf_validator = None
        self.evaluator = None
        self.visualizer = None

        self.results: Dict[str, Any] = {}

        self._init_dirs()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置。

        Returns:
            默认配置字典
        """
        return {
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'start_date': '2011-01-01',
                'end_date': None,
            },
            'label': {
                'horizon': 5,
                'threshold': 0.02,
                'use_quantile': False,
                'quantile': 0.3,
            },
            'model': {
                'random_state': 42,
                'model_type': 'lightgbm',
            },
            'walkforward': {
                'frequency': 'quarterly',
                'window_mode': 'expanding',
                'window_years': 5,
                'min_train_years': 5,
            },
            'portfolio': {
                'risk_on_threshold': 0.6,
                'risk_off_threshold': 0.6,
                'shares_per_trade': 100,
                'use_real_costs': True,
                'commission_bps': 5.0,
                'slippage_bps': 10.0,
                'include_costs': True,
            },
            'evaluation': {
                'risk_free_rate': 0.0,
            },
            'visualization': {
                'save_dir': 'reports/figures',
                'style': 'seaborn-v0_8',
                'figsize': (12, 6),
                'dpi': 150,
            },
        }

    def _init_dirs(self) -> None:
        """初始化必要的目录。"""
        dirs = [
            self.config['data']['raw_dir'],
            self.config['data']['processed_dir'],
            self.config['visualization']['save_dir'],
            'models',
            'reports',
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _check_data_exists(self) -> bool:
        """检查原始数据是否已存在。

        Returns:
            所有必要数据文件是否都存在
        """
        raw_dir = self.config['data']['raw_dir']
        required_files = ['qqq.csv', 'tqqq.csv', 'sqqq.csv', 'spy.csv']
        for f in required_files:
            if not os.path.exists(os.path.join(raw_dir, f)):
                return False
        return True

    def step_load_data(self) -> Dict[str, pd.DataFrame]:
        """步骤 1: 加载/下载数据。

        Returns:
            加载的数据字典
        """
        print("\n" + "=" * 80)
        print("步骤 1/7: 数据加载")
        print("=" * 80)

        self.data_loader = DataLoader(
            save_dir=self.config['data']['raw_dir'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
        )

        if not self._check_data_exists():
            print("原始数据不存在，开始下载...")
            results = self.data_loader.download_all()
            data = results['data']
        else:
            print("原始数据已存在，从本地加载...")
            data = self.data_loader.load_all()

        self.results['raw_data'] = data
        print(f"\n数据加载完成，共 {len(data)} 个标的")
        return data

    def step_build_features(self) -> pd.DataFrame:
        """步骤 2: 构建特征。

        Returns:
            特征 DataFrame
        """
        print("\n" + "=" * 80)
        print("步骤 2/7: 特征工程")
        print("=" * 80)

        self.feature_engineer = FeatureEngineer(
            raw_dir=self.config['data']['raw_dir'],
            processed_dir=self.config['data']['processed_dir'],
        )

        features_df = self.feature_engineer.build_features()
        self.feature_engineer.save_features(features_df)

        self.results['features'] = features_df
        print(f"\n特征构建完成，共 {len(features_df)} 行, {len(features_df.columns)} 列")
        return features_df

    def step_generate_labels(self) -> pd.DataFrame:
        """步骤 3: 生成标签。

        Returns:
            带标签的特征 DataFrame
        """
        print("\n" + "=" * 80)
        print("步骤 3/7: 标签生成")
        print("=" * 80)

        self.label_generator = LabelGenerator(
            horizon=self.config['label']['horizon'],
            threshold=self.config['label']['threshold'],
            threshold_up=self.config['label'].get('threshold_up'),
            threshold_down=self.config['label'].get('threshold_down'),
            use_quantile=self.config['label']['use_quantile'],
            quantile=self.config['label']['quantile'],
        )

        feature_path = os.path.join(
            self.config['data']['processed_dir'], 'features.parquet'
        )
        self.label_generator.load_features(feature_path)

        features_with_labels = self.label_generator.add_labels_to_features()
        self.label_generator.save(feature_path)

        self.results['features_with_labels'] = features_with_labels

        label_counts = features_with_labels['label'].value_counts().sort_index()
        print(f"\n标签生成完成")
        for label_val in [0, 1, 2]:
            count = label_counts.get(label_val, 0)
            pct = count / len(features_with_labels.dropna(subset=['label'])) * 100
            label_name = {0: 'RiskOff', 1: 'Neutral', 2: 'RiskOn'}[label_val]
            print(f"  {label_name} ({label_val}): {count} ({pct:.2f}%)")

        return features_with_labels

    def step_train_model(self) -> Any:
        """步骤 4: 训练模型（全样本，用于特征重要性等）。

        Returns:
            训练好的模型
        """
        print("\n" + "=" * 80)
        print("步骤 4/7: 模型训练")
        print("=" * 80)

        self.model_trainer = ModelTrainer(
            random_state=self.config['model']['random_state']
        )

        feature_path = os.path.join(
            self.config['data']['processed_dir'], 'features.parquet'
        )
        df = self.model_trainer.load_data(feature_path)
        X, y, feature_cols = self.model_trainer.prepare_data(df)

        model_type = self.config['model']['model_type']
        if model_type == 'lightgbm':
            model = self.model_trainer.train_lightgbm(X, y)
        elif model_type == 'random_forest':
            model = self.model_trainer.train_random_forest(X, y)
        elif model_type == 'logistic_regression':
            model = self.model_trainer.train_logistic_regression(X, y)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model_path = os.path.join('models', f'{model_type}_model.pkl')
        self.model_trainer.save_model(model, model_path)

        self.results['model'] = model
        self.results['feature_cols'] = feature_cols

        y_pred_proba = self.model_trainer.predict_proba(model, X)
        y_pred = y_pred_proba.idxmax(axis=1).map({
            'prob_riskoff': 0,
            'prob_neutral': 1,
            'prob_riskon': 2
        })
        accuracy = (y_pred.values == y.values).mean()
        print(f"\n训练集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return model

    def step_walkforward(self) -> pd.DataFrame:
        """步骤 5: Walk-Forward 验证。

        Returns:
            Walk-Forward 结果 DataFrame
        """
        print("\n" + "=" * 80)
        print("步骤 5/7: Walk-Forward 验证")
        print("=" * 80)

        self.wf_validator = WalkForwardValidator(
            frequency=self.config['walkforward']['frequency'],
            window_mode=self.config['walkforward']['window_mode'],
            window_years=self.config['walkforward']['window_years'],
            min_train_years=self.config['walkforward']['min_train_years'],
            random_state=self.config['model']['random_state'],
            model_type=self.config['model']['model_type'],
            portfolio_config=self.config['portfolio'],
        )

        feature_path = os.path.join(
            self.config['data']['processed_dir'], 'features.parquet'
        )
        self.wf_validator.load_data(
            feature_path=feature_path,
            price_dir=self.config['data']['raw_dir'],
        )

        wf_results = self.wf_validator.run()

        wf_save_path = os.path.join(
            self.config['data']['processed_dir'], 'walkforward_results.parquet'
        )
        self.wf_validator.save_results(wf_results, wf_save_path)

        self.results['wf_results'] = wf_results

        print_wf_performance(wf_results)

        return wf_results

    def step_evaluate(self, wf_path: Optional[str] = None) -> Dict[str, Any]:
        """步骤 6: 绩效评估。

        Args:
            wf_path: Walk-Forward 结果文件路径，为 None 时使用默认路径或内存中的结果

        Returns:
            评估结果字典
        """
        print("\n" + "=" * 80)
        print("步骤 6/7: 绩效评估")
        print("=" * 80)

        self.evaluator = PerformanceEvaluator(
            risk_free_rate=self.config['evaluation']['risk_free_rate']
        )

        wf = self.results.get('wf_results')
        if wf is not None:
            self.evaluator.wf_results = wf
            print(f"使用内存中的 Walk-Forward 结果: {len(wf)} 行")
        else:
            if wf_path is None:
                wf_path = os.path.join(
                    self.config['data']['processed_dir'], 'walkforward_results.parquet'
                )
            self.evaluator.load_walkforward_results(wf_path)

        wf = self.evaluator.wf_results

        from evaluate import _load_prices
        prices_df = _load_prices(self.config['data']['raw_dir'])

        benchmarks = self.evaluator.run_benchmarks(prices_df, wf.index)

        daily_returns = (
            wf['daily_return_cost']
            if 'daily_return_cost' in wf.columns
            else wf['daily_return']
        )
        nav_col = (
            'nav_cost_combined'
            if 'nav_cost_combined' in wf.columns
            else None
        )
        nav = wf[nav_col] if nav_col else None

        strategy_metrics = self.evaluator.calc_portfolio_metrics(daily_returns, nav)

        benchmark_metrics = {}
        for name, data in benchmarks.items():
            benchmark_metrics[name] = self.evaluator.calc_portfolio_metrics(
                data['daily_returns'], data['nav']
            )

        positions = wf['actual_position']
        trading_metrics = self.evaluator.calc_trading_metrics(positions, daily_returns)

        feature_path = os.path.join(
            self.config['data']['processed_dir'], 'features.parquet'
        )
        y_true = None
        classification_metrics = None
        if os.path.exists(feature_path):
            features_df = pd.read_parquet(feature_path, engine='pyarrow')
            if 'label' in features_df.columns:
                common_cls_idx = wf.index.intersection(features_df.index)
                y_true = features_df.loc[common_cls_idx, 'label'].astype(int).values

                prob_cols = ['prob_riskoff', 'prob_neutral', 'prob_riskon']
                prob_df = wf.loc[common_cls_idx, prob_cols]
                y_pred = prob_df.idxmax(axis=1).map({
                    'prob_riskoff': 0,
                    'prob_neutral': 1,
                    'prob_riskon': 2,
                }).astype(int).values

                classification_metrics = self.evaluator.calc_classification_metrics(
                    y_true, y_pred
                )

        if classification_metrics is None:
            classification_metrics = {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'confusion_matrix': np.zeros((3, 3), dtype=int),
            }

        self.evaluator.print_report(
            strategy_metrics, trading_metrics, classification_metrics, benchmark_metrics
        )

        eval_results = {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'trading_metrics': trading_metrics,
            'classification_metrics': classification_metrics,
            'benchmarks': benchmarks,
        }
        self.results['evaluation'] = eval_results

        return eval_results

    def step_visualize(self) -> Dict[str, str]:
        """步骤 7: 生成可视化图表。

        Returns:
            图表路径字典
        """
        print("\n" + "=" * 80)
        print("步骤 7/7: 可视化")
        print("=" * 80)

        self.visualizer = Visualizer(
            save_dir=self.config['visualization']['save_dir'],
            style=self.config['visualization']['style'],
            figsize=self.config['visualization']['figsize'],
            dpi=self.config['visualization']['dpi'],
        )

        wf_results = self.results.get('wf_results')
        if wf_results is None:
            wf_path = os.path.join(
                self.config['data']['processed_dir'], 'walkforward_results.parquet'
            )
            wf_results = pd.read_parquet(wf_path, engine='pyarrow')

        eval_results = self.results.get('evaluation', {})
        benchmarks = eval_results.get('benchmarks', {})
        benchmark_navs = {name: data['nav'] for name, data in benchmarks.items()}

        if not benchmark_navs:
            from visualization import _load_prices, _run_benchmarks
            try:
                prices_df = _load_prices(self.config['data']['raw_dir'])
                benchmark_navs = _run_benchmarks(prices_df, wf_results.index)
            except Exception:
                benchmark_navs = {}

        feature_importance = None
        if self.model_trainer and 'feature_cols' in self.results and 'model' in self.results:
            model = self.results['model']
            feature_cols = self.results['feature_cols']
            if hasattr(model, 'feature_importances_'):
                feature_importance = self.model_trainer.get_feature_importance(
                    model, feature_cols
                )

        confusion_matrix = None
        if 'evaluation' in self.results:
            classification_metrics = self.results['evaluation'].get('classification_metrics')
            if classification_metrics:
                confusion_matrix = classification_metrics.get('confusion_matrix')

        saved_paths = self.visualizer.plot_all(
            wf_results=wf_results,
            benchmark_navs=benchmark_navs,
            feature_importance=feature_importance,
            confusion_matrix=confusion_matrix,
        )

        self.results['visualization_paths'] = saved_paths

        print(f"\n可视化完成，共生成 {len(saved_paths)} 张图表")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")

        return saved_paths

    def run_full_backtest(self) -> Dict[str, Any]:
        """运行完整回测流程。

        执行流程：
        1. 加载数据（如果 data/raw/ 没有数据则下载）
        2. 构建特征
        3. 生成标签
        4. Walk-Forward 验证（季频扩张，主方案）
        5. 绩效评估
        6. 生成可视化图表
        7. 打印完整报告

        Returns:
            包含所有结果的字典
        """
        print("\n" + "#" * 80)
        print("# 杠杆择时策略 - 完整回测")
        print("#" * 80)

        try:
            self.step_load_data()
            self.step_build_features()
            self.step_generate_labels()
            self.step_train_model()
            self.step_walkforward()
            self.step_evaluate()
            self.step_visualize()

            print("\n" + "#" * 80)
            print("# 回测完成！")
            print("#" * 80)
            print("\n结果文件位置:")
            print(f"  特征数据: {self.config['data']['processed_dir']}/features.parquet")
            print(f"  WF结果: {self.config['data']['processed_dir']}/walkforward_results.parquet")
            print(f"  图表目录: {self.config['visualization']['save_dir']}/")

        except Exception as e:
            print(f"\n回测过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise

        return self.results

    def run_quick_test(self, n_years: int = 3) -> Dict[str, Any]:
        """快速测试（用少量数据验证流程）。

        Args:
            n_years: 使用最近 N 年数据进行快速测试

        Returns:
            包含测试结果的字典
        """
        print("\n" + "#" * 80)
        print(f"# 杠杆择时策略 - 快速测试 (最近 {n_years} 年)")
        print("#" * 80)

        original_min_train = self.config['walkforward']['min_train_years']
        self.config['walkforward']['min_train_years'] = max(1, n_years - 2)
        self.config['walkforward']['window_years'] = max(1, n_years - 1)

        try:
            self.step_load_data()
            self.step_build_features()
            self.step_generate_labels()
            self.step_train_model()

            features = self.results.get('features_with_labels')
            if features is not None:
                cutoff_date = features.index[-1] - pd.DateOffset(years=n_years)
                quick_features = features.loc[cutoff_date:]
                print(f"\n快速测试: 使用最近 {n_years} 年数据，共 {len(quick_features)} 行")

                quick_path = os.path.join(
                    self.config['data']['processed_dir'], 'features_quick.parquet'
                )
                quick_features.to_parquet(quick_path, engine='pyarrow')

                self.config['walkforward']['min_train_years'] = 1
                wf_validator = WalkForwardValidator(
                    frequency='yearly',
                    window_mode='expanding',
                    window_years=1,
                    min_train_years=1,
                    random_state=self.config['model']['random_state'],
                )
                wf_validator.load_data(
                    feature_path=quick_path,
                    price_dir=self.config['data']['raw_dir'],
                )
                wf_results = wf_validator.run()
                self.results['wf_results'] = wf_results

                wf_save_path = os.path.join(
                    self.config['data']['processed_dir'],
                    'walkforward_results_quick.parquet'
                )
                wf_validator.save_results(wf_results, wf_save_path)

                self.step_evaluate()
                self.step_visualize()

            print("\n" + "#" * 80)
            print("# 快速测试完成！")
            print("#" * 80)

        except Exception as e:
            print(f"\n快速测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.config['walkforward']['min_train_years'] = original_min_train

        return self.results


def main():
    """命令行入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(description='杠杆择时策略回测引擎')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='运行快速测试模式',
    )
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='快速测试使用的年数（默认 3 年）',
    )
    args = parser.parse_args()

    engine = BacktestEngine()

    if args.quick:
        engine.run_quick_test(n_years=args.years)
    else:
        engine.run_full_backtest()


if __name__ == '__main__':
    main()
