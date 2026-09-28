"""可视化模块，用于生成策略结果的各类图表。

提供净值曲线、回撤曲线、滚动夏普、持仓时间线、
特征重要性、混淆矩阵等多种可视化图表生成功能。
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """可视化器，用于生成策略结果图表。

    Attributes:
        save_dir: 图表保存目录
        style: matplotlib 风格
        figsize: 图表大小
        dpi: 图表分辨率
    """

    def __init__(
        self,
        save_dir: str = 'reports/figures',
        style: str = 'seaborn-v0_8',
        figsize: tuple = (12, 6),
        dpi: int = 150,
    ) -> None:
        """初始化可视化器。

        Args:
            save_dir: 图表保存目录，默认 'reports/figures'
            style: matplotlib 风格，默认 'seaborn-v0_8'
            figsize: 图表大小，默认 (12, 6)
            dpi: 图表分辨率，默认 150
        """
        self.save_dir = save_dir
        self.style = style
        self.figsize = figsize
        self.dpi = dpi

        plt.style.use(style)
        os.makedirs(save_dir, exist_ok=True)

    def plot_equity_curve(
        self,
        nav_dict: Dict[str, pd.Series],
        title: str = 'Equity Curve',
        log_scale: bool = False,
    ) -> str:
        """绘制净值曲线。

        Args:
            nav_dict: 净值字典，key 为策略名，value 为净值 Series
            title: 图表标题，默认 'Equity Curve'
            log_scale: 是否使用对数坐标，默认 False

        Returns:
            保存的图片文件路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, nav in nav_dict.items():
            ax.plot(nav.index, nav.values, label=name, linewidth=1.5)

        if log_scale:
            ax.set_yscale('log')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Net Asset Value', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'equity_curve.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"净值曲线已保存到: {save_path}")
        return save_path

    def plot_drawdown(
        self,
        nav_dict: Dict[str, pd.Series],
        title: str = 'Drawdown',
    ) -> str:
        """绘制回撤曲线。

        Args:
            nav_dict: 净值字典，key 为策略名，value 为净值 Series
            title: 图表标题，默认 'Drawdown'

        Returns:
            保存的图片文件路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, nav in nav_dict.items():
            peak = nav.cummax()
            drawdown = (nav - peak) / peak
            ax.plot(drawdown.index, drawdown.values, label=name, linewidth=1.2)

        ax.fill_between(
            list(nav_dict.values())[0].index,
            0,
            min([d.min() for d in [
                (nav - nav.cummax()) / nav.cummax()
                for nav in nav_dict.values()
            ]]),
            alpha=0.05,
            color='red',
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'drawdown.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"回撤曲线已保存到: {save_path}")
        return save_path

    def plot_rolling_sharpe(
        self,
        daily_returns_dict: Dict[str, pd.Series],
        window: int = 252,
        title: str = 'Rolling Sharpe Ratio (1Y)',
    ) -> str:
        """绘制滚动夏普比率曲线。

        Args:
            daily_returns_dict: 日收益率字典，key 为策略名，value 为日收益率 Series
            window: 滚动窗口大小（交易日），默认 252（1年）
            title: 图表标题，默认 'Rolling Sharpe Ratio (1Y)'

        Returns:
            保存的图片文件路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, daily_ret in daily_returns_dict.items():
            rolling_mean = daily_ret.rolling(window=window).mean()
            rolling_std = daily_ret.rolling(window=window).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
            ax.plot(
                rolling_sharpe.index,
                rolling_sharpe.values,
                label=name,
                linewidth=1.2,
            )

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'rolling_sharpe.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"滚动夏普曲线已保存到: {save_path}")
        return save_path

    def plot_position_timeline(
        self,
        positions: pd.Series,
        nav: Optional[pd.Series] = None,
        title: str = 'Position Timeline',
    ) -> str:
        """绘制持仓时间线。

        用不同颜色背景表示不同持仓状态，并叠加净值曲线。

        Args:
            positions: 持仓 Series，值为 'TQQQ', 'SQQQ', 'Cash'
            nav: 净值 Series，叠加显示，默认 None
            title: 图表标题，默认 'Position Timeline'

        Returns:
            保存的图片文件路径
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)

        position_colors = {
            'TQQQ': '#2ecc71',
            'SQQQ': '#e74c3c',
            'Cash': '#95a5a6',
        }

        dates = positions.index
        current_pos = positions.iloc[0]
        start_idx = 0

        for i in range(1, len(positions)):
            if positions.iloc[i] != current_pos:
                color = position_colors.get(current_pos, '#95a5a6')
                ax1.axvspan(
                    dates[start_idx],
                    dates[i - 1],
                    alpha=0.2,
                    color=color,
                )
                current_pos = positions.iloc[i]
                start_idx = i

        color = position_colors.get(current_pos, '#95a5a6')
        ax1.axvspan(
            dates[start_idx],
            dates[-1],
            alpha=0.2,
            color=color,
        )

        if nav is not None:
            ax1.plot(nav.index, nav.values, color='#2c3e50', linewidth=1.5, label='NAV')
            ax1.set_ylabel('Net Asset Value', fontsize=12, color='#2c3e50')
            ax1.tick_params(axis='y', labelcolor='#2c3e50')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=position_colors['TQQQ'], alpha=0.4, label='TQQQ (Long)'),
            Patch(facecolor=position_colors['SQQQ'], alpha=0.4, label='SQQQ (Short)'),
            Patch(facecolor=position_colors['Cash'], alpha=0.4, label='Cash'),
        ]
        if nav is not None:
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color='#2c3e50', linewidth=1.5, label='NAV')
            )

        ax1.legend(handles=legend_elements, loc='best', fontsize=10)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'position_timeline.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"持仓时间线已保存到: {save_path}")
        return save_path

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = 'Top 20 Feature Importance',
    ) -> str:
        """绘制特征重要性横向柱状图。

        Args:
            importance_df: 特征重要性 DataFrame，包含 feature 和 importance 列
            top_n: 显示前 N 个特征，默认 20
            title: 图表标题，默认 'Top 20 Feature Importance'

        Returns:
            保存的图片文件路径
        """
        df = importance_df.sort_values('importance', ascending=True).tail(top_n).copy()

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        ax.barh(df['feature'], df['importance'], color=colors)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'feature_importance.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"特征重要性图已保存到: {save_path}")
        return save_path

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[list] = None,
        title: str = 'Confusion Matrix',
    ) -> str:
        """绘制混淆矩阵热力图。

        Args:
            cm: 3x3 混淆矩阵 numpy array
            labels: 标签列表，默认 ['RiskOff', 'Neutral', 'RiskOn']
            title: 图表标题，默认 'Confusion Matrix'

        Returns:
            保存的图片文件路径
        """
        if labels is None:
            labels = ['RiskOff', 'Neutral', 'RiskOn']

        fig, ax = plt.subplots(figsize=(8, 6))

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Proportion'},
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"混淆矩阵图已保存到: {save_path}")
        return save_path

    def plot_all(
        self,
        wf_results: pd.DataFrame,
        benchmark_navs: Dict[str, pd.Series],
        feature_importance: Optional[pd.DataFrame] = None,
        confusion_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """一键生成所有图表。

        Args:
            wf_results: Walk-Forward 结果 DataFrame
            benchmark_navs: 基准净值字典
            feature_importance: 特征重要性 DataFrame，默认 None
            confusion_matrix: 混淆矩阵 3x3 numpy array，默认 None

        Returns:
            所有图表保存路径的字典
        """
        saved_paths = {}

        nav_col = 'nav_cost_combined' if 'nav_cost_combined' in wf_results.columns else 'nav_combined'
        ret_col = 'daily_return_cost' if 'daily_return_cost' in wf_results.columns else 'daily_return'

        strategy_nav = wf_results[nav_col]
        strategy_ret = wf_results[ret_col]

        nav_dict = {'Strategy': strategy_nav}
        nav_dict.update(benchmark_navs)

        returns_dict = {'Strategy': strategy_ret}
        for name, nav in benchmark_navs.items():
            returns_dict[name] = nav.pct_change().fillna(0.0)

        saved_paths['equity_curve'] = self.plot_equity_curve(nav_dict)
        saved_paths['drawdown'] = self.plot_drawdown(nav_dict)
        saved_paths['rolling_sharpe'] = self.plot_rolling_sharpe(returns_dict)

        positions = wf_results['actual_position'] if 'actual_position' in wf_results.columns else None
        if positions is not None:
            saved_paths['position_timeline'] = self.plot_position_timeline(
                positions, strategy_nav
            )

        if feature_importance is not None:
            saved_paths['feature_importance'] = self.plot_feature_importance(
                feature_importance
            )

        if confusion_matrix is not None:
            saved_paths['confusion_matrix'] = self.plot_confusion_matrix(
                confusion_matrix
            )

        return saved_paths


def _load_prices(data_dir: str = 'data/raw') -> pd.DataFrame:
    """加载价格数据。

    Args:
        data_dir: 原始数据目录

    Returns:
        包含 QQQ 和 TQQQ 收盘价的 DataFrame
    """
    qqq_path = os.path.join(data_dir, 'qqq.csv')
    tqqq_path = os.path.join(data_dir, 'tqqq.csv')

    qqq_df = pd.read_csv(qqq_path, index_col=0, parse_dates=True)
    tqqq_df = pd.read_csv(tqqq_path, index_col=0, parse_dates=True)

    prices_df = pd.DataFrame({
        'QQQ': qqq_df['Close'],
        'TQQQ': tqqq_df['Close'],
    })
    prices_df = prices_df.dropna()
    return prices_df


def _run_benchmarks(
    prices_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> Dict[str, pd.Series]:
    """运行三个基准策略并返回净值。

    基准策略：
    - Benchmark A: Buy & Hold TQQQ
    - Benchmark B: MA200 择时
    - Benchmark C: MA20/60 金叉

    Args:
        prices_df: 价格 DataFrame
        dates: 回测日期索引

    Returns:
        基准净值字典
    """
    common_dates = prices_df.index.intersection(dates)
    prices = prices_df.loc[common_dates].copy()

    tqqq_ret = prices['TQQQ'].pct_change().fillna(0.0)
    qqq_close = prices['QQQ']

    ma200 = qqq_close.rolling(window=200).mean()
    ma20 = qqq_close.rolling(window=20).mean()
    ma60 = qqq_close.rolling(window=60).mean()

    pos_b = pd.Series('Cash', index=common_dates)
    pos_b[qqq_close > ma200] = 'TQQQ'
    pos_b = pos_b.shift(1).fillna('Cash')

    pos_c = pd.Series('Cash', index=common_dates)
    pos_c[ma20 > ma60] = 'TQQQ'
    pos_c = pos_c.shift(1).fillna('Cash')

    nav_a = (1.0 + tqqq_ret).cumprod()

    ret_b = pd.Series(0.0, index=common_dates)
    ret_b[pos_b == 'TQQQ'] = tqqq_ret[pos_b == 'TQQQ']
    ret_b = ret_b.fillna(0.0)
    nav_b = (1.0 + ret_b).cumprod()

    ret_c = pd.Series(0.0, index=common_dates)
    ret_c[pos_c == 'TQQQ'] = tqqq_ret[pos_c == 'TQQQ']
    ret_c = ret_c.fillna(0.0)
    nav_c = (1.0 + ret_c).cumprod()

    return {
        'Benchmark_A_BuyHold': nav_a,
        'Benchmark_B_MA200': nav_b,
        'Benchmark_C_MA20_60': nav_c,
    }


def main() -> None:
    """命令行入口函数。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("可视化模块 - 主程序")
    print("=" * 80)

    wf_path = 'data/processed/walkforward_results.parquet'
    if not os.path.exists(wf_path):
        print(f"错误: Walk-Forward 结果文件不存在: {wf_path}")
        print("\n请先运行 Walk-Forward 验证生成结果，例如:")
        print("  python src/walkforward.py")
        sys.exit(1)

    print("\n1. 加载 Walk-Forward 结果")
    print("-" * 80)
    wf_results = pd.read_parquet(wf_path, engine='pyarrow')
    print(f"已加载: {len(wf_results)} 行, {len(wf_results.columns)} 列")
    print(f"时间区间: {wf_results.index[0].date()} ~ {wf_results.index[-1].date()}")

    print("\n2. 运行基准策略")
    print("-" * 80)
    try:
        prices_df = _load_prices('data/raw')
        benchmark_navs = _run_benchmarks(prices_df, wf_results.index)
        for name, nav in benchmark_navs.items():
            print(f"  {name}: 最终净值 = {nav.iloc[-1]:.4f}")
    except FileNotFoundError as e:
        print(f"  警告: {e}，跳过基准策略")
        benchmark_navs = {}

    print("\n3. 获取特征重要性")
    print("-" * 80)
    feature_importance = None
    model_path = 'models/lgb_model.pkl'
    feature_path = 'data/processed/features.parquet'

    try:
        from train_model import ModelTrainer
        trainer = ModelTrainer()

        if os.path.exists(model_path):
            model = trainer.load_model(model_path)
            if os.path.exists(feature_path):
                feature_df = trainer.load_data(feature_path)
                _, _, feature_cols = trainer.prepare_data(feature_df)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = trainer.get_feature_importance(
                        model, feature_cols
                    )
                    print(f"已获取特征重要性: {len(feature_importance)} 个特征")
                else:
                    print("  警告: 模型不支持特征重要性")
            else:
                print(f"  警告: 特征文件不存在: {feature_path}")
        else:
            print(f"  警告: 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"  警告: 获取特征重要性失败: {e}")

    print("\n4. 计算混淆矩阵")
    print("-" * 80)
    cm = None
    try:
        if os.path.exists(feature_path):
            features_df = pd.read_parquet(feature_path, engine='pyarrow')
            if 'label' in features_df.columns:
                common_idx = wf_results.index.intersection(features_df.index)
                y_true = features_df.loc[common_idx, 'label'].astype(int).values

                prob_cols = ['prob_riskoff', 'prob_neutral', 'prob_riskon']
                if all(col in wf_results.columns for col in prob_cols):
                    prob_df = wf_results.loc[common_idx, prob_cols]
                    y_pred = prob_df.idxmax(axis=1).map({
                        'prob_riskoff': 0,
                        'prob_neutral': 1,
                        'prob_riskon': 2,
                    }).astype(int).values

                    from sklearn.metrics import confusion_matrix as sk_cm
                    cm = sk_cm(y_true, y_pred, labels=[0, 1, 2])
                    print(f"已计算混淆矩阵，样本数: {len(y_true)}")
                else:
                    print("  警告: Walk-Forward 结果中缺少概率列")
            else:
                print("  警告: 特征数据中无 label 列")
        else:
            print(f"  警告: 特征文件不存在: {feature_path}")
    except Exception as e:
        print(f"  警告: 计算混淆矩阵失败: {e}")

    print("\n5. 生成所有图表")
    print("-" * 80)
    visualizer = Visualizer(
        save_dir='reports/figures',
        style='seaborn-v0_8',
        figsize=(12, 6),
        dpi=150,
    )

    saved_paths = visualizer.plot_all(
        wf_results=wf_results,
        benchmark_navs=benchmark_navs,
        feature_importance=feature_importance,
        confusion_matrix=cm,
    )

    print("\n" + "=" * 80)
    print("所有图表生成完成")
    print("=" * 80)
    for chart_name, path in saved_paths.items():
        print(f"  {chart_name:<25s} -> {path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
