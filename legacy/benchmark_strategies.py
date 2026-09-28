"""
基准策略实现

包含：
1. Buy-Hold TQQQ
2. MA20-60 均线交叉
"""

import pandas as pd
import numpy as np


class BuyHoldStrategy:
    """Buy-Hold 策略：始终持有 TQQQ"""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def run(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        运行 Buy-Hold 策略

        Args:
            prices_df: 价格数据，包含 tqqq_close 列

        Returns:
            结果 DataFrame，包含 nav（净值）
        """
        if 'tqqq_close' not in prices_df.columns:
            raise ValueError("prices_df 需要包含 tqqq_close 列")

        tqqq = prices_df['tqqq_close'].copy()
        daily_ret = tqqq.pct_change().fillna(0)
        nav = self.initial_capital * (1 + daily_ret).cumprod()

        return pd.DataFrame({
            'position': 'TQQQ',
            'daily_return': daily_ret,
            'nav': nav,
        }, index=tqqq.index)


class MA60Strategy:
    """MA20-60 均线交叉策略

    - 金叉（MA20 > MA60）→ 做多 TQQQ
    - 死叉（MA20 < MA60）→ 持有现金 Cash
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        ma_short: int = 20,
        ma_long: int = 60,
        use_short: bool = False,  # 是否使用做空，默认 False = Cash
    ):
        self.initial_capital = initial_capital
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.use_short = use_short

    def run(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        运行 MA20-60 均线交叉策略

        Args:
            prices_df: 价格数据

        Returns:
            结果 DataFrame
        """
        if 'tqqq_close' not in prices_df.columns:
            raise ValueError("prices_df 需要包含 tqqq_close 列")

        tqqq = prices_df['tqqq_close'].copy()

        if self.use_short and 'sqqq_close' in prices_df.columns:
            sqqq = prices_df['sqqq_close'].copy()
            common_idx = tqqq.index.intersection(sqqq.index)
            tqqq = tqqq.loc[common_idx]
            sqqq = sqqq.loc[common_idx]
        else:
            common_idx = tqqq.index

        # 计算均线
        ma_short = tqqq.rolling(self.ma_short).mean()
        ma_long = tqqq.rolling(self.ma_long).mean()

        # 生成信号：1 = 做多 TQQQ, 0 = 持有现金, -1 = 做空 SQQQ
        if self.use_short and 'sqqq_close' in prices_df.columns:
            raw_signal = np.where(ma_short > ma_long, 1, -1)
        else:
            raw_signal = np.where(ma_short > ma_long, 1, 0)

        # 均线需预热期
        raw_signal[:self.ma_long] = 0

        # 执行滞后 (避免未来函数：T 日收盘均线信号在 T+1 日生效)
        signal = pd.Series(raw_signal, index=common_idx).shift(1).fillna(0).astype(int)

        # 计算每日收益
        tqqq_ret = tqqq.pct_change().fillna(0)

        if self.use_short and 'sqqq_close' in prices_df.columns:
            sqqq_ret = sqqq.pct_change().fillna(0)
            daily_ret = np.where(signal == 1, tqqq_ret.values, np.where(signal == -1, sqqq_ret.values, 0.0))
            position = np.where(signal == 1, 'TQQQ', np.where(signal == -1, 'SQQQ', 'Cash'))
        else:
            daily_ret = np.where(signal == 1, tqqq_ret.values, 0.0)
            position = np.where(signal == 1, 'TQQQ', 'Cash')

        nav = self.initial_capital * (1 + pd.Series(daily_ret, index=common_idx)).cumprod()

        return pd.DataFrame({
            'position': position,
            'signal': signal,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'daily_return': daily_ret,
            'nav': nav,
        }, index=common_idx)


class BenchmarkRunner:
    """
    基准策略运行器

    运行多个基准策略并对比结果
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def run_all(
        self,
        prices_df: pd.DataFrame,
        feature_df: pd.DataFrame = None,
    ) -> dict:
        """
        运行所有基准策略

        Args:
            prices_df: 价格数据
            feature_df: 特征数据（用于 ML 模型）

        Returns:
            各策略结果字典
        """
        results = {}

        # 1. Buy-Hold TQQQ
        buyhold = BuyHoldStrategy(self.initial_capital)
        results['buyhold_tqqq'] = buyhold.run(prices_df)

        # 2. MA20-60
        ma60 = MA60Strategy(self.initial_capital, ma_short=20, ma_long=60)
        results['ma20_60'] = ma60.run(prices_df)

        return results

    @staticmethod
    def calculate_metrics(nav_series: pd.Series) -> dict:
        """计算绩效指标"""
        nav = nav_series.dropna()
        if len(nav) == 0:
            return {}

        # 总收益
        total_return = nav.iloc[-1] / nav.iloc[0] - 1

        # 年化收益
        days = (nav.index[-1] - nav.index[0]).days
        years = days / 365.25
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

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_nav': nav.iloc[-1],
        }


if __name__ == "__main__":
    # 测试
    prices = pd.DataFrame({
        'tqqq_close': np.random.randn(100).cumsum() + 100,
        'sqqq_close': -np.random.randn(100).cumsum() + 50,
    }, index=pd.date_range('2020-01-01', periods=100))

    runner = BenchmarkRunner()
    results = runner.run_all(prices)

    for name, df in results.items():
        metrics = runner.calculate_metrics(df['nav'])
        print(f"{name}: 总收益={metrics['total_return']:.2%}, 夏普={metrics['sharpe_ratio']:.2f}")
