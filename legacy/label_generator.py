import os
import pandas as pd
import numpy as np


class LabelGenerator:
    """标签生成器，用于基于未来收益率生成五分类标签。

    根据收盘价计算未来 horizon 天的收益率，通过非对称多级阈值将收益率映射为五类：
    - 4: Strong RiskOn (涨幅极大, 持有 TQQQ)
    - 3: Weak RiskOn (涨幅适中, 持有 QQQ)
    - 2: Neutral (震荡微波, 现金 Cash)
    - 1: Weak RiskOff (跌幅适中, 持有 PSQ)
    - 0: Strong RiskOff (暴跌, 持有 SQQQ)
    """

    def __init__(
        self,
        horizon: int = 5,
        threshold_up: float = 0.008,
        threshold_very_up: float = 0.018,
        threshold_down: float = 0.010,
        threshold_very_down: float = 0.020,
    ):
        self.horizon = horizon
        self.threshold_up = threshold_up
        self.threshold_very_up = threshold_very_up
        self.threshold_down = threshold_down
        self.threshold_very_down = threshold_very_down
        self.use_quantile = False
        self.features_df = None

    def load_features(self, path: str = 'data/processed/features.parquet') -> pd.DataFrame:
        """加载特征数据。

        Args:
            path: 特征文件路径，默认 'data/processed/features.parquet'

        Returns:
            加载的特征 DataFrame

        Raises:
            FileNotFoundError: 当文件不存在时抛出
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征文件不存在: {path}")

        self.features_df = pd.read_parquet(path, engine='pyarrow')
        print(f"已加载特征数据: {len(self.features_df)} 行, {len(self.features_df.columns)} 列")
        return self.features_df

    def generate_future_returns(self, close_prices: pd.Series) -> pd.Series:
        """计算未来收益率。

        future_ret = close.shift(-horizon) / close - 1

        注：使用 shift(-horizon) 表示取未来第 horizon 天的价格，
        最后 horizon 行会因无未来数据而为 NaN。

        Args:
            close_prices: 收盘价序列，索引为日期

        Returns:
            未来收益率序列
        """
        future_ret = close_prices.shift(-self.horizon) / close_prices - 1
        return future_ret

    def generate_labels(self, future_returns: pd.Series) -> pd.Series:
        labels = pd.Series(np.nan, index=future_returns.index, name='label')
        valid_mask = future_returns.notna()
        valid_returns = future_returns[valid_mask]

        up1 = self.threshold_up
        up2 = self.threshold_very_up
        down1 = -self.threshold_down
        down2 = -self.threshold_very_down

        cond4 = valid_returns > up2
        cond3 = (valid_returns > up1) & (valid_returns <= up2)
        cond1 = (valid_returns >= down2) & (valid_returns < down1)
        cond0 = valid_returns < down2
        
        # 默认中性(2)
        vals = np.full(len(valid_returns), 2)
        vals[cond4] = 4
        vals[cond3] = 3
        vals[cond1] = 1
        vals[cond0] = 0

        labels.loc[valid_mask] = vals
        return labels

    def add_labels_to_features(self) -> pd.DataFrame:
        """将标签列合并到特征表。

        从特征表中提取 qqq_close 列计算未来收益率和标签，
        然后将 label 列添加到特征 DataFrame。

        Returns:
            带有 label 列的特征 DataFrame

        Raises:
            ValueError: 当特征数据未加载时抛出
            KeyError: 当特征表中缺少 qqq_close 列时抛出
        """
        if self.features_df is None:
            raise ValueError("请先调用 load_features() 加载特征数据")

        if 'qqq_close' not in self.features_df.columns:
            raise KeyError("特征表中缺少 qqq_close 列，无法计算标签")

        close_prices = self.features_df['qqq_close']
        future_returns = self.generate_future_returns(close_prices)
        labels = self.generate_labels(future_returns)

        self.features_df = self.features_df.copy()
        self.features_df['label'] = labels
        self.features_df['future_ret'] = future_returns

        print(f"已添加标签列，NaN 标签数: {labels.isna().sum()}")
        return self.features_df

    def save_features(self, path: str = 'data/processed/features.parquet') -> None:
        if self.features_df is None:
            raise ValueError("请先调用 add_labels_to_features() 生成带标签的特征数据")
        self.features_df.to_parquet(path, engine='pyarrow')
        print(f"带标签特征已保存到: {path}")

if __name__ == '__main__':
    generator = LabelGenerator()
    try:
        df = generator.load_features()
        df = generator.add_labels_to_features()
        print(df['label'].value_counts().sort_index())
        generator.save_features()
    except FileNotFoundError:
        print("Features not found, run feature engineering first.")