"""三重屏障法 (Triple-Barrier Method) 动态标签生成模块。

为 Meta-Labeling 模型提供真实盈亏比的高胜率训练目标：
- 结合标的 14日真实波幅 (ATR) 设立动态止盈与止损屏障
- 动态上屏障：+1.5 * ATR (止盈正样本 Label=1)
- 动态下屏障：-1.0 * ATR (止损负样本 Label=0)
- 时间屏障：最多持有 10 个交易日
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TripleBarrierLabeler:
    """三重屏障标签生成器。

    Attributes:
        atr_window: ATR 计算窗口，默认 14
        upper_atr_mult: 上屏障 ATR 乘数，默认 1.5
        lower_atr_mult: 下屏障 ATR 乘数，默认 1.0
        max_holding_days: 时间屏障持有上限，默认 10
    """

    def __init__(
        self,
        atr_window: int = 14,
        upper_atr_mult: float = 1.5,
        lower_atr_mult: float = 1.0,
        max_holding_days: int = 10,
    ):
        self.atr_window = atr_window
        self.upper_atr_mult = upper_atr_mult
        self.lower_atr_mult = lower_atr_mult
        self.max_holding_days = max_holding_days

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """计算资产的真实波幅 (ATR)。

        Args:
            df: 包含 'high', 'low', 'close' 的 DataFrame
        """
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()
        return atr

    def generate_single_asset_labels(self, df: pd.DataFrame) -> pd.Series:
        """对单个标的计算逐日的三重屏障标签。

        Args:
            df: 包含 ['high', 'low', 'close'] 的按日期升序 DataFrame

        Returns:
            pd.Series (0 或 1，样本不足或末尾为 NaN)
        """
        atr = self.compute_atr(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = atr.values
        n = len(df)

        labels = np.full(n, np.nan)

        for i in range(n - self.max_holding_days):
            cur_p = closes[i]
            cur_atr = atrs[i]

            if np.isnan(cur_p) or np.isnan(cur_atr) or cur_atr <= 0:
                continue

            upper_barrier = cur_p + self.upper_atr_mult * cur_atr
            lower_barrier = cur_p - self.lower_atr_mult * cur_atr

            label = np.nan
            for step in range(1, self.max_holding_days + 1):
                idx = i + step
                h = highs[idx]
                l = lows[idx]

                # 优先检查是否触及上屏障（若同日剧烈穿透，以更靠近开盘或严格逻辑处理）
                hit_upper = h >= upper_barrier
                hit_lower = l <= lower_barrier

                if hit_upper and not hit_lower:
                    label = 1.0
                    break
                elif hit_lower and not hit_upper:
                    label = 0.0
                    break
                elif hit_upper and hit_lower:
                    # 同日均触及：若收盘高于开盘倾向于先上，保守起见视为止损失败或中性
                    label = 0.0
                    break

            # 若 10 天内未触碰任何屏障，以到期日盈亏定输赢
            if np.isnan(label):
                final_p = closes[i + self.max_holding_days]
                label = 1.0 if final_p > cur_p else 0.0

            labels[i] = label

        return pd.Series(labels, index=df.index, name='label')

    def generate_universe_labels(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对标的池所有资产生成三重屏障标签矩阵。

        Returns:
            DataFrame (index: date, columns: tickers, values: 0/1/NaN)
        """
        labels_dict = {}
        for ticker, df in data_dict.items():
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                labels_dict[ticker] = self.generate_single_asset_labels(df)

        df_labels = pd.DataFrame(labels_dict)
        return df_labels
