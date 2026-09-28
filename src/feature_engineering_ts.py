"""时序与量价微结构特征工程模块。

专注于捕捉动量真实质量与假突破识别：
1. 动量二阶加速度 (5日动量 - 20日动量)
2. 量能爆发比率 (5日均量 / 20日均量)
3. 真实 VWAP 基差偏离率 ((Close - VWAP) / VWAP)
4. 14日相对强弱指标 (RSI 14)
5. 真实波动比率 (ATR 14 / Close)
6. 20日收益率偏度 (Vol Skewness)
7. 大盘 20日滚动 Beta 协同度
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class TimeSeriesFeatureExtractor:
    """时序量价特征提取器。"""

    def __init__(self, spy_ticker: str = 'US.SPY'):
        self.spy_ticker = spy_ticker

    @staticmethod
    def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """计算 14 日 RSI。"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # 边界修正：连续单边上涨(avg_loss==0)应为100，单边下跌(avg_gain==0)应为0
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
        return rsi.fillna(50.0)

    def extract_single_asset_features(
        self,
        df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """提取单个标的的多维时序与量价微结构特征。

        Args:
            df: 包含 ['open', 'high', 'low', 'close', 'volume', 'vwap']
            spy_df: 包含 SPY 的 ['close']

        Returns:
            DataFrame 包含各项时序特征 (每列一个特征)
        """
        feats = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']
        vwap = df.get('vwap', close)
        high = df['high']
        low = df['low']

        # 1. 动量多周期与二阶加速度
        ret_5 = close.pct_change(5, fill_method=None)
        ret_20 = close.pct_change(20, fill_method=None)
        ret_60 = close.pct_change(60, fill_method=None)
        feats['ret_5'] = ret_5
        feats['ret_20'] = ret_20
        feats['ret_60'] = ret_60
        # 动量加速度 (加速冲刺为正，衰竭滞涨为负)
        feats['mom_accel'] = ret_5 - (ret_20 * 0.25)

        # 2. 量能爆发比率 (Volume Surge)
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        feats['volume_surge'] = (vol_ma5 / vol_ma20.replace(0, np.nan)).fillna(1.0)

        # 3. 价格相对 VWAP 偏离率
        feats['vwap_basis'] = ((close - vwap) / vwap.replace(0, np.nan)).fillna(0.0)

        # 4. 14日 RSI (超买风险防守)
        feats['rsi_14'] = self.compute_rsi(close, 14) / 100.0  # 归一化至 [0, 1]

        # 5. ATR 真实波幅比率
        close_prev = close.shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        feats['atr_ratio'] = (atr_14 / close.replace(0, np.nan)).fillna(0.02)

        # 6. 20日收益率偏度 (下行尾部风险测度)
        daily_ret = close.pct_change(1, fill_method=None)
        feats['vol_skew_20'] = daily_ret.rolling(20).skew().fillna(0.0)

        # 7. 价格相对 MA60 距离
        ma60 = close.rolling(60).mean()
        feats['dist_ma60'] = ((close - ma60) / ma60.replace(0, np.nan)).fillna(0.0)

        # 8. 相对 SPY 协同 Beta
        if spy_df is not None and 'close' in spy_df.columns:
            spy_ret = spy_df['close'].pct_change(1, fill_method=None)
            cov = daily_ret.rolling(20).cov(spy_ret)
            var = spy_ret.rolling(20).var()
            feats['spy_beta_20'] = (cov / var.replace(0, np.nan)).fillna(1.0).clip(-3.0, 3.0)
        else:
            feats['spy_beta_20'] = 1.0

        return feats

    def extract_universe_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        eligible_tickers: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量提取指定标的池的时序特征集。

        Returns:
            Dict[ticker, feature_df]
        """
        spy_df = data_dict.get(self.spy_ticker)
        tickers = eligible_tickers or list(data_dict.keys())

        result = {}
        for t in tickers:
            if t in data_dict:
                result[t] = self.extract_single_asset_features(data_dict[t], spy_df)

        return result
