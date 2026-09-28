"""
情绪代理指标计算模块

基于现有市场数据计算情绪代理指标，用于 Regime 判断和消息面分析。

情绪代理指标：
1. VIX 期限结构 (Contango/Backwardation)
2. 信用利差 (HYG vs IEF)
3. 股权风险溢价代理
4. 市场宽度指标（如果有多只成分股数据）
"""

import pandas as pd
import numpy as np
from typing import Optional


class SentimentFeatures:
    """
    情绪代理特征计算

    基于市场数据计算各种情绪代理指标
    """

    def __init__(self, data: dict[str, pd.DataFrame]):
        """
        初始化情绪特征计算器

        Args:
            data: 原始数据字典，key 为标的名称（如 'VIX', 'HYG' 等）
        """
        self.data = data

    def calculate_all(self, base_index: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有情绪代理特征

        Args:
            base_index: 基准指数 DataFrame（如 QQQ 数据）

        Returns:
            添加了情绪代理特征的 DataFrame
        """
        df = base_index.copy()

        df = self._calc_vix_term_structure(df)
        df = self._calc_credit_spread(df)
        df = self._calc_equity_risk_premium(df)
        df = self._calc_market_regime_indicators(df)

        return df

    def _calc_vix_term_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 VIX 期限结构

        VIX 期限结构反映市场对未来波动的预期：
        - Contango (VIX < VIX_M1): 正常市场，看跌期权对冲需求低
        - Backwardation (VIX > VIX_M1): 恐慌市场，即期波动更高

        注：Yahoo Finance 没有 VIX 期货数据，这里用 VIX 自身变化率作为代理
        """
        if 'vix_close' not in df.columns:
            return df

        vix = df['vix_close']

        # VIX 变化率
        df['vix_change_1d'] = vix.pct_change(1)
        df['vix_change_5d'] = vix.pct_change(5)
        df['vix_change_20d'] = vix.pct_change(20)

        # VIX 移动平均
        df['vix_ma5'] = vix.rolling(5).mean()
        df['vix_ma20'] = vix.rolling(20).mean()
        df['vix_ma60'] = vix.rolling(60).mean()

        # VIX 相对均线
        df['vix_ma5_ratio'] = vix / df['vix_ma5']
        df['vix_ma20_ratio'] = vix / df['vix_ma20']

        # VIX 分位数
        df['vix_quantile_60d'] = vix.rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        df['vix_quantile_120d'] = vix.rolling(120).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

        # VIX 期限结构代理（简化版：使用 VIX 变化趋势）
        # 如果 VIX 快速上升，市场可能处于恐慌状态
        df['vix_term_structure_proxy'] = np.where(
            df['vix_change_5d'] > 0.1, -1,  # 快速上升 = Backwardation 代理
            np.where(df['vix_change_5d'] < -0.1, 1, 0)  # 快速下降 = Contango 代理
        )

        return df

    def _calc_credit_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算信用利差

        HYG (高收益债) vs IEF (7-10年国债)
        - 利差扩大: 风险偏好下降
        - 利差收窄: 风险偏好上升
        """
        if 'hyg_close' not in df.columns or 'ief_close' not in df.columns:
            return df

        hyg = df['hyg_close']
        ief = df['ief_close']

        # 信用利差 = HYG 收益率 - IEF 收益率
        # 注意：债券价格与收益率反向，这里用倒数近似
        hyg_yield_proxy = 1 / hyg * 100
        ief_yield_proxy = 1 / ief * 100

        df['credit_spread'] = hyg_yield_proxy - ief_yield_proxy
        df['credit_spread_5d_change'] = df['credit_spread'].pct_change(5)
        df['credit_spread_20d_change'] = df['credit_spread'].pct_change(20)

        # 信用利差分位数
        df['credit_spread_quantile_60d'] = df['credit_spread'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

        return df

    def _calc_equity_risk_premium(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算股权风险溢价代理

        使用股债收益率差异作为风险偏好指标
        """
        if 'spy_close' not in df.columns or 'tlt_close' not in df.columns:
            return df

        spy = df['spy_close']
        tlt = df['tlt_close']

        # 收益率
        spy_ret = spy.pct_change(20)  # 20日收益率
        tlt_ret = tlt.pct_change(20)

        # 股债收益率差异
        df['erp_proxy'] = spy_ret - tlt_ret
        df['erp_proxy_ma20'] = df['erp_proxy'].rolling(20).mean()

        # 风险偏好指标（简化版）
        # erp > 0: 股票表现好于债券，风险偏好高
        # erp < 0: 债券表现好于股票，风险偏好低
        df['risk_appetite'] = np.where(
            df['erp_proxy'] > 0, 1,
            np.where(df['erp_proxy'] < -0.02, -1, 0)
        )

        return df

    def _calc_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场 Regime 指标

        综合多个指标判断当前市场状态
        """
        # 市场宽度代理（SPY vs IWM）
        if 'spy_close' in df.columns and 'iwm_close' in df.columns:
            spy_iwm_ratio = df['spy_close'] / df['iwm_close']
            df['spy_iwm_ratio'] = spy_iwm_ratio
            df['spy_iwm_ratio_20d_change'] = spy_iwm_ratio.pct_change(20)

        # 防御性 vs 进攻性（QQQ vs SPY）
        if 'qqq_close' in df.columns and 'spy_close' in df.columns:
            qqq_spy_ratio = df['qqq_close'] / df['spy_close']
            df['rs_qv_spy_ret_20d'] = qqq_spy_ratio.pct_change(20)
            df['rs_qv_spy_ret_60d'] = qqq_spy_ratio.pct_change(60)

        # 综合情绪指标（简化版）
        sentiment_components = []

        if 'vix_ma20_ratio' in df.columns:
            # VIX 偏高 = 负面情绪
            sentiment_components.append(
                (df['vix_ma20_ratio'] - 1) * -1  # 反转
            )

        if 'credit_spread' in df.columns:
            # 信用利差偏高 = 负面情绪
            spread_zscore = (df['credit_spread'] - df['credit_spread'].rolling(60).mean()) / \
                           df['credit_spread'].rolling(60).std()
            sentiment_components.append(spread_zscore * -1)

        if 'risk_appetite' in df.columns:
            sentiment_components.append(df['risk_appetite'])

        if sentiment_components:
            # 简单平均
            sentiment_df = pd.concat(sentiment_components, axis=1)
            df['composite_sentiment'] = sentiment_df.mean(axis=1)

            # 标准化到 -1 到 1
            df['composite_sentiment'] = df['composite_sentiment'].clip(-2, 2) / 2
        else:
            df['composite_sentiment'] = 0.0

        return df

    def get_sentiment_summary(self, date: pd.Timestamp) -> dict:
        """
        获取特定日期的情绪摘要

        Args:
            date: 查询日期

        Returns:
            包含各情绪指标的字典
        """
        if not hasattr(self, '_cached_df'):
            return {}

        df = self._cached_df
        mask = df.index <= date
        if not mask.any():
            return {}

        row = df.loc[mask].iloc[-1]

        return {
            'vix_level': row.get('vix_close', np.nan),
            'vix_sentiment': 'high' if row.get('vix_ma20_ratio', 1) > 1.2 else
                           'low' if row.get('vix_ma20_ratio', 1) < 0.8 else 'normal',
            'vix_5d_change': row.get('vix_change_5d', 0) * 100,
            'credit_spread': row.get('credit_spread', np.nan),
            'credit_sentiment': 'tight' if row.get('credit_spread_quantile_60d', 0.5) < 0.3 else
                              'wide' if row.get('credit_spread_quantile_60d', 0.5) > 0.7 else 'normal',
            'risk_appetite': 'high' if row.get('risk_appetite', 0) > 0 else
                           'low' if row.get('risk_appetite', 0) < 0 else 'neutral',
            'composite_sentiment': row.get('composite_sentiment', 0),
        }


def add_sentiment_features(
    base_df: pd.DataFrame,
    raw_data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    添加情绪特征的便捷函数

    Args:
        base_df: 基础特征 DataFrame
        raw_data: 原始数据字典

    Returns:
        添加了情绪特征的 DataFrame
    """
    calculator = SentimentFeatures(raw_data)
    result_df = calculator.calculate_all(base_df)
    calculator._cached_df = result_df
    return result_df


if __name__ == "__main__":
    # 简单测试
    import os

    raw_dir = "data/raw"
    from data_loader import DataLoader

    # 加载数据
    loader = DataLoader(save_dir=raw_dir)
    data = loader.load_all()

    # 创建基础 DataFrame
    import feature_engineering
    fe = feature_engineering.FeatureEngineer(raw_dir=raw_dir)
    base_df = fe.load_raw_data()

    # 添加情绪特征
    enhanced_df = add_sentiment_features(base_df, data)

    # 显示情绪特征
    sentiment_cols = [
        'vix_close', 'vix_ma20_ratio', 'vix_quantile_60d',
        'credit_spread', 'credit_spread_quantile_60d',
        'risk_appetite', 'composite_sentiment'
    ]

    available_cols = [c for c in sentiment_cols if c in enhanced_df.columns]
    print("\n情绪代理特征 (最近5行):")
    print(enhanced_df[available_cols].tail())
