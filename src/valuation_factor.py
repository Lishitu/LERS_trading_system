"""Valuation Factor Engine based on Moomoo get_valuation_detail API.

从 Moomoo 估值详情接口提取行业锚定标的的历史市盈率 (P/E)、行业市盈率 (Plate P/E) 与盈利增速，
构建滚动估值分位数、估值 Z-Score、盈利收益率 (Earnings Yield E/P) 综合估值因子。
"""

import os
import glob
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# 行业 ETF 与其核心权重锚定个股的映射关系
SECTOR_VALUATION_ANCHORS: Dict[str, List[str]] = {
    'US.XLK': ['US.AAPL', 'US.MSFT'],          # 科技权重锚: 苹果、微软
    'US.SMH': ['US.NVDA', 'US.AVGO'],          # 半导体权重锚: 英伟达、博通
    'US.XLE': ['US.XOM', 'US.CVX'],            # 能源权重锚: 埃克森美孚、雪佛龙
    'US.XOP': ['US.XOM', 'US.CVX'],            # 油气勘探开采权重锚
    'US.XLF': ['US.JPM'],                      # 金融权重锚: 摩根大通
    'US.XLI': ['US.CAT'],                      # 工业制造权重锚: 卡特彼勒
    'US.COPX': ['US.FCX'],                     # 铜矿采选权重锚: 自由港麦克莫兰
    'US.URA': ['US.CCJ'],                      # 铀能开采权重锚: 卡梅科
}


class ValuationFactorEngine:
    """基于 Moomoo get_valuation_detail 数据源的多因子估值引擎。"""

    def __init__(
        self,
        valuation_dir: str = 'data/moomoo/valuation',
        rolling_window: int = 252,
    ):
        self.valuation_dir = valuation_dir
        self.rolling_window = rolling_window
        self.stock_val_dict: Dict[str, pd.DataFrame] = {}
        self.sector_pe_df: Optional[pd.DataFrame] = None
        self.sector_pe_zscore: Optional[pd.DataFrame] = None
        self.sector_earnings_yield: Optional[pd.DataFrame] = None

    def load_and_compute_factors(self, dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """加载本地 Parquet 缓存并对齐至指定交易日历，计算估值因子矩阵。"""
        val_files = glob.glob(os.path.join(self.valuation_dir, '*.parquet'))
        for vf in val_files:
            ticker = os.path.basename(vf).replace('_', '.').replace('.parquet', '')
            try:
                df = pd.read_parquet(vf)
                df = df.reindex(dates).ffill()
                self.stock_val_dict[ticker] = df
            except Exception as e:
                print(f"[ValuationEngine] 加载 {vf} 异常: {e}")

        # 合成行业 P/E 序列
        sec_pe = {}
        for sec, anchors in SECTOR_VALUATION_ANCHORS.items():
            valid_series = [
                self.stock_val_dict[a]['value']
                for a in anchors
                if a in self.stock_val_dict and 'value' in self.stock_val_dict[a]
            ]
            if valid_series:
                sec_pe[sec] = pd.concat(valid_series, axis=1).mean(axis=1)

        self.sector_pe_df = pd.DataFrame(sec_pe).reindex(dates).ffill()

        # 1. 滚动 252 日估值 Z-Score: (PE - Mean) / Std (Clip 避免离群点)
        rolling_mean = self.sector_pe_df.rolling(self.rolling_window, min_periods=40).mean()
        rolling_std = self.sector_pe_df.rolling(self.rolling_window, min_periods=40).std().replace(0, np.nan).fillna(1.0)
        self.sector_pe_zscore = ((self.sector_pe_df - rolling_mean) / rolling_std).clip(-3.0, 3.0).fillna(0.0)

        # 2. 盈利收益率 (Earnings Yield = 1 / PE)
        self.sector_earnings_yield = (1.0 / self.sector_pe_df.replace(0, np.nan)).fillna(0.0)

        # 3. 综合估值吸引力评分 (Value Score: 奖励估值折价与高盈利收益率，惩罚极端估值泡沫)
        # 估值越处于历史低位(折价)，得分越高；估值泡沫越严重，得分扣除
        val_composite_score = (-1.0 * self.sector_pe_zscore)

        return {
            'pe_zscore': self.sector_pe_zscore,
            'earnings_yield': self.sector_earnings_yield,
            'valuation_score': val_composite_score,
        }

    def get_sector_valuation_score(self, sector: str, date: pd.Timestamp) -> float:
        """获取指定行业在 T-1 日的估值综合调整打分。"""
        if self.sector_pe_zscore is not None and sector in self.sector_pe_zscore.columns:
            if date in self.sector_pe_zscore.index:
                z = self.sector_pe_zscore.loc[date, sector]
                return float(-1.0 * z) if not np.isnan(z) else 0.0
        return 0.0
