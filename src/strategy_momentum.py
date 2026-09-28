"""截面多周期风险调整动量与大盘环境过滤策略。

包含：
1. 多周期动量与已实现波动率计算 (20/60/120日加权收益 / 20日年化波动率)
2. 个券自身 MA60 绝对均线过滤 (趋势防守，不买左侧下跌股)
3. 大盘基准 (SPY) 双均线 (MA50/MA200) 与波动率环境状态机判断
4. 进攻性杠杆 ETF 自适应升级映射 (Bull 环境升级为 SOXL/TQQQ/FAS 等)
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from data_moomoo import DEFAULT_SECTORS, LEVERAGED_MAP, DEFAULT_DEFENSE


class MomentumRegimeStrategy:
    """截面动量与杠杆轮动策略引擎。

    Attributes:
        top_n: 选取截面动量排名前 N 名标的，默认 3
        ma_filter_window: 个券自身均线过滤窗口，默认 60
        vol_window: 波动率计算窗口，默认 20
        spy_benchmark: 宏观市场环境参考标的，默认 'US.SPY'
        vol_threshold: 市场恐慌波动率阈值，默认 0.25 (25% 年化波动率)
    """

    def __init__(
        self,
        top_n: int = 3,
        exit_rank: Optional[int] = None,
        ma_filter_window: int = 60,
        vol_window: int = 20,
        spy_benchmark: str = 'US.SPY',
        vol_threshold: float = 0.25,
    ):
        self.top_n = top_n
        self.exit_rank = exit_rank or top_n
        self.ma_filter_window = ma_filter_window
        self.vol_window = vol_window
        self.spy_benchmark = spy_benchmark
        self.vol_threshold = vol_threshold
        self._current_held_sectors: List[str] = []

    def compute_asset_indicators(self, df_close: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """计算所有资产的基础动量、波动率与均线。

        Args:
            df_close: 各资产收盘价 DataFrame (index: date, columns: tickers)

        Returns:
            scores: 多周期风险调整动量评分 DataFrame
            vols: 20日年化已实现波动率 DataFrame
            ma_trend: 价格相对 MA60 的趋势布尔值 DataFrame (close > MA60)
        """
        # 收益率计算
        ret_20 = df_close.pct_change(20, fill_method=None)
        ret_60 = df_close.pct_change(60, fill_method=None)
        ret_120 = df_close.pct_change(120, fill_method=None)

        # 日度收益率与滚动年化波动率 (20日)
        daily_ret = df_close.pct_change(1, fill_method=None)
        vols = daily_ret.rolling(self.vol_window).std() * np.sqrt(252)
        # 避免波动率除零
        vols_safe = vols.replace(0, np.nan).fillna(0.15)

        # 多周期加权动量除以波动率
        raw_mom = 0.4 * ret_20 + 0.4 * ret_60 + 0.2 * ret_120
        scores = raw_mom / vols_safe

        # 个券自身 MA60 绝对均线
        ma60 = df_close.rolling(self.ma_filter_window).mean()
        ma_trend = df_close > ma60

        return scores, vols, ma_trend

    def get_market_regimes(self, spy_close: pd.Series) -> pd.Series:
        """评估大盘基准环境状态 (Bull / Neutral / Bear)。

        状态定义：
        - BULL (多头强进攻): SPY > MA50 且 SPY > MA200 且 20日波动率 < 0.25
        - NEUTRAL (震荡/弱势): SPY > MA200 但未满足 BULL 条件
        - BEAR (熊市破位): SPY <= MA200

        Returns:
            Series of str: 'BULL', 'NEUTRAL', 'BEAR'
        """
        ma50 = spy_close.rolling(50).mean()
        ma200 = spy_close.rolling(200).mean()
        spy_ret = spy_close.pct_change(1, fill_method=None)
        spy_vol = spy_ret.rolling(self.vol_window).std() * np.sqrt(252)

        regimes = pd.Series(index=spy_close.index, dtype=object)

        is_bull = (spy_close > ma50) & (spy_close > ma200) & (spy_vol < self.vol_threshold)
        is_neutral = (spy_close > ma200) & (~is_bull)
        is_bear = spy_close <= ma200

        regimes[is_bull] = 'BULL'
        regimes[is_neutral] = 'NEUTRAL'
        regimes[is_bear] = 'BEAR'

        return regimes

    def generate_daily_signals(
        self,
        date: pd.Timestamp,
        current_scores: pd.Series,
        current_trend: pd.Series,
        current_vols: pd.Series,
        regime: str,
        eligible_sectors: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """在给定日期生成最终的目标持仓标的及其实际对应波动率。

        Args:
            date: 当前交易日
            current_scores: 当日各行业 1x 标的的动量得分
            current_trend: 当日各行业 1x 标的是否 > MA60
            current_vols: 当日各标的 (含杠杆标的) 的已实现波动率
            regime: 当日大盘环境 ('BULL', 'NEUTRAL', 'BEAR')
            eligible_sectors: 参与截面排名的行业池，默认 DEFAULT_SECTORS

        Returns:
            List[Tuple[selected_ticker, ticker_vol]]: 选中的资产代码及对应波动率
        """
        sectors = eligible_sectors or DEFAULT_SECTORS

        # 1. 若大盘处于 BEAR 状态，全部退守防御性资产
        if regime == 'BEAR':
            self._current_held_sectors = []
            def_vol = current_vols.get(DEFAULT_DEFENSE, 0.02)
            if np.isnan(def_vol) or def_vol <= 0:
                def_vol = 0.02
            return [(DEFAULT_DEFENSE, def_vol)]

        # 2. 截面筛选：仅保留 > MA60 且有有效得分的 1x 标的
        valid_candidates = []
        for sec in sectors:
            if sec in current_scores.index and sec in current_trend.index:
                score = current_scores[sec]
                is_uptrend = current_trend[sec]
                if is_uptrend and not np.isnan(score):
                    valid_candidates.append((sec, score))

        # 按得分从高到低排序
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_sectors = [x[0] for x in valid_candidates]

        # 若启用了 exit_rank 迟滞缓冲区 (exit_rank > top_n):
        if self.exit_rank > self.top_n:
            kept = [s for s in self._current_held_sectors if s in ranked_sectors[:self.exit_rank]]
            for s in ranked_sectors:
                if len(kept) >= self.top_n:
                    break
                if s not in kept:
                    kept.append(s)
            selected_sectors = kept
        else:
            selected_sectors = ranked_sectors[:self.top_n]

        self._current_held_sectors = selected_sectors

        # 3. 杠杆升级与资产映射
        selected_assets = []
        for sec in selected_sectors:
            target_asset = sec
            if regime == 'BULL' and sec in LEVERAGED_MAP:
                # 升级为进攻性杠杆 ETF
                target_asset = LEVERAGED_MAP[sec]

            vol = current_vols.get(target_asset, np.nan)
            if np.isnan(vol) or vol <= 0:
                base_vol = current_vols.get(sec, 0.20)
                vol = base_vol * (3.0 if target_asset in LEVERAGED_MAP.values() else 1.0)
            
            selected_assets.append((target_asset, vol))

        # 4. 若合格标的少于 top_n，用防御资产补足席位
        slots_needed = self.top_n - len(selected_assets)
        if slots_needed > 0:
            def_vol = current_vols.get(DEFAULT_DEFENSE, 0.02)
            if np.isnan(def_vol) or def_vol <= 0:
                def_vol = 0.02
            for _ in range(slots_needed):
                selected_assets.append((DEFAULT_DEFENSE, def_vol))

        return selected_assets
