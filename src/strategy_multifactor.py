"""多因子 ETF 截面选基与组合优化策略引擎。

将动量与波动率解构为多因子评分体系中的独立维度，融合多维量价与微结构因子：
1. 动量因子 (Momentum Factor): 多周期加权收益率 (20/60/120日)
2. 时序动量加速度因子 (Acceleration Factor): 5日动量 - 20日动量*0.25
3. 低波动/质量因子 (Low Vol Factor): 20日年化波动率负向打分
4. 量价配合因子 (Volume Surge Factor): 5日均成交量 / 20日均成交量
5. VWAP基差防御因子 (VWAP Basis Defense): 负向惩罚脱离当日VWAP过大的追高买点

支持：
- 截面 Z-Score 标准化与自适应加权
- 个券 MA60 绝对趋势防守
- QQQ 大盘基准双均线与波动率环境滤网
- 进攻杠杆标的升级映射 (SOXL/TQQQ/FAS)
- 因子倾斜的波动率倒数组合优化 (Factor-Tilted Inverse Volatility Allocation)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from data_moomoo import DEFAULT_SECTORS, LEVERAGED_MAP, DEFAULT_DEFENSE


class MultiFactorETFStrategy:
    """多因子 ETF 选基与组合优化策略引擎。

    Attributes:
        benchmark: 宏观基准，默认 'US.QQQ'
        top_n: 入选标的席位数，默认 3
        exit_rank: 退出迟滞排名，默认 4
        ma_filter_window: 个券自身均线防守窗口，默认 60
        vol_threshold: QQQ 恐慌波动率阈值，默认 0.28
    """

    def __init__(
        self,
        benchmark: str = 'US.QQQ',
        top_n: int = 3,
        exit_rank: int = 4,
        ma_filter_window: int = 60,
        vol_threshold: float = 0.28,
        factor_weights: Optional[Dict[str, float]] = None,
    ):
        self.benchmark = benchmark
        self.top_n = top_n
        self.exit_rank = exit_rank
        self.ma_filter_window = ma_filter_window
        self.vol_threshold = vol_threshold
        self.factor_weights = factor_weights or {
            'momentum': 0.55,        # 风险调整多周期复合动量
            'acceleration': 0.30,    # 风险调整动量二阶加速度 (启动 vs 衰竭)
            'trend_strength': 0.15,  # 波动率归一化均线通道强度
        }
        self._current_held_sectors: List[str] = []

    def compute_factors_for_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """批量计算所有标的的多因子矩阵。

        Returns:
            factors_dict: Dict[factor_name, DataFrame(date x ticker)]
            vols: DataFrame(date x ticker) 20日年化波动率
            ma_trend: DataFrame(date x ticker) Price > MA60
            df_close: DataFrame(date x ticker) 收盘价
        """
        close_dict = {t: df['close'] for t, df in data_dict.items()}
        df_close = pd.DataFrame(close_dict).dropna(subset=[self.benchmark])

        # 波动率基准与安全防零
        daily_ret = df_close.pct_change(1, fill_method=None)
        vols = daily_ret.rolling(20).std() * np.sqrt(252)
        vols_safe = vols.replace(0, np.nan).fillna(0.20)

        # 1. 动量因子 (风险调整 20/60/120日复合收益)
        r20 = df_close.pct_change(20, fill_method=None)
        r60 = df_close.pct_change(60, fill_method=None)
        r120 = df_close.pct_change(120, fill_method=None)
        f_mom = (0.4 * r20 + 0.4 * r60 + 0.2 * r120) / vols_safe

        # 2. 动量加速度因子 (风险调整 5日动量 - 20日动量/4)
        r5 = df_close.pct_change(5, fill_method=None)
        f_accel = (r5 - (r20 * 0.25)) / vols_safe

        # 3. 趋势通道强度 (波动率归一化的 MA60 距离)
        ma60 = df_close.rolling(self.ma_filter_window).mean()
        f_trend_strength = (df_close - ma60) / (vols_safe * df_close)

        # 个券 MA60 趋势防守布尔矩阵
        ma_trend = df_close > ma60

        factors = {
            'momentum': f_mom,
            'acceleration': f_accel,
            'trend_strength': f_trend_strength,
        }

        return factors, vols, ma_trend, df_close

    def get_market_regimes(self, qqq_close: pd.Series) -> pd.Series:
        """根据 QQQ 评估大盘环境状态 (Bull / Neutral / Bear)。"""
        ma50 = qqq_close.rolling(50).mean()
        ma200 = qqq_close.rolling(200).mean()
        qqq_ret = qqq_close.pct_change(1, fill_method=None)
        qqq_vol = qqq_ret.rolling(20).std() * np.sqrt(252)

        regimes = pd.Series(index=qqq_close.index, dtype=object)

        is_bull = (qqq_close > ma50) & (qqq_close > ma200) & (qqq_vol < self.vol_threshold)
        is_neutral = (qqq_close > ma200) & (~is_bull)
        is_bear = qqq_close <= ma200

        regimes[is_bull] = 'BULL'
        regimes[is_neutral] = 'NEUTRAL'
        regimes[is_bear] = 'BEAR'

        return regimes

    def compute_daily_composite_score(
        self,
        date: pd.Timestamp,
        factors_dict: Dict[str, pd.DataFrame],
        candidate_sectors: List[str],
    ) -> pd.Series:
        """在给定日期对候选池执行截面 Z-Score 标准化并合成多因子综合得分。"""
        sub_scores = []

        for f_name, weight in self.factor_weights.items():
            df_f = factors_dict[f_name]
            if date not in df_f.index:
                continue
            raw_vals = df_f.loc[date, candidate_sectors]

            # 截面 Z-Score 标准化
            mean_v = raw_vals.mean()
            std_v = raw_vals.std()
            if np.isnan(std_v) or std_v <= 1e-6:
                z_vals = pd.Series(0.0, index=candidate_sectors)
            else:
                z_vals = (raw_vals - mean_v) / std_v

            sub_scores.append(z_vals * weight)

        if not sub_scores:
            return pd.Series(0.0, index=candidate_sectors)

        composite_score = sum(sub_scores)
        return composite_score

    def generate_daily_signals(
        self,
        date: pd.Timestamp,
        factors_dict: Dict[str, pd.DataFrame],
        current_trend: pd.Series,
        current_vols: pd.Series,
        regime: str,
        eligible_sectors: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        """生成目标配置，包含：(资产代码, 波动率, 多因子得分)。

        Returns:
            List[Tuple[ticker, vol, score]]
        """
        sectors = eligible_sectors or DEFAULT_SECTORS

        # 1. 熊市破位状态：退守超短债
        if regime == 'BEAR':
            self._current_held_sectors = []
            def_vol = current_vols.get(DEFAULT_DEFENSE, 0.02)
            if np.isnan(def_vol) or def_vol <= 0:
                def_vol = 0.02
            return [(DEFAULT_DEFENSE, def_vol, 0.0)]

        # 2. 计算当日截面多因子得分
        comp_scores = self.compute_daily_composite_score(date, factors_dict, sectors)

        # 3. 趋势过滤：自身必须高于 MA60
        valid_candidates = []
        for sec in sectors:
            if sec in comp_scores.index and sec in current_trend.index:
                score = comp_scores[sec]
                is_uptrend = current_trend[sec]
                if is_uptrend and not np.isnan(score):
                    valid_candidates.append((sec, score))

        # 按多因子综合得分降序排序
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_sectors = [x[0] for x in valid_candidates]

        # 4. 退出排名迟滞带过滤
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

        # 5. 杠杆升级与波动率映射
        selected_assets = []
        for sec in selected_sectors:
            target_asset = sec
            sec_score = comp_scores.get(sec, 0.0)

            if regime == 'BULL' and sec in LEVERAGED_MAP:
                target_asset = LEVERAGED_MAP[sec]

            vol = current_vols.get(target_asset, np.nan)
            if np.isnan(vol) or vol <= 0:
                base_vol = current_vols.get(sec, 0.20)
                vol = base_vol * (3.0 if target_asset in LEVERAGED_MAP.values() else 1.0)

            selected_assets.append((target_asset, vol, sec_score))

        # 6. 席位不足时用防御资产补齐
        slots_needed = self.top_n - len(selected_assets)
        if slots_needed > 0:
            def_vol = current_vols.get(DEFAULT_DEFENSE, 0.02)
            if np.isnan(def_vol) or def_vol <= 0:
                def_vol = 0.02
            for _ in range(slots_needed):
                selected_assets.append((DEFAULT_DEFENSE, def_vol, 0.0))

        return selected_assets
