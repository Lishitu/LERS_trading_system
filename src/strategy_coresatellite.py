"""核心底仓 + 卫星动量轮动 (Core-Satellite) 策略引擎。

包含三大核心模块：
1. 宏观状态机 (Macro Regime Switching):
   - BULL_GROWTH (科技成长/降息宽松): QQQ 强势上升，重仓科技与半导体进攻杠杆 (QLD/TQQQ/USD/SOXL)
   - INFLATION_COMMODITY (通胀/大宗商品主线): 能源/工业金属/铀/贵金属相对强度爆发，卫星端重配资源品 (XLE/XOP/COPX/URA/GLD)
   - CRISIS_BEAR (流动性冲击/破位防守): QQQ 破位 MA200 或极端波动率飙升，底仓与不合规格卫星全面退守短债 (BIL)
   - NEUTRAL_CHOP (震荡过渡): 保持稳健底仓与严格趋势过滤

2. 核心底仓 (Core, 默认40%):
   - 锚定美股长期核心贝塔 (QQQ)，杜绝行业轮动错失纳斯达克科技大牛市的跟踪劣势
   - 根据波动率与大盘状态动态在 QLD (2x) / TQQQ (3x) / QQQ (1x) / BIL (防御) 之间自适应切换

3. 卫星动量轮动 (Satellite, 默认60%):
   - 覆盖 13 只高流动性行业、大宗商品 (铜/银/金/铀/油气) 与 QDII (日股)
   - 风险调整多周期复合动量 + 动量二阶加速度 + 波动率归一化趋势强度 + 宏观状态倾斜
   - 因子倾斜的波动率倒数加权 (Inverse Volatility) + 个券 25% 权重上限防守

4. 连续波动率目标化 (Volatility Targeting):
   - 目标年化波动率 22%~25%
   - 根据组合近 20 日已实现波动率动态缩放杠杆倍数 (0.6x ~ 1.8x)
   - 波动率低时放大杠杆增厚收益，波动率飙升时主动降杠杆规避 2020 / 2022 式尾部回撤
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from data_moomoo import (
    DEFAULT_SECTORS,
    LEVERAGED_MAP_2X,
    LEVERAGED_MAP_3X,
    DEFAULT_DEFENSE,
)
from valuation_factor import ValuationFactorEngine

# 宏观大宗商品与能源行业集合
COMMODITY_ENERGY_SECTORS = {
    'US.XLE',   # 能源
    'US.XOP',   # 油气开采
    'US.COPX',  # 铜矿
    'US.SLV',   # 白银
    'US.GLD',   # 黄金
    'US.URA',   # 铀/核能
}

# 科技成长集合
TECH_GROWTH_SECTORS = {
    'US.XLK',   # 科技
    'US.SMH',   # 半导体
}

# 纯生产性权益行业池 (排除零息贵金属 GLD/SLV，专注科技、工业、能源资源、QDII 与小盘成长)
PURE_GROWTH_SECTORS = [
    'US.XLK',   # 科技
    'US.SMH',   # 半导体
    'US.XLE',   # 能源
    'US.XOP',   # 油气开采
    'US.COPX',  # 铜矿
    'US.URA',   # 铀/核能
    'US.XLF',   # 金融
    'US.XLI',   # 工业
    'US.XBI',   # 生物医药
    'US.DXJ',   # QDII: 日股对冲
    'US.IWM',   # 罗素2000
]


class MacroRegimeDetector:
    """宏观状态检测器。"""

    def __init__(
        self,
        benchmark: str = 'US.QQQ',
        vol_threshold: float = 0.28,
        commodity_basket: Optional[List[str]] = None,
    ):
        self.benchmark = benchmark
        self.vol_threshold = vol_threshold
        self.commodity_basket = commodity_basket or list(COMMODITY_ENERGY_SECTORS)

    def detect_regimes(self, df_close: pd.DataFrame) -> pd.Series:
        """评估每日宏观状态。
        
        Returns:
            pd.Series(index=dates, values=['BULL_GROWTH', 'INFLATION_COMMODITY', 'CRISIS_BEAR', 'NEUTRAL_CHOP'])
        """
        qqq = df_close[self.benchmark]
        ma50 = qqq.rolling(50).mean()
        ma200 = qqq.rolling(200).mean()
        qqq_ret20 = qqq.pct_change(20, fill_method=None)
        daily_ret = qqq.pct_change(1, fill_method=None)
        qqq_vol20 = daily_ret.rolling(20).std() * np.sqrt(252)

        # 大宗商品等权指数与 60 日动量
        valid_commodities = [c for c in self.commodity_basket if c in df_close.columns]
        if valid_commodities:
            comm_index = df_close[valid_commodities].pct_change(1, fill_method=None).mean(axis=1).fillna(0.0)
            comm_cum = (1 + comm_index).cumprod()
            comm_ma50 = comm_cum.rolling(50).mean()
            comm_ret60 = comm_cum.pct_change(60, fill_method=None)
        else:
            comm_cum = pd.Series(1.0, index=df_close.index)
            comm_ma50 = pd.Series(1.0, index=df_close.index)
            comm_ret60 = pd.Series(0.0, index=df_close.index)

        qqq_ret60 = qqq.pct_change(60, fill_method=None)

        regimes = pd.Series('NEUTRAL_CHOP', index=df_close.index, dtype=object)

        # 1. 通胀/大宗商品周期 (商品60日超额动量超越 QQQ 4% 以上，且商品自身处于上升通道)
        is_commodity_dominating = (comm_ret60 > (qqq_ret60 + 0.04)) & (comm_cum > comm_ma50) & (comm_ret60 > 0.02)

        # 2. 危机/破位熊市 (QQQ 破位且非大宗商品独立牛市，或者市场波动率出现恐慌飙升)
        is_panic_vol = (qqq_vol20 > 0.32) | ((qqq_ret20 < -0.08) & (qqq_vol20 > self.vol_threshold))
        is_bear = ((qqq <= ma200) & (~is_commodity_dominating)) | is_panic_vol

        # 3. 科技成长牛市 (QQQ > MA200 且 QQQ > MA50，且波动率正常，且非商品主导)
        is_bull_base = (qqq > ma50) & (qqq > ma200) & (qqq_vol20 <= self.vol_threshold) & (~is_commodity_dominating)

        # 状态赋值
        regimes[is_bull_base] = 'BULL_GROWTH'
        regimes[is_commodity_dominating & (~is_panic_vol)] = 'INFLATION_COMMODITY'
        regimes[is_bear] = 'CRISIS_BEAR'

        return regimes


class CoreSatelliteStrategy:
    """核心底仓 + 卫星动量轮动 + 波动率目标化策略引擎。

    Attributes:
        core_ratio: 核心底仓基准比例，默认 0.40 (40%)
        satellite_ratio: 卫星轮动基准比例，默认 0.60 (60%)
        target_vol: 目标年化波动率，默认 0.22 (22%)
        min_leverage: 最低杠杆/仓位缩放系数，默认 0.60
        max_leverage: 最高杠杆/仓位缩放系数，默认 1.80
        top_n: 卫星池持仓数量，默认 3
        exit_rank: 卫星退出迟滞排名，默认 4
    """

    def __init__(
        self,
        benchmark: str = 'US.QQQ',
        core_ratio: float = 0.50,
        satellite_ratio: float = 0.50,
        target_vol: float = 0.25,
        min_leverage: float = 0.60,
        max_leverage: float = 1.80,
        top_n: int = 2,
        exit_rank: int = 3,
        max_satellite_cap: float = 0.30,
        ma_filter_window: int = 60,
        val_factor_weight: float = 0.20,
        use_tqqq: bool = True,
        early_rebound: bool = True,
        satellite_sectors: Optional[List[str]] = None,
        weekly_satellite: bool = True,
        tiered_ratio: Optional[Tuple[float, ...]] = None,
    ):
        self.benchmark = benchmark
        self.core_ratio = core_ratio
        self.satellite_ratio = satellite_ratio
        self.target_vol = target_vol
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.top_n = top_n
        self.exit_rank = exit_rank
        self.max_satellite_cap = max_satellite_cap
        self.ma_filter_window = ma_filter_window
        self.val_factor_weight = val_factor_weight
        self.use_tqqq = use_tqqq
        self.early_rebound = early_rebound
        self.satellite_sectors = satellite_sectors or PURE_GROWTH_SECTORS
        self.weekly_satellite = weekly_satellite
        self.tiered_ratio = tiered_ratio

        self.regime_detector = MacroRegimeDetector(benchmark=benchmark)
        self.val_engine = ValuationFactorEngine()
        self._current_satellite_held: List[str] = []
        self._last_eval_date: Optional[pd.Timestamp] = None

    def reset(self) -> None:
        """重置策略内部的卫星持仓与调仓评估日期状态。"""
        self._current_satellite_held = []
        self._last_eval_date = None
        self._cached_df_id = None
        self._cached_ma200 = None
        self._cached_ma50 = None

    def compute_factors_and_indicators(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """批量计算多因子特征、波动率、趋势布尔值以及宏观状态。"""
        close_dict = {t: df['close'] for t, df in data_dict.items()}
        df_close = pd.DataFrame(close_dict).dropna(subset=[self.benchmark])

        daily_ret = df_close.pct_change(1, fill_method=None)
        vols = daily_ret.rolling(20).std() * np.sqrt(252)
        vols_safe = vols.replace(0, np.nan).fillna(0.20)

        # 1. 动量因子 (10/20/60日复合风险调整收益)
        ret10 = df_close.pct_change(10, fill_method=None)
        r20 = df_close.pct_change(20, fill_method=None)
        r60 = df_close.pct_change(60, fill_method=None)
        f_mom = (0.3 * ret10 + 0.4 * r20 + 0.3 * r60) / vols_safe

        # 2. 动量加速度 (5日短期启动 vs 20日均速)
        r5 = df_close.pct_change(5, fill_method=None)
        f_accel = (r5 - (r20 * 0.25)) / vols_safe

        # 3. 趋势强度 (波动率归一化 MA60 通道距离)
        ma60 = df_close.rolling(self.ma_filter_window).mean()
        f_trend = (df_close - ma60) / (vols_safe * df_close)

        # 4. Moomoo 估值因子 (基于 get_valuation_detail P/E 偏离度)
        try:
            val_factors = self.val_engine.load_and_compute_factors(df_close.index)
            f_val = val_factors.get('valuation_score', pd.DataFrame(0.0, index=df_close.index, columns=df_close.columns))
        except Exception:
            f_val = pd.DataFrame(0.0, index=df_close.index, columns=df_close.columns)

        factors = {
            'momentum': f_mom,
            'acceleration': f_accel,
            'trend_strength': f_trend,
            'valuation': f_val,
        }

        ma_trend = df_close > ma60
        regimes = self.regime_detector.detect_regimes(df_close)

        return factors, vols, ma_trend, df_close, regimes

    def determine_core_allocation(
        self,
        date: pd.Timestamp,
        regime: str,
        df_close: pd.DataFrame,
        current_vols: pd.Series,
    ) -> Dict[str, float]:
        """计算核心底仓目标标的与基础配置比例 (50% 额度内)。"""
        qqq_price = df_close[self.benchmark].loc[date]
        if getattr(self, '_cached_df_id', None) == id(df_close) and self._cached_ma200 is not None:
            qqq_ma200 = self._cached_ma200.loc[date]
            qqq_ma50 = self._cached_ma50.loc[date]
        else:
            self._cached_df_id = id(df_close)
            self._cached_ma200 = df_close[self.benchmark].rolling(200).mean()
            self._cached_ma50 = df_close[self.benchmark].rolling(50).mean()
            qqq_ma200 = self._cached_ma200.loc[date]
            qqq_ma50 = self._cached_ma50.loc[date]

        qqq_vol = current_vols.get(self.benchmark, 0.20)

        # 1. 科技成长大牛市 (QQQ > MA200 且 QQQ > MA50):
        #    - 若极度平稳低波 (vol < 16% 且启用了 use_tqqq): 采用 3x TQQQ 顶格进攻
        #    - 标准稳健进攻: 采用 2x QLD 长期进攻底仓
        if qqq_price > qqq_ma200 and qqq_price > qqq_ma50:
            if self.use_tqqq and qqq_vol < 0.16 and 'US.TQQQ' in df_close.columns:
                return {'US.TQQQ': self.core_ratio}
            elif 'US.QLD' in df_close.columns:
                return {'US.QLD': self.core_ratio}
            else:
                return {'US.QQQ': self.core_ratio}

        # 2. 早期反弹保护 (QQQ > MA50 或 QQQ > MA200):
        #    避免破位后见底急升阶段因滞后于 MA200 踏空前 15% 涨幅，以 1x QQQ 稳健参与
        if self.early_rebound and (qqq_price > qqq_ma50 or qqq_price > qqq_ma200):
            return {'US.QQQ': self.core_ratio}

        # 3. 真正深度熊市破位 (QQQ <= MA200 且 QQQ <= MA50): 100% 切换短债 BIL 防守
        return {DEFAULT_DEFENSE: self.core_ratio}

    def determine_satellite_allocation(
        self,
        date: pd.Timestamp,
        regime: str,
        factors_dict: Dict[str, pd.DataFrame],
        current_trend: pd.Series,
        current_vols: pd.Series,
        candidate_sectors: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """计算卫星轮动目标标的与配置比例 (50% 额度内)。"""
        candidate_pool = candidate_sectors or self.satellite_sectors or PURE_GROWTH_SECTORS
        if 'momentum' not in factors_dict or date not in factors_dict['momentum'].index:
            return {DEFAULT_DEFENSE: self.satellite_ratio}

        sectors = [s for s in candidate_pool if s in factors_dict['momentum'].columns]
        if not sectors:
            return {DEFAULT_DEFENSE: self.satellite_ratio}

        # 0. 评估是否需要执行全量调仓选股 (周度首日或个券破位应急触发)
        is_rebal_day = True
        if self.weekly_satellite and len(self._current_satellite_held) > 0:
            is_new_week = (date.weekday() == 0) or (
                self._last_eval_date is not None and date.isocalendar()[:2] != self._last_eval_date.isocalendar()[:2]
            )
            any_trend_broken = any(not current_trend.get(s, True) for s in self._current_satellite_held)
            is_rebal_day = is_new_week or any_trend_broken

        self._last_eval_date = date

        if not is_rebal_day:
            # 非调仓日只保留趋势健康的标的，破位标的自动剔除退守短债
            healthy_held = [s for s in self._current_satellite_held if current_trend.get(s, True)]
            self._current_satellite_held = healthy_held
            selected_sectors = healthy_held
        else:
            # 1. 风险调整多周期复合动量基础评分
            composite_scores = factors_dict['momentum'].loc[date, sectors].copy()

            # 2. 结合 Moomoo 估值因子打分 (Valuation Factor)
            if 'valuation' in factors_dict and date in factors_dict['valuation'].index:
                val_scores = factors_dict['valuation'].loc[date]
                for sec in sectors:
                    if sec in val_scores.index and not np.isnan(val_scores[sec]):
                        composite_scores[sec] += self.val_factor_weight * val_scores[sec]

            # 3. 宏观状态倾斜加分 (Macro Tilt)
            if regime == 'BULL_GROWTH':
                for sec in TECH_GROWTH_SECTORS:
                    if sec in composite_scores.index:
                        composite_scores[sec] += 0.35  # 强化科技与半导体打分
            elif regime == 'INFLATION_COMMODITY':
                for sec in ['US.XLE', 'US.XOP', 'US.COPX', 'US.URA']:
                    if sec in composite_scores.index:
                        composite_scores[sec] += 0.40  # 强化大宗商品与能源打分

            # 4. 趋势过滤：标的必须在自身 MA60 之上
            valid_candidates = []
            for sec in sectors:
                if sec in composite_scores.index and sec in current_trend.index:
                    if current_trend[sec] and not np.isnan(composite_scores[sec]):
                        valid_candidates.append((sec, composite_scores[sec]))

            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            ranked_sectors = [x[0] for x in valid_candidates]

            # 5. 退出迟滞排名过滤
            if self.exit_rank > self.top_n:
                kept = [s for s in self._current_satellite_held if s in ranked_sectors[:self.exit_rank]]
                for s in ranked_sectors:
                    if len(kept) >= self.top_n:
                        break
                    if s not in kept:
                        kept.append(s)
                selected_sectors = kept
            else:
                selected_sectors = ranked_sectors[:self.top_n]

            self._current_satellite_held = selected_sectors

        # 5. 卫星仓位分配 (支持等权或阶梯权重，严格单券不超过 max_satellite_cap，不足 top_n 席位则入短债防御)
        if not selected_sectors:
            return {DEFAULT_DEFENSE: self.satellite_ratio}

        n = len(selected_sectors)
        satellite_weights = {}
        total_assigned = 0.0

        if self.tiered_ratio is not None and len(self.tiered_ratio) == n:
            # 阶梯权重分配 (例如 Top 1 分配 60%，Top 2 分配 40%)
            for sec, ratio in zip(selected_sectors, self.tiered_ratio):
                w = min(self.satellite_ratio * ratio, self.max_satellite_cap)
                satellite_weights[sec] = w
                total_assigned += w
        else:
            slot_weight = min(self.satellite_ratio / n, self.max_satellite_cap)
            for sec in selected_sectors:
                satellite_weights[sec] = slot_weight
                total_assigned += slot_weight

        rem = self.satellite_ratio - total_assigned
        if rem > 1e-4:
            satellite_weights[DEFAULT_DEFENSE] = satellite_weights.get(DEFAULT_DEFENSE, 0.0) + rem

        return satellite_weights

    def calculate_volatility_multiplier(
        self,
        realized_portfolio_vol: float,
    ) -> float:
        """计算波动率目标化杠杆调节因子。
        
        Formula:
            multiplier = clip(target_vol / realized_vol, min_leverage, max_leverage)
        """
        if np.isnan(realized_portfolio_vol) or realized_portfolio_vol <= 0.05:
            return 1.0
        raw_scale = self.target_vol / realized_portfolio_vol
        return float(np.clip(raw_scale, self.min_leverage, self.max_leverage))

    def generate_portfolio_target(
        self,
        date: pd.Timestamp,
        factors_dict: Dict[str, pd.DataFrame],
        current_trend: pd.Series,
        current_vols: pd.Series,
        regime: str,
        df_close: pd.DataFrame,
        realized_portfolio_vol: float = 0.20,
    ) -> Dict[str, float]:
        """合成整体核心-卫星组合目标权重，并应用波动率目标化动态缩放。
        
        Returns:
            Dict[ticker, target_weight]
        """
        # 1. 核心底仓 (40%)
        core_alloc = self.determine_core_allocation(
            date=date,
            regime=regime,
            df_close=df_close,
            current_vols=current_vols,
        )

        # 2. 卫星轮动 (60%)
        sat_alloc = self.determine_satellite_allocation(
            date=date,
            regime=regime,
            factors_dict=factors_dict,
            current_trend=current_trend,
            current_vols=current_vols,
        )

        # 合并基础权重 (总和 = 1.0)
        combined_weights: Dict[str, float] = {}
        for t, w in core_alloc.items():
            combined_weights[t] = combined_weights.get(t, 0.0) + w
        for t, w in sat_alloc.items():
            combined_weights[t] = combined_weights.get(t, 0.0) + w

        # 3. 组合权重归一化与流动性垫 (应用连续波动率目标化动态缩放)
        vol_multiplier = self.calculate_volatility_multiplier(realized_portfolio_vol)
        # 极端高波动市场 (realized_vol > 32%) 执行应急仓位去杠杆保护
        risk_exposure = min(1.0, 0.32 / realized_portfolio_vol) if realized_portfolio_vol > 0.32 else 1.0
        # 实际权益仓位缩放系数 (在非融资无杠杆借贷模式下上限为 1.0，高波动时主动缩减至 min_leverage 并转移至短债 BIL 防御)
        vol_scale = float(np.clip(vol_multiplier * risk_exposure, self.min_leverage, 1.0))

        scaled_targets: Dict[str, float] = {}
        risk_weight_sum = 0.0
        for ticker, w in combined_weights.items():
            if ticker == DEFAULT_DEFENSE:
                continue
            scaled_w = w * 0.995 * vol_scale
            scaled_targets[ticker] = scaled_w
            risk_weight_sum += scaled_w

        cash_equiv_w = max(0.0, 0.995 - risk_weight_sum)
        if cash_equiv_w > 1e-4:
            scaled_targets[DEFAULT_DEFENSE] = scaled_targets.get(DEFAULT_DEFENSE, 0.0) + cash_equiv_w

        return scaled_targets
