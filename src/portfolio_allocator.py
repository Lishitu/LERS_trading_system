"""组合头寸与权重分配引擎。

实现：
1. 20日滚动波动率倒数加权 (Inverse Volatility Weighting)
2. 单资产权重上限截断与迭代重平衡 (Cap = 35%)
3. 5% 调仓换手缓冲区 (Turnover Buffer)，避免每日微小变动带来过度摩擦
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class PortfolioAllocator:
    """组合权重分配与换手控制分配器。

    Attributes:
        max_asset_weight: 单资产最高权重上限，默认 0.35
        rebalance_buffer: 调仓偏离容忍阈值，默认 0.05 (5%)
        cash_ticker: 防御标的，默认 'US.BIL'
    """

    def __init__(
        self,
        max_asset_weight: float = 0.35,
        rebalance_buffer: float = 0.05,
        cash_ticker: str = 'US.BIL',
    ):
        self.max_asset_weight = max_asset_weight
        self.rebalance_buffer = rebalance_buffer
        self.cash_ticker = cash_ticker

    def compute_target_weights(
        self,
        selected_assets: List[Tuple[str, float]],
        total_capital: float = 1.0,
        win_rates: Optional[Dict[str, float]] = None,
        min_win_rate: float = 0.50,
    ) -> Dict[str, float]:
        """根据选中的标的与其波动率计算目标权重。

        Args:
            selected_assets: [(ticker, vol), ...]
            total_capital: 总资金占比，默认 1.0
            win_rates: 各标的预测胜率 Dict[ticker, P]
            min_win_rate: 最低入选胜率门槛，默认 0.50

        Returns:
            Dict[ticker, weight]
        """
        if not selected_assets:
            return {self.cash_ticker: 1.0}

        # 若只有防守标的，100% 配置防御资产
        if len(selected_assets) == 1 and selected_assets[0][0] == self.cash_ticker:
            return {self.cash_ticker: 1.0}

        # 区分风险资产与防御资产
        risk_assets = [item for item in selected_assets if item[0] != self.cash_ticker]
        defense_count = sum(1 for item in selected_assets if item[0] == self.cash_ticker)

        # 1. 风险资产按波动率倒数与凯利置信度加权
        if not risk_assets:
            return {self.cash_ticker: 1.0}

        inv_vols = {}
        for item in risk_assets:
            ticker = item[0]
            vol = item[1]
            factor_score = item[2] if len(item) > 2 else 0.0

            # 确保波动率有合理的正下界
            safe_vol = max(vol, 0.05) if not np.isnan(vol) else 0.20
            # 因子倾斜：多因子综合得分更高的标的，赋予适度正向加权偏置
            tilt = float(np.exp(0.15 * np.clip(factor_score, -2.0, 2.0)))
            score = (1.0 / safe_vol) * tilt

            # 若接入胜率质检，执行硬门槛过滤与凯利置信度乘数
            if win_rates is not None:
                p = win_rates.get(ticker, 0.50)
                if p < min_win_rate:
                    # 胜率不达标，直接否决，不予分配仓位
                    continue
                confidence = max(0.05, 2.0 * p - 1.0)
                score *= confidence

            inv_vols[ticker] = inv_vols.get(ticker, 0.0) + score

        if not inv_vols:
            # 所有风险资产均未通过胜率门槛，全额退守防御
            return {self.cash_ticker: 1.0}

        total_inv_vol = sum(inv_vols.values())
        raw_weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}

        # 2. 迭代 Cap 平截算法 (单风险资产 <= max_asset_weight)
        target_weights = self._apply_cap_constraint(raw_weights, self.max_asset_weight)

        # 3. 若有未满足席位的防御资产配额，将剩余未分配比例归入防御资产
        allocated_risk_sum = sum(target_weights.values())
        remaining = max(0.0, 1.0 - allocated_risk_sum)
        if remaining > 1e-4 or defense_count > 0:
            target_weights[self.cash_ticker] = target_weights.get(self.cash_ticker, 0.0) + remaining

        return target_weights

    def _apply_cap_constraint(self, weights: Dict[str, float], cap: float) -> Dict[str, float]:
        """对风险资产权重施加上限约束，并将溢出权重按比例分配给未受限资产。"""
        # 如果资产数量少导致 1/N > cap，则退化为平均分配或尽量靠近 cap
        n = len(weights)
        if n * cap < 1.0:
            # 允许平分或按上限
            pass

        adjusted = weights.copy()
        capped = set()

        for _ in range(len(weights) + 1):
            excess = 0.0
            uncapped = []
            for t, w in adjusted.items():
                if t in capped:
                    continue
                if w > cap:
                    capped.add(t)
                    excess += (w - cap)
                    adjusted[t] = cap
                else:
                    uncapped.append(t)

            if excess <= 1e-6 or not uncapped:
                break

            uncapped_sum = sum(adjusted[t] for t in uncapped)
            if uncapped_sum > 0:
                for t in uncapped:
                    adjusted[t] += excess * (adjusted[t] / uncapped_sum)
            else:
                for t in uncapped:
                    adjusted[t] += excess / len(uncapped)

        return adjusted

    def apply_turnover_buffer(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """应用 5% 换手缓冲区判断，过滤微小的仓位微调。

        规则：
        1. 退出标的 (current > 0 且 target == 0): 必须完全卖出
        2. 新进标的 (current == 0 且 target > 0): 必须买入至目标权重
        3. 保留标的 (current > 0 且 target > 0):
           - 若 |current - target| <= buffer，维持 current_weight 不变
           - 若 |current - target| > buffer，调整为 target_weight
        4. 调整后剩余未使用的仓位归入现金/BIL

        Returns:
            Dict[ticker, execution_weight]
        """
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        buffered_weights = {}

        for t in all_tickers:
            curr_w = current_weights.get(t, 0.0)
            targ_w = target_weights.get(t, 0.0)

            # 情况 1: 退出标的，彻底清仓
            if targ_w <= 1e-4:
                continue

            # 情况 2: 新进标的，直接按目标开仓
            if curr_w <= 1e-4:
                buffered_weights[t] = targ_w
                continue

            # 情况 3: 连续持仓标的，检查是否超出缓冲区
            if abs(targ_w - curr_w) <= self.rebalance_buffer:
                # 在容忍范围内，维持现有仓位不变
                buffered_weights[t] = curr_w
            else:
                # 超出容忍范围，调仓至目标权重
                buffered_weights[t] = targ_w

        # 校验总仓位是否超出 100%
        total_w = sum(buffered_weights.values())
        if total_w > 1.0:
            # 缩放至 1.0
            buffered_weights = {t: w / total_w for t, w in buffered_weights.items()}
        elif total_w < 1.0:
            # 剩余部分存入现金/BIL
            gap = 1.0 - total_w
            buffered_weights[self.cash_ticker] = buffered_weights.get(self.cash_ticker, 0.0) + gap

        return buffered_weights
