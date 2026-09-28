"""
规则化 Regime Agent - 用于历史回测验证

使用简单的规则来判断 Regime，不需要 LLM。
这个版本用于验证 "Regime 调整" 这个假设是否有效。

规则设计思路：
- VIX 是最核心的恐慌指标
- 利率趋势反映宏观周期
- 组合多个指标做 Regime 判断
"""

from dataclasses import dataclass
from typing import Literal, Optional

from .base import (
    AgentOutput,
    MacroSnapshot,
    RegimeParams,
    SentimentSnapshot,
    SentimentType,
)


class RuleBasedRegimeAgent:
    """
    基于规则的 Regime 判断

    用于历史回测，验证 Regime 调整假设是否有效。
    后续可以替换成 LLM 驱动的 Agent。

    Regime 判断规则（增强版）：
    - 综合 VIX、利率、美元、股指趋势
    - 5 种状态: 强多、弱多、中性、弱空、强空
    """

    def __init__(
        self,
        # VIX 阈值
        vix_extreme_high: float = 30.0,
        vix_high: float = 22.0,
        vix_low: float = 16.0,
        vix_extreme_low: float = 12.0,
        # 调整参数幅度
        adj_strong: float = 0.15,
        adj_moderate: float = 0.08,
        # 概率缩放
        scale_high_vol: float = 0.85,
        scale_low_vol: float = 1.1,
    ):
        self.vix_extreme_high = vix_extreme_high
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.vix_extreme_low = vix_extreme_low
        self.adj_strong = adj_strong
        self.adj_moderate = adj_moderate
        self.scale_high_vol = scale_high_vol
        self.scale_low_vol = scale_low_vol

    def predict(
        self,
        macro_snapshot: Optional[MacroSnapshot] = None,
        sentiment_snapshot: Optional[SentimentSnapshot] = None,
    ) -> AgentOutput:
        """
        根据规则判断当前 Regime

        Args:
            macro_snapshot: 宏观数据快照
            sentiment_snapshot: 消息面快照（可选）

        Returns:
            Agent 输出
        """
        if macro_snapshot is None:
            return AgentOutput(
                date="unknown",
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=1.0,
                reasoning="No macro data, using neutral",
                source_name="rule_based",
            )

        regime_name, sentiment, intensity, reasoning = self._classify_regime(
            macro_snapshot
        )

        params = self._get_params_for_regime(regime_name)

        return AgentOutput(
            date=macro_snapshot.date,
            sentiment=sentiment,
            intensity=intensity,
            confidence=0.8,  # 规则版本置信度较高
            reasoning=reasoning,
            source_name="rule_based",
            params=params,
        )

    def _classify_regime(
        self, snapshot: MacroSnapshot
    ) -> tuple[str, SentimentType, float, str]:
        """
        分类 Regime（增强版评分系统）

        综合 VIX、SPY趋势、利率、美元等因素打分
        分数范围: -2 ~ +2

        Returns:
            (regime_name, sentiment, intensity, reasoning)
        """
        vix = snapshot.vix_level
        vix_change = snapshot.vix_5d_change
        spy_return = snapshot.spy_20d_return

        score = 0.0
        factors = []

        # VIX 水平打分
        if vix > self.vix_extreme_high:
            score -= 2.0
            factors.append(f"VIX={vix:.1f} 极高")
        elif vix > self.vix_high:
            score -= 1.0
            factors.append(f"VIX={vix:.1f} 偏高")
        elif vix < self.vix_extreme_low:
            score += 2.0
            factors.append(f"VIX={vix:.1f} 极低")
        elif vix < self.vix_low:
            score += 1.0
            factors.append(f"VIX={vix:.1f} 偏低")
        else:
            factors.append(f"VIX={vix:.1f} 中性")

        # VIX 变化趋势
        if vix_change > 5:
            score -= 0.5
            factors.append(f"VIX 5日+{vix_change:.1f}%")
        elif vix_change < -5:
            score += 0.5
            factors.append(f"VIX 5日{vix_change:.1f}%")

        # SPY 趋势
        if spy_return > 0.05:
            score += 1.0
            factors.append(f"SPY 20日+{spy_return:.1%}")
        elif spy_return > 0.02:
            score += 0.5
            factors.append(f"SPY 20日+{spy_return:.1%}")
        elif spy_return < -0.05:
            score -= 1.0
            factors.append(f"SPY 20日{spy_return:.1%}")
        elif spy_return < -0.02:
            score -= 0.5
            factors.append(f"SPY 20日{spy_return:.1%}")

        # 确定 Regime
        score = max(-2.0, min(2.0, score))

        if score >= 1.5:
            regime = "strong_risk_on"
            sentiment = SentimentType.POSITIVE
            intensity = min(1.0, (score - 1.5) + 0.7)
        elif score >= 0.5:
            regime = "moderate_risk_on"
            sentiment = SentimentType.POSITIVE
            intensity = 0.5 + (score - 0.5)
        elif score > -0.5:
            regime = "neutral"
            sentiment = SentimentType.NEUTRAL
            intensity = 0.5 - abs(score)
        elif score > -1.5:
            regime = "moderate_risk_off"
            sentiment = SentimentType.NEGATIVE
            intensity = 0.5 + (-score - 0.5)
        else:
            regime = "strong_risk_off"
            sentiment = SentimentType.NEGATIVE
            intensity = min(1.0, (-score - 1.5) + 0.7)

        reasoning = f"Score={score:+.1f}: {', '.join(factors)}"

        return (regime, sentiment, intensity, reasoning)

    def _get_params_for_regime(self, regime_name: str) -> RegimeParams:
        """获取对应 Regime 的参数"""
        params_map = {
            "strong_risk_on": RegimeParams(
                risk_on_threshold_adjustment=-self.adj_strong,
                risk_off_threshold_adjustment=self.adj_strong,
                probability_scale=self.scale_low_vol,
                overall_sentiment=SentimentType.POSITIVE,
            ),
            "moderate_risk_on": RegimeParams(
                risk_on_threshold_adjustment=-self.adj_moderate,
                risk_off_threshold_adjustment=self.adj_moderate,
                probability_scale=1.05,
                overall_sentiment=SentimentType.POSITIVE,
            ),
            "neutral": RegimeParams(
                risk_on_threshold_adjustment=0.0,
                risk_off_threshold_adjustment=0.0,
                probability_scale=1.0,
                overall_sentiment=SentimentType.NEUTRAL,
            ),
            "moderate_risk_off": RegimeParams(
                risk_on_threshold_adjustment=self.adj_moderate,
                risk_off_threshold_adjustment=-self.adj_moderate,
                probability_scale=0.95,
                overall_sentiment=SentimentType.NEGATIVE,
            ),
            "strong_risk_off": RegimeParams(
                risk_on_threshold_adjustment=self.adj_strong,
                risk_off_threshold_adjustment=-self.adj_strong,
                probability_scale=self.scale_high_vol,
                overall_sentiment=SentimentType.NEGATIVE,
            ),
        }
        return params_map.get(regime_name, params_map["neutral"])


# =============================================================================
# 简化版本：用于快速测试
# =============================================================================


class SimpleRuleBasedAgent:
    """
    简化版规则 Agent - 只用 VIX 和 SPY

    便于快速验证想法
    """

    def predict(self, vix_level: float, spy_return_20d: float) -> RegimeParams:
        """
        简化判断

        Args:
            vix_level: VIX 水平
            spy_return_20d: SPY 20日收益率

        Returns:
            Regime 参数
        """
        # 基准
        risk_on_adj = 0.0
        risk_off_adj = 0.0
        prob_scale = 1.0
        sentiment = SentimentType.NEUTRAL

        # VIX 影响
        if vix_level > 30:
            risk_on_adj = 0.15
            risk_off_adj = 0.10
            prob_scale = 0.8
            sentiment = SentimentType.NEGATIVE
        elif vix_level > 25:
            risk_on_adj = 0.10
            risk_off_adj = 0.05
            prob_scale = 0.9
        elif vix_level > 20:
            if spy_return_20d < -0.02:
                risk_on_adj = 0.08
                risk_off_adj = -0.05
                sentiment = SentimentType.NEGATIVE
            else:
                risk_on_adj = 0.0
                risk_off_adj = 0.0
        elif vix_level < 15:
            risk_on_adj = -0.10
            risk_off_adj = 0.10
            sentiment = SentimentType.POSITIVE
        elif vix_level < 18:
            if spy_return_20d > 0.02:
                risk_on_adj = -0.08
                risk_off_adj = 0.05
                sentiment = SentimentType.POSITIVE

        return RegimeParams(
            risk_on_threshold_adjustment=risk_on_adj,
            risk_off_threshold_adjustment=risk_off_adj,
            probability_scale=prob_scale,
            overall_sentiment=sentiment,
        )


if __name__ == "__main__":
    # 快速测试
    agent = RuleBasedRegimeAgent()

    test_cases = [
        MacroSnapshot(
            date="2024-01-15",
            vix_level=18.5,
            vix_5d_change=2.1,
            vix_20d_ma_ratio=1.1,
            spy_20d_return=0.03,
        ),
        MacroSnapshot(
            date="2020-03-15",
            vix_level=82.7,
            vix_5d_change=25.0,
            vix_20d_ma_ratio=2.5,
            spy_20d_return=-0.12,
        ),
        MacroSnapshot(
            date="2021-11-15",
            vix_level=28.5,
            vix_5d_change=5.0,
            spy_20d_return=-0.02,
        ),
    ]

    for snapshot in test_cases:
        output = agent.predict(macro_snapshot=snapshot)
        print(f"\n{snapshot.date}:")
        print(f"  Regime: {output.source_name}")
        print(f"  Sentiment: {output.sentiment.value}")
        print(f"  Intensity: {output.intensity:.2f}")
        print(f"  Reasoning: {output.reasoning}")
        if hasattr(output, "params") and output.params:
            print(
                f"  Params: risk_on_adj={output.params.risk_on_threshold_adjustment:+.2f}, "
                f"risk_off_adj={output.params.risk_off_threshold_adjustment:+.2f}, "
                f"prob_scale={output.params.probability_scale:.2f}"
            )
