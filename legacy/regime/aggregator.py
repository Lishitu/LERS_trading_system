"""
Regime 参数聚合器

将多个 Agent（宏观 Agent、消息面 Agent）的输出聚合为统一的 Regime 参数

聚合策略：
1. 加权平均：根据 Agent 置信度加权
2. 配置权重：可配置的固定权重
3. 保守策略：取较保守的参数
"""

from dataclasses import dataclass
from typing import Optional, Literal

from .base import AgentOutput, RegimeParams, SentimentType


@dataclass
class AggregatedRegime:
    """
    聚合后的 Regime 输出
    """
    date: str
    sentiment: SentimentType
    intensity: float
    confidence: float
    reasoning: str
    params: RegimeParams

    # 各 Agent 贡献度
    source_weights: dict[str, float]
    macro_sentiment: Optional[SentimentType] = None
    sentiment_agent_sentiment: Optional[SentimentType] = None


class RegimeAggregator:
    """
    多 Agent 参数聚合器

    将宏观 Agent 和消息面 Agent 的输出聚合为统一参数

    使用方式:
        aggregator = RegimeAggregator(macro_weight=0.6, sentiment_weight=0.4)
        result = aggregator.aggregate(macro_output, sentiment_output)
    """

    def __init__(
        self,
        macro_weight: float = 0.6,
        sentiment_weight: float = 0.4,
        use_confidence_weighting: bool = False,
    ):
        """
        初始化聚合器

        Args:
            macro_weight: 宏观 Agent 权重
            sentiment_weight: 消息面 Agent 权重
            use_confidence_weighting: 是否使用置信度动态加权
        """
        self.macro_weight = macro_weight
        self.sentiment_weight = sentiment_weight
        self.use_confidence_weighting = use_confidence_weighting

    def aggregate(
        self,
        macro_output: Optional[AgentOutput] = None,
        sentiment_output: Optional[AgentOutput] = None,
        macro_snapshot: Optional = None,
        sentiment_snapshot: Optional = None,
    ) -> AggregatedRegime:
        """
        聚合多个 Agent 输出

        Args:
            macro_output: 宏观 Agent 输出
            sentiment_output: 消息面 Agent 输出

        Returns:
            聚合后的 Regime
        """
        date = (
            macro_output.date
            if macro_output
            else (sentiment_output.date if sentiment_output else "unknown")
        )

        # 计算实际权重
        total_weight = 0.0
        macro_effective_weight = 0.0
        sentiment_effective_weight = 0.0

        if macro_output is not None:
            if self.use_confidence_weighting:
                macro_effective_weight = self.macro_weight * macro_output.confidence
            else:
                macro_effective_weight = self.macro_weight
            total_weight += macro_effective_weight

        if sentiment_output is not None:
            if self.use_confidence_weighting:
                sentiment_effective_weight = self.sentiment_weight * sentiment_output.confidence
            else:
                sentiment_effective_weight = self.sentiment_weight
            total_weight += sentiment_effective_weight

        # 归一化
        if total_weight > 0:
            macro_effective_weight /= total_weight
            sentiment_effective_weight /= total_weight
        else:
            # 全部无效，返回中性
            return AggregatedRegime(
                date=date,
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=0.0,
                reasoning="No valid agent outputs",
                params=RegimeParams.neutral(),
                source_weights={},
            )

        # 聚合参数
        risk_on_adj = 0.0
        risk_off_adj = 0.0
        prob_scale = 1.0
        total_weight_for_params = 0.0

        if macro_output and macro_output.params:
            w = macro_effective_weight
            risk_on_adj += macro_output.params.risk_on_threshold_adjustment * w
            risk_off_adj += macro_output.params.risk_off_threshold_adjustment * w
            prob_scale += (macro_output.params.probability_scale - 1.0) * w
            total_weight_for_params += w

        if sentiment_output and sentiment_output.params:
            w = sentiment_effective_weight
            risk_on_adj += sentiment_output.params.risk_on_threshold_adjustment * w
            risk_off_adj += sentiment_output.params.risk_off_threshold_adjustment * w
            prob_scale += (sentiment_output.params.probability_scale - 1.0) * w
            total_weight_for_params += w

        # 归一化参数
        if total_weight_for_params > 0:
            risk_on_adj /= total_weight_for_params
            risk_off_adj /= total_weight_for_params

        # 聚合情绪
        sentiment_score = self._aggregate_sentiment(
            macro_output, sentiment_output,
            macro_effective_weight, sentiment_effective_weight
        )

        # 聚合强度
        intensity = 0.5
        if macro_output and sentiment_output:
            intensity = (
                macro_output.intensity * macro_effective_weight +
                sentiment_output.intensity * sentiment_effective_weight
            )
        elif macro_output:
            intensity = macro_output.intensity
        elif sentiment_output:
            intensity = sentiment_output.intensity

        # 聚合置信度
        confidence = 0.0
        if macro_output and sentiment_output:
            confidence = (
                macro_output.confidence * macro_effective_weight +
                sentiment_output.confidence * sentiment_effective_weight
            )
        elif macro_output:
            confidence = macro_output.confidence
        elif sentiment_output:
            confidence = sentiment_output.confidence

        # 生成推理
        reasoning_parts = []
        if macro_output:
            reasoning_parts.append(f"宏观: {macro_output.reasoning}")
        if sentiment_output:
            reasoning_parts.append(f"消息: {sentiment_output.reasoning}")
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No analysis available"

        return AggregatedRegime(
            date=date,
            sentiment=sentiment_score,
            intensity=intensity,
            confidence=confidence,
            reasoning=reasoning,
            params=RegimeParams(
                risk_on_threshold_adjustment=risk_on_adj,
                risk_off_threshold_adjustment=risk_off_adj,
                probability_scale=prob_scale,
                overall_sentiment=sentiment_score,
                confidence=confidence,
            ),
            source_weights={
                "macro": macro_effective_weight,
                "sentiment": sentiment_effective_weight,
            },
            macro_sentiment=macro_output.sentiment if macro_output else None,
            sentiment_agent_sentiment=sentiment_output.sentiment if sentiment_output else None,
        )

    def _aggregate_sentiment(
        self,
        macro_output: Optional[AgentOutput],
        sentiment_output: Optional[AgentOutput],
        macro_weight: float,
        sentiment_weight: float,
    ) -> SentimentType:
        """
        聚合情绪

        使用加权投票
        """
        sentiment_scores = {
            SentimentType.POSITIVE: 0.0,
            SentimentType.NEGATIVE: 0.0,
            SentimentType.NEUTRAL: 0.0,
        }

        sentiment_map = {
            SentimentType.POSITIVE: 1.0,
            SentimentType.NEGATIVE: -1.0,
            SentimentType.NEUTRAL: 0.0,
        }

        total = 0.0

        if macro_output:
            w = macro_weight * (macro_output.intensity + 0.5)  # 强度影响权重
            sentiment_scores[macro_output.sentiment] += w
            total += w

        if sentiment_output:
            w = sentiment_weight * (sentiment_output.intensity + 0.5)
            sentiment_scores[sentiment_output.sentiment] += w
            total += w

        if total == 0:
            return SentimentType.NEUTRAL

        # 计算加权分数
        weighted_score = sum(
            sentiment_scores[s] * sentiment_map[s]
            for s in SentimentType
        ) / total

        if weighted_score > 0.2:
            return SentimentType.POSITIVE
        elif weighted_score < -0.2:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL


# =============================================================================
# 便捷函数
# =============================================================================


def aggregate_regimes(
    macro_output: Optional[AgentOutput] = None,
    sentiment_output: Optional[AgentOutput] = None,
    macro_weight: float = 0.6,
    sentiment_weight: float = 0.4,
) -> AggregatedRegime:
    """
    聚合 Regime 的便捷函数
    """
    aggregator = RegimeAggregator(
        macro_weight=macro_weight,
        sentiment_weight=sentiment_weight,
    )
    return aggregator.aggregate(macro_output, sentiment_output)


if __name__ == "__main__":
    from .macro_agent import MacroAgent
    from .sentiment_agent import SentimentAgent
    from .base import MacroSnapshot, SentimentSnapshot

    # 测试聚合
    macro_agent = MacroAgent(provider="mock")
    sentiment_agent = SentimentAgent(provider="mock")
    aggregator = RegimeAggregator(macro_weight=0.6, sentiment_weight=0.4)

    macro_snapshot = MacroSnapshot(
        date="2024-01-15",
        vix_level=25.0,
        vix_5d_change=5.0,
        spy_20d_return=-0.03,
    )

    sentiment_snapshot = SentimentSnapshot(
        date="2024-01-15",
        news_count_7d=50,
        news_sentiment_score=-0.3,
        put_call_ratio=1.2,
        wsb_sentiment="bearish",
    )

    macro_output = macro_agent.predict(macro_snapshot)
    sentiment_output = sentiment_agent.predict(sentiment_snapshot)

    print("Macro Agent Output:")
    print(f"  Sentiment: {macro_output.sentiment.value}")
    print(f"  Intensity: {macro_output.intensity:.2f}")
    print(f"  Params: risk_on_adj={macro_output.params.risk_on_threshold_adjustment:+.3f}")

    print("\nSentiment Agent Output:")
    print(f"  Sentiment: {sentiment_output.sentiment.value}")
    print(f"  Intensity: {sentiment_output.intensity:.2f}")
    print(f"  Params: risk_on_adj={sentiment_output.params.risk_on_threshold_adjustment:+.3f}")

    result = aggregator.aggregate(macro_output, sentiment_output)

    print("\n" + "=" * 50)
    print("Aggregated Result:")
    print(f"  Date: {result.date}")
    print(f"  Sentiment: {result.sentiment.value}")
    print(f"  Intensity: {result.intensity:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Source Weights: {result.source_weights}")
    print(f"\n  Params:")
    print(f"    risk_on_adj: {result.params.risk_on_threshold_adjustment:+.3f}")
    print(f"    risk_off_adj: {result.params.risk_off_threshold_adjustment:+.3f}")
    print(f"    prob_scale: {result.params.probability_scale:.3f}")
    print(f"\n  Reasoning: {result.reasoning[:100]}...")
