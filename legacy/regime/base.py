"""
Regime 模块 - 基础数据类型和接口定义
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class SentimentType(Enum):
    """情绪类型"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class RegimeParams:
    """
    Regime 参数 - 将被注入到 ML 模型中

    这些参数调整模型的决策边界，使得在不同市场状态下
    模型的行为更加适配当前宏观/情绪环境。
    """
    # 基础阈值调整 (范围约 -0.15 到 +0.15)
    risk_on_threshold_adjustment: float = 0.0   # 正值 = 提高阈值 = 更难做多
    risk_off_threshold_adjustment: float = 0.0  # 正值 = 提高阈值 = 更难做空

    # 波动率调整 (范围约 0.8 到 1.2)
    # > 1.0 = 放大概率差异，< 1.0 = 缩小概率差异（更保守）
    probability_scale: float = 1.0

    # 综合情绪偏向 (可选，用于日志和调试)
    overall_sentiment: SentimentType = SentimentType.NEUTRAL

    # 置信度
    confidence: float = 1.0

    def get_adjusted_risk_on_threshold(self, base: float = 0.60) -> float:
        """获取调整后的 Risk-On 阈值"""
        return max(0.3, min(0.9, base + self.risk_on_threshold_adjustment))

    def get_adjusted_risk_off_threshold(self, base: float = 0.60) -> float:
        """获取调整后的 Risk-Off 阈值"""
        return max(0.3, min(0.9, base + self.risk_off_threshold_adjustment))

    def to_feature_dict(self) -> dict:
        """转为特征字典，可作为 ML 模型的额外特征"""
        return {
            "regime_risk_on_adj": self.risk_on_threshold_adjustment,
            "regime_risk_off_adj": self.risk_off_threshold_adjustment,
            "regime_prob_scale": self.probability_scale,
            "regime_sentiment_score": {
                SentimentType.POSITIVE: 1.0,
                SentimentType.NEGATIVE: -1.0,
                SentimentType.NEUTRAL: 0.0,
            }[self.overall_sentiment],
            "regime_confidence": self.confidence,
        }

    @classmethod
    def neutral(cls) -> RegimeParams:
        """中性参数（不调整）"""
        return cls()

    @classmethod
    def risk_on_aggressive(cls) -> RegimeParams:
        """积极做多参数"""
        return cls(
            risk_on_threshold_adjustment=-0.10,
            risk_off_threshold_adjustment=0.10,
            overall_sentiment=SentimentType.POSITIVE,
        )

    @classmethod
    def risk_off_defensive(cls) -> RegimeParams:
        """防御做空参数"""
        return cls(
            risk_on_threshold_adjustment=0.10,
            risk_off_threshold_adjustment=-0.10,
            overall_sentiment=SentimentType.NEGATIVE,
        )


@dataclass
class AgentOutput:
    """
    Agent 输出基类
    """
    date: str
    sentiment: SentimentType
    intensity: float = 0.5  # 0.0 - 1.0
    confidence: float = 0.5  # 0.0 - 1.0
    reasoning: str = ""
    source_name: str = ""
    params: Optional["RegimeParams"] = None  # Regime 参数

    def __post_init__(self):
        if isinstance(self.sentiment, str):
            self.sentiment = SentimentType(self.sentiment.lower())


@dataclass
class MacroSnapshot:
    """
    宏观数据快照 - 格式化后供 Agent 使用
    """
    date: str

    # VIX
    vix_level: float = 20.0
    vix_5d_change: float = 0.0
    vix_20d_ma_ratio: float = 1.0
    vix_60d_percentile: float = 0.5

    # TNX
    tnx_level: float = 4.0
    tnx_5d_change: float = 0.0
    tnx_trend: str = "flat"  # rising/falling/flat

    # DXY
    dxy_level: float = 100.0
    dxy_20d_change: float = 0.0

    # SPY
    spy_20d_return: float = 0.0
    spy_60d_trend: str = "sideways"  # uptrend/downtrend/sideways

    # QQQ vs SPY
    qqq_spy_relative: str = "neutral"  # outperforming/underperforming/neutral

    def to_dict(self) -> dict:
        """转为字典"""
        return {
            "date": self.date,
            "vix": {
                "level": self.vix_level,
                "change_5d": f"{self.vix_5d_change:+.1%}",
                "ma20_ratio": f"{self.vix_20d_ma_ratio:.2f}",
                "percentile_60d": f"{self.vix_60d_percentile:.0%}",
            },
            "tnx": {
                "level": f"{self.tnx_level:.2f}%",
                "change_5d": f"{self.tnx_5d_change:+.1f}bps",
                "trend": self.tnx_trend,
            },
            "dxy": {
                "level": f"{self.dxy_level:.2f}",
                "change_20d": f"{self.dxy_20d_change:+.2f}%",
            },
            "spy": {
                "return_20d": f"{self.spy_20d_return:+.2f}%",
                "trend_60d": self.spy_60d_trend,
            },
            "qqq_vs_spy": self.qqq_spy_relative,
        }


@dataclass
class SentimentSnapshot:
    """
    消息面数据快照 - 格式化后供 Agent 使用
    """
    date: str

    # 新闻
    news_count_7d: int = 0
    news_count_1d: int = 0
    news_headlines: list[str] = field(default_factory=list)
    news_sentiment_score: float = 0.0  # -1 到 1

    # 社区情绪
    wsb_sentiment: str = "neutral"  # bullish/bearish/neutral
    stocktwits_sentiment: str = "neutral"
    put_call_ratio: float = 1.0

    # 事件标记
    earnings_week: bool = False
    fed_week: bool = False
    macro_event: Optional[str] = None  # cpi/nfp/fomc/None

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "news": {
                "count_7d": self.news_count_7d,
                "count_1d": self.news_count_1d,
                "sentiment": f"{self.news_sentiment_score:+.2f}",
            },
            "social": {
                "wsb": self.wsb_sentiment,
                "stocktwits": self.stocktwits_sentiment,
                "put_call_ratio": f"{self.put_call_ratio:.2f}",
            },
            "events": {
                "earnings_week": self.earnings_week,
                "fed_week": self.fed_week,
                "macro_event": self.macro_event,
            },
        }


class BaseRegimeAgent(ABC):
    """
    Regime Agent 基类
    """

    @abstractmethod
    def predict(
        self,
        macro_snapshot: Optional[MacroSnapshot] = None,
        sentiment_snapshot: Optional[SentimentSnapshot] = None,
    ) -> AgentOutput:
        """
        预测当前 Regime 状态

        Args:
            macro_snapshot: 宏观数据快照
            sentiment_snapshot: 消息面数据快照

        Returns:
            Agent 输出
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        pass
