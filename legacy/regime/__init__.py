"""
Regime Agent 模块 - 宏观/消息面 Regime 判断与参数生成

架构：
┌─────────────────────────────────────────────────────────────────┐
│   Macro Agent (宏观面)     │  Sentiment Agent (消息面)         │
│   - VIX, TNX, DXY, SPY   │  - 新闻, Reddit, StockTwits        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    RegimeAggregator (参数聚合)
                              ↓
                    RegimeParams (参数)
                              ↓
                    LGB/RF Model + Params
                              ↓
                         交易信号
"""

from .base import RegimeParams, AgentOutput, SentimentType, MacroSnapshot, SentimentSnapshot
from .rule_based_regime import RuleBasedRegimeAgent
from .macro_agent import MacroAgent
from .sentiment_agent import SentimentAgent
from .aggregator import RegimeAggregator
from .regime_aware_model import RegimeAwareModelWrapper

__all__ = [
    "RegimeParams",
    "AgentOutput",
    "SentimentType",
    "MacroSnapshot",
    "SentimentSnapshot",
    "RuleBasedRegimeAgent",
    "MacroAgent",
    "SentimentAgent",
    "RegimeAggregator",
    "RegimeAwareModelWrapper",
]
