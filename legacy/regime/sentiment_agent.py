"""
消息面 Agent - 基于 LLM 分析新闻和社交媒体情绪

职责：
- 接收消息面数据快照（新闻、社区情绪、技术面情绪指标）
- 使用 LLM 分析短期市场情绪
- 输出情绪方向、强度和参数调整建议

注意：
- 历史回测中，新闻数据可能不可得，此时可使用情绪代理指标
- PUT/CALL Ratio 等技术面指标可作为消息面的代理
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from .base import (
    AgentOutput,
    MacroSnapshot,
    RegimeParams,
    SentimentSnapshot,
    SentimentType,
)


# =============================================================================
# Prompt 模板
# =============================================================================

SENTIMENT_AGENT_PROMPT = """
你是市场情绪分析师。基于以下消息面信息，判断短期市场情绪并给出交易参数调整建议。

数据时点: {date}

📰 新闻情绪:
- 7天新闻数量: {news_count_7d}
- 1天新闻数量: {news_count_1d}
- 情感打分: {news_sentiment:.2f} (-1=极度看空, 0=中性, +1=极度看多)
{news_headlines}

💬 社区情绪:
- Reddit r/wallstreetbets: {wsb_sentiment} (bullish/bearish/neutral)
- StockTwits: {stocktwits_sentiment} (bullish/bearish/neutral)
- PUT/CALL Ratio: {put_call_ratio:.2f} (>1.2=看跌情绪浓厚, <0.8=看涨情绪浓厚)

📅 重要事件:
- 财报周: {earnings_week} (是/否)
- 美联储会议: {fed_week} (是/否)
- 宏观事件: {macro_event} (cpi/nfp/fomc/无)

请分析以上信息，判断：
1. 当前消息面短期情绪方向（positive/negative/neutral）
2. 情绪强度（0.0-1.0）
3. 你对判断的置信度（0.0-1.0）
4. 给出交易参数调整建议

请严格按以下 JSON 格式输出（不要包含任何其他内容）：
{{
    "sentiment": "positive/negative/neutral",
    "intensity": 0.0到1.0的浮点数,
    "confidence": 0.0到1.0的浮点数,
    "news_sentiment": "你对新闻情绪的判断",
    "social_sentiment": "你对社区情绪的判断",
    "reasoning": "一句话总结你的消息面判断",
    "params": {{
        "risk_on_threshold_adjustment": -0.15到0.15的浮点数,
        "risk_off_threshold_adjustment": -0.15到0.15的浮点数,
        "probability_scale": 0.8到1.2的浮点数
    }}
}}
"""


# =============================================================================
# LLM Client（复用 macro_agent 中的实现）
# =============================================================================

class LLMClient:
    """
    LLM 客户端抽象 - 与 macro_agent.py 中的实现保持一致
    """

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider.lower()
        self.model = model

    def call(self, prompt: str, temperature: float = 0.3) -> str:
        if self.provider == "openai":
            return self._call_openai(prompt, temperature)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, temperature)
        elif self.provider == "mock":
            return self._mock_response(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str, temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        client = OpenAI(api_key=api_key)
        model_name = self.model or "gpt-4o"

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("请设置 ANTHROPIC_API_KEY 环境变量")

        client = Anthropic(api_key=api_key)
        model_name = self.model or "claude-3-5-sonnet-20241022"

        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _mock_response(self, prompt: str) -> str:
        """Mock 响应 - 用于开发和测试"""
        import re

        # 提取 PUT/CALL Ratio
        pcr_match = re.search(r"PUT/CALL Ratio: ([\d.]+)", prompt)
        put_call_ratio = float(pcr_match.group(1)) if pcr_match else 1.0

        # 提取新闻数量
        news_match = re.search(r"7天新闻数量: (\d+)", prompt)
        news_count = int(news_match.group(1)) if news_match else 0

        # 提取情绪关键词
        wsb_bullish = "bullish" in prompt.lower()
        wsb_bearish = "bearish" in prompt.lower()
        fed_week = "是" in prompt and "美联储" in prompt
        earnings = "是" in prompt and "财报" in prompt

        # 简单规则
        if put_call_ratio > 1.3:
            sentiment = "negative"
            intensity = min(1.0, (put_call_ratio - 1.0) / 0.5 + 0.4)
        elif put_call_ratio < 0.7:
            sentiment = "positive"
            intensity = min(1.0, (1.0 - put_call_ratio) / 0.3 + 0.4)
        elif wsb_bearish and put_call_ratio > 1.0:
            sentiment = "negative"
            intensity = 0.6
        elif wsb_bullish and put_call_ratio < 1.0:
            sentiment = "positive"
            intensity = 0.6
        else:
            sentiment = "neutral"
            intensity = 0.4

        # 事件影响
        prob_adj = 0.0
        if fed_week:
            prob_adj = 0.05  # 央行会议更保守
        if earnings:
            prob_adj = max(prob_adj, 0.05)

        response = {
            "sentiment": sentiment,
            "intensity": min(1.0, intensity),
            "confidence": 0.65,
            "news_sentiment": "News volume and sentiment assessment",
            "social_sentiment": "Community sentiment reflects retail positioning",
            "reasoning": f"Based on PUT/CALL={put_call_ratio:.2f} and social sentiment",
            "params": {
                "risk_on_threshold_adjustment": -0.05 if sentiment == "positive" else 0.05,
                "risk_off_threshold_adjustment": 0.05 if sentiment == "negative" else 0.0,
                "probability_scale": 1.0 - prob_adj,
            },
        }

        return json.dumps(response, indent=2)


# =============================================================================
# 消息面 Agent
# =============================================================================


class SentimentAgent:
    """
    消息面 Agent

    使用 LLM 分析新闻和社交媒体情绪，输出短期情绪判断

    使用方式:
        agent = SentimentAgent(provider="mock")
        snapshot = SentimentSnapshot(
            date="2024-01-15",
            news_count_7d=42,
            news_sentiment_score=-0.2,
            ...
        )
        output = agent.predict(snapshot)
    """

    def __init__(
        self,
        provider: str = "mock",
        model: str = None,
        temperature: float = 0.3,
    ):
        self.llm = LLMClient(provider=provider, model=model)
        self.temperature = temperature

    def predict(self, snapshot: SentimentSnapshot) -> AgentOutput:
        """
        分析消息面数据，输出判断

        Args:
            snapshot: 消息面数据快照

        Returns:
            Agent 输出
        """
        prompt = self._build_prompt(snapshot)

        try:
            response_text = self.llm.call(prompt, self.temperature)
            return self._parse_response(response_text, snapshot.date)
        except Exception as e:
            return AgentOutput(
                date=snapshot.date,
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=0.0,
                reasoning=f"LLM call failed: {str(e)}",
                source_name="sentiment_agent",
            )

    def _build_prompt(self, snapshot: SentimentSnapshot) -> str:
        """构建 prompt"""
        # 格式化新闻标题
        headlines_str = ""
        if snapshot.news_headlines:
            headlines_str = "- 新闻标题:\n"
            for i, headline in enumerate(snapshot.news_headlines[:5], 1):
                headlines_str += f"  {i}. {headline}\n"

        return SENTIMENT_AGENT_PROMPT.format(
            date=snapshot.date,
            news_count_7d=snapshot.news_count_7d,
            news_count_1d=snapshot.news_count_1d,
            news_sentiment=snapshot.news_sentiment_score,
            news_headlines=headlines_str or "- 无新闻数据",
            wsb_sentiment=snapshot.wsb_sentiment,
            stocktwits_sentiment=snapshot.stocktwits_sentiment,
            put_call_ratio=snapshot.put_call_ratio,
            earnings_week="是" if snapshot.earnings_week else "否",
            fed_week="是" if snapshot.fed_week else "否",
            macro_event=snapshot.macro_event or "无",
        )

    def _parse_response(self, response_text: str, date: str) -> AgentOutput:
        """解析 LLM 响应"""
        try:
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)

            sentiment = SentimentType(data["sentiment"].lower())
            params = RegimeParams(
                risk_on_threshold_adjustment=data["params"]["risk_on_threshold_adjustment"],
                risk_off_threshold_adjustment=data["params"]["risk_off_threshold_adjustment"],
                probability_scale=data["params"]["probability_scale"],
                overall_sentiment=sentiment,
                confidence=data["confidence"],
            )

            return AgentOutput(
                date=date,
                sentiment=sentiment,
                intensity=float(data["intensity"]),
                confidence=float(data["confidence"]),
                reasoning=data["reasoning"],
                source_name="sentiment_agent",
                params=params,
            )

        except (json.JSONDecodeError, KeyError) as e:
            return AgentOutput(
                date=date,
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                source_name="sentiment_agent",
            )

    def _extract_json(self, text: str) -> str:
        """从 LLM 输出中提取 JSON"""
        import re

        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return text.strip()


# =============================================================================
# 便捷函数
# =============================================================================


def create_sentiment_agent(
    provider: str = "mock",
    model: str = None,
    api_key: str = None,
) -> SentimentAgent:
    """
    创建消息面 Agent 的便捷函数
    """
    if api_key:
        env_key = f"{provider.upper()}_API_KEY"
        os.environ[env_key] = api_key

    return SentimentAgent(provider=provider, model=model)


if __name__ == "__main__":
    # 快速测试
    agent = SentimentAgent(provider="mock")

    test_cases = [
        SentimentSnapshot(
            date="2024-01-15",
            news_count_7d=42,
            news_count_1d=8,
            news_sentiment_score=0.1,
            news_headlines=[
                "科技股财报超预期",
                "美联储暗示暂停加息",
            ],
            wsb_sentiment="bullish",
            stocktwits_sentiment="bullish",
            put_call_ratio=0.75,
            earnings_week=False,
            fed_week=True,
        ),
        SentimentSnapshot(
            date="2020-03-15",
            news_count_7d=150,
            news_count_1d=45,
            news_sentiment_score=-0.8,
            news_headlines=[
                "全球市场暴跌",
                "熔断机制触发",
            ],
            wsb_sentiment="bearish",
            stocktwits_sentiment="bearish",
            put_call_ratio=1.5,
            earnings_week=False,
            fed_week=False,
        ),
    ]

    for snapshot in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {snapshot.date}")
        print(f"  PUT/CALL: {snapshot.put_call_ratio:.2f}, WSB: {snapshot.wsb_sentiment}")

        output = agent.predict(snapshot)
        print(f"\n  Sentiment: {output.sentiment.value}")
        print(f"  Intensity: {output.intensity:.2f}")
        print(f"  Confidence: {output.confidence:.2f}")
        print(f"  Reasoning: {output.reasoning}")

        if output.params:
            print(f"\n  Params:")
            print(f"    risk_on_adj: {output.params.risk_on_threshold_adjustment:+.3f}")
            print(f"    risk_off_adj: {output.params.risk_off_threshold_adjustment:+.3f}")
            print(f"    prob_scale: {output.params.probability_scale:.3f}")
