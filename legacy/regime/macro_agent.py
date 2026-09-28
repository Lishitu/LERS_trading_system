"""
宏观面 Agent - 基于 LLM 判断市场宏观状态

职责：
- 接收宏观数据快照 (VIX, TNX, DXY, SPY)
- 使用 LLM 分析宏观状态
- 输出情绪方向、强度、置信度和调整参数

注意：
- 在历史回测中使用时，需要确保 snapshot 只包含回测时点可用的数据
- LLM 调用需要配置 API key (通过环境变量或 config)
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from .base import (
    AgentOutput,
    MacroSnapshot,
    RegimeParams,
    SentimentType,
)


# =============================================================================
# Prompt 模板
# =============================================================================

MACRO_AGENT_PROMPT = """
你是宏观经济策略分析师。基于以下宏观数据，判断当前市场状态并给出交易参数调整建议。

数据时点: {date}

📊 VIX（恐慌指数）:
- 当前水平: {vix_level:.2f}
- 5日变化: {vix_5d_change:+.2f}%
- 20日均线比: {vix_ma20_ratio:.2f} (>1表示高于均线，<1表示低于均线)
- 60日分位数: {vix_60d_percentile:.0%} (0%=历史最低, 100%=历史最高)

📈 TNX（10年期国债收益率）:
- 当前水平: {tnx_level:.3f}%
- 5日变化: {tnx_5d_change:+.3f}bps
- 20日趋势: {tnx_trend} (rising/falling/flat)

💵 DXY（美元指数）:
- 当前水平: {dxy_level:.2f}
- 20日变化: {dxy_20d_change:+.2f}%

📉 SPY 市场状态:
- 20日收益率: {spy_20d_return:+.2f}%
- 60日趋势: {spy_trend} (uptrend/downtrend/sideways)

🔍 纳指 vs 标普:
- 相对表现: {qqq_spy_relative} (outperforming/underperforming/neutral)

请分析以上数据，判断：
1. 当前宏观市场情绪方向（positive/negative/neutral）
2. 情绪强度（0.0-1.0，0=极度恐慌/中性，1=极度乐观）
3. 你对判断的置信度（0.0-1.0）
4. 给出交易参数调整建议

请严格按以下 JSON 格式输出（不要包含任何其他内容）：
{{
    "sentiment": "positive/negative/neutral",
    "intensity": 0.0到1.0的浮点数,
    "confidence": 0.0到1.0的浮点数,
    "vix_sentiment": "你对VIX的判断",
    "tnx_sentiment": "你对利率的判断",
    "dxy_sentiment": "你对美元的判断",
    "reasoning": "一句话总结你的宏观判断逻辑",
    "params": {{
        "risk_on_threshold_adjustment": -0.15到0.15的浮点数（正值=更难做多，负值=更容易做多）,
        "risk_off_threshold_adjustment": -0.15到0.15的浮点数（正值=更难做空，负值=更容易做空）,
        "probability_scale": 0.8到1.2的浮点数（<1=更保守缩小概率差异，>1=更激进放大概率差异）
    }}
}}
"""


# =============================================================================
# LLM Client 抽象
# =============================================================================

class LLMClient:
    """
    LLM 客户端抽象

    支持多种 LLM 提供商。
    当前支持: OpenAI, Anthropic, 本地模型
    """

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider.lower()
        self.model = model

    def call(self, prompt: str, temperature: float = 0.3) -> str:
        """
        调用 LLM

        Args:
            prompt: 输入 prompt
            temperature: 采样温度

        Returns:
            LLM 输出文本
        """
        if self.provider == "openai":
            return self._call_openai(prompt, temperature)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, temperature)
        elif self.provider == "mock":
            return self._mock_response(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str, temperature: float) -> str:
        """OpenAI API 调用"""
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
        """Anthropic API 调用"""
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
        """
        Mock 响应 - 用于开发和测试

        基于 prompt 中的数值生成一个合理的响应
        """
        # 提取 VIX level
        import re

        vix_match = re.search(r"VIX.*?当前水平.*?(\d+\.?\d*)", prompt)
        vix_level = float(vix_match.group(1)) if vix_match else 20.0

        # 简单规则生成响应
        if vix_level > 30:
            sentiment = "negative"
            intensity = min(1.0, (vix_level - 25) / 20 + 0.5)
        elif vix_level > 20:
            sentiment = "neutral"
            intensity = 0.5
        elif vix_level < 15:
            sentiment = "positive"
            intensity = 0.7
        else:
            sentiment = "neutral"
            intensity = 0.4

        response = {
            "sentiment": sentiment,
            "intensity": intensity,
            "confidence": 0.7,
            "vix_sentiment": f"VIX={vix_level:.1f} indicates {'high' if vix_level > 25 else 'normal' if vix_level > 18 else 'low'} volatility",
            "tnx_sentiment": "Treasury yields showing mixed signals",
            "dxy_sentiment": "Dollar strength neutral for risk assessment",
            "reasoning": f"Based on VIX={vix_level:.1f} market regime assessment",
            "params": {
                "risk_on_threshold_adjustment": -0.05 if sentiment == "positive" else 0.05,
                "risk_off_threshold_adjustment": 0.05 if sentiment == "negative" else 0.0,
                "probability_scale": 0.9 if vix_level > 25 else 1.0,
            },
        }

        return json.dumps(response, indent=2)


# =============================================================================
# 宏观 Agent
# =============================================================================


class MacroAgent:
    """
    宏观面 Agent

    使用 LLM 分析宏观数据，输出情绪判断和参数调整建议

    使用方式:
        agent = MacroAgent(provider="mock")  # 开发测试
        agent = MacroAgent(provider="openai", model="gpt-4o")  # 生产

        snapshot = MacroSnapshot(date="2024-01-15", vix_level=18.5, ...)
        output = agent.predict(snapshot)
    """

    def __init__(
        self,
        provider: str = "mock",
        model: str = None,
        temperature: float = 0.3,
    ):
        """
        初始化宏观 Agent

        Args:
            provider: LLM 提供商 ("openai", "anthropic", "mock")
            model: 模型名称
            temperature: 采样温度
        """
        self.llm = LLMClient(provider=provider, model=model)
        self.temperature = temperature

    def predict(self, snapshot: MacroSnapshot) -> AgentOutput:
        """
        分析宏观数据，输出判断

        Args:
            snapshot: 宏观数据快照

        Returns:
            Agent 输出
        """
        # 构建 prompt
        prompt = self._build_prompt(snapshot)

        # 调用 LLM
        try:
            response_text = self.llm.call(prompt, self.temperature)
            return self._parse_response(response_text, snapshot.date)
        except Exception as e:
            # LLM 调用失败时返回默认中性
            return AgentOutput(
                date=snapshot.date,
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=0.0,
                reasoning=f"LLM call failed: {str(e)}",
                source_name="macro_agent",
            )

    def _build_prompt(self, snapshot: MacroSnapshot) -> str:
        """构建 prompt"""
        return MACRO_AGENT_PROMPT.format(
            date=snapshot.date,
            vix_level=snapshot.vix_level,
            vix_5d_change=snapshot.vix_5d_change,
            vix_ma20_ratio=snapshot.vix_20d_ma_ratio,
            vix_60d_percentile=snapshot.vix_60d_percentile,
            tnx_level=snapshot.tnx_level,
            tnx_5d_change=snapshot.tnx_5d_change,
            tnx_trend=snapshot.tnx_trend,
            dxy_level=snapshot.dxy_level,
            dxy_20d_change=snapshot.dxy_20d_change,
            spy_20d_return=snapshot.spy_20d_return,
            spy_trend=snapshot.spy_60d_trend,
            qqq_spy_relative=snapshot.qqq_spy_relative,
        )

    def _parse_response(self, response_text: str, date: str) -> AgentOutput:
        """解析 LLM 响应"""
        try:
            # 提取 JSON
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
                source_name="macro_agent",
                params=params,
            )

        except (json.JSONDecodeError, KeyError) as e:
            # 解析失败
            return AgentOutput(
                date=date,
                sentiment=SentimentType.NEUTRAL,
                intensity=0.5,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                source_name="macro_agent",
            )

    def _extract_json(self, text: str) -> str:
        """
        从 LLM 输出中提取 JSON

        处理 LLM 可能输出的 markdown 格式
        """
        import re

        # 尝试直接解析
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ```
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 尝试提取 ``` ... ```
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 尝试提取 {...}
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return text.strip()


# =============================================================================
# 便捷函数
# =============================================================================


def create_macro_agent(
    provider: str = "mock",
    model: str = None,
    api_key: str = None,
) -> MacroAgent:
    """
    创建宏观 Agent 的便捷函数

    Args:
        provider: LLM 提供商
        model: 模型名称
        api_key: API Key（会设置到环境变量）

    Returns:
        MacroAgent 实例
    """
    if api_key:
        env_key = f"{provider.upper()}_API_KEY"
        os.environ[env_key] = api_key

    return MacroAgent(provider=provider, model=model)


if __name__ == "__main__":
    # 快速测试
    agent = MacroAgent(provider="mock")

    test_cases = [
        MacroSnapshot(
            date="2024-01-15",
            vix_level=18.5,
            vix_5d_change=2.1,
            vix_20d_ma_ratio=1.1,
            vix_60d_percentile=0.55,
            spy_20d_return=0.03,
            spy_60d_trend="uptrend",
            qqq_spy_relative="outperforming",
        ),
        MacroSnapshot(
            date="2020-03-15",
            vix_level=82.7,
            vix_5d_change=25.0,
            vix_20d_ma_ratio=2.5,
            vix_60d_percentile=0.98,
            spy_20d_return=-0.12,
            spy_60d_trend="downtrend",
            qqq_spy_relative="underperforming",
        ),
    ]

    for snapshot in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {snapshot.date}")
        print(f"  VIX: {snapshot.vix_level:.1f}, SPY 20d: {snapshot.spy_20d_return:+.2%}")

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
