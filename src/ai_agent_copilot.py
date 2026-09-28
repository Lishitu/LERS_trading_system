"""AI 智能投顾与多代理投研决策助手 (AI Agent Copilot)。

参考 TradingAgents 多代理架构并适配 Moomoo 美股策略标的池与行情数据：
1. 市场技术派分析师 (Market Technical Analyst): 基于 Moomoo 行情抽取均线、波动率、动量与相对强度
2. 宏观地缘与政策分析师 (Macro & Fed Policy Analyst): 研判中东/伊朗战争、高油价通胀与美联储加息紧缩预期
3. 多空博弈辩论席 (Bull vs Bear Researcher Debate): 对冲多空论据
4. 首席风控与投资决策官 (CIO & Risk Manager): 输出是否适合策略建仓/加仓/减仓的具体量化建议

底层接入 DeepSeek 大模型 (读取 config/apikey.txt)。
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import requests

from data_moomoo import (
    MoomooDataLoader,
    DEFAULT_BENCHMARKS,
    DEFAULT_SECTORS,
    LEVERAGED_MAP,
    DEFAULT_DEFENSE,
)

logger = logging.getLogger("LERS.AIAgentCopilot")


class AIAgentCopilot:
    """基于 TradingAgents 架构理念的 DeepSeek 策略投研与建仓决策副驾驶。"""

    def __init__(
        self,
        api_key_path: str = "config/apikey.txt",
        cache_dir: str = "data/ai_analysis",
    ):
        self.api_key_path = api_key_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.history_dir = os.path.join(self.cache_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.api_key = self._load_api_key()
        self.loader = MoomooDataLoader()

    def _load_api_key(self) -> str:
        """从 config/apikey.txt 或 config.json 读取 DeepSeek API Key。"""
        if os.path.exists(self.api_key_path):
            try:
                with open(self.api_key_path, "r", encoding="utf-8") as f:
                    k = f.read().strip()
                    if k:
                        return k
            except Exception as e:
                logger.warning(f"读取 {self.api_key_path} 异常: {e}")

        # 兜底从 config.json 读取
        cfg_path = "config/config.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("deepseek", {}).get("api_key", "")
            except Exception:
                pass
        return ""

    def collect_moomoo_market_context(self) -> Dict[str, Any]:
        """从 Moomoo 本地缓存与标的池提取最新关键指标事实。"""
        # 加载核心标的数据
        data_dict = self.loader.load_all_universe(start_date="2025-01-01")
        close_dict = {t: df['close'] for t, df in data_dict.items() if 'close' in df.columns}
        df_close = pd.DataFrame(close_dict).dropna(subset=['US.QQQ'])

        last_date = str(df_close.index[-1].strftime("%Y-%m-%d")) if hasattr(df_close.index[-1], 'strftime') else str(df_close.index[-1])

        def _calc_stats(ticker: str):
            if ticker not in df_close.columns:
                return {}
            s = df_close[ticker].dropna()
            if len(s) < 50:
                return {}
            curr = float(s.iloc[-1])
            ma50 = float(s.rolling(50).mean().iloc[-1])
            ma200 = float(s.rolling(min(200, len(s))).mean().iloc[-1])
            ret_5d = float((curr / s.iloc[-5] - 1.0) * 100.0) if len(s) >= 5 else 0.0
            ret_20d = float((curr / s.iloc[-20] - 1.0) * 100.0) if len(s) >= 20 else 0.0
            ret_60d = float((curr / s.iloc[-60] - 1.0) * 100.0) if len(s) >= 60 else 0.0
            daily_ret = s.pct_change(1).dropna()
            vol20 = float(daily_ret.tail(20).std() * np.sqrt(252) * 100.0)
            return {
                "latest_price": round(curr, 2),
                "ret_5d_pct": round(ret_5d, 2),
                "ret_20d_pct": round(ret_20d, 2),
                "ret_60d_pct": round(ret_60d, 2),
                "ma50": round(ma50, 2),
                "ma200": round(ma200, 2),
                "is_above_ma50": curr > ma50,
                "is_above_ma200": curr > ma200,
                "realized_vol20": round(vol20, 1),
            }

        # 核心指数
        qqq_stats = _calc_stats('US.QQQ')
        tqqq_stats = _calc_stats('US.TQQQ')
        spy_stats = _calc_stats('US.SPY')

        # 重点行业与大宗商品/避险
        smh_stats = _calc_stats('US.SMH')   # 半导体
        xlk_stats = _calc_stats('US.XLK')   # 科技
        xle_stats = _calc_stats('US.XLE')   # 能源 (抗通胀/地缘油价晴雨表)
        xop_stats = _calc_stats('US.XOP')   # 油气开采
        gld_stats = _calc_stats('US.GLD')   # 黄金 (避险)
        copx_stats = _calc_stats('US.COPX') # 铜矿
        bil_stats = _calc_stats('US.BIL')   # 超短债

        return {
            "eval_date": last_date,
            "qqq": qqq_stats,
            "tqqq": tqqq_stats,
            "spy": spy_stats,
            "smh": smh_stats,
            "xlk": xlk_stats,
            "xle": xle_stats,
            "xop": xop_stats,
            "gld": gld_stats,
            "copx": copx_stats,
            "bil": bil_stats,
        }

    def get_decision_history(self, limit: int = 30) -> list:
        """获取所有已归档的 AI 决策历史快照 (按时间倒序)。"""
        if not os.path.exists(self.history_dir):
            return []
        files = sorted(
            [f for f in os.listdir(self.history_dir) if f.startswith("decision_") and f.endswith(".json")],
            reverse=True
        )[:limit]
        res = []
        for fn in files:
            path = os.path.join(self.history_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res.append({
                        "id": data.get("id", fn.replace(".json", "")),
                        "filename": fn,
                        "generated_at": data.get("generated_at", ""),
                        "verdict_badge": data.get("verdict_badge", ""),
                        "action_type": data.get("action_type", ""),
                        "recommended_build_pct": data.get("recommended_build_pct", 0),
                        "confidence_score": data.get("confidence_score", 0),
                    })
            except Exception:
                pass
        return res

    def get_decision_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """获取指定的历史 AI 决策快照。"""
        if decision_id == "latest":
            latest_path = os.path.join(self.cache_dir, "latest_decision.json")
            if os.path.exists(latest_path):
                with open(latest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        fn = decision_id if decision_id.endswith(".json") else f"{decision_id}.json"
        path = os.path.join(self.history_dir, fn)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_previous_decision(self) -> Optional[Dict[str, Any]]:
        """获取最近一轮历史研判决策，供本轮决策对比以维持研判稳定性。"""
        if os.path.exists(self.history_dir):
            files = sorted(
                [f for f in os.listdir(self.history_dir) if f.startswith("decision_") and f.endswith(".json")],
                reverse=True
            )
            if files:
                try:
                    with open(os.path.join(self.history_dir, files[0]), "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        latest_path = os.path.join(self.cache_dir, "latest_decision.json")
        if os.path.exists(latest_path):
            try:
                with open(latest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def run_investment_debate_and_decision(
        self,
        strategy_nav: float = 10000.0,
        regime: str = "BULL_GROWTH",
        strategy_orders: Optional[list] = None,
        target_weights: Optional[Dict[str, float]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """运行 TradingAgents 多代理分析辩论并调用 DeepSeek 产出建仓/调仓决策与订单审核。"""
        cache_file = os.path.join(self.cache_dir, "latest_decision.json")

        # 1. 检查缓存 (1小时内有效，除非强制刷新)
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    cache_age = time.time() - cached.get("generated_at_ts", 0)
                    if cache_age < 3600:
                        return cached
            except Exception:
                pass

        if not self.api_key:
            return {
                "status": "error",
                "message": "未在 config/apikey.txt 中找到有效的 DeepSeek API Key",
            }

        # 2. 读取上一轮历史研判 (用于保证判断稳定性和连续性)
        prev_decision = self.get_previous_decision()
        if prev_decision:
            prev_info_str = (
                f"- 上轮投决会研判时间: {prev_decision.get('generated_at', '未知')}\n"
                f"- 上轮决策标签: {prev_decision.get('verdict_badge', '未知')}\n"
                f"- 上轮建议建仓比例: {prev_decision.get('recommended_build_pct', 0)}%\n"
                f"- 上轮CIO关键指导: {prev_decision.get('cio_conclusion', '无')}"
            )
        else:
            prev_info_str = "无上轮历史研判记录（本次为首次启动投决会研判）。"

        # 3. 整理当前量化策略生成的调仓交易计划
        orders_list = strategy_orders or []
        if orders_list:
            order_lines = []
            order_tickers = []
            for idx, o in enumerate(orders_list, 1):
                act_cn = "买入" if o.get("action") == "BUY" else "卖出"
                w_pct = round(o.get("target_weight", 0) * 100, 1)
                sh_int = int(round(float(o.get("shares", 0))))
                o["shares"] = sh_int
                order_tickers.append(f"{o.get('ticker')}({act_cn}{sh_int}股)")
                order_lines.append(
                    f"{idx}. [{act_cn}] {o.get('ticker')}  {sh_int}股  "
                    f"参考价:${o.get('price')}  预估金额:${o.get('est_amount')}  "
                    f"目标权重:{w_pct}%  触发原因:{o.get('reason', '-')}"
                )
            orders_prompt_str = "\n".join(order_lines)
            tickers_summary_str = "、".join(order_tickers)
            review_requirement_str = (
                f"【极其重要的审核约束 - 必须精准对应真实订单，且严禁碎股/小数股】:\n"
                f"上方量化策略当前确切生成的调仓交易清单为：【{tickers_summary_str}】。\n"
                f"你在 order_reviews 列表中必须严格按顺序、一对一地对上述每笔真实订单做出风控审查与裁决！\n"
                f"每一条 order_reviews 的 ticker、strategy_action、strategy_shares 必须与上述计划完全一致！\n"
                f"【实盘交易硬性规则：严禁出现任何小数股/碎股】\n"
                f"- 美股限价实盘不支持碎股交易。裁决给出的建议调整后股数 adjusted_shares 必须严格为非负整数股（例如 0, 1, 2, 5...），绝对严禁出现 0.5 股、1.5 股等任何小数碎股！\n"
                f"- 若建议缩减或减半：对于原计划买入 1 股的标的，可裁决为 1 股（试探性买入整股）或 0 股（暂缓买入）；对于原计划买入 3 股的标的，可裁决为 1 股 或 2 股；对于 22 股的标的，可缩减为 10 股或 0 股！全流程必须为纯整数股！\n"
                f"裁决重点：\n"
                f"1. 针对 3x 杠杆标的（如 TQQQ）：结合当前中东战争、原油逼近百元及加息风险，评估是否允许买入原计划的全部股数；若风险过高，可裁定 TRIMMED（如缩减为整数股）或 HEDGED/REJECTED（建议调整为 0 股暂缓买入，建议核心降级为 1x QQQ 或增加避险短债/黄金对冲）；\n"
                f"2. 针对高动量行业标的（如 SMH、XLK）：评估动量领涨与短期追高回撤风险，裁定是 APPROVED 还是 TRIMMED；\n"
                f"3. 严禁把策略没有生成的买单虚构为'原计划买入0股'，必须严格审查策略真正产生的订单！"
            )
        else:
            orders_prompt_str = "当前各标的持仓权重在缓冲区内（6%偏离度以内），量化算法建议维持现有持仓，暂无调仓订单产生。"
            review_requirement_str = "当前量化策略在缓冲区内无调仓订单产生。order_reviews 列表可为空 []，投决会维持现有底仓观察。"

        # 4. 从 Moomoo 标的池抽取最新行情事实
        market_facts = self.collect_moomoo_market_context()

        # 5. 构造 TradingAgents 风格的多代理辩论 Prompt
        system_prompt = (
            "你是一个华尔街顶级对冲基金的多智能体联合投决会（TradingAgents Multi-Agent Committee）。\n"
            "团队包含四大专家角色：\n"
            "1. 宏观与地缘政治分析师 (Macro & Fed Policy Analyst)：重点关注中东/伊朗战争局势、霍尔木兹海峡与国际原油价格（原油破百）、二次通胀冲击以及美联储加息紧缩预期；\n"
            "2. 量化技术派分析师 (Quantitative Technical Analyst)：基于 Moomoo 美股 ETF 行情（MA50/MA200、20日已实现波动率、各板块动量强弱）做量化客观诊断；\n"
            "3. 多空辩论博弈席 (Bull vs Bear Debate)：由激进多头研究员与严谨空头研究员针对当前市场展开高水平辩论；\n"
            "4. 首席投资官与风控总监 (CIO & Risk Manager)：基于辩论与事实，对量化策略给出的具体调仓计划做二次审核，输出具体到个股/ETF订单的执行批准、缩减或否决建议。\n\n"
            "请注意：必须严格输出符合规范的 JSON 格式（不得带有任何 markdown 包装外的额外散文），便于前端解析。"
        )

        user_prompt = f"""【当前评估时间】: {time.strftime('%Y-%m-%d %H:%M:%S')}

【输入数据：Moomoo 策略标的池最新量化事实】
- 评估基准日期: {market_facts.get('eval_date')}
- 策略当前系统宏观状态: {regime}
- 用户拟分配实盘验证资金: ${strategy_nav:,.2f}
- 核心指数事实 (QQQ):
  * 最新价格: ${market_facts.get('qqq', {}).get('latest_price')}
  * 近5日涨跌: {market_facts.get('qqq', {}).get('ret_5d_pct')}% | 近20日涨跌: {market_facts.get('qqq', {}).get('ret_20d_pct')}% | 近60日涨跌: {market_facts.get('qqq', {}).get('ret_60d_pct')}%
  * 均线位置: 50日线 ${market_facts.get('qqq', {}).get('ma50')} (站上: {market_facts.get('qqq', {}).get('is_above_ma50')}), 200日线 ${market_facts.get('qqq', {}).get('ma200')} (站上: {market_facts.get('qqq', {}).get('is_above_ma200')})
  * 20日年化波动率: {market_facts.get('qqq', {}).get('realized_vol20')}%
- 杠杆标的事实:
  * TQQQ (3x 纳指): ${market_facts.get('tqqq', {}).get('latest_price')}, 近5日: {market_facts.get('tqqq', {}).get('ret_5d_pct')}%
- 科技成长卫星事实:
  * 半导体 (SMH): ${market_facts.get('smh', {}).get('latest_price')}, 近20日: {market_facts.get('smh', {}).get('ret_20d_pct')}%, 站上MA50: {market_facts.get('smh', {}).get('is_above_ma50')}
  * 科技龙头 (XLK): ${market_facts.get('xlk', {}).get('latest_price')}, 近20日: {market_facts.get('xlk', {}).get('ret_20d_pct')}%
- 地缘危机、能源抗通胀与避险事实:
  * 能源行业 (XLE): ${market_facts.get('xle', {}).get('latest_price')}, 近20日: {market_facts.get('xle', {}).get('ret_20d_pct')}%, 站上MA50: {market_facts.get('xle', {}).get('is_above_ma50')}
  * 油气开采 (XOP): ${market_facts.get('xop', {}).get('latest_price')}, 近20日: {market_facts.get('xop', {}).get('ret_20d_pct')}%
  * 避险黄金 (GLD): ${market_facts.get('gld', {}).get('latest_price')}, 近20日: {market_facts.get('gld', {}).get('ret_20d_pct')}%

【上一轮投决会历史研判基准 (请保持判断连贯性与稳定性)】
{prev_info_str}

【当前量化策略算法给出的待执行调仓交易计划】
{orders_prompt_str}

【量化订单风控裁决指令】
{review_requirement_str}

【当前宏观核心焦点】
中东局势紧张（伊朗危机），国际原油高位逼近或突破100美元，二次通胀风险升温，美联储加息紧缩或推迟降息预期抬头。投资人拟用 ${strategy_nav:,.2f} 资金进行策略实盘建仓验证。

【特别指引要求】：
1. 【判断稳定性要求】：金融投资决策需具备策略稳定性与延续性，切忌无重大利空利多时日内反复摇摆。对比上轮历史研判：若宏观地缘与关键均线未发生结构性破位或逆转，请优先延续和微调上轮立场；若确实发生重大改变，请在 stability_comment 中明确说明转变原因。
2. 【量化订单二次审核与裁决】：请投决会 CIO 与风控专家对上述量化调仓计划逐笔审查，在 order_reviews 中明确给出：
   - 裁决状态 review_status (必须为 APPROVED / TRIMMED / REJECTED / HEDGED 之一)
   - 投决会建议的调整后执行股数 adjusted_shares
   - 针对该笔订单的具体裁定理由 audit_reason
3. 在 order_review_summary 中给出本次调仓执行的综合总评与执行先后顺序建议。

请投决会完成分析辩论，并以严格的 JSON 格式输出如下字段：
{{
  "verdict_badge": "简明醒目的决策标签（如：建议轻仓分批试探 / 建议暂缓观望 / 建议防守型建仓）",
  "action_type": "BUILD_LIGHT 或 BUILD_AGGRESSIVE 或 HOLD_OBSERVE 或 TRIM_DEFENSE 或 WAIT_CASH",
  "confidence_score": 整数70~95之间的信心分,
  "is_suitable_for_build": true或false,
  "recommended_build_pct": 建议当前首次建仓占预算的百分比（如 30% 或 50% 或 0%）,
  "stability_comment": "针对前期研判的连贯性与稳定性说明（解释本次判断与上一期是延续还是微调，宏观驱动逻辑是什么）",
  "macro_geopolitical_summary": "宏观地缘与联储利率视角的专业简评（聚焦伊朗冲突、油价破百二次通胀、加息压力）",
  "bull_thesis": "多头研究员的核心辩护理由（科技基本面、均线支撑等）",
  "bear_thesis": "空头研究员的核心风险警告（地缘失控、原油通胀、加息估值杀跌等）",
  "core_allocation_guidance": "核心底仓建议：当前适合用 3x TQQQ 还是降级为 1x QQQ 或避险短债 BIL？",
  "satellite_allocation_guidance": "卫星仓位建议：当前应重配科技半导体 (SMH) 还是增配能源 (XLE) 或避险黄金 (GLD) 对冲？",
  "cio_conclusion": "首席投资官实操落地指引（含分批买入节奏、止损破位硬指标）",
  "order_reviews": [
    {{
      "ticker": "标的代码 (如 US.QQQ)",
      "strategy_action": "BUY 或 SELL",
      "strategy_shares": 策略原计划股数,
      "review_status": "APPROVED (批准执行) 或 TRIMMED (建议缩减) 或 REJECTED (暂缓/否决) 或 HEDGED (替换对冲)",
      "adjusted_shares": 调整后建议买卖股数,
      "audit_reason": "投决会对该标的订单的具体审查裁定理由"
    }}
  ],
  "order_review_summary": "投决会对本次量化调仓计划的整体综合审查评语与执行次序建议"
}}
"""

        # 6. 调用 DeepSeek API
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 2000,
                },
                timeout=45,
            )

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"DeepSeek API 响应异常 HTTP {resp.status_code}: {resp.text}",
                }

            result_json = resp.json()
            content = result_json["choices"][0]["message"]["content"]
            parsed_decision = json.loads(content)

            # 严格碎股清洗：强制将所有 strategy_shares 与 adjusted_shares 规整为纯整数 (无碎股)
            if "order_reviews" in parsed_decision and isinstance(parsed_decision["order_reviews"], list):
                for r in parsed_decision["order_reviews"]:
                    if "strategy_shares" in r and r["strategy_shares"] is not None:
                        try:
                            r["strategy_shares"] = int(round(float(r["strategy_shares"])))
                        except Exception:
                            r["strategy_shares"] = 0
                    if "adjusted_shares" in r and r["adjusted_shares"] is not None:
                        try:
                            val = float(r["adjusted_shares"])
                            # 若模型仍然给出了 0.5 股，向上取整为 1 股试探或根据理由定为整股
                            r["adjusted_shares"] = int(round(val))
                        except Exception:
                            r["adjusted_shares"] = 0

            # 补充元数据
            now_ts = time.strftime("%Y%m%d_%H%M%S")
            parsed_decision["id"] = f"decision_{now_ts}"
            parsed_decision["status"] = "ok"
            parsed_decision["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            parsed_decision["generated_at_ts"] = time.time()
            parsed_decision["market_facts"] = market_facts
            parsed_decision["strategy_nav"] = strategy_nav
            parsed_decision["regime"] = regime
            parsed_decision["strategy_orders"] = orders_list

            # 7. 双重归档：同时归档至 history 历史快照与 latest_decision.json
            history_file = os.path.join(self.history_dir, f"decision_{now_ts}.json")
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(parsed_decision, f, ensure_ascii=False, indent=2)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(parsed_decision, f, ensure_ascii=False, indent=2)

            logger.info(f"[AIAgentCopilot] DeepSeek 决策已成功生成并归档 (ID: {parsed_decision['id']}, 包含 {len(parsed_decision.get('order_reviews', []))} 笔订单审核)")
            return parsed_decision

        except Exception as e:
            logger.error(f"[AIAgentCopilot] 运行多代理辩论异常: {e}")
            return {
                "status": "error",
                "message": f"AI 分析服务异常: {str(e)}",
            }

