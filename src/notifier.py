"""交易信号与调仓执行消息通知模块。

支持将策略生成的调仓信号、交易清单及模拟盘/实盘报单执行结果，
通过飞书自建应用 (OpenAPI)、飞书 Webhook、钉钉及企业微信实时推送到移动端。
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("LERS.Notifier")


def load_config() -> Dict[str, Any]:
    """读取 config/config.json 配置文件。"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取 {config_path} 异常: {e}")
    return {}


@dataclass
class TradeOrder:
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    est_amount: float
    reason: str
    target_weight: float = 0.0


class FeishuAppClient:
    """飞书开放平台企业自建应用 (OpenAPI) 客户端。"""

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        cfg = load_config().get("feishu", {})
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "") or cfg.get("app_id", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "") or cfg.get("app_secret", "")
        self.chat_id = chat_id or os.getenv("FEISHU_CHAT_ID", "") or cfg.get("chat_id", "")
        self.user_id = user_id or os.getenv("FEISHU_USER_ID", "") or cfg.get("user_id", "")

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    def get_tenant_access_token(self) -> Optional[str]:
        """获取或刷新租户访问凭证 (tenant_access_token)。"""
        if not self.is_configured:
            return None

        # 提前 5 分钟主动换新
        if self._token and time.time() < (self._token_expires_at - 300):
            return self._token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == 0:
                    self._token = data.get("tenant_access_token")
                    self._token_expires_at = time.time() + float(data.get("expire", 7200))
                    logger.info("已成功获取飞书 OpenAPI tenant_access_token")
                    return self._token
                else:
                    logger.error(f"获取飞书 token 失败: {data}")
                    return None
        except Exception as e:
            logger.error(f"请求飞书 token 异常: {e}")
            return None

    def get_bot_info(self) -> Dict[str, Any]:
        """查询应用机器人在飞书中的信息与激活状态。"""
        token = self.get_tenant_access_token()
        if not token:
            return {"success": False, "error": "无法获取 tenant_access_token"}

        url = "https://open.feishu.cn/open-apis/bot/v3/info"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"success": data.get("code") == 0, "data": data.get("bot", {}), "raw": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_joined_chats(self) -> List[Dict[str, Any]]:
        """获取该机器人已加入的所有群聊列表。"""
        token = self.get_tenant_access_token()
        if not token:
            return []

        url = "https://open.feishu.cn/open-apis/im/v1/chats"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == 0:
                    return data.get("data", {}).get("items", [])
                return []
        except Exception as e:
            logger.error(f"查询飞书群聊失败: {e}")
            return []

    def send_interactive_card(
        self,
        card: Dict[str, Any],
        receive_id: Optional[str] = None,
        receive_id_type: Optional[str] = None,
    ) -> bool:
        """通过 OpenAPI 发送富文本卡片消息。"""
        token = self.get_tenant_access_token()
        if not token:
            return False

        target_id = receive_id or self.chat_id or self.user_id
        target_type = receive_id_type

        if not target_id:
            # 自动查询已加入的群
            chats = self.list_joined_chats()
            if chats:
                target_id = chats[0].get("chat_id")
                target_type = "chat_id"
                logger.info(f"自动选用机器人已加入的飞书群聊: {chats[0].get('name')} ({target_id})")

        if not target_id:
            logger.warning(
                "未配置 receive_id (chat_id/user_id) 且机器人尚未被拉入任何群聊。"
                "请将机器人添加到飞书群聊，或在 config.json 中配置 chat_id / user_id。"
            )
            return False

        if not target_type:
            target_type = "chat_id" if target_id.startswith("oc_") else "open_id"

        url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={target_type}"
        payload = {
            "receive_id": target_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {token}",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                if res.get("code") == 0:
                    logger.info(f"飞书 OpenAPI 交互卡片发送成功! msg_id={res.get('data', {}).get('message_id')}")
                    return True
                else:
                    logger.error(f"飞书 OpenAPI 卡片发送失败: {res}")
                    return False
        except Exception as e:
            logger.error(f"飞书 OpenAPI 发送异常: {e}")
            return False


class NotificationService:
    """多渠道消息通知中心。"""

    def __init__(
        self,
        feishu_webhook: Optional[str] = None,
        feishu_app: Optional[FeishuAppClient] = None,
        dingtalk_webhook: Optional[str] = None,
        wecom_webhook: Optional[str] = None,
    ):
        cfg = load_config()
        feishu_cfg = cfg.get("feishu", {})

        self.feishu_webhook = feishu_webhook or os.getenv("FEISHU_WEBHOOK", "") or feishu_cfg.get("webhook_url", "")
        self.feishu_app = feishu_app or FeishuAppClient()
        self.dingtalk_webhook = dingtalk_webhook or os.getenv("DINGTALK_WEBHOOK", "")
        self.wecom_webhook = wecom_webhook or os.getenv("WECOM_WEBHOOK", "")

    def test_feishu_connection(self) -> Dict[str, Any]:
        """测试并返回飞书连接诊断报告。"""
        report: Dict[str, Any] = {
            "configured": self.feishu_app.is_configured,
            "app_id": self.feishu_app.app_id[:6] + "..." if self.feishu_app.app_id else "",
            "token_ok": False,
            "bot_info": None,
            "joined_chats": [],
            "webhook_configured": bool(self.feishu_webhook),
        }

        token = self.feishu_app.get_tenant_access_token()
        if token:
            report["token_ok"] = True
            bot_res = self.feishu_app.get_bot_info()
            if bot_res.get("success"):
                report["bot_info"] = bot_res.get("data")
            chats = self.feishu_app.list_joined_chats()
            report["joined_chats"] = [
                {"chat_id": c.get("chat_id"), "name": c.get("name"), "description": c.get("description")}
                for c in chats
            ]

        return report

    def send_rebalance_signal(
        self,
        account_val: float,
        cash: float,
        orders: List[TradeOrder],
        regime: str,
        vol_multiplier: float = 1.0,
        env: str = "SIMULATE",
    ) -> bool:
        """发送调仓信号与交易清单。"""
        is_real = str(env).upper() == "REAL"
        env_label = "美股实盘" if is_real else "模拟账户"

        if not orders:
            msg = (
                f"### 🔔 [LERS 杠杆择时] 调仓信号检测 ({env_label})\n"
                f"- **当前宏观状态**: `{regime}`\n"
                f"- **{env_label}总资产**: `${account_val:,.2f}` | **可用资金**: `${cash:,.2f}`\n"
                f"- **仓位结论**: 当前持仓与策略目标权重偏差在缓冲区内，**无需调仓**。"
            )
            title = f"[LERS] {env_label}调仓检测: 无需操作"
            self._dispatch_all(title=title, text=msg, has_actions=False, is_real=is_real)
            return True

        table_rows = []
        for o in orders:
            side_tag = "🟢 建议买入" if o.action == "BUY" else "🔴 建议卖出"
            table_rows.append(
                f"| `{o.ticker}` | **{side_tag}** | {o.shares:,.0f} 股 | "
                f"${o.price:.2f} | ${o.est_amount:,.2f} | {o.target_weight*100:.1f}% | {o.reason} |"
            )
        table_str = "\n".join(table_rows)

        if is_real:
            title = "【美股实盘】核心-卫星策略 · 调仓建议与执行清单"
            msg = (
                f"**策略状态**：`{regime}` | **波动率乘数**：`{vol_multiplier:.2f}x`\n"
                f"**实盘总资产**：`${account_val:,.2f}` | **可用现金**：`${cash:,.2f}`\n"
                f"**风控状态**：🔒 **实盘只读安全锁生效中** (程序化直接报单已被物理阻断)\n\n"
                f"**📋 本次计划交易订单明细 (共 {len(orders)} 笔)：**\n\n"
                f"| 标的代码 | 方向 | 建议股数 | 参考限价 | 预估金额 | 目标权重 | 调仓原因 |\n"
                f"| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
                f"{table_str}\n\n"
                f"> 💡 **手动操作指引**: 请打开 **富途牛牛 / Moomoo 客户端**，建议【先卖后买】：先卖出非策略股票与现金资产回笼资金，再挂限价单买入目标 ETF。核对完成后请点击下方确认。"
            )
            return self._dispatch_all(title=title, text=msg, has_actions=True, is_real=True)
        else:
            title = "[LERS] 模拟盘调仓交易清单提醒"
            msg = (
                f"### 🚀 [LERS 核心-卫星策略] 触发模拟盘调仓决策\n\n"
                f"- **当前宏观状态**: `{regime}`\n"
                f"- **已实现波动率缩放 (Vol Multiplier)**: `{vol_multiplier:.2f}x`\n"
                f"- **模拟账户总资产**: `${account_val:,.2f}` (可用现金: `${cash:,.2f}`)\n"
                f"- **待执行笔数**: {len(orders)} 笔 (按卖出优先原则依次成交)\n\n"
                f"| 标的代码 | 交易操作 | 计划股数 | 参考限价 | 预估金额 | 目标权重 | 调仓原因 |\n"
                f"| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
                f"{table_str}\n\n"
                f"您可以点击下方卡片按钮直接通过模拟盘报单，或前往网页看板查看明细。"
            )
            return self._dispatch_all(title=title, text=msg, has_actions=True, is_real=False)

    def send_execution_report(
        self,
        results: List[Dict[str, Any]],
        env: str = "SIMULATE",
        success: bool = True,
        **kwargs,
    ) -> bool:
        """发送订单撮合成交执行报告。"""
        is_real = str(env).upper() == "REAL"
        env_label = "美股实盘" if is_real else "模拟账户"

        if is_real:
            title = "[LERS] 🔒 实盘安全锁拦截通知"
            msg = (
                f"### 🔒 [LERS 实盘安全系统] 拦截自动报单请求\n\n"
                f"- **触发事件**: 试图向 Moomoo 实盘环境发送自动交易指令\n"
                f"- **防御状态**: **100% 物理拦截成功 (未产生任何券商报单)**\n"
                f"- **拦截笔数**: {len(results)} 笔订单已全部转为手动调仓指引\n"
                f"- **安全准则**: 请使用官方富途牛牛 / Moomoo 客户端进行真实交易操作。"
            )
            return self._dispatch_all(title=title, text=msg, has_actions=False, is_real=True)

        success_count = sum(1 for r in results if r.get("success", False))
        fail_count = len(results) - success_count

        rows = []
        for r in results:
            tag = "✅ 成功" if r.get("success") else "❌ 失败"
            order_id = r.get("order_id", "-")
            msg_desc = r.get("msg", "-")
            rows.append(f"| `{r.get('code')}` | {r.get('action')} | {r.get('shares')}股 | {tag} | `{order_id}` | {msg_desc} |")
        rows_str = "\n".join(rows)

        title = f"[LERS] {env_label}报单执行完成 ({success_count}成功 / {fail_count}失败)"
        msg = (
            f"### 📋 [LERS 调仓执行报告] ({env_label})\n\n"
            f"- **执行状态**: 共提交 {len(results)} 笔，成功 {success_count} 笔，失败 {fail_count} 笔\n\n"
            f"| 标的代码 | 方向 | 股数 | 状态 | 订单编号 | 备注信息 |\n"
            f"| :--- | :---: | :---: | :---: | :---: | :--- |\n"
            f"{rows_str}"
        )
        return self._dispatch_all(title=title, text=msg, has_actions=False, is_real=False)

    def _dispatch_all(self, title: str, text: str, has_actions: bool, is_real: bool) -> bool:
        """多渠道异步/同步投递消息。"""
        logger.info(f"正在向多渠道广播通知: {title}")

        feishu_ok = self._send_feishu(title, text, has_actions, is_real)
        dingtalk_ok = True
        wecom_ok = True

        if self.dingtalk_webhook:
            dingtalk_ok = self._send_dingtalk(title, text)
        if self.wecom_webhook:
            wecom_ok = self._send_wecom(title, text)

        return feishu_ok and dingtalk_ok and wecom_ok

    def _build_feishu_card(self, title: str, text: str, has_actions: bool, is_real: bool) -> Dict[str, Any]:
        """构建飞书卡片数据结构。"""
        elements: List[Dict[str, Any]] = [
            {"tag": "markdown", "content": text}
        ]

        if has_actions:
            elements.append({"tag": "hr"})
            actions = []
            if is_real:
                actions.extend([
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 已手动报单核对"},
                        "type": "primary",
                        "value": {"action": "confirm_order"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🔄 刷新最新信号"},
                        "type": "default",
                        "value": {"action": "refresh_signal"},
                    },
                ])
            else:
                actions.extend([
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🚀 一键确认执行调仓"},
                        "type": "primary",
                        "value": {"action": "execute_rebalance", "env": "SIMULATE"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "📊 打开网页看板"},
                        "type": "default",
                        "url": "http://localhost:5000",
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "取消本次操作"},
                        "type": "danger",
                        "value": {"action": "cancel_rebalance"},
                    },
                ])

            elements.append({
                "tag": "action",
                "actions": actions,
            })

        header_template = "orange" if is_real else ("indigo" if has_actions else "blue")
        return {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": header_template,
            },
            "elements": elements,
        }

    def _send_feishu(self, title: str, text: str, has_actions: bool, is_real: bool) -> bool:
        card = self._build_feishu_card(title, text, has_actions, is_real)
        success = False

        # 1. 优先通过自建应用 OpenAPI 发送
        if self.feishu_app.is_configured:
            ok = self.feishu_app.send_interactive_card(card)
            if ok:
                success = True

        # 2. 如果配置了 Webhook，也投递 Webhook
        if self.feishu_webhook:
            payload = {
                "msg_type": "interactive",
                "card": card,
            }
            try:
                ok = self._http_post(self.feishu_webhook, payload)
                if ok:
                    success = True
            except Exception as e:
                logger.error(f"飞书 Webhook 推送失败: {e}")

        # 如果没有配置任何具体渠道，返回 True（已记录日志）
        if not self.feishu_app.is_configured and not self.feishu_webhook:
            logger.info("未配置飞书有效渠道，已跳过飞书推送")
            return True

        return success

    def _send_dingtalk(self, title: str, text: str) -> bool:
        try:
            payload = {
                "msg_type": "markdown",
                "markdown": {"title": title, "text": f"## {title}\n\n{text}"},
            }
            return self._http_post(self.dingtalk_webhook, payload)
        except Exception as e:
            logger.error(f"钉钉推送失败: {e}")
            return False

    def _send_wecom(self, title: str, text: str) -> bool:
        try:
            payload = {
                "msg_type": "markdown",
                "markdown": {"content": f"## {title}\n\n{text}"},
            }
            return self._http_post(self.wecom_webhook, payload)
        except Exception as e:
            logger.error(f"企业微信推送失败: {e}")
            return False

    @staticmethod
    def _http_post(url: str, data: dict) -> bool:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
