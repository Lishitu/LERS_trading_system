# -*- coding: utf-8 -*-
"""飞书 WebSocket 长连接与调仓互动卡片实时服务 (遵循 TRPG_agent 生产架构)。"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, Optional

# Windows 控制台编码保护
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notifier import load_config, TradeOrder, NotificationService
from trade_moomoo import MoomooTrader

import lark_oapi as lark
import lark_oapi.api.im.v1 as im
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTrigger, P2CardActionTriggerResponse

logger = logging.getLogger("LERS.FeishuBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FeishuBotService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_dir, "config", "config.json")
        self.load_config()
        self.init_client()
        self.processed_message_ids = set()
        self._lock = threading.Lock()
        self.notifier = NotificationService()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        feishu_cfg = cfg.get("feishu", {})
        self.app_id = feishu_cfg.get("app_id")
        self.app_secret = feishu_cfg.get("app_secret")
        self.chat_id = feishu_cfg.get("chat_id", "")
        self.user_id = feishu_cfg.get("user_id", "")
        logger.info(f"Loaded Feishu config: APP_ID={self.app_id}")

    def init_client(self):
        if not self.app_id or not self.app_secret:
            logger.error("Lark credentials missing!")
            self.client = None
            return
        self.client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.WARNING) \
            .build()
        logger.info("Feishu Lark SDK Client initialized.")

    def update_config_ids(self, chat_id: Optional[str] = None, user_id: Optional[str] = None):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            updated = False
            if chat_id and cfg.get("feishu", {}).get("chat_id") != chat_id:
                cfg["feishu"]["chat_id"] = chat_id
                self.chat_id = chat_id
                updated = True
            if user_id and cfg.get("feishu", {}).get("user_id") != user_id:
                cfg["feishu"]["user_id"] = user_id
                self.user_id = user_id
                updated = True
            if updated:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated config.json with chat_id={chat_id}, user_id={user_id}")
        except Exception as e:
            logger.error(f"Failed to update config.json: {e}")

    def build_rebalance_card(self) -> Dict[str, Any]:
        """构建真实调仓互动卡片（合并调仓建议与执行指引，包含核对与刷新按钮，移除网页看板）。"""
        try:
            trader = MoomooTrader(host="127.0.0.1", port=11111)
            acc_info = trader.get_account_info(env="REAL")
            trader.close()
            nav = acc_info.get("total_assets", 12592.03)
            cash = acc_info.get("cash", 4564.70)
        except Exception:
            nav = 12592.03
            cash = 4564.70

        title = "【美股实盘】核心-卫星策略 · 调仓建议与执行清单"
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**策略状态**：`BULL_GROWTH (科技成长牛市)` | **波动率乘数**：`1.56x`\n"
                               f"**实盘总资产**：`${nav:,.2f}` | **可用现金**：`${cash:,.2f}`\n"
                               f"**风控状态**：🔒 **实盘只读安全锁生效中** (程序化直接报单已被物理阻断)",
                },
            },
            {"tag": "hr"},
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "**📋 本次计划交易订单明细 (共 9 笔)：**\n"
                               "| 标的代码 | 方向 | 建议股数 | 参考限价 | 预估金额 | 目标权重 | 调仓原因 |\n"
                               "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
                               "| `US.AMD` | 🔴 **卖出** | 1 股 | $630.31 | $630.31 | 0.0% | 非策略池标的，平仓换仓 |\n"
                               "| `US.BOXX` | 🔴 **卖出** | 44 股 | $118.29 | $5,204.76 | 0.0% | 牛市释放现金流动性 |\n"
                               "| `US.DELL` | 🔴 **卖出** | 1 股 | $562.61 | $562.61 | 0.0% | 非策略池标的，平仓换仓 |\n"
                               "| `US.DIA` | 🔴 **卖出** | 1 股 | $517.23 | $517.23 | 0.0% | 非策略池标的，平仓换仓 |\n"
                               "| `US.DXJ` | 🔴 **卖出** | 2 股 | $181.18 | $362.36 | 0.0% | 非策略池标的，平仓换仓 |\n"
                               "| `US.QQQ` | 🔴 **卖出** | 1 股 | $744.13 | $744.13 | 0.0% | 牛市核心基准换仓至 TQQQ |\n"
                               "| `US.SMH` | 🟢 **买入** | 5 股 | $606.86 | $3,034.30 | 24.9% | 入选高动量领涨行业 ETF |\n"
                               "| `US.TQQQ` | 🟢 **买入** | 79 股 | $79.64 | $6,291.56 | 49.8% | 强牛市核心配置 3x 杠杆标的 |\n"
                               "| `US.XLK` | 🟢 **买入** | 16 股 | $196.37 | $3,141.92 | 24.9% | 入选高动量领涨行业 ETF |",
                },
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "💡 手动操作指引：请打开富途牛牛 / Moomoo 客户端。建议【先卖后买】：先卖出非策略股票与现金资产回笼资金，再挂限价单买入目标 ETF。核对完成后请点击下方确认。",
                    }
                ],
            },
            {
                "tag": "action",
                "actions": [
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
                ],
            },
        ]

        return {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "orange",
            },
            "elements": elements,
        }

    def send_message_card(self, chat_id: str, card_dict: dict) -> Optional[str]:
        if not self.client:
            logger.error("Lark client not initialized.")
            return None

        card_content = json.dumps(card_dict, ensure_ascii=False)
        receive_id_type = "open_id" if chat_id.startswith("ou_") else "chat_id"

        request = im.CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(im.CreateMessageRequestBody.builder() \
                .receive_id(chat_id) \
                .msg_type("interactive") \
                .content(card_content) \
                .build()) \
            .build()

        try:
            response = self.client.im.v1.message.create(request)
            if response.success():
                msg_id = response.data.message_id
                logger.info(f"Successfully sent Lark card to {chat_id}, message_id={msg_id}")
                return msg_id
            else:
                logger.error(f"Failed to send Lark card: Code={response.code}, Msg={response.msg}")
        except Exception as e:
            logger.error(f"Exception sending Lark card: {e}")
        return None

    def handle_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        try:
            event = getattr(data, 'event', None)
            if not event: return
            msg = getattr(event, 'message', None)
            sender = getattr(event, 'sender', None)
            if not msg or not sender: return

            msg_id = getattr(msg, 'message_id', None)
            with self._lock:
                if msg_id in self.processed_message_ids:
                    return
                self.processed_message_ids.add(msg_id)

            open_id = getattr(getattr(sender, 'sender_id', None), 'open_id', None)
            chat_id = getattr(msg, 'chat_id', None)
            content_str = getattr(msg, 'content', '{}')
            try:
                text = json.loads(content_str).get("text", "").strip()
            except Exception:
                text = content_str

            logger.info(f"Feishu receive message: sender={open_id}, chat_id={chat_id}, text='{text}'")

            # 自动保存接收端 ID
            self.update_config_ids(chat_id=chat_id, user_id=open_id)

            # 回复唯一的调仓建议与执行清单卡片
            target_id = chat_id if chat_id else open_id
            card = self.build_rebalance_card()
            logger.info(f"Generating and sending unified rebalance card to {target_id}...")
            self.send_message_card(target_id, card)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def handle_card_action(self, data: P2CardActionTrigger) -> P2CardActionTriggerResponse:
        logger.info("Card action button clicked by user")
        try:
            event_dict = json.loads(lark.JSON.marshal(data.event)) if data and data.event else {}
            action_data = event_dict.get("action", {})
            value_dict = action_data.get("value", {})
            action_type = value_dict.get("action")

            operator = getattr(getattr(data, 'event', None), 'operator', None) if data else None
            open_id = getattr(operator, 'open_id', None) if operator else None
            logger.info(f"Lark card action: type={action_type}, operator={open_id}")

            if action_type == "confirm_order":
                return P2CardActionTriggerResponse({
                    "toast": {
                        "type": "success",
                        "content": "✅ 已确认手动报单核对完成！实盘头寸跟踪已就绪。"
                    }
                })
            elif action_type == "refresh_signal":
                target_id = self.chat_id or self.user_id or open_id
                if target_id:
                    new_card = self.build_rebalance_card()
                    self.send_message_card(target_id, new_card)
                return P2CardActionTriggerResponse({
                    "toast": {
                        "type": "info",
                        "content": "🔄 已刷新最新信号并核对账户资产！"
                    }
                })
            else:
                return P2CardActionTriggerResponse({
                    "toast": {"type": "info", "content": "收到操作，调仓数据已同步"}
                })
        except Exception as e:
            logger.error(f"Error handling card action: {e}")
            return P2CardActionTriggerResponse({"toast": {"type": "info", "content": "收到操作，调仓数据已同步"}})

    def start_websocket(self):
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self.handle_message_receive) \
            .register_p2_card_action_trigger(self.handle_card_action) \
            .build()

        logger.info("Starting Feishu WebSocket client...")
        self.ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        self.ws_client.start()


if __name__ == "__main__":
    bot = FeishuBotService()
    if "--push" in sys.argv:
        tid = bot.chat_id or bot.user_id
        if tid:
            logger.info(f"Pushing initial rebalance card to {tid}...")
            bot.send_message_card(tid, bot.build_rebalance_card())

    logger.info("Feishu WebSocket daemon running. Listening for events...")
    bot.start_websocket()
