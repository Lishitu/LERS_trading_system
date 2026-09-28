"""Moomoo 实时行情订阅与推送管理模块。

订阅标的池全量 ETF 的实时行情 (QUOTE) 与分时K线 (K_1M)，
在内存中异步维护盘中最新价、盘口买卖一档、盘中累计 VWAP 及 MA60 趋势状态，
并在盘中侦测到个券跌破 MA60 时支持触发应急响应回调。
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np

try:
    from moomoo import (
        OpenQuoteContext,
        SubType,
        StockQuoteHandlerBase,
        CurKlineHandlerBase,
        RET_OK,
    )
except ImportError:
    OpenQuoteContext = None
    SubType = None
    StockQuoteHandlerBase = object
    CurKlineHandlerBase = object
    RET_OK = 0

from data_moomoo import (
    DEFAULT_BENCHMARKS,
    DEFAULT_SECTORS,
    DEFAULT_DEFENSE,
    LEVERAGED_MAP,
)

def _normalize_ticker(code: str) -> str:
    """标准化标的代码格式，确保美股标的统一带有 'US.' 前缀。"""
    c = str(code).strip()
    if not c.startswith("US.") and "." not in c:
        return f"US.{c}"
    return c


logger = logging.getLogger("LERS.MarketSubscriber")


class RealtimeMarketState:
    """实时市场状态存储容器。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_prices: Dict[str, float] = {}
        self.bid_prices: Dict[str, float] = {}
        self.ask_prices: Dict[str, float] = {}
        self.vwap_prices: Dict[str, float] = {}
        self.ma60_values: Dict[str, float] = {}
        self.last_update_times: Dict[str, str] = {}

    def update_quote(
        self,
        code: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        vwap: Optional[float] = None,
    ):
        with self.lock:
            if price > 0:
                self.latest_prices[code] = price
            if bid and bid > 0:
                self.bid_prices[code] = bid
            if ask and ask > 0:
                self.ask_prices[code] = ask
            if vwap and vwap > 0:
                self.vwap_prices[code] = vwap
            self.last_update_times[code] = time.strftime("%H:%M:%S")

    def set_ma60_baseline(self, ma60_dict: Dict[str, float]):
        with self.lock:
            self.ma60_values.update(ma60_dict)

    def get_price(self, code: str, fallback: float = 100.0) -> float:
        with self.lock:
            return self.latest_prices.get(code, fallback)

    def get_order_limit_price(self, code: str, is_buy: bool, fallback_price: float = 100.0) -> float:
        """获取带有微幅买卖缓冲的限价单参考价格。"""
        with self.lock:
            cur = self.latest_prices.get(code, 0.0)
            if cur <= 0:
                cur = fallback_price
            if cur <= 0:
                return 100.0
            if is_buy:
                # 买入以卖一或最新价略微上浮 0.05% 确保成交流畅，同时防滑点
                ask = self.ask_prices.get(code, cur)
                return round(min(ask * 1.0005, cur * 1.002), 2)
            else:
                # 卖出以买一或最新价略微下浮 0.05%
                bid = self.bid_prices.get(code, cur)
                return round(max(bid * 0.9995, cur * 0.998), 2)

    def is_trend_broken(self, code: str) -> bool:
        """检查标的是否跌破 MA60。"""
        with self.lock:
            cur = self.latest_prices.get(code)
            ma60 = self.ma60_values.get(code)
            if cur is not None and ma60 is not None and ma60 > 0:
                return cur < ma60
            return False


class _InternalQuoteHandler(StockQuoteHandlerBase):
    """Moomoo 实时报价推送处理器。"""

    def __init__(self, state: RealtimeMarketState, break_cb: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.state = state
        self.break_cb = break_cb

    def on_recv_rsp(self, rsp_pb):
        if RET_OK == 0 or rsp_pb is None:
            return
        ret_code, df = super().on_recv_rsp(rsp_pb)
        if ret_code != RET_OK or df is None or df.empty:
            return

        for _, row in df.iterrows():
            code = _normalize_ticker(str(row['code']))
            price = float(row.get('last_price', 0.0))
            bid = float(row.get('bid_price', price))
            ask = float(row.get('ask_price', price))
            vwap = float(row.get('vwap', price))

            self.state.update_quote(code=code, price=price, bid=bid, ask=ask, vwap=vwap)

            if self.break_cb and self.state.is_trend_broken(code):
                self.break_cb(code)


class _InternalKlineHandler(CurKlineHandlerBase):
    """Moomoo 分时 K 线推送处理器。"""

    def __init__(self, state: RealtimeMarketState):
        super().__init__()
        self.state = state

    def on_recv_rsp(self, rsp_pb):
        if RET_OK == 0 or rsp_pb is None:
            return
        ret_code, df = super().on_recv_rsp(rsp_pb)
        if ret_code != RET_OK or df is None or df.empty:
            return

        for _, row in df.iterrows():
            code = _normalize_ticker(str(row['code']))
            close_p = float(row.get('close', 0.0))
            if close_p > 0:
                self.state.update_quote(code=code, price=close_p)


class MoomooMarketSubscriber:
    """Moomoo 标的池行情订阅管理器。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        tickers: Optional[List[str]] = None,
        on_emergency_break: Optional[Callable[[str], None]] = None,
        force_mock: bool = False,
    ):
        self.host = host
        self.port = port
        self.force_mock = force_mock
        self.on_emergency_break = on_emergency_break

        # 默认标的池 (覆盖基准、行业卫星、2x/3x杠杆与短债防御)
        if tickers:
            self.tickers = tickers
        else:
            all_t = set(DEFAULT_BENCHMARKS) | set(DEFAULT_SECTORS) | {DEFAULT_DEFENSE} | set(LEVERAGED_MAP.values())
            self.tickers = sorted(list(all_t))

        self.state = RealtimeMarketState()
        self._ctx: Optional[OpenQuoteContext] = None
        self._is_subscribed = False

        self._init_baseline_ma60()

        if not force_mock:
            self._try_connect_and_subscribe()

    def _init_baseline_ma60(self):
        """从本地历史数据加载各标的最新 MA60 基准值作为趋势安全线。"""
        ma60_map = {}
        for t in self.tickers:
            clean_t = t.replace(".", "_")
            p_file = os.path.join("data", "moomoo", f"{clean_t}.parquet")
            if os.path.exists(p_file):
                try:
                    df = pd.read_parquet(p_file)
                    if 'close' in df.columns and len(df) >= 60:
                        ma60 = float(df['close'].iloc[-60:].mean())
                        last_p = float(df['close'].iloc[-1])
                        ma60_map[t] = ma60
                        self.state.update_quote(t, price=last_p)
                except Exception:
                    pass
        self.state.set_ma60_baseline(ma60_map)

    def _try_connect_and_subscribe(self) -> bool:
        if OpenQuoteContext is None:
            logger.warning("[MarketSubscriber] 未安装 moomoo SDK，使用本地持久化行情快照。")
            return False

        try:
            self._ctx = OpenQuoteContext(host=self.host, port=self.port)
            quote_h = _InternalQuoteHandler(self.state, self.on_emergency_break)
            kline_h = _InternalKlineHandler(self.state)
            self._ctx.set_handler(quote_h)
            self._ctx.set_handler(kline_h)

            ret, data = self._ctx.subscribe(
                code_list=self.tickers,
                subtype_list=[SubType.QUOTE, SubType.K_1M],
                is_first_push=True,
            )
            if ret != RET_OK:
                logger.error(f"[MarketSubscriber] 订阅行情失败: {data}")
                return False

            self._is_subscribed = True
            logger.info(f"[MarketSubscriber] 成功订阅 {len(self.tickers)} 只标的实时行情与分时K线")
            return True
        except Exception as e:
            logger.warning(f"[MarketSubscriber] 连接行情 OpenD 异常 ({e})，启用本地离线行情缓存。")
            self._ctx = None
            self._is_subscribed = False
            return False

    @property
    def is_subscribed(self) -> bool:
        return self._is_subscribed

    def get_latest_prices(self) -> Dict[str, float]:
        """获取所有标的的最新价格字典。"""
        with self.state.lock:
            return dict(self.state.latest_prices)

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
            self._ctx = None
            self._is_subscribed = False
