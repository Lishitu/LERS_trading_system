"""Moomoo OpenD 美股模拟与实盘交易接口封装模块。

基于 moomoo-api 的 OpenUSTradeContext，提供美股模拟盘 (Paper Trading, TrdEnv.SIMULATE) 与实盘接口，
包括账户资金查询、持仓同步、调仓限价报单、订单状态追踪与防异常保护。
若本地未运行 OpenD 服务或未配置环境，支持无缝切换至高保真内建模拟器 (Mock Paper Trader)，便于测试与策略演练。
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

try:
    from moomoo import (
        OpenSecTradeContext,
        TrdEnv,
        TrdSide,
        OrderType,
        TrdMarket,
        RET_OK,
    )
except ImportError:
    OpenSecTradeContext = None
    TrdEnv = None
    TrdSide = None
    OrderType = None
    TrdMarket = None
    RET_OK = 0

logger = logging.getLogger("LERS.TradeMoomoo")


def _safe_float(val: Any, default: float = 0.0) -> float:
    """安全转换浮点数，妥善处理 OpenD 返回的 'N/A' 或 None。"""
    if val is None or val == "N/A" or val == "--" or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _normalize_ticker(code: str) -> str:
    """标准化标的代码格式，确保美股标的统一带有 'US.' 前缀。"""
    c = str(code).strip()
    if not c.startswith("US.") and "." not in c:
        return f"US.{c}"
    return c


@dataclass
class AccountBalance:
    """账户资金概况"""
    total_assets: float = 100000.0  # 总资产 (NAV)
    cash: float = 100000.0          # 可用现金
    market_val: float = 0.0         # 证券市值
    buying_power: float = 100000.0  # 购买力
    currency: str = "USD"
    is_mock: bool = False


@dataclass
class PositionInfo:
    """单只证券持仓概况"""
    code: str
    shares: float
    cost_price: float
    current_price: float
    market_val: float
    unrealized_pnl: float = 0.0


class MoomooTrader:
    """Moomoo 美股交易客户端 (支持模拟盘与实盘)。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        is_simulate: bool = True,
        security_firm: Any = None,
        force_mock: bool = False,
    ):
        self.host = host
        self.port = port
        self.is_simulate = is_simulate
        self.security_firm = security_firm
        self.force_mock = force_mock

        self._ctx: Optional[OpenSecTradeContext] = None
        self._connected = False

        # 内建模拟交易账户 (用于 OpenD 离线或 force_mock 状态下的测试与干跑)
        self._mock_cash = 100000.0
        self._mock_positions: Dict[str, PositionInfo] = {}
        self._mock_orders: List[Dict[str, Any]] = []

        if not force_mock:
            self._try_connect()

    def _try_connect(self) -> bool:
        """尝试连接本地 OpenD 交易通道。"""
        if OpenSecTradeContext is None:
            logger.warning("[MoomooTrader] 未安装 moomoo SDK，切换至内建模拟器模式。")
            self._connected = False
            return False

        try:
            kwargs: Dict[str, Any] = {
                'filter_trdmarket': TrdMarket.US if TrdMarket is not None else 2,
                'host': self.host,
                'port': self.port,
            }
            if self.security_firm is not None and not isinstance(self.security_firm, int):
                kwargs['security_firm'] = self.security_firm

            self._ctx = OpenSecTradeContext(**kwargs)
            self._connected = True
            logger.info(f"[MoomooTrader] 成功连接 OpenD 美股交易通道 ({self.host}:{self.port})")
            return True
        except Exception as e:
            logger.warning(f"[MoomooTrader] 连接 OpenD 失败 ({e})，启用内建模拟交易器。")
            self._ctx = None
            self._connected = False
            return False

    @property
    def is_live_connected(self) -> bool:
        """是否成功连接真实/模拟 OpenD 交易服务。"""
        return self._connected and self._ctx is not None

    def unlock_trade(self, password: str = "") -> bool:
        """解锁交易（实盘需交易密码，模拟盘一般不需要）。"""
        if not self.is_live_connected:
            return True
        try:
            pwd = password or os.getenv("MOOMOO_UNLOCK_PWD", "")
            if not pwd:
                return True
            ret, data = self._ctx.unlock_trade(password=pwd)
            if ret != RET_OK:
                logger.error(f"[MoomooTrader] 解锁交易失败: {data}")
                return False
            logger.info("[MoomooTrader] 交易通道解锁成功")
            return True
        except Exception as e:
            logger.error(f"[MoomooTrader] 解锁交易异常: {e}")
            return False

    def _resolve_trd_env(self, env: Optional[str] = None) -> Any:
        """解析交易环境 (TrdEnv.SIMULATE 或 TrdEnv.REAL)。"""
        if env is not None:
            if str(env).upper() in ("REAL", "0"):
                return TrdEnv.REAL if TrdEnv is not None else 0
            return TrdEnv.SIMULATE if TrdEnv is not None else 1
        return (TrdEnv.SIMULATE if self.is_simulate else TrdEnv.REAL) if TrdEnv is not None else (1 if self.is_simulate else 0)

    def get_account_balance(self, env: Optional[str] = None) -> AccountBalance:
        """查询账户资金概况 (支持分别查询实盘 REAL 与模拟盘 SIMULATE)。"""
        if not self.is_live_connected:
            mkt_val = sum(p.market_val for p in self._mock_positions.values())
            tot_assets = self._mock_cash + mkt_val
            return AccountBalance(
                total_assets=tot_assets,
                cash=self._mock_cash,
                market_val=mkt_val,
                buying_power=self._mock_cash,
                is_mock=True,
            )

        trd_env = self._resolve_trd_env(env)
        try:
            # 优先显式指定查询 USD 货币，避免 OpenD 默认按主账户基准货币 (如 HKD) 结算返回汇率折算数值
            ret, df = self._ctx.accinfo_query(trd_env=trd_env, currency='USD')
            if ret != RET_OK or df.empty:
                # 兼容部分 OpenD 版本，若带 currency 报错则回退标准查询
                ret, df = self._ctx.accinfo_query(trd_env=trd_env)

            if ret != RET_OK or df.empty:
                logger.warning(f"[MoomooTrader] 查询账户资金失败 (env={env}): {df}，使用回退数据")
                return AccountBalance(is_mock=True)

            row = df.iloc[0]
            curr = str(row.get('currency', 'USD')).upper()

            # 双重防护：如果返回仍被结算为 HKD，但包含了专属于美股的 usd_assets / us_cash，则提取纯美元数据
            if curr == 'HKD' and _safe_float(row.get('usd_assets')) > 0:
                tot_assets = _safe_float(row.get('usd_assets'))
                cash = _safe_float(row.get('us_cash'))
                positions = self.get_positions(env=env)
                mkt_val = sum(p.market_val for p in positions.values())
                power = _safe_float(row.get('usd_net_cash_power'), cash)
                curr = "USD"
            else:
                cash = _safe_float(row.get('cash', 0.0))
                tot_assets = _safe_float(row.get('total_assets', cash))
                mkt_val = _safe_float(row.get('market_val', 0.0))
                power = _safe_float(row.get('power', cash))

            return AccountBalance(
                total_assets=tot_assets,
                cash=cash,
                market_val=mkt_val,
                buying_power=power,
                currency=curr,
                is_mock=False,
            )
        except Exception as e:
            logger.error(f"[MoomooTrader] 查询账户资金异常 (env={env}): {e}")
            return AccountBalance(is_mock=True)

    def get_positions(self, env: Optional[str] = None) -> Dict[str, PositionInfo]:
        """获取账户当前所有持仓 (支持分别获取实盘 REAL 与模拟盘 SIMULATE 持仓)。"""
        if not self.is_live_connected:
            return {c: p for c, p in self._mock_positions.items() if p.shares > 0.01}

        trd_env = self._resolve_trd_env(env)
        positions: Dict[str, PositionInfo] = {}

        try:
            ret, df = self._ctx.position_list_query(trd_env=trd_env)
            if ret != RET_OK:
                logger.warning(f"[MoomooTrader] 查询持仓列表失败 (env={env}): {df}")
                return {}

            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = _normalize_ticker(str(row['code']))
                    qty = _safe_float(row.get('qty', 0.0))
                    cost_p = _safe_float(row.get('cost_price', 0.0))
                    cur_p = _safe_float(row.get('nominal_price', cost_p))
                    mkt_v = _safe_float(row.get('market_val', qty * cur_p))
                    pnl = _safe_float(row.get('pl_val'), _safe_float(row.get('unrealized_pl', (cur_p - cost_p) * qty)))

                    if qty > 0.01:
                        positions[code] = PositionInfo(
                            code=code,
                            shares=qty,
                            cost_price=cost_p,
                            current_price=cur_p,
                            market_val=mkt_v,
                            unrealized_pnl=pnl,
                        )
        except Exception as e:
            logger.error(f"[MoomooTrader] 查询持仓列表异常 (env={env}): {e}")

        return positions

    def place_order(
        self,
        code: str,
        action: str,  # 'BUY' or 'SELL'
        shares: float,
        price: float,
        order_type: str = "NORMAL",  # NORMAL = 限价单 (Limit Order)
        env: Optional[str] = None,
    ) -> Dict[str, Any]:
        """报送单笔订单（支持模拟与真实 OpenD）。
        
        【实盘安全防御机制】:
        实盘模式 (TrdEnv.REAL) 下严格禁止程序化直接报单，调用将立即拦截并报错。
        实盘调仓一律采用生成建议清单 + 消息推送，由用户手动在富途/Moomoo App 完成下单。
        """
        if shares <= 0:
            return {'success': False, 'error': '股数必须大于0', 'code': code}

        trd_env = self._resolve_trd_env(env)

        # 🔒 实盘下单最高安全防线：绝对拦截任何实盘程序下单
        if (TrdEnv is not None and trd_env == TrdEnv.REAL) or (str(env).upper() == "REAL"):
            err_msg = "【实盘安全锁触发】系统已被严格配置为只读/建议模式，严禁实盘程序直接下单。请在富途牛牛/Moomoo客户端手动下单。"
            logger.warning(f"[MoomooTrader] 🔒 拦截实盘自动报单尝试: {code} {action} {shares}股 @ ${price:.2f}！实盘仅支持持仓读取与调仓建议，已阻止下单。")
            return {
                'success': False,
                'error': err_msg,
                'code': code,
                'action': action,
                'is_real_blocked': True,
            }

        if not self.is_live_connected:
            # 内建模拟交易器执行
            return self._mock_execute_order(code, action, shares, price)

        trd_side = TrdSide.BUY if action.upper() == "BUY" else TrdSide.SELL
        o_type = OrderType.NORMAL if order_type == "NORMAL" else OrderType.MARKET

        try:
            ret, data = self._ctx.place_order(
                price=round(price, 2),
                qty=int(round(shares)),
                code=code,
                trd_side=trd_side,
                order_type=o_type,
                trd_env=trd_env,
            )

            if ret != RET_OK:
                err_msg = str(data)
                logger.error(f"[MoomooTrader] 报单失败 {code} {action} {shares}股 @ ${price:.2f}: {err_msg}")
                return {'success': False, 'error': err_msg, 'code': code, 'action': action}

            order_id = data.iloc[0]['order_id'] if hasattr(data, 'iloc') else str(data)
            logger.info(f"[MoomooTrader] 模拟盘报单成功: {code} {action} {shares}股 @ ${price:.2f}, OrderID={order_id}")
            return {
                'success': True,
                'order_id': order_id,
                'code': code,
                'action': action,
                'shares': int(round(shares)),
                'price': price,
            }
        except Exception as e:
            logger.error(f"[MoomooTrader] 报单接口调用异常: {e}")
            return {'success': False, 'error': str(e), 'code': code, 'action': action}

    def _mock_execute_order(
        self,
        code: str,
        action: str,
        shares: float,
        price: float,
    ) -> Dict[str, Any]:
        """内建模拟撮合引擎。"""
        cost = shares * price
        if action.upper() == "BUY":
            if self._mock_cash < cost:
                return {
                    'success': False,
                    'error': f'现金不足 (可用 ${self._mock_cash:,.2f} < 所需 ${cost:,.2f})',
                    'code': code,
                    'action': action,
                }
            self._mock_cash -= cost
            cur_pos = self._mock_positions.get(code, PositionInfo(code=code, shares=0, cost_price=0, current_price=price, market_val=0))
            new_shares = cur_pos.shares + shares
            new_cost = (cur_pos.cost_price * cur_pos.shares + cost) / new_shares
            self._mock_positions[code] = PositionInfo(
                code=code,
                shares=new_shares,
                cost_price=new_cost,
                current_price=price,
                market_val=new_shares * price,
            )
        else:
            cur_pos = self._mock_positions.get(code)
            if cur_pos is None or cur_pos.shares < shares:
                avail = cur_pos.shares if cur_pos else 0
                return {
                    'success': False,
                    'error': f'持仓不足 (可卖 {avail}股 < 计划 {shares}股)',
                    'code': code,
                    'action': action,
                }
            self._mock_cash += cost
            cur_pos.shares -= shares
            cur_pos.market_val = cur_pos.shares * price
            if cur_pos.shares <= 0.01:
                del self._mock_positions[code]

        order_record = {
            'order_id': f"MOCK-{int(time.time()*1000)}",
            'code': code,
            'action': action,
            'shares': shares,
            'price': price,
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'success': True,
        }
        self._mock_orders.append(order_record)
        logger.info(f"[MockTrader] 模拟撮合成交: {code} {action} {shares}股 @ ${price:.2f}")
        return order_record

    def close(self):
        """关闭交易通道。"""
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
            self._ctx = None
            self._connected = False

    def __del__(self):
        self.close()
