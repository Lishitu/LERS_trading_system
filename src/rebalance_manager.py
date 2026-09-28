"""调仓管理与交易执行编排模块 (Rebalance Manager)。

连接实时行情、Moomoo 模拟/实盘账户及 Core-Satellite 策略引擎：
1. 提取当前账户真实权益与实时持仓
2. 调用策略生成目标权重及波动率目标化缩放
3. 遵循 6% 调仓缓冲区与卖先买后顺序，精准计算买卖差额股数与限价
4. 输出结构化交易清单与信号报告，落盘历史记录，并通过多渠道推送提醒
5. 支持手动确认与全自动报单执行
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from data_moomoo import (
    MoomooDataLoader,
    DEFAULT_DEFENSE,
    DEFAULT_BENCHMARKS,
    DEFAULT_SECTORS,
    LEVERAGED_MAP,
)
from strategy_coresatellite import CoreSatelliteStrategy
from trade_moomoo import MoomooTrader, AccountBalance, PositionInfo
from market_subscriber import MoomooMarketSubscriber
from notifier import NotificationService, TradeOrder
from trading_costs import TradingCostCalculator

logger = logging.getLogger("LERS.RebalanceManager")


class RebalanceManager:
    """调仓编排与订单管理中心。"""

    def __init__(
        self,
        strategy: Optional[CoreSatelliteStrategy] = None,
        trader: Optional[MoomooTrader] = None,
        subscriber: Optional[MoomooMarketSubscriber] = None,
        notifier: Optional[NotificationService] = None,
        buffer: float = 0.06,
        records_dir: str = "data/trading_records",
    ):
        self.strategy = strategy or CoreSatelliteStrategy()
        self.trader = trader or MoomooTrader(is_simulate=True)
        self.subscriber = subscriber or MoomooMarketSubscriber()
        self.notifier = notifier or NotificationService()
        self.cost_calc = TradingCostCalculator()
        self.buffer = buffer
        self.records_dir = records_dir
        os.makedirs(self.records_dir, exist_ok=True)

        self.last_rebalance_plan: List[TradeOrder] = []
        self.last_regime: str = "NEUTRAL_CHOP"
        self.last_vol_multiplier: float = 1.0

        # 策略资金规模预算与非策略持仓隔离配置
        self.capital_mode: str = "FULL_NAV"  # "FULL_NAV", "CUSTOM_AMOUNT", "PERCENT"
        self.custom_amount: float = 30000.0
        self.capital_percent: float = 50.0
        self.ignore_outside_holdings: bool = True
        self.satellite_weight_mode: str = "EQUAL"  # "EQUAL", "TIERED_60_40", "TIERED_65_35"
        self.last_effective_strat_nav: float = 0.0

    def set_capital_parameters(
        self,
        mode: str = "FULL_NAV",
        custom_amount: float = 30000.0,
        percent: float = 50.0,
        ignore_outside_holdings: bool = True,
        satellite_weight_mode: str = "EQUAL",
    ):
        """动态更新策略运行规模与卫星持仓阶梯配置。"""
        self.capital_mode = str(mode).upper()
        self.custom_amount = float(custom_amount)
        self.capital_percent = float(percent)
        self.ignore_outside_holdings = bool(ignore_outside_holdings)
        self.satellite_weight_mode = str(satellite_weight_mode).upper()

        if self.satellite_weight_mode == "TIERED_60_40":
            self.strategy.tiered_ratio = (0.60, 0.40)
        elif self.satellite_weight_mode == "TIERED_65_35":
            self.strategy.tiered_ratio = (0.65, 0.35)
        else:
            self.strategy.tiered_ratio = None

        logger.info(
            f"已更新策略规模与配置: 模式={self.capital_mode}, 资金=${self.custom_amount:.2f}, "
            f"比例={self.capital_percent:.1f}%, 隔离外部持仓={self.ignore_outside_holdings}, 卫星分配={self.satellite_weight_mode}"
        )

    def set_strategy_parameters(
        self,
        core_ratio: float,
        satellite_ratio: float,
        target_vol: float,
        min_leverage: float,
        max_leverage: float,
        buffer: float,
        top_n: int = 2,
        tiered_ratio: Optional[Tuple[float, ...]] = None,
    ):
        """动态更新策略参数与调仓缓冲。"""
        self.strategy.core_ratio = core_ratio
        self.strategy.satellite_ratio = satellite_ratio
        self.strategy.target_vol = target_vol
        self.strategy.min_leverage = min_leverage
        self.strategy.max_leverage = max_leverage
        self.strategy.top_n = top_n
        if tiered_ratio is not None:
            self.strategy.tiered_ratio = tiered_ratio
        self.buffer = buffer
        logger.info(
            f"已切换策略参数: 核心 {core_ratio*100:.0f}%, 卫星 {satellite_ratio*100:.0f}%, "
            f"目标波 {target_vol*100:.0f}%, 杠杆 [{min_leverage}x, {max_leverage}x], 缓冲 {buffer*100:.0f}%"
        )

    def generate_rebalance_plan(
        self,
        as_of_date: Optional[pd.Timestamp] = None,
        force_rebalance: bool = False,
        env: str = "SIMULATE",
    ) -> Tuple[AccountBalance, List[TradeOrder], Dict[str, float]]:
        """计算最新的调仓方案与交易清单。
        
        Args:
            as_of_date: 评估日期 (可选)
            force_rebalance: 是否忽略缓冲区强制再平衡
            env: 交易环境 ("SIMULATE" 模拟盘 或 "REAL" 实盘)

        Returns:
            (account_balance, trade_orders, target_weights)
        """
        # 1. 查询账户资金与持仓 (根据指定环境)
        acc_bal = self.trader.get_account_balance(env=env)
        tot_nav = acc_bal.total_assets
        cash = acc_bal.cash
        positions = self.trader.get_positions(env=env)

        # 2. 提取标的最新行情价格
        latest_prices = self.subscriber.get_latest_prices()

        # 3. 运行策略引擎获取目标配置
        loader = MoomooDataLoader()
        data_dict = loader.load_all_universe(start_date="2020-01-01")
        (
            factors_dict,
            vols,
            ma_trend,
            df_close,
            regimes,
        ) = self.strategy.compute_factors_and_indicators(data_dict)

        eval_date = as_of_date or df_close.index[-1]
        regime = regimes.loc[eval_date]
        trend = ma_trend.loc[eval_date]
        vol_t = vols.loc[eval_date]

        # 估算近期波动率
        recent_vol = float(vols[self.strategy.benchmark].iloc[-1]) if self.strategy.benchmark in vols else self.strategy.target_vol
        self.last_vol_multiplier = self.strategy.calculate_volatility_multiplier(recent_vol)
        self.last_regime = regime

        target_weights = self.strategy.generate_portfolio_target(
            date=eval_date,
            factors_dict=factors_dict,
            current_trend=trend,
            current_vols=vol_t,
            regime=regime,
            df_close=df_close,
            realized_portfolio_vol=recent_vol,
        )

        # 4. 计算策略运行资金预算 (Strategy Allocated NAV)
        if self.capital_mode == "CUSTOM_AMOUNT":
            strat_nav = min(float(self.custom_amount), tot_nav) if tot_nav > 0 else float(self.custom_amount)
        elif self.capital_mode == "PERCENT":
            strat_nav = tot_nav * (float(self.capital_percent) / 100.0)
        else:
            strat_nav = tot_nav
        self.last_effective_strat_nav = strat_nav

        # 5. 计算当前真实持仓在策略预算内的权重
        current_weights: Dict[str, float] = {}
        for code, pos in positions.items():
            p = latest_prices.get(code, pos.current_price)
            if p <= 0:
                p = pos.cost_price if pos.cost_price > 0 else 100.0
            val = pos.shares * p
            current_weights[code] = (val / strat_nav) if strat_nav > 0 else 0.0

        strat_universe = set(
            DEFAULT_BENCHMARKS
            + DEFAULT_SECTORS
            + list(LEVERAGED_MAP.values())
            + [DEFAULT_DEFENSE, 'US.BOXX', 'US.BIL', 'US.QQQ', 'US.SPY', 'US.QLD', 'US.TQQQ']
        )

        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        sell_orders: List[TradeOrder] = []
        buy_orders: List[TradeOrder] = []

        for code in sorted(list(all_tickers)):
            # 隔离非策略自主持仓：若标的不在策略池且不在目标中，不强制报单平仓
            if code not in target_weights and code not in strat_universe:
                if self.ignore_outside_holdings or self.capital_mode != "FULL_NAV":
                    continue

            cur_w = current_weights.get(code, 0.0)
            targ_w = target_weights.get(code, 0.0)
            pos_obj = positions.get(code)
            cur_shares = int(round(pos_obj.shares)) if pos_obj else 0
            pos_price = pos_obj.current_price if (pos_obj and pos_obj.current_price > 0) else 100.0

            ref_p = latest_prices.get(code, pos_price)
            if ref_p <= 0:
                ref_p = 100.0

            # 调仓缓冲区过滤：如果目标权重与当前权重偏离 <= buffer 且非清仓，则维持现状
            if not force_rebalance and targ_w > 1e-4 and cur_shares > 0:
                if abs(targ_w - cur_w) <= self.buffer:
                    continue

            target_dollars = targ_w * strat_nav
            target_shares = int(round(target_dollars / ref_p))

            if target_shares < cur_shares:
                # 卖出
                delta_s = int(round(cur_shares - target_shares))
                if delta_s >= 1:
                    sell_limit_p = self.subscriber.state.get_order_limit_price(code, is_buy=False, fallback_price=ref_p)
                    est_amt = delta_s * sell_limit_p
                    if targ_w <= 1e-4:
                        if code == 'US.QQQ':
                            if target_weights.get('US.TQQQ', 0) > 0.1:
                                reason = "牛市环境核心升级为 3x 杠杆 TQQQ，建议换仓至 TQQQ"
                            elif target_weights.get('US.QLD', 0) > 0.1:
                                reason = "震荡牛市核心升级为 2x 杠杆 QLD，建议换仓至 QLD"
                            else:
                                reason = "核心基准暂时减仓避险"
                        elif code == 'US.BOXX':
                            reason = "牛市进攻期现金资产释放流动性，建议换仓至权益多头"
                        elif code in (DEFAULT_BENCHMARKS + DEFAULT_SECTORS + list(LEVERAGED_MAP.values()) + [DEFAULT_DEFENSE]):
                            reason = "动量席位让位于当前领涨标的，建议止盈换仓"
                        else:
                            reason = "非策略标的池股票，建议手动平仓换仓"
                    else:
                        reason = "止盈或超额仓位再平衡"

                    sell_orders.append(TradeOrder(
                        ticker=code,
                        action="SELL",
                        shares=int(delta_s),
                        price=sell_limit_p,
                        est_amount=est_amt,
                        reason=reason,
                        target_weight=targ_w,
                    ))
            elif target_shares > cur_shares:
                # 买入
                delta_s = int(round(target_shares - cur_shares))
                if delta_s >= 1:
                    buy_limit_p = self.subscriber.state.get_order_limit_price(code, is_buy=True, fallback_price=ref_p)
                    est_amt = delta_s * buy_limit_p
                    if cur_shares == 0:
                        if code in ('US.TQQQ', 'US.QLD'):
                            reason = "强牛市环境配置核心多头杠杆标的"
                        else:
                            reason = "入选月度高动量领涨行业 (排名前列)"
                    else:
                        reason = "按目标权重加仓"

                    buy_orders.append(TradeOrder(
                        ticker=code,
                        action="BUY",
                        shares=int(delta_s),
                        price=buy_limit_p,
                        est_amount=est_amt,
                        reason=reason,
                        target_weight=targ_w,
                    ))

        # 资金安全校验：防止买单因限价上浮或规费产生现金透支
        est_sell_cash = sum(o.shares * o.price for o in sell_orders)
        max_buy_budget = max(0.0, cash + est_sell_cash * 0.995 - 50.0)  # 预留 $50 规费/摩擦缓冲垫
        total_buy_est = sum(o.est_amount for o in buy_orders)

        if total_buy_est > max_buy_budget and total_buy_est > 0:
            scale = max_buy_budget / total_buy_est
            scaled_buy_orders = []
            for bo in buy_orders:
                new_shares = int(bo.shares * scale)
                if new_shares >= 1:
                    bo.shares = new_shares
                    bo.est_amount = new_shares * bo.price
                    scaled_buy_orders.append(bo)
            buy_orders = scaled_buy_orders

        # 卖单优先，回笼资金后执行买单
        all_orders = sell_orders + buy_orders
        self.last_rebalance_plan = all_orders

        # 归档保存本轮调仓建议，方便后续按日期回顾
        self._archive_rebalance_advice(
            orders=all_orders,
            target_weights=target_weights,
            regime=regime,
            vol_multiplier=self.last_vol_multiplier,
            tot_nav=tot_nav,
            strat_nav=strat_nav,
            cash=cash,
            env=env,
        )

        # 推送调仓通知
        self.notifier.send_rebalance_signal(
            account_val=tot_nav,
            cash=cash,
            orders=all_orders,
            regime=regime,
            vol_multiplier=self.last_vol_multiplier,
            env=env,
        )

        return acc_bal, all_orders, target_weights

    def _archive_rebalance_advice(
        self,
        orders: List[TradeOrder],
        target_weights: Dict[str, float],
        regime: str,
        vol_multiplier: float,
        tot_nav: float,
        strat_nav: float,
        cash: float,
        env: str = "REAL",
    ):
        """保存每轮建议交易快照至 data/advice_history/ 目录，支持按日期回顾。"""
        advice_dir = os.path.join(os.path.dirname(self.records_dir), "advice_history")
        os.makedirs(advice_dir, exist_ok=True)
        now_ts = time.strftime("%Y%m%d_%H%M%S")
        date_str = time.strftime("%Y-%m-%d %H:%M:%S")

        snapshot = {
            "id": f"advice_{now_ts}",
            "timestamp": now_ts,
            "created_at": date_str,
            "env": env,
            "regime": regime,
            "vol_multiplier": round(vol_multiplier, 3),
            "total_nav": round(tot_nav, 2),
            "strategy_nav": round(strat_nav, 2),
            "cash": round(cash, 2),
            "capital_mode": self.capital_mode,
            "satellite_weight_mode": self.satellite_weight_mode,
            "target_weights": {k: round(v, 4) for k, v in target_weights.items()},
            "order_count": len(orders),
            "orders": [
                {
                    "ticker": o.ticker,
                    "action": o.action,
                    "shares": o.shares,
                    "price": round(o.price, 2),
                    "est_amount": round(o.est_amount, 2),
                    "target_weight": round(o.target_weight, 4),
                    "reason": o.reason,
                }
                for o in orders
            ],
        }

        filename = os.path.join(advice_dir, f"advice_{now_ts}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            logger.info(f"[RebalanceManager] 已保存第 {now_ts} 轮调仓建议快照 ({len(orders)} 笔订单)")
        except Exception as e:
            logger.error(f"[RebalanceManager] 保存调仓建议快照失败: {e}")

    def get_advice_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取已归档的历史调仓建议列表 (按时间降序)。"""
        advice_dir = os.path.join(os.path.dirname(self.records_dir), "advice_history")
        if not os.path.exists(advice_dir):
            return []
        files = sorted(
            [f for f in os.listdir(advice_dir) if f.startswith("advice_") and f.endswith(".json")],
            reverse=True
        )[:limit]

        summaries = []
        for fn in files:
            path = os.path.join(advice_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summaries.append(data)
            except Exception:
                pass
        return summaries

    def execute_rebalance_plan(
        self,
        orders: Optional[List[TradeOrder]] = None,
        env: str = "SIMULATE",
    ) -> List[Dict[str, Any]]:
        """报送执行调仓计划。
        
        【实盘安全防御】:
        实盘模式 (env='REAL') 下严禁任何程序自动下单，返回受阻说明并记录日志。
        """
        if str(env).upper() == "REAL":
            err_msg = "【实盘安全锁触发】实盘模式不支持程序化直接报单，请在富途牛牛/Moomoo客户端按调仓清单手动下单。"
            logger.warning(f"[RebalanceManager] 🔒 拦截实盘自动报单执行请求！{err_msg}")
            return [{
                'success': False,
                'error': err_msg,
                'is_real_blocked': True,
            }]

        orders_to_run = orders if orders is not None else self.last_rebalance_plan
        if not orders_to_run:
            logger.info("[RebalanceManager] 无待执行调仓订单")
            return []

        results = []
        for o in orders_to_run:
            res = self.trader.place_order(
                code=o.ticker,
                action=o.action,
                shares=o.shares,
                price=o.price,
                order_type="NORMAL",
                env="SIMULATE",
            )
            res['target_weight'] = o.target_weight
            results.append(res)
            time.sleep(0.05)  # 礼貌间隔

        # 归档本次调仓记录
        self._archive_execution(results)

        # 发送执行结果回报 (容错保护，通知发送即使遇到网络异常也不阻断报单成功返回)
        all_success = all(r.get('success', False) for r in results)
        try:
            self.notifier.send_execution_report(results, success=all_success, env=env)
        except Exception as e:
            logger.warning(f"[RebalanceManager] 发送调仓执行回报通知失败 (报单已成功): {e}")

        return results

    def _archive_execution(self, results: List[Dict[str, Any]]):
        """将成交执行记录持久化至本地 JSON 与 CSV。"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_json = os.path.join(self.records_dir, f"rebalance_{timestamp}.json")
        try:
            with open(filename_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            df_rec = pd.DataFrame(results)
            filename_csv = os.path.join(self.records_dir, "all_trade_history.csv")
            if os.path.exists(filename_csv):
                df_rec.to_csv(filename_csv, mode="a", header=False, index=False)
            else:
                df_rec.to_csv(filename_csv, index=False)

            logger.info(f"[RebalanceManager] 调仓执行明细已归档至 {filename_json}")
        except Exception as e:
            logger.error(f"[RebalanceManager] 归档执行明细异常: {e}")
