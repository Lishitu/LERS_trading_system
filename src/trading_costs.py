"""真实交易成本计算模块。

模拟券商真实费用结构，包括：
- 佣金：0.0049美元/股，最低0.99美元，最高0.5%成交金额
- 平台使用费：0.005美元/股，最低1美元，最高0.5%成交金额
- 交收费：0.003美元/股
- 证监会规费（仅卖出）：0.0000206×成交金额，最低0.01美元
- 交易活动费（仅卖出）：0.000195美元/股，最低0.01美元，最高9.79美元

融资成本不计入（杠杆ETF已内置）。
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TradingCostBreakdown:
    """交易成本明细"""
    commission: float = 0.0
    platform_fee: float = 0.0
    clearing_fee: float = 0.0
    sec_fee: float = 0.0  # 仅卖出
    ta_fee: float = 0.0  # 仅卖出
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'commission': self.commission,
            'platform_fee': self.platform_fee,
            'clearing_fee': self.clearing_fee,
            'sec_fee': self.sec_fee,
            'ta_fee': self.ta_fee,
            'total': self.total,
        }


class TradingCostCalculator:
    """真实交易成本计算器。

    根据券商真实费用结构计算每笔交易的成本。
    支持固定股数模式，方便计算按股计费的成本。

    Attributes:
        shares_per_trade: 每笔交易的固定股数
        commission_per_share: 佣金费率（美元/股）
        commission_min: 佣金最低收费
        commission_max_pct: 佣金最高比例（成交金额的百分比）
        platform_fee_per_share: 平台使用费费率（美元/股）
        platform_fee_min: 平台使用费最低收费
        platform_fee_max_pct: 平台使用费最高比例
        clearing_fee_per_share: 交收费费率（美元/股）
        sec_fee_pct: 证监会规费比例
        sec_fee_min: 证监会规费最低收费
        ta_fee_per_share: 交易活动费费率（美元/股）
        ta_fee_min: 交易活动费最低收费
        ta_fee_max: 交易活动费最高收费
    """

    # 券商费用参数（可根据实际券商调整）
    COMMISSION_PER_SHARE = 0.0049
    COMMISSION_MIN = 0.99
    COMMISSION_MAX_PCT = 0.005  # 0.5%

    PLATFORM_FEE_PER_SHARE = 0.005
    PLATFORM_FEE_MIN = 1.0
    PLATFORM_FEE_MAX_PCT = 0.005  # 0.5%

    CLEARING_FEE_PER_SHARE = 0.003

    SEC_FEE_PCT = 0.0000206
    SEC_FEE_MIN = 0.01

    TA_FEE_PER_SHARE = 0.000195
    TA_FEE_MIN = 0.01
    TA_FEE_MAX = 9.79

    def __init__(self, shares_per_trade: int = 100):
        """初始化交易成本计算器。

        Args:
            shares_per_trade: 每笔交易的固定股数，默认100股
        """
        self.shares_per_trade = shares_per_trade

    def calc_buy_cost(self, price: float) -> TradingCostBreakdown:
        """计算买入成本。

        买入费用包括：佣金、平台使用费、交收费

        Args:
            price: 买入价格

        Returns:
            TradingCostBreakdown 包含各项费用明细
        """
        shares = self.shares_per_trade
        amount = shares * price

        # 佣金: max(0.0049*shares, 0.99), 不超过0.5%成交金额
        commission = min(
            max(self.COMMISSION_PER_SHARE * shares, self.COMMISSION_MIN),
            self.COMMISSION_MAX_PCT * amount
        )

        # 平台使用费: max(0.005*shares, 1), 不超过0.5%成交金额
        platform_fee = min(
            max(self.PLATFORM_FEE_PER_SHARE * shares, self.PLATFORM_FEE_MIN),
            self.PLATFORM_FEE_MAX_PCT * amount
        )

        # 交收费: 0.003*shares
        clearing_fee = self.CLEARING_FEE_PER_SHARE * shares

        total = commission + platform_fee + clearing_fee

        return TradingCostBreakdown(
            commission=commission,
            platform_fee=platform_fee,
            clearing_fee=clearing_fee,
            total=total,
        )

    def calc_sell_cost(self, price: float) -> TradingCostBreakdown:
        """计算卖出成本。

        卖出费用包括：佣金、平台使用费、交收费、证监会规费、交易活动费

        Args:
            price: 卖出价格

        Returns:
            TradingCostBreakdown 包含各项费用明细
        """
        shares = self.shares_per_trade
        amount = shares * price

        # 买入时的三项费用
        commission = min(
            max(self.COMMISSION_PER_SHARE * shares, self.COMMISSION_MIN),
            self.COMMISSION_MAX_PCT * amount
        )
        platform_fee = min(
            max(self.PLATFORM_FEE_PER_SHARE * shares, self.PLATFORM_FEE_MIN),
            self.PLATFORM_FEE_MAX_PCT * amount
        )
        clearing_fee = self.CLEARING_FEE_PER_SHARE * shares

        # 证监会规费（仅卖出）: max(0.0000206*amount, 0.01)
        sec_fee = max(self.SEC_FEE_PCT * amount, self.SEC_FEE_MIN)

        # 交易活动费（仅卖出）: max(min(0.000195*shares, 9.79), 0.01)
        ta_fee = max(
            min(self.TA_FEE_PER_SHARE * shares, self.TA_FEE_MAX),
            self.TA_FEE_MIN
        )

        total = commission + platform_fee + clearing_fee + sec_fee + ta_fee

        return TradingCostBreakdown(
            commission=commission,
            platform_fee=platform_fee,
            clearing_fee=clearing_fee,
            sec_fee=sec_fee,
            ta_fee=ta_fee,
            total=total,
        )

    def calc_trade_cost(self, shares: float, price: float, is_buy: bool = True) -> TradingCostBreakdown:
        """根据动态股数与成交价格计算真实交易成本。

        Args:
            shares: 交易股数
            price: 成交价格
            is_buy: 是否买入 (True 买入, False 卖出)

        Returns:
            TradingCostBreakdown 费用明细
        """
        if shares <= 0 or price <= 0:
            return TradingCostBreakdown()

        amount = shares * price

        commission = min(
            max(self.COMMISSION_PER_SHARE * shares, self.COMMISSION_MIN),
            self.COMMISSION_MAX_PCT * amount
        )
        platform_fee = min(
            max(self.PLATFORM_FEE_PER_SHARE * shares, self.PLATFORM_FEE_MIN),
            self.PLATFORM_FEE_MAX_PCT * amount
        )
        clearing_fee = self.CLEARING_FEE_PER_SHARE * shares

        if is_buy:
            total = commission + platform_fee + clearing_fee
            return TradingCostBreakdown(
                commission=commission,
                platform_fee=platform_fee,
                clearing_fee=clearing_fee,
                sec_fee=0.0,
                ta_fee=0.0,
                total=total,
            )
        else:
            sec_fee = max(self.SEC_FEE_PCT * amount, self.SEC_FEE_MIN)
            ta_fee = max(
                min(self.TA_FEE_PER_SHARE * shares, self.TA_FEE_MAX),
                self.TA_FEE_MIN
            )
            total = commission + platform_fee + clearing_fee + sec_fee + ta_fee
            return TradingCostBreakdown(
                commission=commission,
                platform_fee=platform_fee,
                clearing_fee=clearing_fee,
                sec_fee=sec_fee,
                ta_fee=ta_fee,
                total=total,
            )

    def calc_turnover_cost(self, buy_price: float, sell_price: float) -> float:
        """计算完整换手成本（买入+卖出）。

        Args:
            buy_price: 买入价格
            sell_price: 卖出价格

        Returns:
            总交易成本（美元）
        """
        buy_cost = self.calc_buy_cost(buy_price)
        sell_cost = self.calc_sell_cost(sell_price)
        return buy_cost.total + sell_cost.total

    def get_cost_as_return_pct(self, price: float, is_sell: bool = False) -> float:
        """计算成本占成交金额的比例（用于收益率扣除）。

        Args:
            price: 交易价格
            is_sell: 是否为卖出

        Returns:
            成本占成交金额的比例
        """
        shares = self.shares_per_trade
        amount = shares * price

        if is_sell:
            cost = self.calc_sell_cost(price)
        else:
            cost = self.calc_buy_cost(price)

        return cost.total / amount if amount > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """获取费用参数摘要。

        Returns:
            费用参数字典
        """
        return {
            'shares_per_trade': self.shares_per_trade,
            'commission': {
                'per_share': self.COMMISSION_PER_SHARE,
                'min': self.COMMISSION_MIN,
                'max_pct': self.COMMISSION_MAX_PCT,
            },
            'platform_fee': {
                'per_share': self.PLATFORM_FEE_PER_SHARE,
                'min': self.PLATFORM_FEE_MIN,
                'max_pct': self.PLATFORM_FEE_MAX_PCT,
            },
            'clearing_fee': {
                'per_share': self.CLEARING_FEE_PER_SHARE,
            },
            'sec_fee': {
                'pct': self.SEC_FEE_PCT,
                'min': self.SEC_FEE_MIN,
            },
            'ta_fee': {
                'per_share': self.TA_FEE_PER_SHARE,
                'min': self.TA_FEE_MIN,
                'max': self.TA_FEE_MAX,
            },
        }


def print_cost_example(shares: int = 100, price: float = 50.0):
    """打印成本计算示例。

    Args:
        shares: 股数
        price: 价格
    """
    calc = TradingCostCalculator(shares_per_trade=shares)

    print(f"\n交易成本示例（{shares}股，价格${price:.2f}）")
    print("=" * 50)

    buy_cost = calc.calc_buy_cost(price)
    print(f"\n买入成本:")
    print(f"  佣金:       ${buy_cost.commission:.2f}")
    print(f"  平台使用费: ${buy_cost.platform_fee:.2f}")
    print(f"  交收费:     ${buy_cost.clearing_fee:.2f}")
    print(f"  合计:       ${buy_cost.total:.2f}")
    print(f"  占成交金额: {buy_cost.total / (shares * price) * 100:.2f}%")

    sell_cost = calc.calc_sell_cost(price)
    print(f"\n卖出成本:")
    print(f"  佣金:       ${sell_cost.commission:.2f}")
    print(f"  平台使用费: ${sell_cost.platform_fee:.2f}")
    print(f"  交收费:     ${sell_cost.clearing_fee:.2f}")
    print(f"  证监会规费: ${sell_cost.sec_fee:.2f}")
    print(f"  交易活动费: ${sell_cost.ta_fee:.2f}")
    print(f"  合计:       ${sell_cost.total:.2f}")
    print(f"  占成交金额: {sell_cost.total / (shares * price) * 100:.2f}%")

    total_turnover = buy_cost.total + sell_cost.total
    print(f"\n完整换手成本: ${total_turnover:.2f}")
    print(f"占成交金额:   {total_turnover / (shares * price) * 100:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    # 测试不同价格下的成本
    print_cost_example(shares=100, price=50.0)
    print_cost_example(shares=100, price=20.0)
    print_cost_example(shares=100, price=100.0)