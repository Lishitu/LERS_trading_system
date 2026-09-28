"""LERS 策略 Moomoo 模拟盘实时监控与交互调仓命令行工具。

使用方式:
    # 1. 交互式调仓（推荐）：计算信号、输出清晰交易清单，由用户手动键入 Y 确认执行
    python src/live_trader_cli.py

    # 2. 全自动无人值守模式：自动报送限价单至模拟盘
    python src/live_trader_cli.py --auto-trade

    # 3. 试运行模式 (Dry-Run)：仅生成并查看交易清单，不产生任何报单
    python src/live_trader_cli.py --dry-run

    # 4. 离线/模拟器模式：无需 OpenD 连接，使用内建模拟账户演练
    python src/live_trader_cli.py --mock
"""

import argparse
import sys
import os
import time

# 兼容 Windows 控制台 GBK 编码输出
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 确保 src 在模块搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_moomoo import MoomooTrader
from market_subscriber import MoomooMarketSubscriber
from rebalance_manager import RebalanceManager
from notifier import NotificationService


def parse_args():
    parser = argparse.ArgumentParser(description="LERS Moomoo 模拟与实盘调仓监控控制台")
    parser.add_argument("--real", action="store_true", help="实盘模式：读取真实持仓并生成手动调仓清单 (安全锁已生效，严禁程序直接报单)")
    parser.add_argument("--auto-trade", action="store_true", help="启用模拟盘无人值守全自动报单模式")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式，仅展示交易清单不执行报单")
    parser.add_argument("--mock", action="store_true", help="强制使用内建模拟器，无需连接 OpenD")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="OpenD 监听主机")
    parser.add_argument("--port", type=int, default=11111, help="OpenD 监听端口")
    parser.add_argument("--buffer", type=float, default=0.06, help="调仓权重缓冲区，默认 6%%")
    parser.add_argument("--test-feishu", action="store_true", help="测试飞书 OpenAPI / 机器人连接状态")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.test_feishu:
        print("=" * 80)
        print(" [FEISHU] 正在测试飞书自建应用 OpenAPI 连接与机器人状态... ")
        print("=" * 80)
        notifier = NotificationService()
        diag = notifier.test_feishu_connection()
        print(f"  * 凭据配置状态: {'[OK] 已配置' if diag['configured'] else '[FAIL] 未配置'}")
        print(f"  * App ID:       {diag.get('app_id')}")
        print(f"  * Token 鉴权:   {'[OK] 成功获取 tenant_access_token' if diag['token_ok'] else '[FAIL] 获取失败'}")
        bot = diag.get('bot_info') or {}
        if bot:
            status_code = bot.get('activate_status')
            status_desc = "[OK] 正常激活 (Status 2)" if status_code == 2 else f"状态码: {status_code}"
            print(f"  * 机器人名称:   {bot.get('app_name')} (OpenID: {bot.get('open_id')})")
            print(f"  * 激活状态:     {status_desc}")
        else:
            print("  * 机器人信息:   未获取到")
        chats = diag.get('joined_chats', [])
        print(f"  * 已加入群聊数: {len(chats)} 个")
        for c in chats:
            print(f"     - 群名称: {c.get('name')} | ChatID: {c.get('chat_id')}")
        if not chats:
            bot_name = bot.get('app_name', '美股策略助手')
            print(f"  * 提示: 机器人当前尚未被拉入任何群聊。请在飞书群中点击【添加群机器人】搜索并添加【{bot_name}】，系统即可自动捕获群ID并推送卡片。")
        print("=" * 80)
        return

    target_env = "REAL" if args.real else "SIMULATE"
    env_label = "【美股实盘·只读建议模式】" if args.real else "【模拟盘·可报单模式】"

    print("=" * 85)
    print(f"   [LERS] 核心底仓 + 卫星动量轮动系统 · Moomoo 控制台 {env_label}")
    print("=" * 85)

    if args.real:
        print("\n🔒 [实盘安全防御已激活]: 系统将读取您真实账户资产与持仓，计算调仓建议并推送，")
        print("                      但已物理禁用程序直接报单功能，请在 Moomoo 客户端手动下单。")

    print("\n[1/4] 初始化 Moomoo 交易与行情客户端...")
    trader = MoomooTrader(host=args.host, port=args.port, is_simulate=not args.real, force_mock=args.mock)
    subscriber = MoomooMarketSubscriber(host=args.host, port=args.port, force_mock=args.mock)
    notifier = NotificationService()

    conn_status = "已连接 OpenD 交易通道" if trader.is_live_connected else "已启用内建模拟撮合引擎 (Mock)"
    print(f"-> 交易通道状态: {conn_status}")
    print(f"-> 行情订阅状态: {'已实时订阅' if subscriber.is_subscribed else '使用本地行情缓存'}")

    print(f"\n[2/4] 获取 {target_env} 资金与最新持仓概况...")
    bal = trader.get_account_balance(env=target_env)
    positions = trader.get_positions(env=target_env)
    print(f"-> 总资产 (NAV):   ${bal.total_assets:,.2f}")
    print(f"-> 可用现金 (Cash): ${bal.cash:,.2f}")
    print(f"-> 证券持仓市值:   ${bal.market_val:,.2f}")

    if positions:
        print(f"\n当前 {target_env} 账户持仓明细:")
        print(f"{'代码':<12} {'持股数':>8} {'成本价':>10} {'现价':>10} {'市值':>12} {'盈亏':>10}")
        print("-" * 68)
        for code, pos in positions.items():
            print(f"{code:<12} {pos.shares:>8.0f} ${pos.cost_price:>9.2f} ${pos.current_price:>9.2f} ${pos.market_val:>11.2f} ${pos.unrealized_pnl:>+9.2f}")
    else:
        print(f"-> 当前 {target_env} 账户无任何证券持仓（100% 现金就绪）")

    print("\n[3/4] 运行核心-卫星量化策略，生成调仓清单与交易信号...")
    mgr = RebalanceManager(
        trader=trader,
        subscriber=subscriber,
        notifier=notifier,
        buffer=args.buffer,
    )

    acc_bal, orders, target_w = mgr.generate_rebalance_plan(env=target_env)

    print(f"\n当前策略宏观状态: 【{mgr.last_regime}】")
    print(f"已实现波动率动态缩放系数: 【{mgr.last_vol_multiplier:.2f}x】")
    print("\n策略最新目标资产权重分布:")
    for ticker, w in sorted(target_w.items(), key=lambda x: x[1], reverse=True):
        print(f"  * {ticker:<12}: {w*100:>5.1f}%")

    if not orders:
        print("\n" + "=" * 85)
        print(f"[OK] 检查完毕: 当前 {target_env} 持仓与目标仓位完全吻合（或在 6% 缓冲期内），无需调仓。")
        print("=" * 85)
        trader.close()
        subscriber.close()
        return

    print("\n" + "=" * 85)
    print(f"          [待执行调仓交易清单 ({target_env})] (共 {len(orders)} 笔建议订单)")
    print("=" * 85)
    print(f"{'标的代码':<12} {'方向':^8} {'计划股数':>8} {'参考限价':>10} {'预估金额':>12} {'目标权重':>8} {'调仓原因'}")
    print("-" * 85)
    total_est_trade = 0.0
    for o in orders:
        side_sym = "[BUY]" if o.action == "BUY" else "[SELL]"
        print(f"{o.ticker:<12} {side_sym:^8} {o.shares:>8.0f} ${o.price:>9.2f} ${o.est_amount:>11.2f} {o.target_weight*100:>7.1f}%  {o.reason}")
        total_est_trade += o.est_amount
    print("-" * 85)
    print(f"总交易意向金额: ${total_est_trade:,.2f}")
    print("=" * 85)

    # 🔒 如果是实盘模式：严格禁止程序下单，直接提示用户在 App 手动操作
    if args.real:
        print("\n" + "=" * 85)
        print("📢 [🔒 实盘安全锁已生效 - 严禁程序化直接报单]")
        print("已成功将调仓建议同步推送至飞书 / Web 看板。")
        print("请根据上方清单，打开【富途牛牛 / Moomoo 官方客户端】手动报单：")
        print("  1. 建议先卖后买，卖出非核心标的与超额仓位以回笼资金；")
        print("  2. 再挂限价单买入目标加仓与新入选动量 ETF。")
        print("=" * 85)
        trader.close()
        subscriber.close()
        return

    if args.dry_run:
        print("\n[Dry-Run] 试运行模式结束，未向模拟盘发送报单。")
        trader.close()
        subscriber.close()
        return

    # 4. 执行决策 (仅模拟盘支持自动 vs 交互确认)
    should_execute = False
    if args.auto_trade:
        print("\n[Auto-Trade] 处于无人值守全自动模式，立即向模拟账户报单...")
        should_execute = True
    else:
        user_input = input("\n[PROMPT] 是否立即向 Moomoo 模拟账户报送以上调仓订单？[Y/n]: ").strip().lower()
        should_execute = (user_input in ['y', 'yes', ''])

    if should_execute:
        print("\n[4/4] 正在依次报单至模拟盘（卖单优先回笼资金，随后买单）...")
        results = mgr.execute_rebalance_plan(orders, env="SIMULATE")
        print("\n" + "=" * 85)
        print("                         [订单报送执行结果]")
        print("=" * 85)
        for r in results:
            stat = "[成功]" if r.get('success', False) else f"[失败]: {r.get('error', '')}"
            oid = f"(ID: {r.get('order_id', '-')})" if r.get('success', False) else ""
            print(f"  * {r.get('code'):<12} {r.get('action'):^6} {r.get('shares', 0)}股 @ ${r.get('price', 0):.2f} -> {stat} {oid}")
        print("=" * 85)
        print("调仓执行明细已同步持久化归档至: data/trading_records/")
    else:
        print("\n[Cancel] 用户已取消本次调仓报单。")

    trader.close()
    subscriber.close()


if __name__ == "__main__":
    main()

