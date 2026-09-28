# -*- coding: utf-8 -*-
"""LERS 核心-卫星杠杆策略统一命令行管理工具 (CLI Entrypoint).

支持命令示例:
    # 1. 运行策略并查看最新调仓信号 (默认实盘模式，支持推送到飞书)
    python cli.py signal
    python cli.py signal --real --push
    python cli.py signal --simulate

    # 2. 查看当前 Moomoo 账户资金与持仓
    python cli.py account
    python cli.py account --real

    # 3. 启动飞书长连接 (WebSocket) 后台守护进程
    python cli.py daemon

    # 4. 诊断连通性 (Moomoo OpenD / 飞书 OpenAPI)
    python cli.py test feishu
    python cli.py test moomoo

    # 5. 运行宏观压力测试套件
    python cli.py stress-test
"""

import sys
import os
import argparse

# 兼容 Windows 控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = os.path.dirname(os.path.abspath(__file__))
# 确保 src 在导入路径中
sys.path.insert(0, os.path.join(base_dir, "src"))


def cmd_signal(args):
    """生成并展示调仓信号。"""
    from trade_moomoo import MoomooTrader
    from market_subscriber import MoomooMarketSubscriber
    from rebalance_manager import RebalanceManager
    from notifier import NotificationService

    target_env = "SIMULATE" if getattr(args, "simulate", False) else "REAL"
    env_label = "【美股实盘·只读建议模式】" if target_env == "REAL" else "【模拟盘·演练模式】"

    print("=" * 85)
    print(f"   [LERS] 核心-卫星量化策略 · 调仓信号引擎 {env_label}")
    print("=" * 85)

    if target_env == "REAL":
        print("🔒 [实盘只读安全锁已生效]: 仅读取真实资产并输出调仓建议，已物理阻断程序直接下单。")

    trader = MoomooTrader(host=args.host, port=args.port, is_simulate=(target_env == "SIMULATE"))
    subscriber = MoomooMarketSubscriber(host=args.host, port=args.port)
    notifier = NotificationService()

    mgr = RebalanceManager(
        trader=trader,
        subscriber=subscriber,
        notifier=notifier,
        buffer=args.buffer,
    )

    acc_bal, orders, target_w = mgr.generate_rebalance_plan(env=target_env)

    print(f"\n宏观市场状态:       【{mgr.last_regime}】")
    print(f"动态波动率缩放乘数: 【{mgr.last_vol_multiplier:.2f}x】")
    print(f"账户总资产 (NAV):   ${acc_bal.total_assets:,.2f}")
    print(f"账户可用现金:       ${acc_bal.cash:,.2f}")

    print("\n策略最新目标资产权重分布:")
    for ticker, w in sorted(target_w.items(), key=lambda x: x[1], reverse=True):
        print(f"  * {ticker:<12}: {w*100:>5.1f}%")

    if not orders:
        print("\n" + "=" * 85)
        print(f"[OK] 当前持仓与目标仓位完全吻合（或在 {args.buffer*100:.0f}% 缓冲阈值内），无需调仓。")
        print("=" * 85)
    else:
        print("\n" + "=" * 85)
        print(f"          [本次计划调仓订单清单 ({target_env})] (共 {len(orders)} 笔建议订单)")
        print("=" * 85)
        print(f"{'标的代码':<12} {'方向':^8} {'计划股数':>8} {'参考限价':>10} {'预估金额':>12} {'目标权重':>8} {'调仓原因'}")
        print("-" * 85)
        total_est = 0.0
        for o in orders:
            side = "🟢 [BUY]" if o.action == "BUY" else "🔴 [SELL]"
            print(f"{o.ticker:<12} {side:^8} {o.shares:>8.0f} ${o.price:>9.2f} ${o.est_amount:>11.2f} {o.target_weight*100:>7.1f}%  {o.reason}")
            total_est += o.est_amount
        print("-" * 85)
        print(f"总意向交易金额: ${total_est:,.2f}")
        print("=" * 85)

        if getattr(args, "push", False):
            print("\n[Feishu] 正在推送最新互动卡片到飞书...")
            notifier.send_rebalance_signal(
                account_val=acc_bal.total_assets,
                cash=acc_bal.cash,
                orders=orders,
                regime=mgr.last_regime,
                vol_multiplier=mgr.last_vol_multiplier,
                env=target_env,
            )
            print("[Feishu] 已成功投递至飞书群聊！")

    trader.close()
    subscriber.close()


def cmd_account(args):
    """查询并展示当前账户持仓与资产概况。"""
    from trade_moomoo import MoomooTrader

    target_env = "SIMULATE" if getattr(args, "simulate", False) else "REAL"
    env_label = "美股实盘账户" if target_env == "REAL" else "模拟交易账户"

    print("=" * 75)
    print(f"   [LERS] Moomoo {env_label} 资金与资产概况")
    print("=" * 75)

    trader = MoomooTrader(host=args.host, port=args.port, is_simulate=(target_env == "SIMULATE"))
    bal = trader.get_account_balance(env=target_env)
    positions = trader.get_positions(env=target_env)

    print(f"总资产 (Total NAV):   ${bal.total_assets:,.2f}")
    print(f"可用现金 (Cash):      ${bal.cash:,.2f}")
    print(f"证券持仓市值:         ${bal.market_val:,.2f}")
    print(f"购买力 (BuyingPower): ${bal.buying_power:,.2f}")

    if positions:
        print(f"\n当前持仓明细 (共 {len(positions)} 支):")
        print(f"{'标的代码':<12} {'持股数':>8} {'成本价':>10} {'现价':>10} {'持仓市值':>12} {'浮动盈亏':>10}")
        print("-" * 75)
        for code, pos in positions.items():
            print(f"{code:<12} {pos.shares:>8.0f} ${pos.cost_price:>9.2f} ${pos.current_price:>9.2f} ${pos.market_val:>11.2f} ${pos.unrealized_pnl:>+9.2f}")
        print("-" * 75)
    else:
        print("\n当前账户无任何股票/ETF持仓（全现金状态）。")
    print("=" * 75)
    trader.close()


def cmd_daemon(args):
    """启动飞书 WebSocket 守护进程。"""
    from feishu_bot import FeishuBotService

    print("=" * 75)
    print("   [LERS] 正在启动飞书 WebSocket 长连接交互服务 (TRPG 架构)...")
    print("=" * 75)
    bot = FeishuBotService()
    if getattr(args, "push", False):
        tid = bot.chat_id or bot.user_id
        if tid:
            print(f"正在向已绑定的目标 {tid} 推送初始卡片...")
            bot.send_message_card(tid, bot.build_rebalance_card())
    print("飞书 WebSocket 监听中，按 Ctrl+C 可停止...")
    bot.start_websocket()


def cmd_test(args):
    """执行连通性诊断。"""
    from notifier import NotificationService
    from trade_moomoo import MoomooTrader

    target = getattr(args, "target", "all").lower()

    if target in ("feishu", "all"):
        print("\n--- [1] 飞书 OpenAPI 诊断 ---")
        notifier = NotificationService()
        diag = notifier.test_feishu_connection()
        print(f"  * 凭据配置状态:   {'[OK] 已配置' if diag['configured'] else '[FAIL] 未配置'}")
        print(f"  * App ID:         {diag.get('app_id')}")
        print(f"  * Token 鉴权状态: {'[OK] 正常' if diag['token_ok'] else '[FAIL] 鉴权失败'}")
        bot_info = diag.get("bot_info") or {}
        print(f"  * 机器人名称:     {bot_info.get('app_name', '未获取')}")
        print(f"  * 机器人状态:     {bot_info.get('activate_status', '未知')}")
        chats = diag.get("joined_chats", [])
        print(f"  * 已加入群聊数:   {len(chats)} 个")
        for c in chats:
            print(f"     - {c.get('name')} (ID: {c.get('chat_id')})")

    if target in ("moomoo", "all"):
        print("\n--- [2] Moomoo OpenD 诊断 ---")
        try:
            trader = MoomooTrader(host=args.host, port=args.port, is_simulate=False)
            if trader.is_live_connected:
                bal = trader.get_account_balance(env="REAL")
                print(f"  * OpenD 连接状态: [OK] 正常连接 (Host: {args.host}:{args.port})")
                print(f"  * 实盘账户总资产: ${bal.total_assets:,.2f}")
                print(f"  * 实盘可用资金:   ${bal.cash:,.2f}")
            else:
                print(f"  * OpenD 连接状态: [WARN] 未连接到本地 OpenD，已降级至 Mock 模拟器")
            trader.close()
        except Exception as e:
            print(f"  * OpenD 连接失败: {e}")


def cmd_stress_test(args):
    """运行极端压力测试套件。"""
    print("=" * 75)
    print("   [LERS] 正在执行 2018-2026 历史极端危机与多维压力测试...")
    print("=" * 75)
    from stress_test import run_stress_test_suite
    report = run_stress_test_suite()
    print(report)


def cmd_version(args):
    """查看或切换实盘策略固化版本。"""
    import json
    cfg_path = os.path.join(base_dir, 'config', 'config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    strat = cfg.get('strategy', {})
    active = strat.get('active_version', 'v1.0-production')
    versions = strat.get('versions', {})

    if args.switch:
        if args.switch not in versions:
            print(f"[ERROR] 未知策略版本: {args.switch}。可用版本: {list(versions.keys())}")
            return
        strat['active_version'] = args.switch
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[OK] 已成功将实盘策略版本切换为: {args.switch} ({versions[args.switch].get('name')})")
        return

    print("=" * 75)
    print("       LERS 核心-卫星量化策略 · 实盘固化版本管理 (v1.0.1)")
    print("=" * 75)
    for k, v in versions.items():
        is_cur = " ★ [当前激活]" if k == active else ""
        print(f"\n* 版本代码: {k}{is_cur}")
        print(f"  版本名称: {v.get('name')}")
        print(f"  策略定位: {v.get('tag')}")
        print(f"  核心描述: {v.get('description')}")
        print(f"  配置参数: 核心 {v.get('core_ratio')*100:.0f}% | 卫星 {v.get('satellite_ratio')*100:.0f}% | 目标波 {v.get('target_vol')*100:.0f}% | 杠杆 [{v.get('min_leverage')}x, {v.get('max_leverage')}x] | 调仓缓冲 {v.get('rebalance_buffer')*100:.0f}%")
        print(f"  全期表现: CAGR {v.get('cagr')}% | Sharpe {v.get('sharpe')} | MaxDD {v.get('max_dd')}%")
    print("\n" + "=" * 75)
    print("切换版本指令: python cli.py version --switch <版本代码>")
    print("示例: python cli.py version --switch v1.0-aggressive")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lers",
        description="LERS 核心-卫星杠杆择时策略统一命令行工具 (CLI)",
        epilog="示例: python cli.py signal --real --push"
    )
    parser.add_argument("--version", action="version", version="LERS Quantitative Trading System v1.0.1")
    parser.add_argument("--host", default="127.0.0.1", help="Moomoo OpenD IP (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11111, help="Moomoo OpenD 端口 (默认 11111)")

    subparsers = parser.add_subparsers(dest="command", help="子命令列表")

    # 1. signal
    p_signal = subparsers.add_parser("signal", aliases=["rebalance"], help="运行策略并输出最新调仓信号")
    p_signal.add_argument("--real", action="store_true", default=True, help="美股实盘模式 (默认)")
    p_signal.add_argument("--simulate", action="store_true", help="模拟盘模式")
    p_signal.add_argument("--push", action="store_true", help="是否同时推送互动卡片到飞书")
    p_signal.add_argument("--buffer", type=float, default=0.06, help="调仓权重缓冲阈值 (默认 0.06)")
    p_signal.set_defaults(func=cmd_signal)

    # 2. account
    p_acc = subparsers.add_parser("account", aliases=["status"], help="查询当前账户资金与持仓")
    p_acc.add_argument("--real", action="store_true", default=True, help="美股实盘 (默认)")
    p_acc.add_argument("--simulate", action="store_true", help="模拟交易盘")
    p_acc.set_defaults(func=cmd_account)

    # 3. daemon
    p_daemon = subparsers.add_parser("daemon", aliases=["bot"], help="启动飞书长连接 (WebSocket) 交互守护进程")
    p_daemon.add_argument("--push", action="store_true", help="启动时向飞书推送一张最新调仓卡片")
    p_daemon.set_defaults(func=cmd_daemon)

    # 4. test
    p_test = subparsers.add_parser("test", help="测试服务连通性 (feishu / moomoo / all)")
    p_test.add_argument("target", nargs="?", default="all", choices=["feishu", "moomoo", "all"], help="测试目标")
    p_test.set_defaults(func=cmd_test)

    # 5. stress-test
    p_stress = subparsers.add_parser("stress-test", help="运行策略压力测试与危机检验")
    p_stress.set_defaults(func=cmd_stress_test)

    # 6. version
    p_ver = subparsers.add_parser("version", help="查看或切换实盘策略固化版本")
    p_ver.add_argument("--switch", help="切换至目标策略版本代码 (如 v1.0-aggressive, v1.0-conservative, v1.0-production)")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
