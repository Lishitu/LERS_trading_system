# 核心-卫星杠杆择时量化交易系统（LERS）技术文档

---

## 1. 系统架构与数学模型

本系统采用**“核心底仓（Core）+ 卫星动量轮动（Satellite）+ 组合波动率动态目标化（Volatility Targeting）”**三位一体架构。旨在利用美股纳斯达克长期向上的科技红利，同时通过多维宏观状态识别与动量轮动控制下行回撤。

```
                          ┌────────────────────────┐
                          │   美股行情与估值数据源    │
                          │   (Moomoo OpenD API)   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    特征工程与宏观状态检测器       │
                      │  (MacroRegimeDetector & FE)    │
                      └───────┬────────────────┬───────┘
                              │                │
            ┌─────────────────┴─┐            ┌─┴────────────────┐
            │   核心底仓决策引擎  │            │  卫星轮动决策引擎 │
            │  (Core Allocation)│            │(Satellite Engine)│
            │  QQQ / QLD / TQQQ │            │ 动量+估值+宏观倾斜 │
            └─────────────────┬─┘            └─┬────────────────┘
                              │                │
                              └───────┬────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │    组合目标波动率动态缩放      │
                      │ (Target Volatility Allocator)  │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    双轨执行与物理风控隔离层     │
                      │   - 模拟盘 (SIMULATE): 自动下单 │
                      │   - 实盘 (REAL): 物理硬拦截锁   │
                      │     └ 卡片推送 / Web 确认     │
                      └────────────────────────────────┘
```

---

### 1.1 宏观状态检测数学模型（Macro Regime Detection）

系统每日收盘后评估基准指数（`US.QQQ`）及其相对大宗商品等权指数的强弱关系：

1. **基准技术指标**：
   $$\text{MA}_{50}(t) = \frac{1}{50}\sum_{i=0}^{49} P_{qqq}(t-i), \quad \text{MA}_{200}(t) = \frac{1}{200}\sum_{i=0}^{199} P_{qqq}(t-i)$$
   $$\sigma_{20}(t) = \sqrt{252} \times \text{Std}\Big(\{R_{qqq}(t-i)\}_{i=0}^{19}\Big)$$

2. **商品等权动量**：
   选取大宗商品与能源篮子（`US.XLE`, `US.XOP`, `US.COPX`, `US.URA`）：
   $$R_{comm, 60}(t) = \frac{I_{comm}(t)}{I_{comm}(t-60)} - 1$$

3. **四类状态判定转移矩阵**：
   - **科技成长牛市 (`BULL_GROWTH`)**：
     $$P_{qqq} > \text{MA}_{50} \land P_{qqq} > \text{MA}_{200} \land \sigma_{20} \le 28\% \land \neg \text{CommodityDominating}$$
   - **通胀大宗商品周期 (`INFLATION_COMMODITY`)**：
     $$R_{comm, 60} > (R_{qqq, 60} + 4\%) \land I_{comm} > \text{MA}_{50}(I_{comm}) \land R_{comm, 60} > 2\%$$
   - **危机与深度破位熊市 (`CRISIS_BEAR`)**：
     $$(P_{qqq} \le \text{MA}_{200} \land \neg \text{CommodityDominating}) \lor (\sigma_{20} > 32\%) \lor (R_{qqq, 20} < -8\% \land \sigma_{20} > 28\%)$$
   - **震荡市 (`NEUTRAL_CHOP`)**：不满足上述条件的其余过渡状态。

---

### 1.2 核心底仓决策逻辑（Core Allocation）

基准分配额度 $w_{core} = 50\%$：
- **牛市平稳期 ($\sigma_{20} < 16\%$)**：配置 **3x 杠杆 TQQQ**，享受低波动主升浪进攻；
- **牛市标准期 ($\sigma_{20} \ge 16\%$)**：配置 **2x 杠杆 QLD**，兼顾高弹性与抗波动损耗；
- **早期反弹期 ($P_{qqq} > \text{MA}_{50} \lor P_{qqq} > \text{MA}_{200}$)**：配置 **1x 现货 QQQ**，防止踏空反弹初段；
- **破位熊市 ($P_{qqq} \le \text{MA}_{200} \land P_{qqq} \le \text{MA}_{50}$)**：100% 切换至超短债 **US.BIL** 防守。

---

### 1.3 卫星动量与估值打分模型（Satellite Engine）

基准分配额度 $w_{sat} = 50\%$，持有前 $N=2$ 只标的，单券上限 $w_{max} \le 30\%$。

1. **复合动量评分 (Multi-Horizon Risk-Adjusted Momentum)**：
   $$S_{mom}(i, t) = \frac{0.3 \times R_{10}(i, t) + 0.4 \times R_{20}(i, t) + 0.3 \times R_{60}(i, t)}{\max(\sigma_{20}(i, t), 0.20)}$$
2. **动量加速度 (Acceleration)**：
   $$S_{accel}(i, t) = \frac{R_{5}(i, t) - 0.25 \times R_{20}(i, t)}{\max(\sigma_{20}(i, t), 0.20)}$$
3. **Moomoo 估值偏离度评分 (Valuation Score)**：
   提取底层权重股历史市盈率（P/E），计算滚动 252 日 Z-Score：
   $$Z_{pe}(i, t) = \text{clip}\left(\frac{\text{PE}(i, t) - \mu_{252}(\text{PE})}{\sigma_{252}(\text{PE})}, -3.0, 3.0\right), \quad S_{val}(i, t) = -1.0 \times Z_{pe}(i, t)$$
4. **宏观倾向加分 (Macro Tilt)**：
   - 若 `BULL_GROWTH`：科技与半导体标的加分 $+0.35$；
   - 若 `INFLATION_COMMODITY`：能源大宗标的加分 $+0.40$。
5. **综合打分与趋势过滤**：
   $$S_{composite} = S_{mom} + 0.20 \times S_{val} + \text{Tilt}$$
   **硬约束**：仅当标的收盘价大于自身 $\text{MA}_{60}$ 时方可入选，否则一票否决；不足 $N$ 个席位时，剩余资金全部退守 `US.BIL`。
6. **调仓迟滞缓冲（Hysteresis Exit）**：
   已持有标的只有跌出前 $K=3$ 名时才触发卖出换仓，降低周度调仓摩擦。

---

### 1.4 组合波动率动态目标化（Volatility Targeting）

组合基于最近 20 个交易日已实现年化波动率 $\sigma_{port}$，自适应计算总体仓位缩放乘数：
$$M_{vol}(t) = \text{clip}\left(\frac{\sigma_{target}}{\max(\sigma_{port}(t), 0.10)}, L_{min}, L_{max}\right)$$
- 默认参数：$\sigma_{target} = 25\%$，$L_{min} = 0.60$，$L_{max} = 1.80$。
- **最终资产目标权重**：
  $$W_{final}(i, t) = W_{target}(i, t) \times M_{vol}(t)$$
  若总权益权重 $\sum W_{final} < 1.0$，差额作为现金沉淀或申购 `US.BIL`。

---

## 2. 无未来函数与回测撮合机制

```
  T-1 日收盘 (16:00 EST)
    ├─ 获取 T-1 收盘价、计算均线、动量打分、宏观状态
    ├─ 生成 T 日目标持仓与权重 Target_W
    └─ 输出调仓差额报告 (Rebalance Diff)
  -------------------------------------------------------------
  T 日开盘至收盘 (09:30 ~ 16:00 EST)
    ├─ 采用 T 日盘中成交量加权均价 (VWAP) 撮合成交
    ├─ 优先卖出破位与超配资产，释放可用资金
    ├─ 扣除真实 Moomoo 佣金与 SEC/FINRA 监管费
    └─ 扣除 0.06 (6%) 的调仓死区缓冲区 (Buffer Zone)，抑制微量调仓
```

- **数据隔离**：特征矩阵计算严格向后滑动，无居中窗口，无全局标准化；
- **标签隔离**：机器学习 Meta-labeling 引入 10 日三重屏障冷却期（Purge & Embargo），切断跨期泄露。

---

## 3. 实盘对接与物理安全硬锁设计

```
                    ┌─────────────────────────┐
                    │  RebalanceManager 调仓引擎 │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 环境校验: TrdEnv.REAL? │
                     └───────┬───────┬───────┘
                             │       │
                   [是: 实盘] │       │ [否: 模拟盘]
                             │       │
                             ▼       ▼
         ┌─────────────────────────┐   ┌─────────────────────────┐
         │     物理硬安全锁拦截     │   │      OpenD 自动撮合成交  │
         │  (RealTradeSafetyError) │   │ (trader.place_order...) │
         └───────────┬─────────────┘   └─────────────────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │ 1. 结构化日志写入 logs/  │
         │ 2. 飞书交互式卡片 Webhook│
         │ 3. Web 看板调仓清单呈现 │
         │ 4. 用户手动核对后执行    │
         └─────────────────────────┘
```

### 3.1 资产币种自适应归一化
富途/Moomoo OpenD 接口默认以港币（HKD）汇总多币种账户。为确保美元资产核算准确：
1. `MoomooTrader.get_account_funds()` 显式指定 `currency='USD'`；
2. 内部优先提取 `usd_assets` 与 `us_cash` 字段；
3. 彻底杜绝实盘中 \$12,500 美元总资产被误识别为 9.8 万港币导致的计算偏差。

---

## 4. 接口与服务规范

### 4.1 核心 Python 模块说明

| 文件路径 | 模块名称 | 核心职责 |
| :--- | :--- | :--- |
| `src/strategy_coresatellite.py` | 策略核心算法 | 宏观状态机、核心底仓切换、卫星轮动打分、波动率缩放 |
| `src/rebalance_manager.py` | 调仓差异引擎 | 抓取当前持仓、计算买卖股数、生成换仓理由、格式化消息 |
| `src/trade_moomoo.py` | 交易网关接口 | 封装 Moomoo OpenD 连接、多币种资金持仓查询、安全锁拦截 |
| `src/data_moomoo.py` | 数据管理模块 | 历史 K 线离线缓存（Parquet）、估值与 K 线增量同步 |
| `src/notifier.py` | 消息通知引擎 | 飞书 Webhook 交互富文本卡片格式化与异步投递 |
| `src/logger.py` | 统一日志系统 | 结构化滚动文件日志（`logs/lers_*.log`）与控制台双向输出 |
| `web/app.py` | 监控看板后端 | Flask Web 服务，提供 REST API、图表数据与调仓交互 |

### 4.2 Webhook 飞书卡片数据规范

```json
{
  "msg_type": "interactive",
  "card": {
    "header": {
      "title": {"tag": "plain_text", "content": "🎯 LERS 核心-卫星量化调仓建议 [实盘]"},
      "template": "blue"
    },
    "elements": [
      {
        "tag": "div",
        "text": {"tag": "lark_md", "content": "**宏观状态**: 🟢 BULL_GROWTH | **总资产**: $12,592.03"}
      },
      {
        "tag": "table",
        "columns": [
          {"name": "action", "display_name": "操作"},
          {"name": "symbol", "display_name": "标的"},
          {"name": "shares", "display_name": "股数"},
          {"name": "reason", "display_name": "换仓理由"}
        ]
      },
      {
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": "🔒 实盘物理安全锁已激活：本系统不直接下单，请核对后手动执行。"}]
      }
    ]
  }
}
```

---

## 5. 测试与质量保障（Test Suite）

系统配备完整的自动化单元测试与集成测试（Pytest），涵盖 29 项测试用例：
```bash
pytest -v
```
- `tests/test_coresatellite.py`：测试状态机切换、单券上限、波动率缩放边界；
- `tests/test_data_loader.py`：测试 Parquet 离线缓存、复权处理、时间对齐；
- `tests/test_live_trading.py`：**重点验证实盘物理安全锁对直接下单的 100% 拦截率**；
- `tests/test_web_api.py`：测试 Flask 看板所有 REST API 路由与参数安全性。
