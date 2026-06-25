# Helios - 美股杠杆 ETF 择时指增与日内 T0 增强系统

> Project: Helios | 核心-卫星（Core-Satellite）架构量化交易系统

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [核心模块详解](#核心模块详解)
- [开发里程碑](#开发里程碑)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [风险管理](#风险管理)
- [技术栈](#技术栈)
- [开发规范](#开发规范)
- [Agent 协作指南](#agent-协作指南)

---

## 项目概述

### 项目定位

Helios 是一个基于核心-卫星（Core-Satellite）架构的美股量化交易系统，目标是在控制杠杆损耗的前提下，通过宏观择时与日内增强实现超额收益。

### 投资理念

- **主策略（Core）**：通过 Quant Agent 进行宏观与价量择时，动态调仓美股高流动性杠杆 ETF（如 TQQQ/SOXL），捕获指数增强 Beta。
- **卫星策略（Satellite）**：利用底仓与日内剩余现金，在低摩擦时段进行日内 T0 统计套利与夜盘网格挂单，钝化持仓成本。

### 核心标的

| 代码 | 名称 | 杠杆倍数 | 跟踪指数 |
|------|------|----------|----------|
| TQQQ | ProShares UltraPro QQQ | 3x 做多 | Nasdaq-100 |
| SOXL | Direxion Daily Semiconductor Bull 3X | 3x 做多 | PHLX Semiconductor |
| UPRO | ProShares UltraPro S&P500 | 3x 做多 | S&P 500 |
| SQQQ | ProShares UltraPro Short QQQ | 3x 做空 | Nasdaq-100 |
| QQQ | Invesco QQQ Trust | 1x | Nasdaq-100 |

---

## 系统架构

系统采用四大分层架构，通过 Docker 环境隔离，Python 异步驱动：

```
┌─────────────────────────────────────────────────────────────┐
│                    1. 数据层 (Data Engine)                   │
│   [yfinance (日线/分钟)]  |  [AkShare (宏观)]  |  [Moomoo L2] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               2. 因子与决策层 (Quant Agent Engine)           │
│   Data-Agent  →  Risk-Agent (损耗/波动)  →  Exec-Agent       │
│   (Alpha158因子)                    (尾盘15:55调仓)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 3. 回测与验证层 (Qlib Engine)                │
│   Qlib Dataset  →  Portfolio Provider  →  滑点/成本分析       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   4. 实盘与交易层 (Execution)                │
│   Moomoo Open API (底仓T0)   |  夜盘网格现金管理 (被动单)      │
└─────────────────────────────────────────────────────────────┘
```

### 各层职责

#### 1. 数据层
- 多数据源接入：yfinance、AkShare、Moomoo Open API
- 数据频率：日线（5年历史）、1分钟线（近1个月）、L2 订单薄（实盘录制）
- 数据清洗与格式统一：转换为 Qlib 兼容格式

#### 2. 因子与决策层
- 基于魔改 Alpha158 的因子计算
- 多 Agent 协同决策（Data / Risk / Execution）
- 波动率损耗拦截机制

#### 3. 回测与验证层
- Qlib 框架集成
- 严格滑点与交易成本模拟
- 绩效归因分析

#### 4. 实盘与交易层
- Moomoo Open API 对接
- 尾盘集中调仓（15:55 ET）
- 日内 T0 与夜盘网格执行

---

## 核心模块详解

### 数据引擎 (Data Engine)

#### 数据源
- **yfinance**：日线与分钟线行情数据
- **AkShare**：宏观经济指标（美债收益率、VIX、恐惧贪婪指数）
- **Moomoo L2**：实盘 Level 2 订单薄数据录制

#### 核心功能
- 自动增量更新
- 数据质量校验
- Qlib 格式转换
- 本地持久化存储

### 因子引擎 (Factor Engine)

#### 魔改 Alpha158
在原版 Alpha158 基础上，针对杠杆 ETF 特性进行优化：

| 因子类别 | 说明 |
|----------|------|
| 动量因子 | 不同周期的收益率、动量强度 |
| 反转因子 | 短期超买超卖、均值回归信号 |
| 量价背离 | 价涨量缩/价跌量增 |
| 波动率因子 | 已实现波动率、ATR、VIX 联动 |

#### 波动率损耗拦截器
- **触发条件**：当 30 日波动率超过阈值且无明显趋势时
- **执行动作**：因子输出强制归零，主策略切换为 1x 正股（QQQ）或空仓避险
- **参数配置**：可在 `config/agent_config.json` 中调整阈值

### Quant Agent 系统

采用三 Agent 流水线架构，每日美东时间 15:45 - 15:55 执行：

```
15:45  Data-Agent     拉取当日盘中数据 + Qlib 因子预测
   │
   ▼
15:50  Risk-Agent     计算波动率损耗暴露，判断是否触发熔断
   │
   ▼
15:55  Exec-Agent     生成最终交易指令，尾盘执行
```

#### Data-Agent
- 触发时间：美东 15:45
- 职责：拉取实时行情、更新因子值、加载模型预测
- 输出：当日候选交易信号

#### Risk-Agent
- 触发时间：美东 15:50
- 职责：
  - 计算当前多头持仓的复利损耗暴露（Volatility Decay Exposure）
  - 检查 VIX / ATR 等风险指标
  - 判断是否触发强制平仓或对冲
- 输出：风险调整后的信号

#### Exec-Agent
- 触发时间：美东 15:55（尾盘流动性最深时段）
- 职责：生成具体交易指令，对接 Moomoo API 执行
- 输出：成交回报与持仓更新

### 卫星策略

#### 底仓 T0（日内策略）
- **原理**：基于分钟线或 L2 订单薄流不平衡度（Order Imbalance）
- **操作**：日内均值回归节点做 Buy-To-Cover 或 Sell-To-Open
- **约束**：收盘前必须平仓，保持原始底仓股数不变
- **目标**：降低底仓持仓成本

#### 现金网格（夜盘策略）
- **原理**：利用夜盘流动性枯竭导致的异常长下影线
- **操作**：在远离现价 1.5% - 3% 处挂被动买单（Limit Buy）
- **止盈**：触发后反弹 0.5% 立刻分批吐出
- **适用场景**：乌龙指、流动性瞬断

---

## 开发里程碑

### Phase 1: 数据基础设施搭建

**目标**：建立稳定、自动化的本地高频与日级数据库。

**交付物**：
- `src/data_engine/fetcher.py` - 数据下载模块
- `src/data_engine/processor.py` - 数据处理与格式转换
- `data/raw/` - 原始数据目录
- `data/qlib_bin/` - Qlib 格式数据目录

**关键任务**：
1. 编写数据下载脚本 `data_fetcher.py`，利用 yfinance 下载指定杠杆 ETF（TQQQ, SOXL, UPRO, SQQQ）过去 5 年的日线及近 1 个月的 1 分钟线数据
2. 统一转化为 Qlib 兼容的 CSV 格式
3. 利用 akshare 编写宏观流动性指标（美债收益率、VIX 指数、恐惧贪婪指数）的爬取脚本 `macro_fetcher.py`

### Phase 2: Qlib 回测框架集成与因子挖掘

**目标**：在 Qlib 中复现并魔改 Alpha158 因子，使其适应杠杆 ETF 的择时场景。

**交付物**：
- `config/qlib_config.yaml` - Qlib 配置文件
- `src/factors/leveraged_alpha.py` - 魔改 Alpha158 因子
- `backtest/run_backtest.py` - 回测运行脚本
- `backtest/analyzer.py` - 绩效分析工具

**关键任务**：
1. 初始化 Qlib 配置文件 `config.yaml`，配置数据集
2. 编写自定义模型类 `LeveragedETFModel`
3. 构建基础价量因子：动量、反转、量价背离
4. 实现自适应波动率过滤机制

### Phase 3: 多 Agent 决策协同

**目标**：利用 LangGraph 或轻量级状态机，构建尾盘逻辑处理 Agents。

**交付物**：
- `config/agent_config.json` - Agent 配置与风控阈值
- `src/agents/data_agent.py` - 数据 Agent
- `src/agents/risk_agent.py` - 风控 Agent
- `src/agents/exec_agent.py` - 执行 Agent

**关键任务**：
1. 实现 Data-Agent：每日美东 15:45 触发，拉取当日盘中表现及 Qlib 因子预测值
2. 实现 Risk-Agent：计算当前多头持仓的复利损耗暴露，判断是否触发强制平仓或对冲
3. 实现 Execution-Agent：生成最终交易指令，15:55 尾盘执行

### Phase 4: 日内 T0 与夜盘卫星策略开发

**目标**：在不改变日终（End of Day）持仓股数的前提下，利用 Moomoo API 进行日内与隔夜套利。

**交付物**：
- `src/strategies/intra_t0.py` - 日内 T0 策略
- `src/strategies/overnight_grid.py` - 夜盘网格策略

**关键任务**：
1. 底仓 T0：基于订单薄流不平衡度的日内均值回归交易
2. 现金网格：夜盘被动挂单，捕捉流动性异常

---

## 目录结构

```
杠杆择时策略/
│
├── config/                          # 配置文件目录
│   ├── qlib_config.yaml            # Qlib 回测参数配置
│   └── agent_config.json           # Agent 提示词与风控阈值
│
├── data/                            # 数据目录
│   ├── raw/                        # yfinance/akshare 原始下载数据
│   └── qlib_bin/                   # Qlib 转换后的二进制格式数据
│
├── src/                             # 源代码目录
│   ├── data_engine/                # 数据下载与清洗模块
│   │   ├── fetcher.py              # 数据下载器
│   │   └── processor.py            # 数据处理器
│   │
│   ├── factors/                    # 因子计算逻辑（魔改Alpha158）
│   │   └── leveraged_alpha.py      # 杠杆ETF专用因子
│   │
│   ├── agents/                     # Quant Agent 核心
│   │   ├── data_agent.py           # 数据Agent
│   │   ├── risk_agent.py           # 风控Agent
│   │   └── exec_agent.py           # 执行Agent
│   │
│   └── strategies/                 # T0 与夜盘策略模块
│       ├── intra_t0.py             # 日内T0策略
│       └── overnight_grid.py       # 夜盘网格策略
│
├── backtest/                        # 回测目录
│   ├── run_backtest.py             # Qlib 回测运行脚本
│   └── analyzer.py                 # 绩效分析工具
│
├── logs/                            # 日志目录
│
├── docker-compose.yml               # Docker 编排文件
├── requirements.txt                 # Python 依赖
├── .gitignore                       # Git 忽略文件
├── project.txt                      # 原始项目计划书
└── README.md                        # 项目说明文档
```

---

## 快速开始

### 环境要求

- Python 3.9+
- Docker / Docker Compose（可选，用于环境隔离）
- 可用的 Moomoo 账户（实盘阶段需要）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd 杠杆择时策略
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   # 复制配置模板并填写
   cp config/agent_config.json.example config/agent_config.json
   ```

5. **下载数据**
   ```bash
   python src/data_engine/fetcher.py
   ```

6. **运行回测**
   ```bash
   python backtest/run_backtest.py
   ```

---

## 风险管理

### 风险矩阵

| 风险项 | 风险描述 | 技术/策略解决方案 |
|--------|----------|-------------------|
| **震荡损耗 (Decay)** | 杠杆 ETF 在震荡市中由于每日复利计算导致净值天然流失 | Risk-Agent 熔断机制：当 ATR 或 VIX 处于高位区间震荡时，Agent 自动将标的降维至 QQQ（1x）或全仓持币 |
| **洗售规则 (Wash Sale)** | 频繁 T0 触发美股 Wash Sale，导致亏损无法抵税，增加报税成本 | 系统记账隔离：实盘时 T0 模块尽量使用独立子账户操作，或者在策略回测端引入税务摩擦惩罚因子 |
| **回测幸存者偏差** | 免费的 yfinance 分钟线数据没有买卖点差记录，T0 回测可能过度乐观 | Qlib 严格滑点模拟：在 Qlib 的 Slippage 组件中，为 T0 策略强制加入固定 1-2 个基点的滑点，且夜盘网格仅采用被动挂单撮合逻辑（Limit Order） |
| **执行风险** | 尾盘流动性不足导致滑点过大 | 选择 15:55 流动性最深时段执行，使用限价单保护 |
| **系统风险** | 程序崩溃、网络中断 | Docker 容器化部署，健康检查，告警机制 |

### 风控参数配置

在 `config/agent_config.json` 中可配置以下风控参数：

```json
{
  "risk_management": {
    "volatility_threshold": 0.4,
    "vix_high_threshold": 30,
    "max_position_pct": 1.0,
    "max_single_position_pct": 0.5,
    "stop_loss_pct": 0.08,
    "t0_slippage_bps": 2
  }
}
```

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **编程语言** | Python 3.9+ | 主开发语言 |
| **数据获取** | yfinance | 美股行情数据 |
| | AkShare | 宏观经济数据 |
| | Moomoo Open API | 实盘交易 & L2 数据 |
| **回测框架** | Qlib | 因子回测与模型训练 |
| **Agent 框架** | LangGraph / 状态机 | 多 Agent 协同 |
| **异步框架** | asyncio / aiohttp | 异步数据获取与执行 |
| **容器化** | Docker / Docker Compose | 环境隔离与部署 |
| **数据存储** | CSV / Parquet / Qlib Bin | 本地数据持久化 |

---

## 开发规范

### 代码风格
- 遵循 PEP 8 规范
- 使用 type hints 类型注解
- 编写 docstring 文档字符串

### 命名约定
- 模块：小写 + 下划线（如 `data_fetcher.py`）
- 类：大驼峰（如 `LeveragedETFModel`）
- 函数/变量：小写 + 下划线（如 `calculate_volatility()`）
- 常量：全大写 + 下划线（如 `MAX_POSITION_PCT`）

### Git 工作流
- `main` / `master`：主分支，生产就绪代码
- `dev`：开发分支
- `feature/xxx`：功能分支
- `fix/xxx`：修复分支

### Commit 规范
```
<type>: <subject>

type 可选值:
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- test: 测试相关
- chore: 构建/工具相关
```

---

## Agent 协作指南

### 给 AI Agent 的工作说明

本项目设计为可由 AI Agent 逐步推进开发。以下是各阶段的任务指令参考：

#### Phase 1 指令
> "编写一个数据下载脚本 `data_fetcher.py`，利用 yfinance 下载指定杠杆 ETF（TQQQ, SOXL, UPRO, SQQQ）过去 5 年的日线及近 1 个月的 1 分钟线数据，并统一转化为 Qlib 兼容的 CSV 格式。"
>
> "利用 akshare 编写宏观流动性指标（如美债收益率、VIX 指数、恐惧贪婪指数）的爬取脚本 `macro_fetcher.py`。"

#### Phase 2 指令
> "初始化 Qlib 配置文件 `config.yaml`，配置初始化数据集，编写自定义模型类 `LeveragedETFModel`，使其能够载入修改后的 Alpha158 因子进行日级择时训练。"

#### Phase 3 指令
> "实现三个 Agent：Data-Agent（每日美东15:45拉取数据）、Risk-Agent（波动率损耗检查）、Exec-Agent（15:55尾盘执行），使用 LangGraph 或轻量级状态机实现协同。"

#### Phase 4 指令
> "实现日内 T0 策略（基于订单薄不平衡度的均值回归）和夜盘网格策略（被动挂单捕捉流动性异常），对接 Moomoo API。"

### 注意事项
1. **数据优先**：始终确保数据层的正确性，再进行上层开发
2. **回测验证**：任何策略修改都必须经过回测验证
3. **风险第一**：开发新功能时，优先考虑风险控制
4. **渐进式开发**：按 Phase 顺序推进，每阶段完成后再进入下一阶段

---

## 许可证

本项目仅供学习研究使用，不构成任何投资建议。使用本系统进行实盘交易的风险由使用者自行承担。

---

*Project Helios - 量化交易系统 | 最后更新: 2026-06-25*
