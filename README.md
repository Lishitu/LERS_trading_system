# LERS - 杠杆 ETF 状态切换系统

> **L**everaged **E**TF **R**egime **S**witching System
>
> 基于市场状态预测的杠杆 ETF 动态择时交易系统

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 交易标的与基准策略](#2-交易标的与基准策略)
- [3. 项目目录结构](#3-项目目录结构)
- [4. 开发里程碑（9个Phase详解）](#4-开发里程碑9个phase详解)
- [5. 快速开始（Step-by-Step）](#5-快速开始step-by-step)
- [6. 技术栈与依赖](#6-技术栈与依赖)
- [7. 成功标准](#7-成功标准)
- [8. 开发规范](#8-开发规范)
- [9. Agent 工作指南](#9-agent-工作指南)
- [10. 未来扩展](#10-未来扩展)

---

## 1. 项目概述

### 1.1 项目目标

构建一个基于**市场状态（Regime）预测**的量化交易系统，在杠杆 ETF 之间动态分配资金。

**核心原则**：这不是一个选股系统，而是一个**择时系统**。

### 1.2 核心思路

系统识别三种市场状态，并在三种仓位之间切换：

| 市场状态 | 预测含义 | 对应仓位 |
|----------|----------|----------|
| **Risk-On** | 未来上涨概率高 | 持有 **TQQQ**（3x 做多纳指） |
| **Neutral** | 方向不明，震荡市 | 持有 **Cash**（现金） |
| **Risk-Off** | 未来下跌概率高 | 持有 **SQQQ**（3x 做空纳指） |

### 1.3 预测目标

- **主要预测周期**：未来 **5 个交易日**的 QQQ 收益率
- **可选对比周期**：1 天、10 天

### 1.4 业绩目标

在**风险调整后收益**上超越买入持有 TQQQ：
- 更高的夏普比率（Sharpe Ratio）
- 显著更低的最大回撤（Max Drawdown）
- 正的样本外年化收益率

---

## 2. 交易标的与基准策略

### 2.1 交易标的（MVP 阶段）

| 代码 | 名称 | 类型 | 说明 |
|------|------|------|------|
| TQQQ | ProShares UltraPro QQQ | 3x 做多 | 风险开启时持有 |
| SQQQ | ProShares UltraPro Short QQQ | 3x 做空 | 风险关闭时持有 |
| Cash | 现金 | — | 中性状态时持有 |

> **注意**：Phase 1 MVP 阶段仅使用以上三个标的，不包含 SOXL、UPRO 等其他资产。

### 2.2 基准策略

系统将与以下三个基准策略进行对比：

#### 基准 A：买入持有 TQQQ（Buy & Hold）
- **规则**：始终满仓持有 TQQQ
- **用途**：核心对比基准，衡量择时是否创造价值

#### 基准 B：200 日均线择时
- **规则**：
  ```
  若 QQQ 收盘价 > MA200: 持有 TQQQ
  否则: 持有现金
  ```
- **用途**：经典趋势跟踪策略对比

#### 基准 C：20/60 日均线金叉
- **规则**：
  ```
  若 MA20 > MA60: 持有 TQQQ
  否则: 持有现金
  ```
- **用途**：双均线策略对比

---

## 3. 项目目录结构

```
杠杆择时策略/
│
├── data/                              # 数据目录
│   ├── raw/                          # 原始数据（yfinance下载）
│   │   ├── qqq.csv
│   │   ├── tqqq.csv
│   │   ├── sqqq.csv
│   │   ├── spy.csv
│   │   ├── vix.csv
│   │   ├── tnx.csv
│   │   └── dxy.csv
│   │
│   └── processed/                    # 处理后的数据
│       └── features.parquet          # 特征工程后的总表
│
├── src/                               # 源代码目录
│   ├── data_loader.py                # 数据下载与加载
│   ├── feature_engineering.py        # 特征工程
│   ├── label_generator.py            # 标签生成
│   ├── train_model.py                # 模型训练
│   ├── portfolio_engine.py           # 组合决策引擎
│   ├── walkforward.py                # 滚动Walk-Forward验证
│   ├── backtest.py                   # 回测引擎
│   ├── evaluate.py                   # 绩效评估
│   └── visualization.py              # 可视化
│
├── notebooks/                         # Jupyter 笔记本（探索用）
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── tests/                             # 单元测试
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_backtest.py
│
├── reports/                           # 报告输出
│   └── figures/                      # 图表输出目录
│       ├── equity_curve.png
│       ├── drawdown.png
│       ├── rolling_sharpe.png
│       ├── position_timeline.png
│       ├── feature_importance.png
│       └── confusion_matrix.png
│
├── models/                            # 训练好的模型
│   └── .gitkeep
│
├── requirements.txt                   # Python 依赖
├── .gitignore                         # Git 忽略文件
├── project.txt                        # 原始项目计划书
└── README.md                          # 项目说明文档
```

---

## 4. 开发里程碑（9个Phase详解）

### Phase 1: MVP 定义与项目初始化

**目标**：明确项目范围，搭建项目骨架。

**输入**：项目计划书

**输出**：项目目录结构、依赖文件、配置框架

**具体步骤**：
1. 创建完整的目录结构（data/, src/, notebooks/, tests/, reports/）
2. 编写 `requirements.txt`，列出所有依赖包
3. 编写 `.gitignore` 文件
4. 创建各模块的占位文件（空文件或带骨架代码）

**交付文件**：
- `requirements.txt`
- 完整的目录结构
- 各 `.py` 文件的骨架代码

**验证方式**：
- 目录结构与第 3 节一致
- `pip install -r requirements.txt` 能成功执行

---

### Phase 2: 数据层

**目标**：建立稳定的数据下载与存储机制。

**输入**：yfinance API

**输出**：`data/raw/` 下的 7 个 CSV 文件

**数据标的**：
| 代码 | 说明 | yfinance 代码 |
|------|------|---------------|
| QQQ | 纳斯达克100 ETF | QQQ |
| TQQQ | 3x 做多纳指 | TQQQ |
| SQQQ | 3x 做空纳指 | SQQQ |
| SPY | S&P 500 ETF | SPY |
| VIX | 波动率指数 | ^VIX |
| TNX | 10年期美债收益率 | ^TNX |
| DXY | 美元指数 | DX-Y.NYB |

**时间范围**：
- 起始：2011-01-01
- 结束：最新交易日

**具体步骤**：
1. 在 `src/data_loader.py` 中实现 `download_data()` 函数
2. 函数参数：`tickers` 列表、`start_date`、`end_date`、`save_dir`
3. 逐个下载每个标的的日线数据
4. 保存为 CSV 格式到 `data/raw/` 目录
5. 实现数据校验：检查数据连续性、缺失值处理
6. 编写 `python -m src.data_loader` 命令行入口

**交付文件**：
- `src/data_loader.py`
- `data/raw/qqq.csv` ~ `data/raw/dxy.csv`（共7个文件）

**验证方式**：
- 运行 `python src/data_loader.py` 能成功下载所有数据
- 每个 CSV 文件包含：Date, Open, High, Low, Close, Volume 列
- 数据起始日期不晚于 2011-01-03
- 数据行数与日期范围匹配

---

### Phase 3: 特征工程

**目标**：计算所有技术特征，构建特征矩阵。

**输入**：`data/raw/` 中的原始 CSV 数据

**输出**：`data/processed/features.parquet`

**特征清单（共 11 大类，约 60+ 个特征）**：

#### 3.1 动量特征（Momentum）
| 特征名 | 计算公式 |
|--------|----------|
| ret_1d | close / close.shift(1) - 1 |
| ret_3d | close / close.shift(3) - 1 |
| ret_5d | close / close.shift(5) - 1 |
| ret_10d | close / close.shift(10) - 1 |
| ret_20d | close / close.shift(20) - 1 |
| ret_60d | close / close.shift(60) - 1 |
| ret_120d | close / close.shift(120) - 1 |
| mom_5_20 | ret_5d - ret_20d（动量加速度） |
| mom_20_60 | ret_20d - ret_60d |

#### 3.2 均线特征（Moving Average）
| 特征名 | 说明 |
|--------|------|
| ma5 | 5 日均线 |
| ma10 | 10 日均线 |
| ma20 | 20 日均线 |
| ma60 | 60 日均线 |
| ma120 | 120 日均线 |
| ma200 | 200 日均线 |
| ma5_ma20_ratio | ma5 / ma20 - 1（短长均线比） |
| ma20_ma60_ratio | ma20 / ma60 - 1 |
| ma20_ma200_ratio | ma20 / ma200 - 1 |

#### 3.3 距离特征（Distance to MA）
| 特征名 | 计算公式 |
|--------|----------|
| dist_ma5 | close / ma5 - 1 |
| dist_ma20 | close / ma20 - 1 |
| dist_ma60 | close / ma60 - 1 |
| dist_ma200 | close / ma200 - 1 |

#### 3.4 波动率特征（Volatility）
| 特征名 | 说明 |
|--------|------|
| vol_5 | 5 日收益率标准差（年化） |
| vol_20 | 20 日收益率标准差（年化） |
| vol_60 | 60 日收益率标准差（年化） |
| vol_120 | 120 日收益率标准差（年化） |
| vol_5_vol_20_ratio | vol_5 / vol_20（短期/长期波动比） |
| vol_20_vol_60_ratio | vol_20 / vol_60 |

#### 3.5 价量形态特征（Price-Volume Pattern）
| 特征名 | 计算公式 / 说明 |
|--------|-----------------|
| volume_ret_1d | volume / volume.shift(1) - 1（成交量变化率） |
| volume_ma5 | 5 日均量 |
| volume_ma20 | 20 日均量 |
| volume_ratio_5_20 | volume / volume_ma20（量比） |
| obv | 累积能量潮（On-Balance Volume） |
| obv_5d_change | obv / obv.shift(5) - 1 |
| price_volume_divergence | ret_5d 与 volume_ret_5d 的符号是否一致（量价背离标记） |
| close_position | (close - low) / (high - low)（收盘价在当日K线位置，0~1） |
| upper_shadow | (high - max(open, close)) / (high - low)（上影线比例） |
| lower_shadow | (min(open, close) - low) / (high - low)（下影线比例） |
| body_size | abs(close - open) / (high - low)（实体比例） |
| high_low_range | (high - low) / close（振幅） |
| high_low_range_ma20 | 20 日均振幅 |

#### 3.6 VIX 特征
| 特征名 | 计算公式 |
|--------|----------|
| vix_level | VIX 当日收盘价 |
| vix_change_1d | vix / vix.shift(1) - 1 |
| vix_change_5d | vix / vix.shift(5) - 1 |
| vix_change_20d | vix / vix.shift(20) - 1 |
| vix_ma20_ratio | vix / vix_ma20 - 1 |
| vix_ma60_ratio | vix / vix_ma60 - 1 |
| vix_quantile_60d | VIX 在过去 60 天的分位数（0~1） |

#### 3.7 相对强弱特征（Relative Strength / Breadth）
| 特征名 | 计算公式 |
|--------|----------|
| rs_qv_spy | QQQ / SPY 比值 |
| rs_qv_spy_ret_5d | (QQQ/SPY) / (QQQ/SPY).shift(5) - 1 |
| rs_qv_spy_ret_20d | (QQQ/SPY) / (QQQ/SPY).shift(20) - 1 |
| rs_qv_spy_ret_60d | (QQQ/SPY) / (QQQ/SPY).shift(60) - 1 |

#### 3.8 市场广度特征（Market Breadth）
> 使用 SPY 代理整体市场广度

| 特征名 | 计算公式 / 说明 |
|--------|-----------------|
| spy_ret_1d | SPY 当日收益率 |
| spy_ret_5d | SPY 5日收益率 |
| spy_ret_20d | SPY 20日收益率 |
| spy_vol_20 | SPY 20日波动率 |
| spy_dist_ma200 | SPY 距 200 日均线距离 |
| spy_high_252d_ratio | SPY 收盘价 / 过去252日最高价（接近高点程度） |
| spy_low_252d_ratio | SPY 收盘价 / 过去252日最低价（远离低点程度） |
| new_high_low_diff | （代理：SPY 52周新高/新低状态） |

#### 3.9 利率特征（Interest Rate / Macro）
| 特征名 | 计算公式 |
|--------|----------|
| tnx_level | TNX 10年期美债收益率 |
| tnx_change_1d | tnx / tnx.shift(1) - 1 |
| tnx_change_5d | tnx / tnx.shift(5) - 1 |
| tnx_change_20d | tnx / tnx.shift(20) - 1 |
| tnx_ma20_ratio | tnx / tnx_ma20 - 1 |
| yield_curve_slope | （代理：TNX 自身水平 + 变化率，反映利率环境） |

#### 3.10 美元特征（Dollar / Macro）
| 特征名 | 计算公式 |
|--------|----------|
| dxy_level | DXY 美元指数 |
| dxy_change_1d | dxy / dxy.shift(1) - 1 |
| dxy_change_5d | dxy / dxy.shift(5) - 1 |
| dxy_change_20d | dxy / dxy.shift(20) - 1 |
| dxy_ma20_ratio | dxy / dxy_ma20 - 1 |
| dxy_trend_60d | dxy / dxy.shift(60) - 1（美元中期趋势） |

#### 3.11 交叉宏观因子（Cross-Asset Macro）
| 特征名 | 计算公式 / 说明 |
|--------|-----------------|
| vix_spy_corr_20d | VIX 变化率 与 SPY 收益率 20日滚动相关（恐慌联动性） |
| dxy_tnx_corr_20d | DXY 变化率 与 TNX 变化率 20日滚动相关 |
| gold_silver_ratio | （可选，暂不纳入 MVP，用 DXY 代理） |

**具体步骤**：
1. 在 `src/feature_engineering.py` 中实现 `FeatureEngineer` 类
2. 实现 `load_raw_data()` 方法：加载所有原始 CSV（QQQ, TQQQ, SQQQ, SPY, VIX, TNX, DXY）
3. 为每类特征实现一个计算方法（如 `_calc_momentum()`, `_calc_ma()`, `_calc_price_volume()` 等）
4. 实现 `build_features()` 主方法：组合所有特征，按日期对齐
5. 实现 `_remove_outliers()` 方法：处理极端值和 Inf
6. 输出为 Parquet 格式保存到 `data/processed/features.parquet`
7. 特征以 QQQ 为基准对齐时间索引

**交付文件**：
- `src/feature_engineering.py`
- `data/processed/features.parquet`

**验证方式**：
- 特征矩阵行数 = QQQ 交易日天数
- 特征列数 >= 50 列
- 无 Inf 或 NaN（去除初始窗口的 NaN 行后）
- 特征值范围合理（如日收益率绝对值 < 20%）
- 每类特征都有对应的列，命名规范统一

---

### Phase 4: 标签生成与阈值敏感性分析

**目标**：基于未来收益生成市场状态标签，并分析不同阈值对标签分布和策略的影响。

**输入**：QQQ 的收盘价序列

**输出**：带有 label 列的特征表 + 阈值敏感性分析报告

**标签规则（默认配置）**：

使用未来 5 个交易日的收益率来定义市场状态：

```python
future_ret_5d = close.shift(-5) / close - 1
```

| 未来 5 日收益率 | 市场状态 | 标签编码 |
|-----------------|----------|----------|
| > 2% | Risk-On | 2 |
| < -2% | Risk-Off | 0 |
| -2% ~ +2% | Neutral | 1 |

**阈值敏感性分析**：

由于 ±2% 的阈值具有主观性，需要对多个阈值进行对比分析，选择最优配置：

| 阈值方案 | 上涨阈值 | 下跌阈值 | 预期标签分布 | 适用场景 |
|----------|----------|----------|--------------|----------|
| 保守 | ±1.0% | ±1.0% | 极端样本少，中性多 | 低换手、稳健型 |
| 偏保守 | ±1.5% | ±1.5% | 平衡偏中性 | 通用型 |
| **基准（默认）** | **±2.0%** | **±2.0%** | **三类均衡** | **标准方案** |
| 偏激进 | ±2.5% | ±2.5% | 极端样本多，中性少 | 高换手、进攻型 |
| 激进 | ±3.0% | ±3.0% | 极端样本很少，中性极多 | 低频交易型 |
| 非对称 | +1.5% | -2.5% | 下跌更严格 | 风险厌恶型 |
| 分位数法 | top 30% | bottom 30% | 各约 30% / 40% / 30% | 分布自适应 |

**分析维度**：
1. **标签分布**：各阈值下三类标签的数量比例
2. **模型准确率**：各阈值下模型的分类准确率
3. **策略收益**：各阈值下回测的年化收益、夏普、回撤
4. **换手率**：各阈值下的交易频率
5. **稳定性**：不同市场环境下各阈值的表现一致性

**具体步骤**：
1. 在 `src/label_generator.py` 中实现 `LabelGenerator` 类
2. 实现 `generate_labels()` 方法：
   - 计算未来收益率
   - 根据阈值打标签
3. 可配置参数：
   - `horizon`: 预测天数（默认 5）
   - `threshold`: 分类阈值（默认 0.02）
   - `threshold_up`: 上涨阈值（非对称时使用）
   - `threshold_down`: 下跌阈值（非对称时使用）
   - `use_quantile`: 是否使用分位数法（默认 False）
   - `quantile`: 分位数比例（默认 0.3）
4. 实现 `threshold_sensitivity_analysis()` 方法：
   - 遍历多个阈值方案
   - 计算各方案的标签分布
   - 输出对比表格
5. 将标签合并到特征表中
6. 输出带标签的完整数据集
7. 保存阈值分析结果到 `reports/threshold_analysis.csv`

**交付文件**：
- `src/label_generator.py`
- 更新后的 `data/processed/features.parquet`（含 label 列）
- `reports/threshold_analysis.csv`（阈值敏感性分析结果）

**验证方式**：
- 三类标签的分布比例合理（没有某类 > 80% 或 < 5%）
- 没有未来函数泄露（检查 shift 方向是否正确）
- 最后 5 行的 label 为 NaN（合理，因为没有未来数据）
- 阈值敏感性分析覆盖至少 5 种方案
- 各阈值方案的标签数量之和 = 总样本数

---

### Phase 5: 模型训练

**目标**：训练分类模型预测市场状态概率。

**输入**：带标签的特征表

**输出**：训练好的模型、每日预测概率

**模型清单**：

| 模型 | 优先级 | 说明 |
|------|--------|------|
| LightGBM | 主要 | 梯度提升树，主模型 |
| Random Forest | 对比 | 随机森林，对比用 |
| Logistic Regression | 基线 | 逻辑回归，简单基线 |

**预测输出**：
每个交易日输出三个概率：
- `P(RiskOn)` - 风险开启概率
- `P(Neutral)` - 中性概率
- `P(RiskOff)` - 风险关闭概率

**具体步骤**：
1. 在 `src/train_model.py` 中实现 `ModelTrainer` 类
2. 实现 `prepare_data()` 方法：
   - 划分特征 X 和标签 y
   - 去除包含 NaN 的行
3. 实现 `train_lightgbm()` 方法：
   - 设置合理的超参数
   - 训练模型
   - 返回模型对象
4. 实现 `train_random_forest()` 和 `train_logistic_regression()` 对比方法
5. 实现 `predict_proba()` 方法：输出每日三类概率
6. 保存训练好的模型到 `models/` 目录

**交付文件**：
- `src/train_model.py`
- `models/lgb_model.pkl`（及其他模型）
- 带预测概率的 DataFrame

**验证方式**：
- 训练集上的准确率 > 随机猜测（33%）
- 三个概率之和 ≈ 1.0
- 模型能成功保存和加载

---

### Phase 6: 组合决策引擎与交易成本模拟

**目标**：将模型预测概率转换为具体的交易仓位，并模拟真实交易成本。

**输入**：每日预测概率 + TQQQ/SQQQ 价格数据

**输出**：每日持仓信号 + 扣除成本后的净值曲线

**决策规则**：

```
若 P(RiskOn) > 0.60:
    目标仓位 = TQQQ（3x 做多）

否则若 P(RiskOff) > 0.60:
    目标仓位 = SQQQ（3x 做空）

否则:
    目标仓位 = Cash（现金）
```

**执行规则**：
- 信号在**收盘后**生成
- 交易在**下一个交易日开盘**执行
- **严禁未来函数**（严禁使用当日收盘价当日交易）

**仓位约束**：
- 每次只持有一种仓位（TQQQ / SQQQ / Cash 三选一）
- 满仓进出，不加杠杆叠加

**交易成本与滑点模拟**：

为了让回测更贴近实盘，必须计入以下成本：

| 成本项 | 默认值 | 说明 | 可调参数 |
|--------|--------|------|----------|
| **佣金 (Commission)** | 0.0005（5 bps） | 每次买卖的交易佣金 | `commission_bps` |
| **滑点 (Slippage)** | 0.0010（10 bps） | 买卖价差与市场冲击成本 | `slippage_bps` |
| **买卖价差 (Spread)** | 已含在滑点中 | 杠杆 ETF 点差较大 | 合并入滑点 |
| **融资成本 (Financing)** | 0.0002 / 日（~5%/年） | 杠杆 ETF 的隐含融资成本（持有期间扣除） | `financing_cost_daily` |

> **注意**：TQQQ/SQQQ 这类杠杆 ETF 的总成本 = 管理费（约 0.95%/年）+ 融资成本（约 3-5%/年）+ 交易成本。
> 回测中简化为：每次换手扣除佣金+滑点，持有期间按日扣除融资成本。

**成本计算公式**：

```
换手成本 = 目标仓位 != 前日持仓 时扣除:
    卖出成本 = 前日仓位价值 × (commission + slippage)
    买入成本 = 新仓位价值 × (commission + slippage)

持有成本（仅持有杠杆ETF时）:
    日融资成本 = 持仓价值 × financing_cost_daily

净值_t = 净值_{t-1} × (1 + 当日收益率) - 当日成本
```

**具体步骤**：
1. 在 `src/portfolio_engine.py` 中实现 `PortfolioEngine` 类
2. 实现 `generate_signals()` 方法：
   - 输入预测概率
   - 根据阈值规则生成每日目标仓位
3. 实现 `apply_execution_lag()` 方法：
   - 将信号后移一天（次日开盘执行）
4. 实现 `calculate_daily_returns()` 方法：
   - 根据持仓计算每日收益率
   - 使用开盘价计算换仓收益，收盘价计算持有收益
5. 实现 `apply_transaction_costs()` 方法：
   - 换手时扣除佣金 + 滑点
   - 持有杠杆 ETF 时扣除每日融资成本
6. 实现 `calculate_nav()` 方法：
   - 计算累计净值曲线
7. 可配置参数：
   - `risk_on_threshold`（默认 0.60）
   - `risk_off_threshold`（默认 0.60）
   - `commission_bps`（默认 5 bps）
   - `slippage_bps`（默认 10 bps）
   - `financing_cost_daily`（默认 0.0002/日）
   - `include_costs`（默认 True，可关闭成本用于对比）

**交付文件**：
- `src/portfolio_engine.py`

**验证方式**：
- 持仓只有三种状态：TQQQ、SQQQ、Cash
- 信号有合理的换手率（不是每天都换）
- 执行滞后正确（T 日信号 → T+1 日持仓）
- 有成本 vs 无成本的净值曲线存在合理差异
- 换手次数 × 单次成本 ≈ 总交易成本（量级校验）
- 融资成本仅在持有 TQQQ/SQQQ 时扣除，Cash 时不扣除

---

### Phase 7: 滚动 Walk-Forward 验证

**目标**：使用滚动时间窗口验证策略，避免过拟合。

**输入**：特征 + 标签 + 模型 + 组合引擎

**输出**：样本外每日收益序列 + 多窗口对比分析

**为什么用 Walk-Forward**：
- ❌ **禁止**使用随机划分训练集/测试集（数据泄露风险）
- ✅ **必须**使用时间序列滚动验证

**Walk-Forward 方案对比**：

提供三种滚动频率和两种窗口模式，灵活组合：

| 配置项 | 选项 | 说明 | 默认 |
|--------|------|------|------|
| **滚动频率** | `yearly` | 每年重训练一次 | 是 |
| | `quarterly` | 每季度重训练一次 | 推荐 |
| | `monthly` | 每月重训练一次 | 可选 |
| **窗口模式** | `expanding` | 扩张窗口（训练数据越来越多） | 是 |
| | `rolling` | 固定滚动窗口（如最近5年） | 可选 |
| **窗口大小** | 自定义 | 仅 rolling 模式需要 | 5 年 |

**推荐方案（默认+对比）**：

```
方案 A - 年频扩张（基准，较慢）
第1轮:  Train: 2011-2018  |  Test: 2019
第2轮:  Train: 2011-2019  |  Test: 2020
...

方案 B - 季频扩张（推荐，更快适应市场）
第1轮:  Train: 2011-2018Q4 |  Test: 2019Q1
第2轮:  Train: 2011-2019Q1 |  Test: 2019Q2
第3轮:  Train: 2011-2019Q2 |  Test: 2019Q3
...

方案 C - 季频滚动 5 年（可选，适应市场结构变化）
第1轮:  Train: 2014Q1-2018Q4 |  Test: 2019Q1
第2轮:  Train: 2014Q2-2019Q1 |  Test: 2019Q2
...
```

**方案选择建议**：
- **方案 A（年频）**：计算量小，用于快速验证
- **方案 B（季频扩张）**：平衡计算量和适应性，**推荐作为主方案**
- **方案 C（季频滚动）**：更贴近实盘，适合最终验证

**具体步骤**：
1. 在 `src/walkforward.py` 中实现 `WalkForwardValidator` 类
2. 实现 `generate_splits()` 方法：
   - 支持按年/季/月划分训练/测试集
   - 支持 expanding 和 rolling 两种窗口模式
   - rolling 模式下窗口大小可配置（`window_years`）
3. 实现 `_split_by_year()` / `_split_by_quarter()` / `_split_by_month()` 子方法
4. 实现 `run()` 方法：
   - 遍历每个 split
   - 在训练集上训练模型（支持 LightGBM 主模型）
   - 在测试集上预测（输出三类概率）
   - 通过组合引擎生成仓位和收益
   - 收集所有样本外预测结果
5. 实现 `compare_schemes()` 方法：
   - 运行多种 Walk-Forward 方案
   - 输出各方案的绩效对比表
6. 输出完整的样本外预测序列和收益序列

**交付文件**：
- `src/walkforward.py`
- 样本外预测结果（Parquet）
- Walk-Forward 方案对比报告（CSV / 控制台输出）

**验证方式**：
- 测试集没有重叠
- 训练集不包含任何测试集日期（无未来函数）
- 样本外结果覆盖所有测试年份/季度
- 能复现每个测试周期的预测
- 不同滚动频率的结果有合理差异（不是完全一致）
- rolling 窗口的训练数据量保持恒定

---

### Phase 8: 绩效评估

**目标**：全面评估策略表现。

**输入**：样本外每日收益、仓位序列

**输出**：绩效指标报告

**评估指标三大类**：

#### 8.1 组合绩效指标
| 指标 | 英文 | 说明 |
|------|------|------|
| 年化收益率 | CAGR | 复合年化增长率 |
| 年化收益 | Annual Return | 简单年化 |
| 波动率 | Volatility | 年化标准差 |
| 夏普比率 | Sharpe Ratio | 收益/波动 |
| 索提诺比率 | Sortino Ratio | 下行风险调整收益 |
| 卡玛比率 | Calmar Ratio | 收益/最大回撤 |
| 最大回撤 | Max Drawdown | 历史最大跌幅 |

#### 8.2 交易指标
| 指标 | 英文 | 说明 |
|------|------|------|
| 胜率 | Win Rate | 盈利交易占比 |
| 换手率 | Turnover | 年均交易次数/仓位变化 |
| 平均持仓期 | Avg Holding Period | 平均每次持仓天数 |
| 交易次数 | Trade Count | 总交易次数 |

#### 8.3 分类指标
| 指标 | 英文 | 说明 |
|------|------|------|
| 准确率 | Accuracy | 预测正确的比例 |
| 精确率 | Precision | 预测为某类中实际正确的比例 |
| 召回率 | Recall | 实际某类中被预测正确的比例 |
| F1 分数 | F1 Score | 精确率和召回率的调和平均 |
| 混淆矩阵 | Confusion Matrix | 3x3 分类结果矩阵 |

**具体步骤**：
1. 在 `src/evaluate.py` 中实现 `PerformanceEvaluator` 类
2. 实现 `calc_portfolio_metrics()` 方法
3. 实现 `calc_trading_metrics()` 方法
4. 实现 `calc_classification_metrics()` 方法
5. 实现 `print_report()` 方法：打印完整报告
6. 与三个基准策略对比输出

**交付文件**：
- `src/evaluate.py`
- 控制台输出的绩效报告

**验证方式**：
- 所有指标都能计算输出
- 与手动计算的结果一致（抽样验证）
- 三个基准策略的结果符合预期

---

### Phase 9: 可视化

**目标**：生成可视化图表，直观展示策略表现。

**输入**：回测结果数据

**输出**：`reports/figures/` 目录下的 6 张图

**图表清单**：

| 序号 | 图表名称 | 文件名 | 说明 |
|------|----------|--------|------|
| 1 | 净值曲线 | equity_curve.png | 策略 vs 基准的累计收益 |
| 2 | 回撤曲线 | drawdown.png | 策略 vs 基准的回撤 |
| 3 | 滚动夏普 | rolling_sharpe.png | 滚动 1 年夏普比率 |
| 4 | 仓位时间线 | position_timeline.png | 每日持仓状态变化 |
| 5 | 特征重要性 | feature_importance.png | LightGBM 特征重要性排序 |
| 6 | 混淆矩阵 | confusion_matrix.png | 分类结果热力图 |

**具体步骤**：
1. 在 `src/visualization.py` 中实现 `Visualizer` 类
2. 为每个图表实现一个方法
3. 使用 matplotlib 或 seaborn 绘图
4. 统一图表风格（标题、坐标轴标签、图例）
5. 保存为 PNG 格式到 `reports/figures/`

**交付文件**：
- `src/visualization.py`
- `reports/figures/` 下的 6 张图片

**验证方式**：
- 6 张图都能成功生成
- 图表清晰、有标题和坐标轴标签
- 数据与绩效评估一致

---

## 5. 快速开始（Step-by-Step）

### 5.1 环境准备

**要求**：
- Python 3.9 或更高版本
- pip 包管理器

### 5.2 安装步骤

**Step 1: 克隆仓库并进入目录**
```bash
cd 杠杆择时策略
```

**Step 2: 创建虚拟环境**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: 安装依赖**
```bash
pip install -r requirements.txt
```

### 5.3 运行项目

**Step 4: 下载数据（Phase 2）**
```bash
python src/data_loader.py
```
预期输出：`data/raw/` 目录下生成 7 个 CSV 文件。

**Step 5: 构建特征（Phase 3）**
```bash
python src/feature_engineering.py
```
预期输出：`data/processed/features.parquet`

**Step 6: 生成标签（Phase 4）**
```bash
python src/label_generator.py
```

**Step 7: 运行 Walk-Forward 回测（Phase 5-7）**
```bash
python src/walkforward.py
```

**Step 8: 查看绩效评估（Phase 8）**
```bash
python src/evaluate.py
```
预期输出：控制台打印完整的绩效报告。

**Step 9: 生成可视化图表（Phase 9）**
```bash
python src/visualization.py
```
预期输出：`reports/figures/` 目录下生成 6 张 PNG 图。

### 5.4 一键运行

如果所有模块都已实现，可以通过主脚本一键运行完整流程：

```bash
# （待实现）python main.py --full
```

---

## 6. 技术栈与依赖

### 6.1 核心库

| 库名 | 用途 |
|------|------|
| `pandas` | 数据处理与分析 |
| `numpy` | 数值计算 |
| `yfinance` | 金融数据下载 |
| `lightgbm` | LightGBM 梯度提升模型 |
| `scikit-learn` | 机器学习工具（逻辑回归、随机森林、指标计算） |
| `matplotlib` | 数据可视化 |
| `seaborn` | 统计可视化（美化图表） |
| `pyarrow` | Parquet 文件读写 |

### 6.2 开发工具

| 库名 | 用途 |
|------|------|
| `pytest` | 单元测试 |
| `jupyter` | 探索性分析 |
| `ipykernel` | Jupyter 内核 |

### 6.3 requirements.txt 参考

```txt
# 数据处理
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# 数据获取
yfinance>=0.2.0

# 机器学习
scikit-learn>=1.3.0
lightgbm>=4.0.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 开发工具
pytest>=7.0.0
jupyter>=1.0.0
```

---

## 7. 成功标准

策略验证通过需要同时满足以下条件：

| 序号 | 标准 | 说明 |
|------|------|------|
| 1 | 夏普比率 > 买入持有 TQQQ | 风险调整后收益更优 |
| 2 | 最大回撤显著低于买入持有 TQQQ | 风险控制更好 |
| 3 | 样本外年化收益率 > 0 | 策略整体盈利 |
| 4 | Walk-Forward 各年度表现一致 | 不是单一年份驱动 |
| 5 | 在多种市场环境下均稳健 | 牛市、熊市、震荡市都有合理表现 |

---

## 8. 开发规范

### 8.1 代码风格
- 遵循 **PEP 8** 规范
- 所有函数和类必须有 **docstring**
- 使用 **type hints** 类型注解
- 每个模块职责单一

### 8.2 命名约定
| 类型 | 规则 | 示例 |
|------|------|------|
| 模块/文件 | 小写 + 下划线 | `data_loader.py` |
| 类 | 大驼峰（PascalCase） | `FeatureEngineer` |
| 函数/方法 | 小写 + 下划线 | `calculate_returns()` |
| 变量 | 小写 + 下划线 | `daily_returns` |
| 常量 | 全大写 + 下划线 | `TRAIN_START_DATE` |

### 8.3 关键原则
1. **严禁未来函数**：所有回测必须严格避免未来信息泄露
2. **可复现性**：固定随机种子，确保每次运行结果一致
3. **模块化设计**：每个 Phase 的代码独立，便于测试和替换
4. **配置化**：关键参数集中配置，不硬编码在逻辑中

### 8.4 Git 工作流
- 每个 Phase 开一个 feature 分支：`feature/phase2-data`
- 完成后合并到 `main` 分支
- Commit 信息格式：`type: subject`

---

## 9. Agent 工作指南

### 9.1 给 AI Agent 的工作方式

本项目按 Phase 顺序开发，**必须严格按顺序推进**，前一个 Phase 完成并验证后再进入下一个 Phase。

### 9.2 各 Phase 详细指令

#### Phase 1: 项目初始化
> **任务**：创建项目骨架
>
> **指令**：
> "创建完整的项目目录结构，包括 data/, src/, notebooks/, tests/, reports/figures/, models/。
> 编写 requirements.txt，包含 pandas, numpy, yfinance, lightgbm, scikit-learn, matplotlib, seaborn, pyarrow, pytest。
> 在 src/ 下创建 9 个模块文件的骨架（空文件或 import 骨架）：data_loader.py, feature_engineering.py, label_generator.py, train_model.py, portfolio_engine.py, walkforward.py, backtest.py, evaluate.py, visualization.py。"
>
> **验收**：运行 `tree /f` 目录结构完整，`pip install -r requirements.txt` 成功。

#### Phase 2: 数据层
> **任务**：实现数据下载
>
> **指令**：
> "在 src/data_loader.py 中实现 DataLoader 类。
> 使用 yfinance 下载 7 个标的的数据：QQQ, TQQQ, SQQQ, SPY, ^VIX, ^TNX, DX-Y.NYB。
> 时间范围：2011-01-01 到今天。
> 每个标的保存为独立 CSV 到 data/raw/ 目录。
> 提供命令行入口：if __name__ == '__main__': 下载所有数据并打印完成信息。"
>
> **验收**：运行 `python src/data_loader.py` 后，data/raw/ 下有 7 个 CSV 文件，每个文件包含 OHLCV 列。

#### Phase 3: 特征工程
> **任务**：实现特征计算（11大类，60+个特征）
>
> **指令**：
> "在 src/feature_engineering.py 中实现 FeatureEngineer 类。
> 实现 11 大类特征的计算：
>   1. 动量特征：ret_1d/3d/5d/10d/20d/60d/120d, mom_5_20, mom_20_60
>   2. 均线特征：ma5/10/20/60/120/200, ma5_ma20_ratio, ma20_ma60_ratio, ma20_ma200_ratio
>   3. 距离特征：dist_ma5/20/60/200
>   4. 波动率特征：vol_5/20/60/120, vol_5_vol_20_ratio, vol_20_vol_60_ratio
>   5. 价量形态：volume_ret_1d, volume_ma5/20, volume_ratio_5_20, obv, obv_5d_change, price_volume_divergence, close_position, upper_shadow, lower_shadow, body_size, high_low_range, high_low_range_ma20
>   6. VIX 特征：vix_level, vix_change_1d/5d/20d, vix_ma20/60_ratio, vix_quantile_60d
>   7. 相对强弱：rs_qv_spy, rs_qv_spy_ret_5d/20d/60d
>   8. 市场广度：spy_ret_1d/5d/20d, spy_vol_20, spy_dist_ma200, spy_high_252d_ratio, spy_low_252d_ratio
>   9. 利率特征：tnx_level, tnx_change_1d/5d/20d, tnx_ma20_ratio
>   10. 美元特征：dxy_level, dxy_change_1d/5d/20d, dxy_ma20_ratio, dxy_trend_60d
>   11. 交叉宏观：vix_spy_corr_20d, dxy_tnx_corr_20d
> 从 data/raw/ 加载所有 7 个 CSV，以 QQQ 日期为基准对齐。
> 实现 _remove_outliers() 方法处理极端值。
> 输出 features.parquet 到 data/processed/。
> 提供命令行入口可直接运行。"
>
> **验收**：运行后生成 features.parquet，包含 >= 50 个特征列，无 NaN/Inf，每类特征都有对应列。

#### Phase 4: 标签生成与阈值敏感性分析
> **任务**：实现标签生成 + 多阈值敏感性分析
>
> **指令**：
> "在 src/label_generator.py 中实现 LabelGenerator 类。
> 基于 QQQ 收盘价计算未来 5 日收益率 future_ret_5d = close.shift(-5)/close - 1。
> 默认按 ±2% 阈值打标签：>2% 为 RiskOn(2)，<-2% 为 RiskOff(0)，中间 Neutral(1)。
> 支持可配置参数：horizon(预测天数), threshold(对称阈值), threshold_up/threshold_down(非对称), use_quantile(分位数法), quantile(分位数比例)。
> 实现 threshold_sensitivity_analysis() 方法，遍历至少 5 种阈值方案：
>   - ±1.0%, ±1.5%, ±2.0%, ±2.5%, ±3.0%
>   - 非对称方案（+1.5%/-2.5%）
>   - 分位数方案（top/bottom 30%）
> 输出各方案的标签分布对比表，保存到 reports/threshold_analysis.csv。
> 将默认标签列合并到特征表，保存回 features.parquet。
> 提供命令行入口。"
>
> **验收**：features.parquet 新增 label 列；阈值分析覆盖 >= 5 种方案；三类标签分布合理；无未来函数。

#### Phase 5: 模型训练
> **任务**：实现模型训练
>
> **指令**：
> "在 src/train_model.py 中实现 ModelTrainer 类。
> 实现三个模型：LightGBM（主模型）、Random Forest、Logistic Regression。
> 输入特征表，划分 X 和 y，去除 NaN 行。
> 训练模型并保存到 models/ 目录。
> 实现 predict_proba() 方法，输出每日 P(RiskOn), P(Neutral), P(RiskOff)。
> 固定随机种子确保可复现。"
>
> **验收**：模型能训练并保存，预测概率之和≈1，训练集准确率>33%。

#### Phase 6: 组合引擎与交易成本
> **任务**：实现组合决策 + 交易成本/滑点/融资成本模拟
>
> **指令**：
> "在 src/portfolio_engine.py 中实现 PortfolioEngine 类。
> 决策规则：P(RiskOn)>0.6 → TQQQ，P(RiskOff)>0.6 → SQQQ，否则 Cash。
> 信号收盘生成，次日开盘执行（持仓后移一天）。
> 实现交易成本模拟：
>   - 佣金 commission_bps = 5 bps（每次买卖）
>   - 滑点 slippage_bps = 10 bps（每次买卖）
>   - 融资成本 financing_cost_daily = 0.0002/日（仅持有 TQQQ/SQQQ 时扣除）
> 换手时扣除佣金+滑点，持有期间按日扣除融资成本。
> 实现 calculate_nav() 方法计算扣除成本后的净值曲线。
> 支持 include_costs 参数，可关闭成本用于对比。
> 输入预测概率，输出每日持仓序列、每日收益率序列、净值序列。
> 可配置阈值和成本参数。"
>
> **验收**：输出三态持仓；有执行滞后；有成本vs无成本净值有合理差异；融资成本仅在杠杆仓位时扣除；换手次数×单次成本≈总成本（量级校验）。

#### Phase 7: Walk-Forward
> **任务**：实现多方案滚动验证（年频/季频 + 扩张/滚动窗口）
>
> **指令**：
> "在 src/walkforward.py 中实现 WalkForwardValidator 类。
> 支持三种滚动频率：yearly（年频）、quarterly（季频）、monthly（月频）。
> 支持两种窗口模式：expanding（扩张窗口）、rolling（固定窗口，window_years 可配置）。
> 实现 generate_splits() 方法：按配置生成所有训练/测试切分。
> 实现 run() 方法：遍历每个 split，训练 LightGBM 模型，预测测试集，通过组合引擎生成收益，收集样本外结果。
> 实现 compare_schemes() 方法：对比多种 Walk-Forward 方案（年频扩张、季频扩张、季频滚动5年）的绩效。
> 严禁使用随机划分，严禁测试集数据泄露到训练集。
> 输出样本外预测和收益序列，保存为 Parquet。"
>
> **验收**：样本外结果覆盖所有测试周期；训练/测试无时间重叠；支持 >=3 种方案对比；不同方案结果有合理差异；可复现。

#### Phase 8: 绩效评估
> **任务**：实现绩效计算
>
> **指令**：
> "在 src/evaluate.py 中实现 PerformanceEvaluator 类。
> 计算三大类指标：
>   1. 组合指标：CAGR、年化收益、波动率、Sharpe、Sortino、Calmar、最大回撤
>   2. 交易指标：胜率、换手率、平均持仓期、交易次数
>   3. 分类指标：Accuracy、Precision、Recall、F1、混淆矩阵
> 与三个基准策略（Buy&Hold、MA200、MA20/60）对比输出。
> 提供 print_report() 方法打印完整报告。"
>
> **验收**：所有指标能计算，结果合理，与基准对比清晰。

#### Phase 9: 可视化
> **任务**：实现图表生成
>
> **指令**：
> "在 src/visualization.py 中实现 Visualizer 类。
> 生成 6 张图并保存到 reports/figures/：
>   1. equity_curve.png - 净值曲线（策略+3个基准）
>   2. drawdown.png - 回撤曲线
>   3. rolling_sharpe.png - 滚动1年夏普
>   4. position_timeline.png - 持仓时间线
>   5. feature_importance.png - 特征重要性（前20个）
>   6. confusion_matrix.png - 混淆矩阵热力图
> 图表要有标题、坐标轴标签、图例，风格统一。"
>
> **验收**：6 张图全部生成，清晰美观，数据正确。

### 9.3 开发注意事项

1. **按顺序推进**：完成一个 Phase 并验证后再开始下一个
2. **每个 Phase 都要测试**：运行示例代码验证输出正确
3. **保持代码质量**：docstring、type hints、合理的函数拆分
4. **注意数据对齐**：多标的数据按日期对齐，避免 NaN 污染
5. **严防未来函数**：所有 shift 方向要正确，回测要模拟真实交易

---

## 10. 未来扩展

> **以下内容为 MVP 验证通过后的扩展方向，当前阶段不实现**

### Phase 2 资产扩展
- 加入 SOXL / SOXS（半导体 3x）
- 加入 UPRO / SPXU（标普 500 3x）
- 多资产配置优化

### 策略进阶
- 连续仓位管理（非满仓进出）
- 波动率目标（Volatility Targeting）
- 强化学习择时
- LLM 生成宏观状态特征
- Macro Agent 集成

---

## 免责声明

本项目仅供学习研究和量化探索使用，**不构成任何投资建议**。
杠杆 ETF 风险极高，使用本系统进行实盘交易的所有风险由使用者自行承担。

---

*LERS - 杠杆 ETF 状态切换系统 | 最后更新: 2026-06-25*
