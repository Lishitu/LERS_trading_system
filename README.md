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

**特征清单（共 8 大类）**：

#### 3.1 动量特征（Momentum）
| 特征名 | 计算公式 |
|--------|----------|
| ret_1d | close / close.shift(1) - 1 |
| ret_5d | close / close.shift(5) - 1 |
| ret_10d | close / close.shift(10) - 1 |
| ret_20d | close / close.shift(20) - 1 |
| ret_60d | close / close.shift(60) - 1 |

#### 3.2 均线特征（Moving Average）
| 特征名 | 说明 |
|--------|------|
| ma5 | 5 日均线 |
| ma10 | 10 日均线 |
| ma20 | 20 日均线 |
| ma60 | 60 日均线 |
| ma120 | 120 日均线 |
| ma200 | 200 日均线 |

#### 3.3 距离特征（Distance to MA）
| 特征名 | 计算公式 |
|--------|----------|
| dist_ma20 | close / ma20 - 1 |
| dist_ma60 | close / ma60 - 1 |
| dist_ma200 | close / ma200 - 1 |

#### 3.4 波动率特征（Volatility）
| 特征名 | 说明 |
|--------|------|
| vol_5 | 5 日收益率标准差（年化） |
| vol_20 | 20 日收益率标准差（年化） |
| vol_60 | 60 日收益率标准差（年化） |

#### 3.5 VIX 特征
| 特征名 | 计算公式 |
|--------|----------|
| vix_level | VIX 当日收盘价 |
| vix_change_5d | vix / vix.shift(5) - 1 |
| vix_ma20_ratio | vix / vix_ma20 - 1 |

#### 3.6 相对强弱特征（Relative Strength）
| 特征名 | 计算公式 |
|--------|----------|
| rs_qv_spy | QQQ / SPY 比值 |
| rs_qv_spy_ret_5d | (QQQ/SPY) / (QQQ/SPY).shift(5) - 1 |
| rs_qv_spy_ret_20d | (QQQ/SPY) / (QQQ/SPY).shift(20) - 1 |

#### 3.7 利率特征（Interest Rate）
| 特征名 | 计算公式 |
|--------|----------|
| tnx_level | TNX 当日收益率 |
| tnx_change_5d | tnx / tnx.shift(5) - 1 |

#### 3.8 美元特征（Dollar）
| 特征名 | 计算公式 |
|--------|----------|
| dxy_level | DXY 当日收盘 |
| dxy_change_5d | dxy / dxy.shift(5) - 1 |

**具体步骤**：
1. 在 `src/feature_engineering.py` 中实现 `FeatureEngineer` 类
2. 实现 `load_raw_data()` 方法：加载所有原始 CSV
3. 为每类特征实现一个计算方法（如 `_calc_momentum()`, `_calc_ma()` 等）
4. 实现 `build_features()` 主方法：组合所有特征
5. 输出为 Parquet 格式保存到 `data/processed/features.parquet`
6. 特征以 QQQ 为基准对齐时间索引

**交付文件**：
- `src/feature_engineering.py`
- `data/processed/features.parquet`

**验证方式**：
- 特征矩阵行数 = QQQ 交易日天数
- 特征列数 >= 25 列
- 无 Inf 或 NaN（去除初始窗口的 NaN 行后）
- 特征值范围合理（如收益率不会超过 50%/天）

---

### Phase 4: 标签生成

**目标**：基于未来收益生成市场状态标签。

**输入**：QQQ 的收盘价序列

**输出**：带有 label 列的特征表

**标签规则**：

使用未来 5 个交易日的收益率来定义市场状态：

```python
future_ret_5d = close.shift(-5) / close - 1
```

| 未来 5 日收益率 | 市场状态 | 标签编码 |
|-----------------|----------|----------|
| > 2% | Risk-On | 2 |
| < -2% | Risk-Off | 0 |
| -2% ~ +2% | Neutral | 1 |

**具体步骤**：
1. 在 `src/label_generator.py` 中实现 `LabelGenerator` 类
2. 实现 `generate_labels()` 方法：
   - 计算未来收益率
   - 根据阈值打标签
3. 可配置参数：
   - `horizon`: 预测天数（默认 5）
   - `threshold`: 分类阈值（默认 0.02）
4. 将标签合并到特征表中
5. 输出带标签的完整数据集

**交付文件**：
- `src/label_generator.py`
- 更新后的 `data/processed/features.parquet`（含 label 列）

**验证方式**：
- 三类标签的分布比例合理（没有某类 > 80% 或 < 5%）
- 没有未来函数泄露（检查 shift 方向是否正确）
- 最后 5 行的 label 为 NaN（合理，因为没有未来数据）

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

### Phase 6: 组合决策引擎

**目标**：将模型预测概率转换为具体的交易仓位。

**输入**：每日预测概率

**输出**：每日持仓信号

**决策规则**：

```
若 P(RiskOn) > 0.60:
    持有 TQQQ（3x 做多）

否则若 P(RiskOff) > 0.60:
    持有 SQQQ（3x 做空）

否则:
    持有 Cash（现金）
```

**执行规则**：
- 信号在**收盘后**生成
- 交易在**下一个交易日开盘**执行
- **严禁未来函数**（严禁使用当日收盘价当日交易）

**仓位约束**：
- 每次只持有一种仓位（TQQQ / SQQQ / Cash 三选一）
- 满仓进出，不加杠杆叠加

**具体步骤**：
1. 在 `src/portfolio_engine.py` 中实现 `PortfolioEngine` 类
2. 实现 `generate_signals()` 方法：
   - 输入预测概率
   - 根据阈值规则生成每日目标仓位
3. 实现 `apply_execution_lag()` 方法：
   - 将信号后移一天（次日开盘执行）
4. 可配置参数：
   - `risk_on_threshold`（默认 0.60）
   - `risk_off_threshold`（默认 0.60）

**交付文件**：
- `src/portfolio_engine.py`

**验证方式**：
- 持仓只有三种状态：TQQQ、SQQQ、Cash
- 信号有合理的换手率（不是每天都换）
- 执行滞后正确（T 日信号 → T+1 日持仓）

---

### Phase 7: 滚动 Walk-Forward 验证

**目标**：使用滚动时间窗口验证策略，避免过拟合。

**输入**：特征 + 标签 + 模型 + 组合引擎

**输出**：样本外每日收益序列

**为什么用 Walk-Forward**：
- ❌ **禁止**使用随机划分训练集/测试集（数据泄露风险）
- ✅ **必须**使用时间序列滚动验证

**Walk-Forward 方案**：

```
示例（年频滚动）：

第1轮:  Train: 2011-2018  |  Test: 2019
第2轮:  Train: 2011-2019  |  Test: 2020
第3轮:  Train: 2011-2020  |  Test: 2021
第4轮:  Train: 2011-2021  |  Test: 2022
第5轮:  Train: 2011-2022  |  Test: 2023
第6轮:  Train: 2011-2023  |  Test: 2024
...
```

- 训练窗口：**扩张式**（从 2011 年开始，逐年累加）
- 测试窗口：**1 年**
- 最终结果：拼接所有年份的样本外预测

**具体步骤**：
1. 在 `src/walkforward.py` 中实现 `WalkForwardValidator` 类
2. 实现 `generate_splits()` 方法：
   - 按年份划分训练/测试集
   - 训练集始终包含 2011 年到测试年前一年
3. 实现 `run()` 方法：
   - 遍历每个 split
   - 在训练集上训练模型
   - 在测试集上预测
   - 收集所有样本外预测结果
4. 输出完整的样本外预测序列

**交付文件**：
- `src/walkforward.py`
- 样本外预测结果（CSV 或 Parquet）

**验证方式**：
- 测试集没有重叠
- 训练集不包含任何测试集日期
- 样本外结果覆盖所有测试年份
- 能复现每个测试年份的预测

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
> **任务**：实现特征计算
>
> **指令**：
> "在 src/feature_engineering.py 中实现 FeatureEngineer 类。
> 实现 8 大类特征的计算：动量(ret_1d/5d/10d/20d/60d)、均线(ma5/10/20/60/120/200)、距离(dist_ma20/60/200)、波动率(vol_5/20/60)、VIX、相对强弱(QQQ/SPY)、利率(TNX)、美元(DXY)。
> 从 data/raw/ 加载所有 CSV，以 QQQ 日期为基准对齐。
> 输出 features.parquet 到 data/processed/。
> 提供命令行入口可直接运行。"
>
> **验收**：运行后生成 features.parquet，包含 >= 25 个特征列，无 NaN/Inf。

#### Phase 4: 标签生成
> **任务**：实现标签生成
>
> **指令**：
> "在 src/label_generator.py 中实现 LabelGenerator 类。
> 基于 QQQ 收盘价计算未来 5 日收益率 future_ret_5d = close.shift(-5)/close - 1。
> 按阈值打标签：>2% 为 RiskOn(2)，<-2% 为 RiskOff(0)，中间 Neutral(1)。
> 将标签列合并到特征表，保存回 features.parquet。
> 提供命令行入口。"
>
> **验收**：features.parquet 新增 label 列，三类标签分布合理，无未来函数。

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

#### Phase 6: 组合引擎
> **任务**：实现组合决策
>
> **指令**：
> "在 src/portfolio_engine.py 中实现 PortfolioEngine 类。
> 决策规则：P(RiskOn)>0.6 → TQQQ，P(RiskOff)>0.6 → SQQQ，否则 Cash。
> 信号收盘生成，次日开盘执行（持仓后移一天）。
> 输入预测概率，输出每日持仓序列和每日收益率序列（需加载 TQQQ/SQQQ 价格计算）。
> 可配置阈值参数。"
>
> **验收**：输出三态持仓，有执行滞后，收益率计算正确。

#### Phase 7: Walk-Forward
> **任务**：实现滚动验证
>
> **指令**：
> "在 src/walkforward.py 中实现 WalkForwardValidator 类。
> 按年份划分：训练集从 2011 到测试年前一年，测试集为该年。
> 遍历每个测试年份，逐年训练预测。
> 拼接所有样本外结果，输出完整的样本外预测序列。
> 严禁使用随机划分，严禁测试集数据泄露到训练集。"
>
> **验收**：样本外结果覆盖所有测试年，训练/测试无时间重叠，可复现。

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
