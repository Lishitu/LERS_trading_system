"""多因子选基与组合优化单元测试。"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategy_multifactor import MultiFactorETFStrategy
from portfolio_allocator import PortfolioAllocator


def test_multifactor_zscore_computation():
    strategy = MultiFactorETFStrategy(benchmark='US.QQQ', top_n=3)

    date = pd.Timestamp('2024-01-02')
    sectors = ['US.XLK', 'US.SMH', 'US.GLD']

    # 构造测试因子矩阵
    factors_dict = {
        'momentum': pd.DataFrame({'US.XLK': [1.0], 'US.SMH': [2.0], 'US.GLD': [0.0]}, index=[date]),
        'acceleration': pd.DataFrame({'US.XLK': [0.2], 'US.SMH': [0.5], 'US.GLD': [-0.1]}, index=[date]),
        'trend_strength': pd.DataFrame({'US.XLK': [0.1], 'US.SMH': [0.3], 'US.GLD': [0.0]}, index=[date]),
    }

    comp = strategy.compute_daily_composite_score(date, factors_dict, sectors)

    # 验证 SMH 得分最高，GLD 得分最低
    assert comp['US.SMH'] > comp['US.XLK'] > comp['US.GLD']


def test_factor_tilted_allocation():
    allocator = PortfolioAllocator(max_asset_weight=0.35, cash_ticker='US.BIL')

    # 给定 4 个标的，附带多因子综合得分 (ticker, vol, score)
    assets = [('A', 0.20, 1.5), ('B', 0.20, 0.2), ('C', 0.20, -0.5), ('D', 0.20, -1.5)]
    weights = allocator.compute_target_weights(assets)

    # 相同波动率下，多因子得分更高的 A, B 应获得更高的倾斜权重
    assert weights['A'] >= weights['B'] > weights['C'] > weights['D']
    # 验证 Cap 约束有效
    assert weights['A'] <= 0.35 + 1e-5
