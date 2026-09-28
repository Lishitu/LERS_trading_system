import os
import numpy as np
import pandas as pd


def generate_mock_data(save_dir: str = 'data/raw', start_date: str = '2015-01-01',
                       end_date: str = '2024-12-31', seed: int = 42) -> None:
    """生成模拟行情数据用于测试。

    Args:
        save_dir: 保存目录
        start_date: 开始日期
        end_date: 结束日期
        seed: 随机种子
    """
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    print(f"生成模拟数据: {start_date} ~ {end_date}, 共 {n_days} 个交易日")

    qqq_close = 100.0
    qqq_closes = []
    for i in range(n_days):
        daily_ret = np.random.normal(0.0005, 0.015)
        qqq_close *= (1 + daily_ret)
        qqq_closes.append(qqq_close)

    qqq_close = np.array(qqq_closes)
    qqq_open = qqq_close * (1 + np.random.normal(0, 0.005, n_days))
    qqq_high = np.maximum(qqq_close, qqq_open) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    qqq_low = np.minimum(qqq_close, qqq_open) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    qqq_volume = np.random.randint(10000000, 50000000, n_days).astype(float)

    qqq_df = pd.DataFrame({
        'Open': qqq_open,
        'High': qqq_high,
        'Low': qqq_low,
        'Close': qqq_close,
        'Adj Close': qqq_close,
        'Volume': qqq_volume,
    }, index=dates)
    qqq_df.to_csv(os.path.join(save_dir, 'qqq.csv'))
    print(f"  QQQ: {len(qqq_df)} 行")

    tqqq_close = 50.0
    tqqq_closes = []
    for i in range(n_days):
        daily_ret = np.random.normal(0.0015, 0.045)
        tqqq_close *= (1 + daily_ret)
        tqqq_closes.append(tqqq_close)

    tqqq_close = np.array(tqqq_closes)
    tqqq_open = tqqq_close * (1 + np.random.normal(0, 0.01, n_days))
    tqqq_high = np.maximum(tqqq_close, tqqq_open) * (1 + np.abs(np.random.normal(0, 0.015, n_days)))
    tqqq_low = np.minimum(tqqq_close, tqqq_open) * (1 - np.abs(np.random.normal(0, 0.015, n_days)))
    tqqq_volume = np.random.randint(5000000, 30000000, n_days).astype(float)

    tqqq_df = pd.DataFrame({
        'Open': tqqq_open,
        'High': tqqq_high,
        'Low': tqqq_low,
        'Close': tqqq_close,
        'Adj Close': tqqq_close,
        'Volume': tqqq_volume,
    }, index=dates)
    tqqq_df.to_csv(os.path.join(save_dir, 'tqqq.csv'))
    print(f"  TQQQ: {len(tqqq_df)} 行")

    sqqq_close = 50.0
    sqqq_closes = []
    for i in range(n_days):
        daily_ret = np.random.normal(-0.0015, 0.045)
        sqqq_close *= (1 + daily_ret)
        sqqq_closes.append(sqqq_close)

    sqqq_close = np.array(sqqq_closes)
    sqqq_open = sqqq_close * (1 + np.random.normal(0, 0.01, n_days))
    sqqq_high = np.maximum(sqqq_close, sqqq_open) * (1 + np.abs(np.random.normal(0, 0.015, n_days)))
    sqqq_low = np.minimum(sqqq_close, sqqq_open) * (1 - np.abs(np.random.normal(0, 0.015, n_days)))
    sqqq_volume = np.random.randint(5000000, 30000000, n_days).astype(float)

    sqqq_df = pd.DataFrame({
        'Open': sqqq_open,
        'High': sqqq_high,
        'Low': sqqq_low,
        'Close': sqqq_close,
        'Adj Close': sqqq_close,
        'Volume': sqqq_volume,
    }, index=dates)
    sqqq_df.to_csv(os.path.join(save_dir, 'sqqq.csv'))
    print(f"  SQQQ: {len(sqqq_df)} 行")

    spy_close = 150.0
    spy_closes = []
    for i in range(n_days):
        daily_ret = np.random.normal(0.0004, 0.012)
        spy_close *= (1 + daily_ret)
        spy_closes.append(spy_close)

    spy_close = np.array(spy_closes)
    spy_df = pd.DataFrame({
        'Close': spy_close,
        'Adj Close': spy_close,
    }, index=dates)
    spy_df.to_csv(os.path.join(save_dir, 'spy.csv'))
    print(f"  SPY: {len(spy_df)} 行")

    vix_close = 20.0
    vix_closes = []
    for i in range(n_days):
        daily_change = np.random.normal(0, 0.05) * vix_close
        vix_close = max(10, min(80, vix_close + daily_change))
        vix_closes.append(vix_close)

    vix_close = np.array(vix_closes)
    vix_df = pd.DataFrame({
        'Close': vix_close,
    }, index=dates)
    vix_df.to_csv(os.path.join(save_dir, 'vix.csv'))
    print(f"  VIX: {len(vix_df)} 行")

    tnx_close = 2.5
    tnx_closes = []
    for i in range(n_days):
        daily_change = np.random.normal(0, 0.02)
        tnx_close = max(0.5, min(6.0, tnx_close + daily_change))
        tnx_closes.append(tnx_close)

    tnx_close = np.array(tnx_closes)
    tnx_df = pd.DataFrame({
        'Close': tnx_close,
    }, index=dates)
    tnx_df.to_csv(os.path.join(save_dir, 'tnx.csv'))
    print(f"  TNX: {len(tnx_df)} 行")

    dxy_close = 100.0
    dxy_closes = []
    for i in range(n_days):
        daily_ret = np.random.normal(0, 0.003)
        dxy_close *= (1 + daily_ret)
        dxy_closes.append(dxy_close)

    dxy_close = np.array(dxy_closes)
    dxy_df = pd.DataFrame({
        'Close': dxy_close,
    }, index=dates)
    dxy_df.to_csv(os.path.join(save_dir, 'dxy.csv'))
    print(f"  DXY: {len(dxy_df)} 行")

    print(f"\n模拟数据已生成到: {save_dir}/")


if __name__ == '__main__':
    generate_mock_data()
