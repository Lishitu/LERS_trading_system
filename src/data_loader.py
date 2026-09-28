import os
import pandas as pd
import numpy as np
import yfinance as yf


class DataLoader:
    """数据加载器，负责从 Yahoo Finance 下载和加载原始行情数据。"""

    TICKERS = {
        'QQQ': 'QQQ',
        'TQQQ': 'TQQQ',
        'SQQQ': 'SQQQ',
        'PSQ': 'PSQ',     # 纳指1倍做空
        'SPY': 'SPY',
        'IWM': 'IWM',     # 罗素2000 (市场广度)
        'DIA': 'DIA',     # 道琼斯 (市场广度)
        'HYG': 'HYG',     # 垃圾债 (贪婪指数代理)
        'IEF': 'IEF',     # 7-10年期国债 (贪婪指数代理分母)
        'TLT': 'TLT',     # 20年+期国债 (恐慌指数代理)
        '^VIX': 'VIX',
        '^TNX': 'TNX',
        'DX-Y.NYB': 'DXY'
    }

    def __init__(
        self,
        save_dir: str = 'data/raw',
        start_date: str = '2011-01-01',
        end_date: str | None = None
    ):
        """
        初始化数据加载器。

        Args:
            save_dir: 数据保存目录路径
            start_date: 数据起始日期，格式 'YYYY-MM-DD'
            end_date: 数据结束日期，格式 'YYYY-MM-DD'，None 表示至今
        """
        self.save_dir = save_dir
        self.start_date = start_date
        self.end_date = end_date
        os.makedirs(self.save_dir, exist_ok=True)

    def _download_single(self, ticker: str, name: str) -> pd.DataFrame | None:
        """
        下载单个标的的行情数据。

        Args:
            ticker: Yahoo Finance 标的代码
            name: 标的名称（用于保存文件名）

        Returns:
            包含行情数据的 DataFrame，下载失败返回 None
        """
        try:
            print(f"正在下载 {name} ({ticker}) ...")
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if df.empty:
                print(f"  警告: {name} 数据为空")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1) # Drop the 'Ticker' level
            df = df.sort_index()
            filepath = os.path.join(self.save_dir, f"{name.lower()}.csv")
            df.to_csv(filepath)
            print(f"  成功: {len(df)} 行数据，已保存到 {filepath}")
            return df
        except Exception as e:
            print(f"  错误: 下载 {name} 失败 - {e}")
            return None

    def validate_data(self, df: pd.DataFrame, name: str) -> dict:
        """
        校验数据质量，检查缺失值和数据量。

        Args:
            df: 待校验的 DataFrame
            name: 标的名称

        Returns:
            包含校验结果的字典
        """
        result = {
            'name': name,
            'rows': len(df),
            'columns': list(df.columns),
            'missing_total': 0,
            'missing_by_column': {},
            'date_range': None,
            'valid': True
        }
        try:
            if df.empty:
                result['valid'] = False
                result['message'] = '数据为空'
                return result
            result['date_range'] = (df.index[0], df.index[-1])
            for col in df.columns:
                missing = df[col].isna().sum()
                result['missing_by_column'][col] = int(missing)
                result['missing_total'] += int(missing)
            if result['missing_total'] > 0:
                print(f"  校验: {name} 存在 {result['missing_total']} 个缺失值")
            else:
                print(f"  校验: {name} 数据完整，共 {result['rows']} 行")
        except Exception as e:
            result['valid'] = False
            result['message'] = f'校验异常: {e}'
        return result

    def download_all(self) -> dict:
        """
        下载所有标的的数据。

        Returns:
            包含下载结果统计的字典
        """
        print(f"开始下载所有标的数据 ({self.start_date} ~ {self.end_date or '至今'})")
        print(f"共 {len(self.TICKERS)} 个标的\n")
        results = {
            'success': [],
            'failed': [],
            'validation': {},
            'data': {}
        }
        for ticker, name in self.TICKERS.items():
            df = self._download_single(ticker, name)
            if df is not None:
                results['success'].append(name)
                results['data'][name] = df
                validation = self.validate_data(df, name)
                results['validation'][name] = validation
            else:
                results['failed'].append(name)
            print()
        print("=" * 50)
        print(f"下载完成: 成功 {len(results['success'])} 个，失败 {len(results['failed'])} 个")
        if results['failed']:
            print(f"失败标的: {', '.join(results['failed'])}")
        print("=" * 50)
        return results

    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        从本地 CSV 加载所有标的的数据。

        Returns:
            键为标的名称，值为 DataFrame 的字典
        """
        data = {}
        print(f"从 {self.save_dir} 加载数据...")
        for ticker, name in self.TICKERS.items():
            filepath = os.path.join(self.save_dir, f"{name.lower()}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    data[name] = df
                    print(f"  已加载 {name}: {len(df)} 行")
                except Exception as e:
                    print(f"  加载 {name} 失败: {e}")
            else:
                print(f"  文件不存在: {filepath}")
        print(f"\n共加载 {len(data)} 个标的")
        return data


if __name__ == '__main__':
    loader = DataLoader(save_dir='data/raw', start_date='2011-01-01')
    results = loader.download_all()
    print("\n--- 下载统计 ---")
    for name, val in results['validation'].items():
        date_range = val['date_range']
        if date_range:
            print(f"{name}: {val['rows']} 行, {date_range[0].date()} ~ {date_range[1].date()}, 缺失值: {val['missing_total']}")
