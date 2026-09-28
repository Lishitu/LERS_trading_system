"""Moomoo OpenD 数据加载与本地增量持久化缓存模块。

提供美股高流动性 ETF 行情下载、自动计算真实 QFQ VWAP（成交额/成交量折算）以及本地 Parquet 缓存管理。
严格遵循 Moomoo OpenD 额度限制（每周 300 只限制），优先读取本地缓存，仅在必要时请求接口。
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from moomoo import (
        OpenQuoteContext,
        KLType,
        AuType,
        RET_OK,
    )
except ImportError:
    OpenQuoteContext = None
    KLType = None
    AuType = None
    RET_OK = 0


# 标的池定义
DEFAULT_BENCHMARKS = ['US.SPY', 'US.QQQ']
DEFAULT_SECTORS = [
    'US.XLK',   # 科技
    'US.SMH',   # 半导体
    'US.XLF',   # 金融
    'US.XLI',   # 工业
    'US.XLE',   # 能源
    'US.XBI',   # 生物医药
    'US.IWM',   # 罗素2000/小盘成长
    'US.DIA',   # 道琼斯30蓝筹/价值
    'US.DXJ',   # QDII: 日本股票对冲
    'US.COPX',  # 商品/有色: 铜矿
    'US.SLV',   # 商品/贵金属: 白银
    'US.GLD',   # 商品/避险: 黄金
    'US.URA',   # 新增: 铀/核能主题
    'US.XOP',   # 新增: 油气开采
]


# 进攻性杠杆映射表
LEVERAGED_MAP_2X = {
    'US.QQQ': 'US.QLD',  # 2x 纳指100
    'US.SMH': 'US.USD',  # 2x 半导体
}

LEVERAGED_MAP_3X = {
    'US.QQQ': 'US.TQQQ', # 3x 纳指100
    'US.SMH': 'US.SOXL', # 3x 半导体
    'US.XLK': 'US.TQQQ', # 3x 科技
    'US.XLF': 'US.FAS',  # 3x 金融
}

# 全局杠杆合并表 (用于数据预加载)
LEVERAGED_MAP = {
    'US.QQQ': 'US.TQQQ',
    'US.SMH': 'US.SOXL',
    'US.XLK': 'US.TQQQ',
    'US.XLF': 'US.FAS',
    'US.QLD_BASE': 'US.QLD',
    'US.USD_BASE': 'US.USD',
}

# 防御型资产（超短债 / 现金等价物）
DEFAULT_DEFENSE = 'US.BIL'


class MoomooDataLoader:
    """Moomoo OpenD 数据加载器。

    Attributes:
        host: OpenD 监听主机地址，默认 127.0.0.1
        port: OpenD 监听端口，默认 11111
        cache_dir: 本地 Parquet 缓存目录，默认 data/moomoo
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        cache_dir: str = "data/moomoo",
    ):
        self.host = host
        self.port = port
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._ctx: Optional[OpenQuoteContext] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _get_context(self) -> OpenQuoteContext:
        """获取或创建 OpenD 引用上下文。"""
        if self._ctx is None:
            if OpenQuoteContext is None:
                raise ImportError("未安装 moomoo SDK，请先安装: pip install moomoo-api")
            self._ctx = OpenQuoteContext(host=self.host, port=self.port)
        return self._ctx

    def close(self):
        """释放 OpenD 链接连接。"""
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
            self._ctx = None

    def _get_cache_path(self, code: str) -> str:
        clean_code = code.replace(".", "_")
        return os.path.join(self.cache_dir, f"{clean_code}.parquet")

    def fetch_ticker_data(
        self,
        code: str,
        start_date: str = "2016-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """获取单个标的的历史日K数据（含精准 QFQ VWAP）。

        优先从本地 Parquet 缓存读取；如需更新则从 Moomoo OpenD 抓取。

        Args:
            code: 标的代码，例如 'US.SPY'
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD，若为 None 则为最新日期
            force_refresh: 是否强制忽略缓存向 OpenD 发起请求

        Returns:
            DataFrame 包含 [time_key, open, high, low, close, volume, turnover, vwap]
            以 time_key (pd.Timestamp) 为索引并按升序排列。
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(code)
        
        # 1. 优先尝试从本地缓存加载
        if not force_refresh and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                df['time_key'] = pd.to_datetime(df['time_key'])
                df = df.sort_values('time_key').set_index('time_key')
                # 检查日期覆盖范围 (首个交易日常遇元旦/周末，允许 5 天以内节假日偏移)
                cache_start_ts = df.index.min()
                cache_end_ts = df.index.max()
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)

                if cache_start_ts <= start_ts + pd.Timedelta(days=5) and (end_ts - cache_end_ts).days <= 5:
                    print(f"[DataLoader] 命中本地持久化缓存: {cache_path} ({cache_start_ts.strftime('%Y-%m-%d')} ~ {cache_end_ts.strftime('%Y-%m-%d')})")
                    return df[start_date:end_date]
            except Exception as e:
                print(f"[DataLoader] 读取本地缓存 {cache_path} 异常: {e}，将重新从 OpenD 下载")

        # 2. 从 OpenD 远程请求
        print(f"[DataLoader] 正在从 Moomoo OpenD 下载 {code} (start={start_date}, end={end_date or 'latest'}) ...")
        ctx = self._get_context()

        # 分页获取全量前复权数据
        page_key = None
        qfq_list = []
        while True:
            ret, data, page_key = ctx.request_history_kline(
                code=code,
                start=start_date,
                end=end_date,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                page_req_key=page_key,
            )
            if ret != RET_OK:
                raise RuntimeError(f"请求 {code} 前复权历史K线失败: {data}")
            qfq_list.append(data)
            if page_key is None:
                break
            time.sleep(0.05)

        # 分页获取未复权数据（用于精准折算 VWAP）
        page_key = None
        raw_list = []
        while True:
            ret, data, page_key = ctx.request_history_kline(
                code=code,
                start=start_date,
                end=end_date,
                ktype=KLType.K_DAY,
                autype=AuType.NONE,
                page_req_key=page_key,
            )
            if ret != RET_OK:
                raise RuntimeError(f"请求 {code} 未复权历史K线失败: {data}")
            raw_list.append(data)
            if page_key is None:
                break
            time.sleep(0.05)

        df_qfq = pd.concat(qfq_list, ignore_index=True)
        df_raw = pd.concat(raw_list, ignore_index=True)

        # 校验对齐
        df_qfq['time_key'] = pd.to_datetime(df_qfq['time_key'])
        df_raw['time_key'] = pd.to_datetime(df_raw['time_key'])

        merged = pd.merge(
            df_qfq[['time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover']],
            df_raw[['time_key', 'close', 'turnover', 'volume']].rename(
                columns={'close': 'raw_close', 'turnover': 'raw_turnover', 'volume': 'raw_volume'}
            ),
            on='time_key',
            how='inner',
        )

        # 计算日度 VWAP：
        # raw_vwap = raw_turnover / raw_volume
        # adj_factor = close_qfq / raw_close
        # qfq_vwap = raw_vwap * adj_factor
        has_volume = (merged['raw_volume'] > 0) & (merged['raw_close'] > 0)
        
        merged['raw_vwap'] = np.where(
            has_volume,
            merged['raw_turnover'] / merged['raw_volume'],
            merged['raw_close']
        )
        merged['adj_factor'] = np.where(
            merged['raw_close'] > 0,
            merged['close'] / merged['raw_close'],
            1.0
        )
        merged['vwap'] = merged['raw_vwap'] * merged['adj_factor']

        # 边界防错：若 vwap 异常脱离 [low, high]，收敛至 [low, high]
        merged['vwap'] = np.clip(merged['vwap'], merged['low'], merged['high'])

        result_df = merged[['time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'vwap']].copy()
        
        # 写入本地 Parquet 缓存
        result_df.to_parquet(cache_path, index=False)
        print(f"[DataLoader] 已持久化缓存至 {cache_path}，共 {len(result_df)} 个交易日数据。")

        result_df = result_df.sort_values('time_key').set_index('time_key')
        return result_df[start_date:end_date]

    def load_all_universe(
        self,
        start_date: str = "2016-01-01",
        end_date: Optional[str] = None,
        additional_tickers: Optional[List[str]] = None,
        force_refresh: bool = False,
        auto_close: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """批量加载整个策略宇宙所有标的的数据。

        包括：基准标的、行业/主题1x标的、进攻性杠杆标的、防御标的。

        Args:
            start_date: 起始日期
            end_date: 结束日期
            additional_tickers: 额外补充标的列表
            force_refresh: 是否强制忽略缓存刷新
            auto_close: 加载完成后是否自动释放并关闭 OpenD 连接，默认 True

        Returns:
            Dict[ticker, DataFrame]
        """
        all_tickers = set(DEFAULT_BENCHMARKS + DEFAULT_SECTORS + list(LEVERAGED_MAP.values()) + [DEFAULT_DEFENSE])
        if additional_tickers:
            all_tickers.update(additional_tickers)

        results = {}
        try:
            for ticker in sorted(all_tickers):
                df = self.fetch_ticker_data(
                    code=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh,
                )
                results[ticker] = df
        finally:
            if auto_close:
                self.close()

        return results


if __name__ == "__main__":
    loader = MoomooDataLoader()
    try:
        data = loader.load_all_universe(start_date="2016-01-01", force_refresh=True)
        print("\n=== 加载完成，标的池列表 ===")
        for k, v in data.items():
            print(f"标的: {k:10s} | 起始: {v.index.min().strftime('%Y-%m-%d')} | 结束: {v.index.max().strftime('%Y-%m-%d')} | 记录数: {len(v)}")
    finally:
        loader.close()
