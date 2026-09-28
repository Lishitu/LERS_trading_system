import os
import pandas as pd
import numpy as np


class FeatureEngineer:
    """特征工程类，用于从原始行情数据中构建预测特征。

    该类从 data/raw/ 目录加载 7 个标的的 CSV 数据，以 QQQ 日期为基准对齐，
    然后计算 11 大类共 70+ 个技术指标和宏观特征。

    Attributes:
        raw_dir: 原始数据目录路径
        processed_dir: 处理后数据保存目录路径
        data: 存储加载的原始数据字典
    """

    TICKERS = [
        'QQQ', 'TQQQ', 'SQQQ', 'PSQ', 'SPY', 'IWM', 'DIA',
        'HYG', 'IEF', 'TLT', 'VIX', 'TNX', 'DXY'
    ]

    def __init__(
        self,
        raw_dir: str = 'data/raw',
        processed_dir: str = 'data/processed'
    ):
        """初始化特征工程类。

        Args:
            raw_dir: 原始数据目录路径
            processed_dir: 处理后数据保存目录路径
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.data: dict[str, pd.DataFrame] = {}
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """从 data/raw/ 加载 7 个 CSV，以 QQQ 日期为基准对齐。

        Returns:
            对齐后的 DataFrame，包含 QQQ 行情数据及其他标的的收盘价列

        Raises:
            FileNotFoundError: 当 QQQ 数据文件不存在时抛出
        """
        print(f"从 {self.raw_dir} 加载原始数据...")

        for ticker in self.TICKERS:
            filepath = os.path.join(self.raw_dir, f"{ticker.lower()}.csv")
            if not os.path.exists(filepath):
                print(f"  警告: {ticker} 文件不存在: {filepath}")
                continue
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            df = df.sort_index()
            self.data[ticker] = df
            print(f"  已加载 {ticker}: {len(df)} 行")

        if 'QQQ' not in self.data:
            raise FileNotFoundError("QQQ 数据文件不存在，无法构建特征")

        qqq = self.data['QQQ'].copy()
        qqq.columns = [f"qqq_{col.lower()}" for col in qqq.columns]

        for ticker in self.TICKERS:
            if ticker == 'QQQ' or ticker not in self.data:
                continue
            ticker_lower = ticker.lower()
            df = self.data[ticker].copy()
            close_col = 'Close' if 'Close' in df.columns else df.columns[0]
            qqq[f"{ticker_lower}_close"] = df[close_col].reindex(qqq.index)

        print(f"  对齐后共 {len(qqq)} 个交易日")
        return qqq

    def _calc_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量特征（基于 QQQ 收盘价）。

        Args:
            df: 包含 qqq_close 列的 DataFrame

        Returns:
            添加动量特征后的 DataFrame
        """
        close = df['qqq_close']

        df['ret_1d'] = close.pct_change(1)
        df['ret_3d'] = close.pct_change(3)
        df['ret_5d'] = close.pct_change(5)
        df['ret_10d'] = close.pct_change(10)
        df['ret_20d'] = close.pct_change(20)
        df['ret_60d'] = close.pct_change(60)
        df['ret_120d'] = close.pct_change(120)

        df['mom_5_20'] = df['ret_5d'] - df['ret_20d']
        df['mom_20_60'] = df['ret_20d'] - df['ret_60d']

        return df

    def _calc_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线特征（QQQ）。

        Args:
            df: 包含 qqq_close 列的 DataFrame

        Returns:
            添加均线特征后的 DataFrame
        """
        close = df['qqq_close']

        df['ma5'] = close.rolling(window=5).mean()
        df['ma10'] = close.rolling(window=10).mean()
        df['ma20'] = close.rolling(window=20).mean()
        df['ma60'] = close.rolling(window=60).mean()
        df['ma120'] = close.rolling(window=120).mean()
        df['ma200'] = close.rolling(window=200).mean()

        df['ma5_ma20_ratio'] = df['ma5'] / df['ma20'] - 1
        df['ma20_ma60_ratio'] = df['ma20'] / df['ma60'] - 1
        df['ma20_ma200_ratio'] = df['ma20'] / df['ma200'] - 1

        return df

    def _calc_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算距离特征（QQQ）。

        Args:
            df: 包含 qqq_close、ma5、ma20、ma60、ma200 列的 DataFrame

        Returns:
            添加距离特征后的 DataFrame
        """
        close = df['qqq_close']

        df['dist_ma5'] = close / df['ma5'] - 1
        df['dist_ma20'] = close / df['ma20'] - 1
        df['dist_ma60'] = close / df['ma60'] - 1
        df['dist_ma200'] = close / df['ma200'] - 1

        return df

    def _calc_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征（QQQ）。

        Args:
            df: 包含 ret_1d 列的 DataFrame

        Returns:
            添加波动率特征后的 DataFrame
        """
        ret = df['ret_1d']

        df['vol_5'] = ret.rolling(window=5).std() * np.sqrt(252)
        df['vol_20'] = ret.rolling(window=20).std() * np.sqrt(252)
        df['vol_60'] = ret.rolling(window=60).std() * np.sqrt(252)
        df['vol_120'] = ret.rolling(window=120).std() * np.sqrt(252)

        df['vol_5_vol_20_ratio'] = df['vol_5'] / df['vol_20']
        df['vol_20_vol_60_ratio'] = df['vol_20'] / df['vol_60']

        return df

    def _calc_volume_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价量形态特征（QQQ）。

        Args:
            df: 包含 qqq 相关 OHLCV 列的 DataFrame

        Returns:
            添加价量形态特征后的 DataFrame
        """
        volume = df['qqq_volume']
        close = df['qqq_close']
        high = df['qqq_high']
        low = df['qqq_low']
        open_price = df['qqq_open']

        df['volume_ret_1d'] = volume.pct_change(1)
        df['volume_ret_5d'] = volume.pct_change(5)
        df['volume_ma5'] = volume.rolling(window=5).mean()
        df['volume_ma20'] = volume.rolling(window=20).mean()
        df['volume_ratio_5_20'] = volume / df['volume_ma20']

        price_dir = np.where(df['ret_1d'] >= 0, 1, -1)
        df['obv'] = (price_dir * volume).cumsum()
        df['obv_5d_change'] = df['obv'].pct_change(5)

        df['price_volume_divergence'] = (
            (df['ret_5d'] > 0) != (df['volume_ret_5d'] > 0)
        ).astype(int)

        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan)

        df['close_position'] = (close - low) / hl_range
        df['upper_shadow'] = (high - np.maximum(open_price, close)) / hl_range
        df['lower_shadow'] = (np.minimum(open_price, close) - low) / hl_range
        df['body_size'] = np.abs(close - open_price) / hl_range
        df['high_low_range'] = (high - low) / close
        df['high_low_range_ma20'] = df['high_low_range'].rolling(window=20).mean()

        return df

    def _calc_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 VIX 特征。

        Args:
            df: 包含 vix_close 列的 DataFrame

        Returns:
            添加 VIX 特征后的 DataFrame
        """
        if 'vix_close' not in df.columns:
            print("  警告: 缺少 VIX 数据，跳过 VIX 特征")
            return df

        vix = df['vix_close']

        df['vix_level'] = vix
        df['vix_change_1d'] = vix.pct_change(1)
        df['vix_change_5d'] = vix.pct_change(5)
        df['vix_change_20d'] = vix.pct_change(20)
        df['vix_ma20'] = vix.rolling(window=20).mean()
        df['vix_ma60'] = vix.rolling(window=60).mean()
        df['vix_ma20_ratio'] = vix / df['vix_ma20'] - 1
        df['vix_ma60_ratio'] = vix / df['vix_ma60'] - 1
        df['vix_quantile_60d'] = vix.rolling(window=60).apply(
            lambda x: (x < x[-1]).mean(), raw=True
        )
        df['vix_zscore_60d'] = (df['vix_level'] - df['vix_ma60']) / df['vix_level'].rolling(window=60).std()

        return df

    def _calc_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算相对强弱特征。

        Args:
            df: 包含 qqq_close 和 spy_close 列的 DataFrame

        Returns:
            添加相对强弱特征后的 DataFrame
        """
        if 'spy_close' not in df.columns:
            print("  警告: 缺少 SPY 数据，跳过相对强弱特征")
            return df

        rs = df['qqq_close'] / df['spy_close']
        df['rs_qv_spy'] = rs
        df['rs_qv_spy_ret_5d'] = rs.pct_change(5)
        df['rs_qv_spy_ret_20d'] = rs.pct_change(20)
        df['rs_qv_spy_ret_60d'] = rs.pct_change(60)

        return df


    def _calc_trend_oscillators(self, df):
        close = df['qqq_close']
        
        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        rsi_14 = rsi_14.where(~((loss == 0) & (gain > 0)), 100.0)
        rsi_14 = rsi_14.where(~((gain == 0) & (loss > 0)), 0.0)
        df['rsi_14'] = rsi_14.fillna(50.0)
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands %b
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df['bb_pct_b'] = (close - lower) / (upper - lower)
        
        return df

    def _calc_fear_greed(self, df):
        if 'hyg_close' in df.columns and 'ief_close' in df.columns:
            df['junk_bond_demand'] = df['hyg_close'] / df['ief_close']
            df['junk_bond_demand_ma20_ratio'] = df['junk_bond_demand'] / df['junk_bond_demand'].rolling(window=20).mean() - 1
            
        if 'tlt_close' in df.columns and 'spy_close' in df.columns:
            df['safe_haven_demand'] = df['tlt_close'] / df['spy_close']
            df['safe_haven_demand_ma20_ratio'] = df['safe_haven_demand'] / df['safe_haven_demand'].rolling(window=20).mean() - 1
            
        return df

    def _calc_market_breadth(self, df):
        if 'spy_close' in df.columns:
            spy = df['spy_close']
            df['spy_ret_1d'] = spy.pct_change(1)
            df['spy_ret_5d'] = spy.pct_change(5)
            df['spy_ret_20d'] = spy.pct_change(20)
            df['spy_vol_20'] = df['spy_ret_1d'].rolling(window=20).std() * np.sqrt(252)
            spy_ma200 = spy.rolling(window=200).mean()
            df['spy_dist_ma200'] = spy / spy_ma200 - 1
            df['spy_high_252d_ratio'] = spy / spy.rolling(window=252).max()
            df['spy_low_252d_ratio'] = spy / spy.rolling(window=252).min()

        if 'iwm_close' in df.columns:
            df['iwm_ret_1d'] = df['iwm_close'].pct_change(1)
            df['qqq_iwm_rs'] = df['qqq_close'] / df['iwm_close']
            df['qqq_iwm_rs_ret_20d'] = df['qqq_iwm_rs'].pct_change(20)

        if 'dia_close' in df.columns:
            df['dia_ret_1d'] = df['dia_close'].pct_change(1)
            df['qqq_dia_rs'] = df['qqq_close'] / df['dia_close']
            df['qqq_dia_rs_ret_20d'] = df['qqq_dia_rs'].pct_change(20)

        return df

    def _calc_tnx(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算利率特征（TNX）。

        Args:
            df: 包含 tnx_close 列的 DataFrame

        Returns:
            添加利率特征后的 DataFrame
        """
        if 'tnx_close' not in df.columns:
            print("  警告: 缺少 TNX 数据，跳过利率特征")
            return df

        tnx = df['tnx_close']

        df['tnx_level'] = tnx
        df['tnx_change_1d'] = tnx.pct_change(1)
        df['tnx_change_5d'] = tnx.pct_change(5)
        df['tnx_change_20d'] = tnx.pct_change(20)
        df['tnx_ma20'] = tnx.rolling(window=20).mean()
        df['tnx_ma20_ratio'] = tnx / df['tnx_ma20'] - 1

        return df

    def _calc_dxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算美元特征（DXY）。

        Args:
            df: 包含 dxy_close 列的 DataFrame

        Returns:
            添加美元特征后的 DataFrame
        """
        if 'dxy_close' not in df.columns:
            print("  警告: 缺少 DXY 数据，跳过美元特征")
            return df

        dxy = df['dxy_close']

        df['dxy_level'] = dxy
        df['dxy_change_1d'] = dxy.pct_change(1)
        df['dxy_change_5d'] = dxy.pct_change(5)
        df['dxy_change_20d'] = dxy.pct_change(20)
        df['dxy_ma20'] = dxy.rolling(window=20).mean()
        df['dxy_ma20_ratio'] = dxy / df['dxy_ma20'] - 1
        df['dxy_trend_60d'] = dxy.pct_change(60)

        return df

    def _calc_cross_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算交叉宏观因子。

        Args:
            df: 包含 vix_change_1d、spy_ret_1d、dxy_change_1d、tnx_change_1d 的 DataFrame

        Returns:
            添加交叉宏观因子后的 DataFrame
        """
        has_vix = 'vix_change_1d' in df.columns
        has_spy = 'spy_ret_1d' in df.columns
        has_dxy = 'dxy_change_1d' in df.columns
        has_tnx = 'tnx_change_1d' in df.columns

        if has_vix and has_spy:
            df['vix_spy_corr_20d'] = df['vix_change_1d'].rolling(window=20).corr(
                df['spy_ret_1d']
            )

        if has_dxy and has_tnx:
            df['dxy_tnx_corr_20d'] = df['dxy_change_1d'].rolling(window=20).corr(
                df['tnx_change_1d']
            )

        return df

    def _calc_advanced_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级振荡器特征。

        KDJ、CCI、Williams %R、ROC、MFI等

        Args:
            df: 包含 QQQ 行情的 DataFrame

        Returns:
            添加高级振荡器特征后的 DataFrame
        """
        close = df['qqq_close']
        high = df.get('qqq_high', close)
        low = df.get('qqq_low', close)

        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(20).mean()
        md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['cci_20'] = (tp - ma_tp) / (0.015 * md)

        # Williams %R
        high_14 = high.rolling(14).max()
        low_14 = low.rolling(14).min()
        df['williams_r_14'] = -100 * (high_14 - close) / (high_14 - low_14)

        # ROC (Rate of Change)
        df['roc_10'] = close.pct_change(10) * 100
        df['roc_20'] = close.pct_change(20) * 100
        df['roc_60'] = close.pct_change(60) * 100

        # Stochastic Oscillator (KDJ 简化版)
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        rsv = (close - low_9) / (high_9 - low_9) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        df['stoch_k'] = k
        df['stoch_d'] = d
        df['stoch_j'] = j

        # 更多 RSI 周期
        for period in [5, 9, 21, 50]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
            rsi = rsi.where(~((gain == 0) & (loss > 0)), 0.0)
            df[f'rsi_{period}'] = rsi.fillna(50.0)

        # RSI 偏离 (RSI 与其 MA 的距离)
        df['rsi_14_ma14_ratio'] = df['rsi_14'] / df['rsi_14'].rolling(14).mean() - 1

        return df

    def _calc_volatility_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级波动率特征。

        ATR、真实波幅、收益率偏度峰度、高低点波动率等

        Args:
            df: 包含 QQQ 行情的 DataFrame

        Returns:
            添加高级波动率特征后的 DataFrame
        """
        close = df['qqq_close']
        high = df.get('qqq_high', close)
        low = df.get('qqq_low', close)
        ret = close.pct_change()

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct_14'] = df['atr_14'] / close

        # 收益率偏度和峰度
        df['ret_skew_20'] = ret.rolling(20).skew()
        df['ret_skew_60'] = ret.rolling(60).skew()
        df['ret_kurt_20'] = ret.rolling(20).kurt()
        df['ret_kurt_60'] = ret.rolling(60).kurt()

        # 上下波动率 (好波动 vs 坏波动)
        up_ret = ret.where(ret > 0, 0)
        down_ret = ret.where(ret < 0, 0)
        df['up_vol_20'] = up_ret.rolling(20).std() * np.sqrt(252)
        df['down_vol_20'] = down_ret.rolling(20).std() * np.sqrt(252)
        df['vol_ratio_20'] = df['up_vol_20'] / df['down_vol_20']

        # 高低点范围
        df['hl_range_5d'] = (high.rolling(5).max() - low.rolling(5).min()) / close
        df['hl_range_20d'] = (high.rolling(20).max() - low.rolling(20).min()) / close
        df['hl_range_60d'] = (high.rolling(60).max() - low.rolling(60).min()) / close

        return df

    def _calc_momentum_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级动量特征。

        更多周期动量、加速动量、相对强弱动量等

        Args:
            df: 包含 QQQ 行情的 DataFrame

        Returns:
            添加高级动量特征后的 DataFrame
        """
        close = df['qqq_close']

        # 更多周期动量
        for period in [3, 10, 21, 63, 126, 252]:
            df[f'mom_{period}d'] = close.pct_change(period)

        # 加速动量 (动量的动量)
        df['mom_accel_5_20'] = close.pct_change(5) - close.pct_change(20).shift(5)
        df['mom_accel_10_60'] = close.pct_change(10) - close.pct_change(60).shift(10)

        # 连续涨跌天数
        ret = close.pct_change()
        direction = np.sign(ret)
        streak = pd.Series(0, index=close.index, dtype=float)
        current_streak = 0
        prev_dir = 0
        for i in range(len(direction)):
            d = direction.iloc[i]
            if d == prev_dir and d != 0:
                current_streak += 1
            else:
                current_streak = 1 if d != 0 else 0
            streak.iloc[i] = current_streak
            prev_dir = d
        df['streak_days'] = streak
        df['streak_direction'] = direction * streak

        # 均线动量 (均线斜率)
        for ma_period in [10, 20, 50, 200]:
            ma = close.rolling(ma_period).mean()
            df[f'ma{ma_period}_slope_5d'] = ma.pct_change(5)
            df[f'ma{ma_period}_slope_20d'] = ma.pct_change(20)

        return df

    def _calc_volume_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级成交量特征。

        OBV、VWAP、量价背离等

        Args:
            df: 包含 QQQ 行情的 DataFrame

        Returns:
            添加高级成交量特征后的 DataFrame
        """
        close = df['qqq_close']
        volume = df.get('qqq_volume', pd.Series(1, index=close.index))
        high = df.get('qqq_high', close)
        low = df.get('qqq_low', close)

        # OBV (On-Balance Volume)
        ret_sign = np.sign(close.pct_change()).fillna(0)
        df['obv'] = (ret_sign * volume).cumsum()
        df['obv_ma20_ratio'] = df['obv'] / df['obv'].rolling(20).mean() - 1

        # VWAP (日内近似)
        typical_price = (high + low + close) / 3
        df['vwap_20'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['vwap_dist_20'] = close / df['vwap_20'] - 1

        # 量价相关性
        df['volume_price_corr_20d'] = volume.pct_change().rolling(20).corr(close.pct_change())

        # 成交量动量
        df['volume_ma5_ratio'] = volume / volume.rolling(5).mean() - 1
        df['volume_ma20_ratio'] = volume / volume.rolling(20).mean() - 1
        df['volume_change_1d'] = volume.pct_change()

        return df

    def _calc_ma_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级均线特征。

        更多均线、均线排列、均线收敛发散等

        Args:
            df: 包含 QQQ 行情的 DataFrame

        Returns:
            添加高级均线特征后的 DataFrame
        """
        close = df['qqq_close']

        # 更多 EMA
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_dist'] = close / df[f'ema_{period}'] - 1

        # 均线排列 (多头/空头排列强度)
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma200 = close.rolling(200).mean()

        # 多头排列得分 (越多头越高)
        df['ma_alignment_score'] = (
            (ma5 > ma10).astype(int) +
            (ma10 > ma20).astype(int) +
            (ma20 > ma60).astype(int) +
            (ma60 > ma200).astype(int)
        )

        # MACD 更多参数
        for fast, slow, signal in [(5, 35, 5), (8, 17, 9), (12, 26, 9), (19, 39, 9)]:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            sig = macd.ewm(span=signal, adjust=False).mean()
            name = f'macd_{fast}_{slow}'
            df[name] = macd
            df[f'{name}_hist'] = macd - sig

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理极端值。

        - 将收益率类特征中绝对值 > 20% 的截断
        - 替换 Inf 为 NaN
        - forward fill 填充 NaN

        Args:
            df: 特征 DataFrame

        Returns:
            处理后的 DataFrame
        """
        df = df.copy()

        df = df.replace([np.inf, -np.inf], np.nan)

        # 分层截断极端异常值：避免暴力截断抹杀中长周期大牛市动量与高波动特征
        short_ret_cols = ['ret_1d', 'ret_3d', 'ret_5d', 'spy_ret_1d', 'spy_ret_5d']
        for col in short_ret_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=-0.25, upper=0.25)

        med_ret_cols = [
            'ret_10d', 'ret_20d', 'rs_qv_spy_ret_5d', 'rs_qv_spy_ret_20d',
            'spy_ret_20d', 'tnx_change_1d', 'tnx_change_5d', 'dxy_change_1d', 'dxy_change_5d'
        ]
        for col in med_ret_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=-0.5, upper=0.5)

        long_ret_cols = ['ret_60d', 'ret_120d', 'rs_qv_spy_ret_60d', 'tnx_change_20d', 'dxy_change_20d', 'dxy_trend_60d']
        for col in long_ret_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=-1.0, upper=1.5)

        vol_vol_cols = ['volume_ret_1d', 'volume_ret_5d', 'obv_5d_change', 'vix_change_1d', 'vix_change_5d', 'vix_change_20d']
        for col in vol_vol_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=-2.0, upper=5.0)

        df = df.ffill()

        return df

    def build_features(self) -> pd.DataFrame:
        """组合所有特征。

        Returns:
            包含所有特征的 DataFrame
        """
        df = self.load_raw_data()

        print("\n开始构建特征...")

        df = self._calc_momentum(df)
        print("  1/16 动量特征完成")

        df = self._calc_ma(df)
        print("  2/16 均线特征完成")

        df = self._calc_distance(df)
        print("  3/16 距离特征完成")

        df = self._calc_volatility(df)
        print("  4/16 波动率特征完成")

        df = self._calc_volume_price(df)
        print("  5/16 价量形态特征完成")

        df = self._calc_vix(df)
        print("  6/16 VIX 特征完成")

        df = self._calc_relative_strength(df)
        print("  7/16 相对强弱特征完成")

        df = self._calc_market_breadth(df)
        print("  8/16 市场广度特征完成")

        df = self._calc_tnx(df)
        print("  9/16 利率特征完成")

        df = self._calc_dxy(df)
        print("  10/16 美元特征完成")

        df = self._calc_cross_macro(df)
        print("  11/16 交叉宏观因子完成")

        df = self._calc_trend_oscillators(df)
        print("  12/16 趋势振荡器完成")

        df = self._calc_fear_greed(df)
        print("  13/16 贪婪恐慌指标完成")

        df = self._calc_advanced_oscillators(df)
        print("  14/16 高级振荡器完成")

        df = self._calc_volatility_advanced(df)
        print("  15/16 高级波动率完成")

        df = self._calc_momentum_advanced(df)
        print("  16/16 高级动量完成")

        df = self._calc_volume_advanced(df)
        print("  - 高级成交量完成")

        df = self._calc_ma_advanced(df)
        print("  - 高级均线完成")

        feature_cols = [col for col in df.columns if not col.startswith('qqq_')
                        and col not in ('tqqq_close', 'sqqq_close', 'psq_close',
                                        'spy_close', 'vix_close', 'iwm_close', 'dia_close',
                                        'hyg_close', 'ief_close', 'tlt_close',
                                        'tnx_close', 'dxy_close')]
        print(f"\n特征列数: {len(feature_cols)}")

        df = self._remove_outliers(df)
        df = df.dropna()

        print(f"去除 NaN 后行数: {len(df)}")
        print(f"最终特征列数: {len(feature_cols)}")

        return df

    def save_features(self, df: pd.DataFrame, filename: str = 'features.parquet') -> str:
        """保存特征为 Parquet 到 data/processed/。

        Args:
            df: 特征 DataFrame
            filename: 保存文件名

        Returns:
            保存文件的完整路径
        """
        filepath = os.path.join(self.processed_dir, filename)
        df.to_parquet(filepath, engine='pyarrow')
        print(f"\n特征已保存到: {filepath}")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        return filepath


if __name__ == '__main__':
    engineer = FeatureEngineer()
    features_df = engineer.build_features()

    print("\n" + "=" * 60)
    print("特征统计")
    print("=" * 60)
    print(f"总行数: {len(features_df)}")
    print(f"总列数: {len(features_df.columns)}")
    print(f"日期范围: {features_df.index[0].date()} ~ {features_df.index[-1].date()}")

    feature_cols = [col for col in features_df.columns
                    if not col.startswith('qqq_')
                    and col not in ('tqqq_close', 'sqqq_close',
                                    'spy_close', 'vix_close',
                                    'tnx_close', 'dxy_close')]
    print(f"特征列数: {len(feature_cols)}")

    print("\n缺失值检查:")
    missing = features_df.isna().sum()
    if missing.sum() == 0:
        print("  无缺失值")
    else:
        print(f"  缺失值总数: {missing.sum()}")
        print(missing[missing > 0])

    print("\nInf 值检查:")
    inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
    if inf_count.sum() == 0:
        print("  无 Inf 值")
    else:
        print(f"  Inf 值总数: {inf_count.sum()}")
        print(inf_count[inf_count > 0])

    print("\n特征描述性统计:")
    print(features_df[feature_cols].describe().T[['mean', 'std', 'min', '25%',
                                                   '50%', '75%', 'max']].round(4))

    engineer.save_features(features_df)
