"""基于 LightGBM 与概率校准的胜率预测模型。

实现：
1. 三重屏障标签驱动的二分类 LightGBM 模型 (浅层树结构防过拟合)
2. Isotonic 概率校准 (输出严谨校准的真实胜率概率 P ∈ [0, 1])
3. Walk-Forward 滚动前向测试引擎 (严格无前视偏差，每半年滚动增量拟合)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


class WinRatePredictor:
    """胜率预测器 (Meta-Labeling Quality Gate)。"""

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        min_child_samples: int = 25,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model: Optional[CalibratedClassifierCV] = None

    def _build_model(self) -> CalibratedClassifierCV:
        base_lgbm = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        # 3折概率校准
        calibrated = CalibratedClassifierCV(
            estimator=base_lgbm,
            method='isotonic',
            cv=3,
        )
        return calibrated

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """拟合模型。"""
        valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(y_clean) < 100 or len(np.unique(y_clean)) < 2:
            return None

        self.model = self._build_model()
        self.model.fit(X_clean, y_clean)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测样本属于胜率类别 (Label=1) 的后验概率。"""
        if self.model is None or len(X) == 0:
            return np.full(len(X), 0.50)

        # 填充缺失值为中位数
        X_filled = X.fillna(0.0)
        probs = self.model.predict_proba(X_filled)[:, 1]
        return probs


class WalkForwardWinRatePipeline:
    """Walk-Forward 增量滚动训练与预测流水线。"""

    def __init__(
        self,
        warmup_days: int = 500,  # 约 2 年初始训练集
        refit_interval: int = 126,  # 每半年 (约 126 个交易日) 重训一次
        barrier_embargo_days: int = 10,  # 10天三重屏障冷却期，避免标签泄漏
    ):
        self.warmup_days = warmup_days
        self.refit_interval = refit_interval
        self.barrier_embargo_days = barrier_embargo_days

    def run_walkforward_predictions(
        self,
        features_dict: Dict[str, pd.DataFrame],
        labels_df: pd.DataFrame,
        candidate_tickers: List[str],
    ) -> pd.DataFrame:
        """在全时间轴上滚动输出各资产完全纯样本外 (OOS) 的预测胜率矩阵。

        Returns:
            DataFrame (index: dates, columns: tickers, values: win_rate [0, 1])
        """
        # 获取共有交易日
        common_dates = labels_df.index
        n_dates = len(common_dates)

        # 初始化输出矩阵 (默认未入模前或冷启动期设为 0.50 中性概率)
        df_win_rates = pd.DataFrame(
            0.50,
            index=common_dates,
            columns=candidate_tickers,
            dtype=float,
        )

        current_predictor = None

        # 滚动窗口循环
        for train_end_idx in range(self.warmup_days, n_dates, self.refit_interval):
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + self.refit_interval, n_dates)

            # ---------------------------
            # 1. 组装截至 train_end 的全部历史样本
            # 必须扣除 barrier_embargo_days (避免标签未来泄漏)
            # ---------------------------
            cutoff_idx = train_end_idx - self.barrier_embargo_days
            if cutoff_idx <= 100:
                continue

            cutoff_date = common_dates[cutoff_idx]

            X_train_list = []
            y_train_list = []

            for ticker in candidate_tickers:
                if ticker not in features_dict or ticker not in labels_df:
                    continue

                feat_t = features_dict[ticker].loc[:cutoff_date]
                lbl_t = labels_df[ticker].loc[:cutoff_date]

                aligned = pd.concat([feat_t, lbl_t.rename('label')], axis=1).dropna()
                if len(aligned) > 0:
                    X_train_list.append(aligned.drop(columns=['label']))
                    y_train_list.append(aligned['label'])

            if not X_train_list:
                continue

            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)

            predictor = WinRatePredictor()
            predictor.fit(X_train, y_train)
            if predictor.model is not None:
                current_predictor = predictor

            # ---------------------------
            # 2. 预测接下来的测试期样本 (test_start ~ test_end)
            # ---------------------------
            if current_predictor is not None:
                for ticker in candidate_tickers:
                    if ticker not in features_dict:
                        continue
                    test_dates = common_dates[test_start_idx:test_end_idx]
                    X_test = features_dict[ticker].reindex(test_dates)

                    probs = current_predictor.predict_proba(X_test)
                    df_win_rates.loc[test_dates, ticker] = probs

        return df_win_rates
