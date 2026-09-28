"""
Regime 感知模型包装器

将 Regime 参数注入到 ML 模型中，调整模型的决策行为

注入方式：
1. 阈值调整：调整 Risk-On / Risk-Off 的概率阈值
2. 概率缩放：缩放各类别的概率差异
3. 特征注入：将 Regime 参数作为额外特征输入模型
"""

from typing import Optional, Literal
import numpy as np
import pandas as pd

from .base import RegimeParams


class RegimeAwareModelWrapper:
    """
    Regime 感知的模型包装器

    将 Regime 参数注入到基础 ML 模型中，调整决策行为

    使用方式:
        wrapper = RegimeAwareModelWrapper(
            base_model=lgbm_model,
            base_risk_on_thresh=0.60,
            base_risk_off_thresh=0.60,
        )

        # 使用默认中性参数
        predictions = wrapper.predict(X)

        # 使用 Regime 调整后的参数
        predictions = wrapper.predict(X, regime_params=regime_params)
    """

    def __init__(
        self,
        base_model,
        base_risk_on_thresh: float = 0.60,
        base_risk_off_thresh: float = 0.60,
        use_feature_injection: bool = False,
    ):
        """
        初始化包装器

        Args:
            base_model: 基础 ML 模型（需要有 predict_proba 方法）
            base_risk_on_thresh: 基础 Risk-On 阈值
            base_risk_off_thresh: 基础 Risk-Off 阈值
            use_feature_injection: 是否将 Regime 参数作为特征注入
        """
        self.base_model = base_model
        self.base_risk_on_thresh = base_risk_on_thresh
        self.base_risk_off_thresh = base_risk_off_thresh
        self.use_feature_injection = use_feature_injection

    def predict(
        self,
        X: pd.DataFrame,
        regime_params: Optional[RegimeParams] = None,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵
            regime_params: Regime 参数（None = 中性参数）
            return_proba: 是否返回概率

        Returns:
            预测类别 (0/1/2) 或 (预测类别, 概率)
        """
        if regime_params is None:
            regime_params = RegimeParams.neutral()

        # 获取基础模型概率
        if hasattr(self.base_model, "predict_proba"):
            proba = self.base_model.predict_proba(X)
        else:
            # 如果模型没有 predict_proba，直接预测
            predictions = self.base_model.predict(X)
            if return_proba:
                return predictions, None
            return predictions

        # 应用 Regime 调整
        adjusted_proba = self._apply_regime_adjustment(proba, regime_params)

        # 决策
        predictions = self._make_decision(
            adjusted_proba,
            regime_params.get_adjusted_risk_on_threshold(self.base_risk_on_thresh),
            regime_params.get_adjusted_risk_off_threshold(self.base_risk_off_thresh),
        )

        if return_proba:
            return predictions, adjusted_proba
        return predictions

    def _apply_regime_adjustment(
        self,
        proba: np.ndarray,
        regime_params: RegimeParams,
    ) -> np.ndarray:
        """
        应用 Regime 调整

        调整策略：
        1. 概率缩放：扩大或缩小类别之间的概率差异
        2. 阈值调整：在决策时体现
        """
        adjusted = proba.copy()

        # 概率缩放
        if regime_params.probability_scale != 1.0:
            scale = regime_params.probability_scale

            # 将概率转换为 logit，缩放，再转回
            eps = 1e-6
            clipped_proba = np.clip(proba, eps, 1 - eps)
            logits = np.log(clipped_proba / (1 - clipped_proba))

            # 缩放
            scaled_logits = logits * scale

            # 转回概率
            adjusted = 1 / (1 + np.exp(-scaled_logits))
            # 归一化
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

        return adjusted

    def _make_decision(
        self,
        proba: np.ndarray,
        risk_on_thresh: float,
        risk_off_thresh: float,
    ) -> np.ndarray:
        """
        基于概率和阈值做决策

        类别: 0 = Risk-Off, 1 = Neutral, 2 = Risk-On
        """
        n_samples = proba.shape[0]
        decisions = np.ones(n_samples, dtype=int)  # 默认 Neutral

        # Risk-On 判断：P(Risk-On) > 阈值
        risk_on_mask = proba[:, 2] > risk_on_thresh
        decisions[risk_on_mask] = 2

        # Risk-Off 判断：P(Risk-Off) > 阈值（但不在 Risk-On 区间）
        risk_off_mask = (proba[:, 0] > risk_off_thresh) & (decisions == 1)
        decisions[risk_off_mask] = 0

        return decisions

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        regime_params: Optional[RegimeParams] = None,
        uncertainty_thresh: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        预测并返回不确定性

        Args:
            X: 特征矩阵
            regime_params: Regime 参数
            uncertainty_thresh: 不确定性阈值（max_prob < 此值视为不确定）

        Returns:
            (predictions, uncertainty_mask)
            uncertainty_mask: True = 不确定，False = 确定
        """
        if regime_params is None:
            regime_params = RegimeParams.neutral()

        proba = self.base_model.predict_proba(X)
        adjusted_proba = self._apply_regime_adjustment(proba, regime_params)

        # 计算不确定性：最大概率与 1/3 的差距
        max_prob = np.max(adjusted_proba, axis=1)
        uncertainty = 1.0 - (max_prob - 1/3) / (2/3)  # 归一化到 0-1
        uncertainty_mask = uncertainty > uncertainty_thresh

        predictions = self._make_decision(
            adjusted_proba,
            regime_params.get_adjusted_risk_on_thresh(self.base_risk_on_thresh),
            regime_params.get_adjusted_risk_off_thresh(self.base_risk_off_thresh),
        )

        return predictions, uncertainty_mask


# =============================================================================
# 特征注入版本
# =============================================================================


class RegimeFeatureInjector:
    """
    Regime 特征注入器

    将 Regime 参数作为额外特征注入到特征矩阵中
    """

    @staticmethod
    def inject(
        X: pd.DataFrame,
        regime_params: RegimeParams,
    ) -> pd.DataFrame:
        """
        将 Regime 参数注入到特征矩阵

        Args:
            X: 原始特征矩阵
            regime_params: Regime 参数

        Returns:
            添加了 Regime 特征的新特征矩阵
        """
        X_augmented = X.copy()

        # 添加 Regime 特征
        X_augmented["regime_risk_on_adj"] = regime_params.risk_on_threshold_adjustment
        X_augmented["regime_risk_off_adj"] = regime_params.risk_off_threshold_adjustment
        X_augmented["regime_prob_scale"] = regime_params.probability_scale
        X_augmented["regime_confidence"] = regime_params.confidence

        sentiment_map = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
        }
        X_augmented["regime_sentiment_score"] = sentiment_map.get(
            regime_params.overall_sentiment.value, 0.0
        )

        return X_augmented


# =============================================================================
# 便捷函数
# =============================================================================


def create_regime_aware_predictor(
    base_model,
    regime_aggregator,
    macro_agent=None,
    sentiment_agent=None,
    macro_snapshot_builder=None,
    sentiment_snapshot_builder=None,
    base_risk_on_thresh: float = 0.60,
    base_risk_off_thresh: float = 0.60,
) -> callable:
    """
    创建 Regime 感知的预测函数

    这是一个便捷工厂函数，将多个组件组合成一个可调用函数

    Args:
        base_model: 基础 ML 模型
        regime_aggregator: Regime 聚合器
        macro_agent: 宏观 Agent（可选）
        sentiment_agent: 消息面 Agent（可选）
        macro_snapshot_builder: 构建宏观快照的函数
        sentiment_snapshot_builder: 构建消息面快照的函数
        base_risk_on_thresh: 基础阈值
        base_risk_off_thresh: 基础阈值

    Returns:
        预测函数 predict(X, date) -> predictions
    """
    wrapper = RegimeAwareModelWrapper(
        base_model=base_model,
        base_risk_on_thresh=base_risk_on_thresh,
        base_risk_off_thresh=base_risk_off_thresh,
    )

    def predict(X: pd.DataFrame, date: str) -> np.ndarray:
        # 获取 Regime 参数
        macro_output = None
        sentiment_output = None

        if macro_agent and macro_snapshot_builder:
            snapshot = macro_snapshot_builder(date)
            macro_output = macro_agent.predict(snapshot)

        if sentiment_agent and sentiment_snapshot_builder:
            snapshot = sentiment_snapshot_builder(date)
            sentiment_output = sentiment_agent.predict(snapshot)

        # 聚合
        aggregated = regime_aggregator.aggregate(macro_output, sentiment_output)

        # 预测
        return wrapper.predict(X, regime_params=aggregated.params)

    return predict


if __name__ == "__main__":
    # 简单测试
    import lightgbm as lgb

    # 创建假数据
    X_train = pd.DataFrame(
        np.random.randn(1000, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )
    y_train = np.random.choice([0, 1, 2], size=1000)

    # 训练简单模型
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(X_train, y_train)

    # 创建包装器
    wrapper = RegimeAwareModelWrapper(
        base_model=model,
        base_risk_on_thresh=0.60,
        base_risk_off_thresh=0.60,
    )

    # 测试预测
    X_test = pd.DataFrame(
        np.random.randn(10, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )

    print("Predictions without regime adjustment:")
    preds = wrapper.predict(X_test)
    print(preds)

    print("\nPredictions with aggressive risk-on regime:")
    regime_params = RegimeParams.risk_on_aggressive()
    preds = wrapper.predict(X_test, regime_params=regime_params)
    print(preds)

    print("\nPredictions with defensive risk-off regime:")
    regime_params = RegimeParams.risk_off_defensive()
    preds = wrapper.predict(X_test, regime_params=regime_params)
    print(preds)
