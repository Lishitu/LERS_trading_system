import os
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


class ModelTrainer:
    """模型训练器，负责训练和管理机器学习模型。

    支持 LightGBM、随机森林、逻辑回归三种分类模型，
    用于五分类（SQQQ/PSQ/Cash/QQQ/TQQQ）的市场状态预测。

    Attributes:
        random_state: 随机种子，用于保证结果可复现
        scaler: 标准化器（仅逻辑回归使用）
    """

    def __init__(self, random_state: int = 42):
        """初始化模型训练器。

        Args:
            random_state: 随机种子，默认 42
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None

    def load_data(self, path: str = 'data/processed/features.parquet') -> pd.DataFrame:
        """加载带标签的特征数据。

        Args:
            path: 数据文件路径，默认 'data/processed/features.parquet'

        Returns:
            加载的特征 DataFrame

        Raises:
            FileNotFoundError: 当文件不存在时抛出
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"特征文件不存在: {path}")

        df = pd.read_parquet(path, engine='pyarrow')
        print(f"已加载数据: {len(df)} 行, {len(df.columns)} 列")
        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """准备训练数据。

        - 去除 label 为 NaN 的行
        - 分离特征 X 和标签 y

        Args:
            df: 包含特征和标签的 DataFrame
            feature_cols: 特征列名列表，为 None 时自动选取所有非标签列

        Returns:
            X: 特征 DataFrame
            y: 标签 Series
            feature_cols: 使用的特征列名列表

        Raises:
            KeyError: 当 DataFrame 中缺少 label 列时抛出
        """
        if 'label' not in df.columns:
            raise KeyError("DataFrame 中缺少 label 列")

        df_clean = df.dropna(subset=['label']).copy()
        print(f"去除 NaN 标签后: {len(df_clean)} 行")

        if feature_cols is None:
            exclude_cols = {'label', 'future_ret'}
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        X = df_clean[feature_cols].copy()
        y = df_clean['label'].astype(int)

        print(f"特征数量: {len(feature_cols)}")
        print(f"标签分布:\n{y.value_counts().sort_index().to_string()}")

        return X, y, feature_cols

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> LGBMClassifier:
        """训练 LightGBM 五分类模型。

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选，用于 early stopping）
            y_val: 验证标签（可选）

        Returns:
            训练好的 LightGBM 模型
        """
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
            ]

        model = LGBMClassifier(
            objective='multiclass',
            num_class=5,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_split_gain=0.01,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, **fit_params)
        print("LightGBM 模型训练完成")
        return model

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """训练随机森林分类模型。

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            训练好的随机森林模型
        """
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print("随机森林模型训练完成")
        return model

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """训练逻辑回归分类模型。

        训练前会对特征进行标准化处理。

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            训练好的逻辑回归模型
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
        )
        model.fit(X_train_scaled, y_train)
        print("逻辑回归模型训练完成")
        return model

    def predict_proba(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率。

        Args:
            model: 训练好的模型（支持 LGBMClassifier、RandomForestClassifier、LogisticRegression）
            X: 特征数据

        Returns:
            预测概率 DataFrame，列名为 prob_riskoff, prob_neutral, prob_riskon
        """
        if isinstance(model, LogisticRegression) and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            proba = model.predict_proba(X_scaled)
        else:
            proba = model.predict_proba(X)

        result = pd.DataFrame(
            proba,
            index=X.index,
            columns=['prob_sqqq', 'prob_psq', 'prob_cash', 'prob_qqq', 'prob_tqqq']
        )
        return result

    def save_model(self, model, path: str) -> str:
        """保存模型为 pkl 文件。

        Args:
            model: 要保存的模型对象
            path: 保存路径

        Returns:
            保存文件的完整路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {path}")
        return path

    def load_model(self, path: str):
        """加载模型。

        Args:
            path: 模型文件路径

        Returns:
            加载的模型对象

        Raises:
            FileNotFoundError: 当文件不存在时抛出
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已加载: {path}")
        return model

    def get_feature_importance(
        self,
        model,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """获取特征重要性。

        仅支持 LightGBM 和随机森林模型。

        Args:
            model: 训练好的模型
            feature_cols: 特征列名列表

        Returns:
            特征重要性 DataFrame，包含 feature 和 importance 两列，按重要性降序排列

        Raises:
            ValueError: 当模型不支持特征重要性时抛出
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("该模型不支持特征重要性分析")

        importance = model.feature_importances_
        result = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        })
        result = result.sort_values('importance', ascending=False).reset_index(drop=True)
        return result


if __name__ == '__main__':
    trainer = ModelTrainer(random_state=42)

    print("=" * 60)
    print("加载数据")
    print("=" * 60)
    df = trainer.load_data()

    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)
    X, y, feature_cols = trainer.prepare_data(df)

    print("\n" + "=" * 60)
    print("训练 LightGBM 模型")
    print("=" * 60)
    lgb_model = trainer.train_lightgbm(X, y)

    print("\n" + "=" * 60)
    print("训练集预测")
    print("=" * 60)
    y_pred_proba = trainer.predict_proba(lgb_model, X)
    y_pred = y_pred_proba.idxmax(axis=1).map({
        'prob_riskoff': 0,
        'prob_neutral': 1,
        'prob_riskon': 2
    })
    accuracy = (y_pred.values == y.values).mean()
    print(f"训练集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    proba_sum = y_pred_proba.sum(axis=1)
    max_error = (proba_sum - 1.0).abs().max()
    print(f"概率和最大误差: {max_error:.2e}")

    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)
    trainer.save_model(lgb_model, 'models/lgb_model.pkl')

    print("\n" + "=" * 60)
    print("Top 10 特征重要性")
    print("=" * 60)
    importance_df = trainer.get_feature_importance(lgb_model, feature_cols)
    print(importance_df.head(10).to_string(index=False))
