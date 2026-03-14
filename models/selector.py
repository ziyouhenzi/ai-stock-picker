"""
models/selector.py - AI 选股模型
使用 RandomForest / XGBoost 多因子模型进行选股打分
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class AIStockSelector:
    """
    AI 选股器
    - 输入：多因子特征矩阵
    - 输出：股票评分 (0~1)，越高越值得买入
    """

    def __init__(self, model_type="random_forest", config=None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def _build_model(self):
        n = self.config.get("n_estimators", 200)
        depth = self.config.get("max_depth", 6)

        if self.model_type == "xgboost" and HAS_XGB:
            return XGBClassifier(
                n_estimators=n, max_depth=depth,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", random_state=42, n_jobs=-1
            )
        elif self.model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=n, max_depth=depth,
                learning_rate=0.05, subsample=0.8, random_state=42
            )
        else:
            return RandomForestClassifier(
                n_estimators=n, max_depth=depth,
                min_samples_leaf=5, max_features="sqrt",
                random_state=42, n_jobs=-1, class_weight="balanced"
            )

    def prepare_data(self, factor_df: pd.DataFrame, label_series: pd.Series):
        """
        准备训练数据，处理缺失值和标准化
        """
        # 对齐
        common_idx = factor_df.index.intersection(label_series.index)
        X = factor_df.loc[common_idx].copy()
        y = label_series.loc[common_idx].copy()

        # 删除全空列
        X = X.dropna(axis=1, how="all")
        # 用中位数填充缺失值
        X = X.fillna(X.median())
        # 去除极值（winsorize at 1% and 99%）
        for col in X.columns:
            p1, p99 = X[col].quantile(0.01), X[col].quantile(0.99)
            X[col] = X[col].clip(p1, p99)

        self.feature_names = X.columns.tolist()
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, cv=5):
        """训练模型"""
        self.model = self._build_model()
        X_scaled = self.scaler.fit_transform(X)

        # 交叉验证
        scores = cross_val_score(self.model, X_scaled, y, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
        print(f"  Cross-val AUC: {scores.mean():.4f} ± {scores.std():.4f}")

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # 训练集评估
        y_pred = self.model.predict(X_scaled)
        print(classification_report(y, y_pred, target_names=["Hold", "Buy"], zero_division=0))

        return scores.mean()

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """预测买入概率得分"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        X_clean = X[self.feature_names].fillna(X[self.feature_names].median())
        X_scaled = self.scaler.transform(X_clean)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return pd.Series(proba, index=X.index, name="score")

    def select_top_n(self, X: pd.DataFrame, n=20) -> list:
        """选出评分最高的 N 只股票"""
        scores = self.predict_scores(X)
        return scores.nlargest(n).index.tolist()

    def get_feature_importance(self) -> pd.Series:
        """获取因子重要性"""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        return pd.Series()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_names}, path)
        print(f"  Model saved to {path}")

    def load(self, path):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.feature_names = obj["features"]
        self.is_trained = True
