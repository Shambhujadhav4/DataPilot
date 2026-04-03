import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class ModelTrainer:
    """Trains and evaluates classification and regression models."""

    # Available model registries
    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    REGRESSION_MODELS = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }

    def __init__(self):
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.feature_names: list = []
        self.task_type: str = ""
        self.model_name: str = ""

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        test_size: float = 0.2,
        random_state: int = 42,
        task_type: str | None = None,
    ):
        X = df[feature_cols]
        y = df[target_col]
        self.feature_names = feature_cols
        stratify = None
        if task_type == "classification" and y.nunique(dropna=False) > 1:
            class_counts = y.value_counts(dropna=False)
            if class_counts.min() < 2:
                raise ValueError(
                    "Each target class needs at least 2 rows for stratified splitting."
                )

            test_count = (
                int(np.ceil(len(y) * test_size))
                if isinstance(test_size, float)
                else int(test_size)
            )
            train_count = len(y) - test_count
            n_classes = y.nunique(dropna=False)
            if test_count < n_classes or train_count < n_classes:
                raise ValueError(
                    "Adjust the test split so both train and test sets can include every target class."
                )

            stratify = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, task_type: str, model_name: str):
        self.task_type = task_type
        self.model_name = model_name

        registry = (
            self.CLASSIFICATION_MODELS
            if task_type == "classification"
            else self.REGRESSION_MODELS
        )

        if model_name not in registry:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        import sklearn.base as base

        self.model = base.clone(registry[model_name])
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        if task_type == "classification" and hasattr(self.model, "predict_proba"):
            self.y_pred_proba = self.model.predict_proba(self.X_test)

        return self.model

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        return (
            self._classification_metrics()
            if self.task_type == "classification"
            else self._regression_metrics()
        )

    def _classification_metrics(self) -> dict:
        metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred, average="weighted"),
            "precision": precision_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred),
            "classification_report": classification_report(self.y_test, self.y_pred),
        }
        if self.y_pred_proba is not None:
            try:
                n_classes = len(np.unique(self.y_test))
                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_test, self.y_pred_proba[:, 1]
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_test,
                        self.y_pred_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
            except Exception:
                pass
        return metrics

    def _regression_metrics(self) -> dict:
        mse = mean_squared_error(self.y_test, self.y_pred)
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(self.y_test, self.y_pred),
            "r2_score": r2_score(self.y_test, self.y_pred),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame | None:
        if hasattr(self.model, "feature_importances_"):
            return (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": self.model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": np.abs(coef),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        return None

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def get_cross_val_scores(self, cv: int = 5) -> np.ndarray:
        scoring = "accuracy" if self.task_type == "classification" else "r2"
        X_all = pd.concat([self.X_train, self.X_test])
        y_all = pd.concat([self.y_train, self.y_test])
        import sklearn.base as base

        fresh_model = base.clone(self.model)
        return cross_val_score(fresh_model, X_all, y_all, cv=cv, scoring=scoring)
