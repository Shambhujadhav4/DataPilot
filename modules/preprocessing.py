import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """Handles all data preprocessing steps."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders: dict = {}
        self.scaler = None
        self.scaled_columns: list = []

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_missing_info(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        return (
            pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
            .sort_values("Missing Count", ascending=False)
        )

    # ------------------------------------------------------------------
    # Step 1 – Drop columns
    # ------------------------------------------------------------------

    def drop_columns(self, columns: list) -> pd.DataFrame:
        self.df = self.df.drop(columns=columns, errors="ignore")
        return self.df

    # ------------------------------------------------------------------
    # Step 2 – Missing values
    # ------------------------------------------------------------------

    def handle_missing_values(
        self,
        strategy: str,
        columns: list | None = None,
        fill_value=None,
    ) -> pd.DataFrame:
        """
        strategy options:
            'drop_rows'   – drop rows that have nulls in *columns*
            'drop_columns'– drop the selected columns entirely
            'mean'        – fill numeric cols with column mean
            'median'      – fill numeric cols with column median
            'mode'        – fill any col with the most-frequent value
            'zero'        – fill with 0
            'custom'      – fill with *fill_value*
        """
        cols = columns if columns else self.df.columns.tolist()

        if strategy == "drop_rows":
            self.df = self.df.dropna(subset=cols)

        elif strategy == "drop_columns":
            self.df = self.df.drop(columns=cols, errors="ignore")

        elif strategy in ("mean", "median"):
            num_cols = [c for c in cols if is_numeric_dtype(self.df[c])]
            for col in num_cols:
                val = self.df[col].mean() if strategy == "mean" else self.df[col].median()
                self.df[col] = self.df[col].fillna(val)

        elif strategy == "mode":
            for col in cols:
                mode_vals = self.df[col].mode()
                if not mode_vals.empty:
                    self.df[col] = self.df[col].fillna(mode_vals[0])

        elif strategy == "zero":
            for col in cols:
                self.df[col] = self.df[col].fillna(0)

        elif strategy == "custom" and fill_value is not None:
            for col in cols:
                self.df[col] = self.df[col].fillna(fill_value)

        return self.df

    # ------------------------------------------------------------------
    # Step 3 – Encode categorical features
    # ------------------------------------------------------------------

    def encode_categorical(self, method: str, columns: list | None = None) -> pd.DataFrame:
        """
        method options:
            'label'  – LabelEncoder (integer codes)
            'onehot' – pd.get_dummies (binary columns, drop_first=False)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not columns:
            return self.df

        if method == "label":
            for col in columns:
                non_null_mask = self.df[col].notna()
                if not non_null_mask.any():
                    self.df[col] = pd.Series(np.nan, index=self.df.index, dtype="float64")
                    continue

                le = LabelEncoder()
                encoded = pd.Series(np.nan, index=self.df.index, dtype="float64")
                encoded.loc[non_null_mask] = le.fit_transform(
                    self.df.loc[non_null_mask, col].astype(str)
                )
                self.df[col] = encoded
                self.label_encoders[col] = le

        elif method == "onehot":
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=False)

        return self.df

    # ------------------------------------------------------------------
    # Step 4 – Scale numeric features
    # ------------------------------------------------------------------

    def scale_features(self, method: str, columns: list | None = None) -> pd.DataFrame:
        """
        method options: 'standard', 'minmax', 'robust'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return self.df

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self.scaler = scalers.get(method, StandardScaler())
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        self.scaled_columns = columns
        return self.df

    # ------------------------------------------------------------------
    # Step 5 – Outlier handling
    # ------------------------------------------------------------------

    def handle_outliers(
        self,
        method: str,
        columns: list | None = None,
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        method options:
            'iqr_remove'    – remove rows outside IQR fence
            'iqr_cap'       – cap values to IQR fence (Winsorize)
            'zscore_remove' – remove rows with |z-score| > 3
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "iqr_remove":
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col in columns:
                q1, q3 = self.df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                mask &= ~(
                    (self.df[col] < q1 - threshold * iqr)
                    | (self.df[col] > q3 + threshold * iqr)
                )
            self.df = self.df[mask]

        elif method == "iqr_cap":
            for col in columns:
                q1, q3 = self.df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.df[col] = self.df[col].clip(
                    lower=q1 - threshold * iqr,
                    upper=q3 + threshold * iqr,
                )

        elif method == "zscore_remove":
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col in columns:
                col_filled = self.df[col].fillna(self.df[col].mean())
                std = col_filled.std()
                if std == 0:
                    continue
                z = (col_filled - col_filled.mean()) / std
                mask &= z.abs() < threshold
            self.df = self.df[mask]

        return self.df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_data(self) -> pd.DataFrame:
        return self.df
