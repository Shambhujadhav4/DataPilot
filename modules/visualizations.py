import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc


class DataVisualizer:
    """Collection of static Plotly-based visualization helpers."""

    # ------------------------------------------------------------------
    # EDA – Univariate
    # ------------------------------------------------------------------

    @staticmethod
    def plot_missing_values(df: pd.DataFrame):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return None
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            title="Missing Values per Column",
            labels={"x": "Column", "y": "Missing Count"},
            color=missing.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str):
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Histogram + KDE", "Box Plot"]
        )
        fig.add_trace(
            go.Histogram(x=df[column], name="Distribution", nbinsx=30, opacity=0.75),
            row=1,
            col=1,
        )
        fig.add_trace(go.Box(y=df[column], name="Box Plot", boxmean=True), row=1, col=2)
        fig.update_layout(
            title_text=f"Distribution of <b>{column}</b>",
            showlegend=False,
            height=420,
        )
        return fig

    @staticmethod
    def plot_countplot(df: pd.DataFrame, column: str):
        counts = df[column].value_counts()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            title=f"Value Counts: <b>{column}</b>",
            labels={"x": column, "y": "Count"},
            color=counts.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    # ------------------------------------------------------------------
    # EDA – Multivariate
    # ------------------------------------------------------------------

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr().round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values,
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        fig.update_layout(title="Correlation Heatmap", height=600)
        return fig

    @staticmethod
    def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col=None):
        kwargs = dict(x=x_col, y=y_col, title=f"{x_col} vs {y_col}", opacity=0.7)
        if color_col:
            kwargs["color"] = color_col
        else:
            kwargs["trendline"] = "ols"
        fig = px.scatter(df, **kwargs)
        return fig

    @staticmethod
    def plot_scatter_matrix(df: pd.DataFrame, columns: list, color=None):
        dims = columns[: min(len(columns), 6)]
        fig = px.scatter_matrix(df, dimensions=dims, color=color, title="Scatter Matrix")
        fig.update_traces(diagonal_visible=False)
        return fig

    @staticmethod
    def plot_boxplot(df: pd.DataFrame, column: str, group_by=None):
        if group_by:
            fig = px.box(df, x=group_by, y=column, title=f"{column} by {group_by}")
        else:
            fig = px.box(df, y=column, title=f"Box Plot – {column}")
        return fig

    @staticmethod
    def plot_data_types(df: pd.DataFrame):
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=[str(d) for d in dtype_counts.index],
            title="Data Types Distribution",
        )
        return fig

    # ------------------------------------------------------------------
    # Model evaluation – Classification
    # ------------------------------------------------------------------

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, labels=None):
        if labels is None:
            labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        str_labels = [str(l) for l in labels]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=str_labels,
                y=str_labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500,
        )
        return fig

    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, classes):
        if len(classes) != 2:
            return None
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
        )
        return fig

    # ------------------------------------------------------------------
    # Model evaluation – Regression
    # ------------------------------------------------------------------

    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred):
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        min_v = min(y_test_arr.min(), y_pred_arr.min())
        max_v = max(y_test_arr.max(), y_pred_arr.max())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_test_arr,
                y=y_pred_arr,
                mode="markers",
                name="Predictions",
                marker=dict(color="#1f77b4", opacity=0.6, size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Actual vs Predicted",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            height=450,
        )
        return fig

    @staticmethod
    def plot_residuals(y_test, y_pred):
        residuals = np.array(y_test) - np.array(y_pred)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Residuals vs Predicted", "Residuals Distribution"],
        )
        fig.add_trace(
            go.Scatter(
                x=np.array(y_pred),
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="#1f77b4", opacity=0.6, size=5),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(
            go.Histogram(x=residuals, name="Distribution", nbinsx=30, opacity=0.75),
            row=1,
            col=2,
        )
        fig.update_layout(title="Residual Analysis", showlegend=False, height=420)
        return fig

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
        top = importance_df.head(top_n)
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {min(top_n, len(top))} Feature Importances",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, top_n * 22),
            coloraxis_showscale=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Cross-validation bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_cv_scores(cv_scores: np.ndarray, scoring_name: str = "Score"):
        folds = [f"Fold {i + 1}" for i in range(len(cv_scores))]
        mean_score = cv_scores.mean()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=folds,
                y=cv_scores,
                name=scoring_name,
                marker_color="#1f77b4",
                opacity=0.8,
            )
        )
        fig.add_hline(
            y=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_score:.4f}",
        )
        fig.update_layout(
            title=f"Cross-Validation {scoring_name}s",
            xaxis_title="Fold",
            yaxis_title=scoring_name,
            height=380,
        )
        return fig
