"""
ML Dashboard – Streamlit Application
Upload a CSV → Explore → Preprocess → Train → Visualise Results
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from modules.models import ModelTrainer
from modules.preprocessing import DataPreprocessor
from modules.visualizations import DataVisualizer

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .big-title  { font-size:2.2rem; font-weight:700; color:#1f77b4; text-align:center; padding:.6rem 0; }
    .sec-header { font-size:1.25rem; font-weight:700; color:#2c3e50;
                  border-bottom:2px solid #1f77b4; padding-bottom:.3rem; margin-bottom:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "raw_data": None,
    "processed_data": None,
    "preprocessing_log": [],
    "model_trainer": None,
    "model_results": None,
    "target_column": None,
    "feature_columns": [],
    "task_type": None,
    "current_page": "📤 Data Upload",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────
PAGES = [
    "📤 Data Upload",
    "🔍 Data Exploration",
    "⚙️ Preprocessing",
    "🧠 Model Training",
    "📊 Results & Visualizations",
]

st.sidebar.markdown('<div class="big-title">🤖 ML Dashboard</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    PAGES,
    index=PAGES.index(st.session_state.current_page),
)
st.session_state.current_page = page

# Status badges in sidebar
if st.session_state.raw_data is not None:
    st.sidebar.markdown("---")
    raw = st.session_state.raw_data
    st.sidebar.success(f"✅ Data: {raw.shape[0]:,} rows × {raw.shape[1]} cols")
    if st.session_state.processed_data is not None:
        proc = st.session_state.processed_data
        st.sidebar.info(f"⚙️ Preprocessed: {proc.shape[0]:,} rows × {proc.shape[1]} cols")
    if st.session_state.model_results is not None:
        st.sidebar.success("🧠 Model trained!")

# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────
def _go(target: str):
    st.session_state.current_page = target
    st.rerun()


def _next_btn(label: str, target: str):
    if st.button(label, use_container_width=True):
        _go(target)


# ======================================================================
# PAGE 1 – Data Upload
# ======================================================================
if page == "📤 Data Upload":
    st.markdown('<div class="big-title">📤 Data Upload</div>', unsafe_allow_html=True)

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload your CSV dataset", type=["csv"],
            help="Supported format: comma-separated values (.csv)",
        )
    with col_opt:
        st.markdown("**Parse options**")
        sep = st.selectbox("Separator", [",", ";", "\\t", "|"], index=0)
        enc = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"], index=0)
        hdr = st.number_input("Header row", min_value=0, max_value=10, value=0)

    if uploaded is not None:
        try:
            actual_sep = "\t" if sep == "\\t" else sep
            df = pd.read_csv(uploaded, sep=actual_sep, encoding=enc, header=int(hdr))
            st.session_state.raw_data = df
            st.session_state.processed_data = df.copy()
            st.session_state.preprocessing_log = []
            st.session_state.model_trainer = None
            st.session_state.model_results = None
            st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data

        # Quick stats
        st.markdown('<div class="sec-header">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))
        c4.metric("Duplicate rows", int(df.duplicated().sum()))

        st.markdown("**Preview**")
        max_preview_rows = max(1, min(200, len(df)))
        default_preview_rows = min(10, max_preview_rows)
        n = st.slider("Rows to display", 1, max_preview_rows, default_preview_rows)
        st.dataframe(df.head(n), use_container_width=True)

        st.markdown("**Column info**")
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-null": df.count().values,
                "Null": df.isnull().sum().values,
                "Unique": df.nunique().values,
            }
        )
        st.dataframe(col_info, use_container_width=True)

        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.markdown("---")
        _next_btn("➡️ Explore Data", "🔍 Data Exploration")


# ======================================================================
# PAGE 2 – Data Exploration
# ======================================================================
elif page == "🔍 Data Exploration":
    st.markdown('<div class="big-title">🔍 Data Exploration</div>', unsafe_allow_html=True)

    if st.session_state.raw_data is None:
        st.warning("⚠️ Please upload a dataset first.")
        if st.button("← Go to Data Upload"):
            _go("📤 Data Upload")
    else:
        df = st.session_state.processed_data
        viz = DataVisualizer()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        tab_dist, tab_corr, tab_scatter, tab_box, tab_cat, tab_dtypes = st.tabs(
            ["📊 Distributions", "🔥 Correlation", "📈 Scatter", "📦 Box Plots", "🔢 Categorical", "🗂️ Data Types"]
        )

        # ── Distributions ──────────────────────────────────────────────
        with tab_dist:
            if num_cols:
                sel = st.selectbox("Select numeric column", num_cols, key="dist_col")
                fig = viz.plot_distribution(df, sel)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found.")

            st.markdown("---")
            mis_fig = viz.plot_missing_values(df)
            if mis_fig:
                st.markdown("**Missing Values Overview**")
                st.plotly_chart(mis_fig, use_container_width=True)
            else:
                st.success("✅ No missing values in this dataset!")

        # ── Correlation ────────────────────────────────────────────────
        with tab_corr:
            corr_fig = viz.plot_correlation_heatmap(df)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.warning("Need ≥ 2 numeric columns for a correlation heatmap.")

        # ── Scatter ────────────────────────────────────────────────────
        with tab_scatter:
            if len(num_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                x_col = c1.selectbox("X-axis", num_cols, key="sc_x")
                y_col = c2.selectbox("Y-axis", num_cols, index=1, key="sc_y")
                color_opts = ["None"] + df.columns.tolist()
                color_sel = c3.selectbox("Colour by", color_opts, key="sc_c")
                color = None if color_sel == "None" else color_sel
                st.plotly_chart(viz.plot_scatter(df, x_col, y_col, color), use_container_width=True)

                if len(num_cols) >= 3:
                    st.markdown("**Scatter Matrix (first 6 numeric cols)**")
                    st.plotly_chart(
                        viz.plot_scatter_matrix(df, num_cols, color=color),
                        use_container_width=True,
                    )
            else:
                st.info("Need ≥ 2 numeric columns for a scatter plot.")

        # ── Box Plots ──────────────────────────────────────────────────
        with tab_box:
            if num_cols:
                c1, c2 = st.columns(2)
                box_col = c1.selectbox("Numeric column", num_cols, key="box_col")
                grp_opts = ["None"] + cat_cols
                grp_col = c2.selectbox("Group by (optional)", grp_opts, key="box_grp")
                grp = None if grp_col == "None" else grp_col
                st.plotly_chart(viz.plot_boxplot(df, box_col, grp), use_container_width=True)
            else:
                st.info("No numeric columns found.")

        # ── Categorical ────────────────────────────────────────────────
        with tab_cat:
            if cat_cols:
                sel_cat = st.selectbox("Select categorical column", cat_cols, key="cat_col")
                st.plotly_chart(viz.plot_countplot(df, sel_cat), use_container_width=True)

                st.markdown(f"**Top 20 values – {sel_cat}**")
                st.dataframe(
                    df[sel_cat].value_counts().head(20).rename("Count").reset_index(),
                    use_container_width=True,
                )
            else:
                st.info("No categorical columns found.")

        # ── Data Types ─────────────────────────────────────────────────
        with tab_dtypes:
            st.plotly_chart(viz.plot_data_types(df), use_container_width=True)

        st.markdown("---")
        _next_btn("➡️ Preprocess Data", "⚙️ Preprocessing")


# ======================================================================
# PAGE 3 – Preprocessing
# ======================================================================
elif page == "⚙️ Preprocessing":
    st.markdown('<div class="big-title">⚙️ Data Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.raw_data is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.processed_data

        # Preprocessing history
        if st.session_state.preprocessing_log:
            with st.expander("📋 Preprocessing History", expanded=False):
                for i, entry in enumerate(st.session_state.preprocessing_log, 1):
                    st.markdown(f"**{i}.** {entry}")

        # Reset button
        _, reset_col = st.columns([5, 1])
        with reset_col:
            if st.button("🔄 Reset to Original"):
                st.session_state.processed_data = st.session_state.raw_data.copy()
                st.session_state.preprocessing_log = []
                st.success("Data reset to original!")
                st.rerun()

        st.markdown("---")

        tab_drop, tab_miss, tab_enc, tab_scale, tab_out = st.tabs(
            ["🗑️ Drop Columns", "🩹 Missing Values", "🔡 Encoding", "📏 Scaling", "📐 Outliers"]
        )

        # ── Drop Columns ───────────────────────────────────────────────
        with tab_drop:
            st.markdown("Select columns you want to remove from the dataset.")
            to_drop = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_cols")
            if st.button("Drop Selected Columns", key="btn_drop"):
                if to_drop:
                    p = DataPreprocessor(st.session_state.processed_data)
                    st.session_state.processed_data = p.drop_columns(to_drop)
                    st.session_state.preprocessing_log.append(f"Dropped columns: {to_drop}")
                    st.success(f"Dropped {len(to_drop)} column(s).")
                    st.rerun()
                else:
                    st.warning("No columns selected.")

        # ── Missing Values ─────────────────────────────────────────────
        with tab_miss:
            miss_info = df.isnull().sum()
            miss_cols = miss_info[miss_info > 0].index.tolist()

            if not miss_cols:
                st.success("✅ No missing values in the current dataset!")
            else:
                miss_df = pd.DataFrame(
                    {
                        "Column": miss_cols,
                        "Missing Count": [miss_info[c] for c in miss_cols],
                        "Missing %": [f"{miss_info[c]/len(df)*100:.1f}%" for c in miss_cols],
                    }
                )
                st.dataframe(miss_df, use_container_width=True)

            c1, c2 = st.columns(2)
            strategy = c1.selectbox(
                "Strategy",
                ["drop_rows", "mean", "median", "mode", "zero", "custom"],
                format_func=lambda x: {
                    "drop_rows": "Drop rows with nulls",
                    "mean": "Fill with mean",
                    "median": "Fill with median",
                    "mode": "Fill with mode",
                    "zero": "Fill with 0",
                    "custom": "Fill with custom value",
                }[x],
            )
            apply_to = c2.multiselect(
                "Apply to columns (empty = all)", df.columns.tolist(), key="miss_cols"
            )
            fill_val = None
            if strategy == "custom":
                fill_val = st.text_input("Custom fill value", "0")

            if st.button("Apply Missing Value Treatment", key="btn_miss"):
                p = DataPreprocessor(st.session_state.processed_data)
                cols = apply_to if apply_to else None
                st.session_state.processed_data = p.handle_missing_values(strategy, cols, fill_val)
                st.session_state.preprocessing_log.append(
                    f"Missing values → {strategy} on {cols or 'all columns'}"
                )
                st.success("Done!")
                st.rerun()

        # ── Encoding ───────────────────────────────────────────────────
        with tab_enc:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if not cat_cols:
                st.info("No categorical columns detected.")
            else:
                st.info(f"Categorical columns detected: **{cat_cols}**")
                c1, c2 = st.columns(2)
                enc_method = c1.selectbox(
                    "Encoding method",
                    ["label", "onehot"],
                    format_func=lambda x: "Label Encoding" if x == "label" else "One-Hot Encoding",
                )
                enc_cols = c2.multiselect(
                    "Columns to encode (empty = all categorical)",
                    cat_cols,
                    key="enc_cols",
                )
                if st.button("Apply Encoding", key="btn_enc"):
                    p = DataPreprocessor(st.session_state.processed_data)
                    cols = enc_cols if enc_cols else None
                    st.session_state.processed_data = p.encode_categorical(enc_method, cols)
                    st.session_state.preprocessing_log.append(
                        f"Encoding → {enc_method} on {cols or 'all categorical'}"
                    )
                    st.success("Done!")
                    st.rerun()

        # ── Scaling ────────────────────────────────────────────────────
        with tab_scale:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            c1, c2 = st.columns(2)
            scale_method = c1.selectbox(
                "Scaling method",
                ["standard", "minmax", "robust"],
                format_func=lambda x: {
                    "standard": "Standard Scaler (Z-score)",
                    "minmax": "Min-Max Scaler [0, 1]",
                    "robust": "Robust Scaler (IQR-based)",
                }[x],
            )
            scale_cols = c2.multiselect(
                "Columns to scale (empty = all numeric)", num_cols, key="scale_cols"
            )
            if st.button("Apply Scaling", key="btn_scale"):
                p = DataPreprocessor(st.session_state.processed_data)
                cols = scale_cols if scale_cols else None
                st.session_state.processed_data = p.scale_features(scale_method, cols)
                st.session_state.preprocessing_log.append(
                    f"Scaling → {scale_method} on {cols or 'all numeric'}"
                )
                st.success("Done!")
                st.rerun()

        # ── Outliers ───────────────────────────────────────────────────
        with tab_out:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            c1, c2, c3 = st.columns(3)
            out_method = c1.selectbox(
                "Method",
                ["iqr_remove", "iqr_cap", "zscore_remove"],
                format_func=lambda x: {
                    "iqr_remove": "IQR – Remove rows",
                    "iqr_cap": "IQR – Cap (Winsorize)",
                    "zscore_remove": "Z-Score – Remove rows",
                }[x],
            )
            threshold_label = (
                "Z-Score threshold"
                if out_method == "zscore_remove"
                else "IQR threshold multiplier"
            )
            threshold_default = 3.0 if out_method == "zscore_remove" else 1.5
            threshold = c2.slider(threshold_label, 1.0, 3.0, threshold_default, 0.1)
            out_cols = c3.multiselect(
                "Columns (empty = all numeric)", num_cols, key="out_cols"
            )
            if st.button("Handle Outliers", key="btn_out"):
                p = DataPreprocessor(st.session_state.processed_data)
                cols = out_cols if out_cols else None
                st.session_state.processed_data = p.handle_outliers(out_method, cols, threshold)
                st.session_state.preprocessing_log.append(
                    f"Outliers → {out_method} on {cols or 'all numeric'}"
                )
                st.success("Done!")
                st.rerun()

        # ── Preview ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-header">Preprocessed Data Preview</div>', unsafe_allow_html=True)
        proc = st.session_state.processed_data
        c1, c2 = st.columns(2)
        c1.info(f"Original: {st.session_state.raw_data.shape[0]:,} rows × {st.session_state.raw_data.shape[1]} cols")
        c2.success(f"After preprocessing: {proc.shape[0]:,} rows × {proc.shape[1]} cols")
        st.dataframe(proc.head(10), use_container_width=True)

        # Download
        buf = io.StringIO()
        proc.to_csv(buf, index=False)
        st.download_button(
            "📥 Download Processed CSV",
            data=buf.getvalue(),
            file_name="processed_data.csv",
            mime="text/csv",
        )

        st.markdown("---")
        _next_btn("➡️ Train a Model", "🧠 Model Training")


# ======================================================================
# PAGE 4 – Model Training
# ======================================================================
elif page == "🧠 Model Training":
    st.markdown('<div class="big-title">🧠 Model Training</div>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.processed_data

        st.markdown('<div class="sec-header">1 · Task & Model Selection</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        task_type = c1.radio(
            "Task type",
            ["classification", "regression"],
            format_func=str.capitalize,
            horizontal=True,
        )
        st.session_state.task_type = task_type

        model_opts = (
            list(ModelTrainer.CLASSIFICATION_MODELS.keys())
            if task_type == "classification"
            else list(ModelTrainer.REGRESSION_MODELS.keys())
        )
        selected_model = c2.selectbox("Algorithm", model_opts)

        st.markdown('<div class="sec-header">2 · Target & Features</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Target column (y)", df.columns.tolist())
        st.session_state.target_column = target_col

        available_feats = [c for c in df.columns if c != target_col]
        feature_cols = c2.multiselect(
            "Feature columns (X)",
            available_feats,
            default=available_feats[: min(len(available_feats), 15)],
        )
        st.session_state.feature_columns = feature_cols

        st.markdown('<div class="sec-header">3 · Training Options</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test split ratio", 0.05, 0.5, 0.2, 0.05)
        random_state = c2.number_input("Random seed", 0, 9999, 42)
        run_cv = c3.checkbox("5-fold Cross-Validation", value=True)

        st.markdown("---")
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if not feature_cols:
                st.error("Select at least one feature column.")
            else:
                non_num = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
                if non_num:
                    st.error(
                        f"Non-numeric features detected: **{non_num}**. "
                        "Please encode them in the Preprocessing step first."
                    )
                elif df[target_col].isnull().any():
                    st.error("Target column contains missing values. Fix them in Preprocessing.")
                else:
                    with st.spinner("Training… please wait"):
                        try:
                            trainer = ModelTrainer()
                            trainer.prepare_data(df, target_col, feature_cols, test_size, int(random_state))
                            trainer.train(task_type, selected_model)
                            metrics = trainer.get_metrics()
                            if run_cv:
                                try:
                                    metrics["cv_scores"] = trainer.get_cross_val_scores(cv=5)
                                except Exception as cv_err:
                                    st.warning(f"CV failed: {cv_err}")
                            st.session_state.model_trainer = trainer
                            st.session_state.model_results = metrics
                            st.success("✅ Model trained successfully!")
                            st.balloons()
                        except Exception as exc:
                            st.error(f"Training failed: {exc}")

        # Quick result summary
        if st.session_state.model_results:
            st.markdown('<div class="sec-header">Quick Results</div>', unsafe_allow_html=True)
            m = st.session_state.model_results

            if task_type == "classification":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                c2.metric("F1-Score", f"{m.get('f1_score', 0):.4f}")
                c3.metric("Precision", f"{m.get('precision', 0):.4f}")
                c4.metric("Recall", f"{m.get('recall', 0):.4f}")
                if "roc_auc" in m:
                    st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² Score", f"{m.get('r2_score', 0):.4f}")
                c2.metric("RMSE", f"{m.get('rmse', 0):.4f}")
                c3.metric("MAE", f"{m.get('mae', 0):.4f}")
                c4.metric("MSE", f"{m.get('mse', 0):.4f}")

            if "cv_scores" in m:
                cv = m["cv_scores"]
                st.info(f"5-Fold CV → mean: **{cv.mean():.4f}**  std: **{cv.std():.4f}**")

            st.markdown("---")
            _next_btn("➡️ View Full Results & Visualizations", "📊 Results & Visualizations")


# ======================================================================
# PAGE 5 – Results & Visualizations
# ======================================================================
elif page == "📊 Results & Visualizations":
    st.markdown('<div class="big-title">📊 Results & Visualizations</div>', unsafe_allow_html=True)

    if st.session_state.model_results is None:
        st.warning("⚠️ Please train a model first.")
        if st.button("← Go to Model Training"):
            _go("🧠 Model Training")
    else:
        m = st.session_state.model_results
        trainer: ModelTrainer = st.session_state.model_trainer
        viz = DataVisualizer()
        task = st.session_state.task_type

        tab_met, tab_plots, tab_feat, tab_cv = st.tabs(
            ["📈 Metrics", "🎯 Model Plots", "🔍 Feature Importance", "🔁 Cross-Validation"]
        )

        # ── Metrics ────────────────────────────────────────────────────
        with tab_met:
            if task == "classification":
                st.markdown('<div class="sec-header">Classification Metrics</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                c2.metric("F1-Score (weighted)", f"{m.get('f1_score', 0):.4f}")
                c3.metric("Precision (weighted)", f"{m.get('precision', 0):.4f}")
                c4.metric("Recall (weighted)", f"{m.get('recall', 0):.4f}")
                if "roc_auc" in m:
                    st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
                st.markdown("**Classification Report**")
                st.code(m.get("classification_report", ""), language="text")
            else:
                st.markdown('<div class="sec-header">Regression Metrics</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² Score", f"{m.get('r2_score', 0):.4f}")
                c2.metric("RMSE", f"{m.get('rmse', 0):.4f}")
                c3.metric("MAE", f"{m.get('mae', 0):.4f}")
                c4.metric("MSE", f"{m.get('mse', 0):.6f}")

        # ── Model Plots ────────────────────────────────────────────────
        with tab_plots:
            if task == "classification":
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Confusion Matrix**")
                    cm_fig = viz.plot_confusion_matrix(trainer.y_test, trainer.y_pred)
                    st.plotly_chart(cm_fig, use_container_width=True)
                with c2:
                    if trainer.y_pred_proba is not None:
                        classes = sorted(trainer.y_test.unique().tolist())
                        roc_fig = viz.plot_roc_curve(trainer.y_test, trainer.y_pred_proba, classes)
                        if roc_fig:
                            st.markdown("**ROC Curve**")
                            st.plotly_chart(roc_fig, use_container_width=True)
                        else:
                            st.info("ROC curve is only shown for binary classification.")
                    else:
                        st.info("This model does not support probability estimates.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Actual vs Predicted**")
                    st.plotly_chart(
                        viz.plot_actual_vs_predicted(trainer.y_test, trainer.y_pred),
                        use_container_width=True,
                    )
                with c2:
                    st.markdown("**Residual Analysis**")
                    st.plotly_chart(
                        viz.plot_residuals(trainer.y_test, trainer.y_pred),
                        use_container_width=True,
                    )

        # ── Feature Importance ─────────────────────────────────────────
        with tab_feat:
            importance = trainer.get_feature_importance()
            if importance is not None:
                top_n = st.slider("Top N features to display", 5, min(50, len(importance)), min(20, len(importance)))
                st.plotly_chart(
                    viz.plot_feature_importance(importance, top_n),
                    use_container_width=True,
                )
                st.markdown("**Full Feature Importance Table**")
                st.dataframe(importance, use_container_width=True)
            else:
                st.info(
                    "Feature importance is not available for this model type. "
                    "Try Random Forest, Decision Tree, or Gradient Boosting."
                )

        # ── Cross-Validation ───────────────────────────────────────────
        with tab_cv:
            if "cv_scores" in m:
                cv = m["cv_scores"]
                scoring_name = "Accuracy" if task == "classification" else "R²"
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean CV Score", f"{cv.mean():.4f}")
                c2.metric("Std CV Score", f"{cv.std():.4f}")
                c3.metric("Min / Max", f"{cv.min():.4f} / {cv.max():.4f}")
                st.plotly_chart(
                    viz.plot_cv_scores(cv, scoring_name),
                    use_container_width=True,
                )
            else:
                st.info("Cross-validation was not run. Enable it on the Model Training page.")
