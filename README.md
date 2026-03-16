# 🤖 ML Dashboard

An interactive, browser-based machine learning dashboard built with **Streamlit**.  
Upload any CSV dataset, explore it, preprocess it, train a model, and visualise the results — all without writing a single line of code.

---

## Features

| Step | What you can do |
|------|----------------|
| **📤 Upload** | Upload a CSV with configurable separator, encoding, and header row |
| **🔍 Explore** | Distributions, correlation heatmap, scatter plots, box plots, categorical counts |
| **⚙️ Preprocess** | Drop columns · Handle missing values · Label/One-Hot encoding · Feature scaling · Outlier removal |
| **🧠 Train** | Classification or Regression · 6–8 algorithms · Configurable train/test split · 5-fold CV |
| **📊 Results** | Metrics · Confusion matrix / ROC curve / Actual-vs-Predicted / Residuals · Feature importance · CV bar chart |

---

## Quick Start

```bash
# 1. Go into the project folder
cd ml-dashboard

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## Project Structure

```
ml-dashboard/
├── app.py                    # Streamlit UI (5 pages)
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py      # DataPreprocessor class
│   ├── models.py             # ModelTrainer class
│   └── visualizations.py    # DataVisualizer class (Plotly)
└── README.md
```

---

## Supported Algorithms

**Classification**
- Logistic Regression
- Random Forest
- Decision Tree
- SVM (with probability)
- K-Nearest Neighbors
- Gradient Boosting

**Regression**
- Linear Regression
- Ridge / Lasso Regression
- Random Forest
- Decision Tree
- SVR
- Gradient Boosting
- K-Nearest Neighbors

---

## Notes

- All features fed to the model must be **numeric**. Use the Encoding step to convert categorical columns first.  
- The preprocessing pipeline is applied in-memory; download the processed CSV any time from the Preprocessing page.  
- Cross-validation clones the model and re-fits on the full dataset, so it does not leak test-set information.
