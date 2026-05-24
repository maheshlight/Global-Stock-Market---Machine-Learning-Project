# 📈 Global Stock Market — Machine Learning Project

> **Team 1 | Machine Learning Mini Project | Python & Jupyter Notebook**

---

## 📌 Project Overview

This project performs an end-to-end **Machine Learning analysis on Global Stock Market data** covering **200 companies across 10 countries** from **January 2023 to March 2026** — a dataset of **90,040 records**.

The project tackles **10 real-world financial problem statements** using supervised, unsupervised, and statistical ML techniques, with fully interactive visualizations.

---

## 👥 Team Members

- Ankit
- Samir
- Sushant
- Mahesh
- Pooja

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|------------------|
| **Language** | Python 3 |
| **Notebook Environment** | Jupyter Notebook |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Plotly Express, Plotly Graph Objects, Plotly Subplots, Plotly Figure Factory, Seaborn |
| **Machine Learning** | Scikit-learn (sklearn) |
| **ML Models** | Linear Regression, Logistic Regression, Random Forest (Regressor & Classifier), Gradient Boosting Regressor, Decision Tree Classifier, KMeans Clustering, Isolation Forest, Local Outlier Factor, Ridge Regression |
| **ML Utilities** | StandardScaler, LabelEncoder, PCA, train_test_split |
| **ML Metrics** | MAE, RMSE, R², ROC-AUC, Confusion Matrix, Classification Report |
| **Optimization** | SciPy (`scipy.optimize.minimize`) |
| **Data Source** | `Global_Stock_Market_Master_Dataset.xlsx` |

---

## 📂 Dataset

| Field | Detail |
|-------|--------|
| **File** | `Global_Stock_Market_Master_Dataset.xlsx` |
| **Records** | 90,040 rows |
| **Companies** | 200 |
| **Countries** | 10 |
| **Date Range** | January 2023 – March 2026 |

**Columns Used:**
`Date`, `Country`, `Company`, `Sector`, `Sub_Sector`, `Open`, `High`, `Low`, `Close`, `Volume`, `BUY`, `SELL`, `Daily_Return`, `War_Period`

**Engineered Features:**
`Year`, `Month`, `Price_Range`, `BuySell_Ratio`, Lag features (`Lag1_Close`, `Lag2_Close`), Moving Averages (`MA5`, `MA10`)

---

## 🧠 Problem Statements Covered

| # | Problem Statement | ML Method | Models Used |
|---|------------------|-----------|-------------|
| 1 | **Stock Price Prediction** | Regression | Linear Regression, Random Forest, Gradient Boosting |
| 2 | **Buy or Sell Signal Classification** | Classification | Logistic Regression, Decision Tree, Random Forest |
| 3 | **Cross-Sector & Cross-Country Market Analysis** | Clustering + EDA | KMeans, PCA |
| 4 | **Investor Behavior & Market Sentiment Analysis** | Feature Engineering + Regression | Random Forest |
| 5 | **Portfolio Optimization** | Modern Portfolio Theory | SciPy Optimization |
| 6 | **Stock Volatility Forecasting** | Regression | Random Forest, Ridge Regression |
| 7 | **Anomaly Detection in Stock Data** | Unsupervised | Isolation Forest, Local Outlier Factor |
| 8 | **Stock Market Trend Classification (Multi-Class)** | Multi-Class Classification | Random Forest Classifier |
| 9 | **War Period Impact Analysis** | Statistical Testing | Hypothesis Testing |
| 10 | **Sector Rotation Strategy using Return Momentum** | Momentum Strategy | Return Analysis |

---

## 🚀 How to Run

1. **Clone or download** this repository.
2. Make sure you have Python 3 installed.
3. Install required libraries:
   ```bash
   pip install pandas numpy plotly seaborn scikit-learn scipy openpyxl
   ```
4. Place `Global_Stock_Market_Master_Dataset.xlsx` in the same folder as the notebook.
5. Open the notebook:
   ```bash
   jupyter notebook Global_Stock_Market_ML_Project.ipynb
   ```
6. Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## 📊 Key Outputs

- Interactive Plotly charts for every problem statement
- Model performance comparison tables (MAE, RMSE, R², ROC-AUC)
- Feature importance plots
- Geographic and sector-wise heatmaps
- Optimized portfolio weights and efficient frontier
- Anomaly flagging on stock data
- War period impact statistical report

---

## 📅 Project Info

| Field | Detail |
|-------|--------|
| **Project Type** | Academic Mini Project |
| **Language** | Python 3 |
| **Notebook Format** | `.ipynb` (Jupyter Notebook) |
| **Dataset Period** | Jan 2023 – Mar 2026 |
| **Total Problem Statements** | 10 |

---

## 📝 License

This project is created for **educational purposes only**. All stock market data used is for academic demonstration and analysis.
