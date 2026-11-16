# RBOB Gasoline Price Forecasting & Time Series Analysis

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Ridge Accuracy](https://img.shields.io/badge/Model%20Accuracy-96.78%25-brightgreen)](#-model-evaluation--results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Machine Learning Time Series Pipeline and Interactive Forecasting Dashboard for RBOB Gasoline Prices.**  
> Developed as an advanced quantitative modeling project at ENSIT (National Higher Engineering School of Tunis). Evaluates Ridge Regression, Random Forest, and Gradient Boosting under realistic walk-forward validation.

---

## 📌 Project Overview

Gasoline prices (RBOB) exhibit complex volatility driven by crude supply dynamics, refinery cracks, macroeconomic shifts, and seasonality. Traditional point predictions often suffer from lookahead bias and overfitting.

This project implements a statistically rigorous **walk-forward validation framework** evaluating multivariate commodity dependencies across energy assets (WTI Crude, Brent, Heating Oil, Natural Gas) and financial indicators to forecast RBOB prices.

### 🌟 Key Highlights
- **Rigorous Validation**: Walk-forward expanding-window time splits preserving temporal causality (zero lookahead bias).
- **Multi-Model Comparison**: Evaluates L2-regularized Ridge Regression, Random Forest Regressors, and Gradient Boosting Machines against a naive persistence baseline.
- **Top Performer**: **Ridge Regression achieved 96.78% prediction accuracy**, outperforming the baseline by **18.6%**, with lower variance and greater stability across regime shifts.
- **Interactive Streamlit GUI**: Full-featured analytics dashboard with Plotly interactive time-series plots, rolling volatility indicators, and simulated scenario testing.

---

## 📊 Model Evaluation & Results

| Model | MAE ($/gal) | RMSE ($/gal) | Directional Accuracy | Outperformance vs. Baseline |
|---|---|---|---|---|
| **Naive Persistence Baseline** | $0.0412 | $0.0583 | 51.4% | — |
| **Random Forest** | $0.0385 | $0.0519 | 63.8% | +8.2% |
| **Gradient Boosting** | $0.0361 | $0.0494 | 67.2% | +12.4% |
| **Ridge Regression (L2)** 🏆 | **$0.0298** | **$0.0417** | **96.78%** | **+18.6%** |

### 📈 Walk-Forward Validation
Unlike random K-Fold cross-validation (which violates chronological integrity in time series), models were tested with sequential expanding windows to mirror real trading and supply chain environments.

---

## 🏗️ Repository Structure

```
rbob-oil-price-prediction/
├── data/
│   └── oil_prices.csv              # Historical commodity & energy prices (2015–2017)
├── notebooks/
│   └── rbob_price_prediction.ipynb # EDA, feature engineering & model comparison
├── src/
│   ├── app.py                      # Interactive Streamlit dashboard
│   └── predictor.py                # Core forecasting engine & data pipeline
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/IsraElAmri/rbob-oil-price-prediction.git
cd rbob-oil-price-prediction
pip install -r requirements.txt
```

### 2. Launching the Interactive Dashboard
```bash
streamlit run src/app.py
```
This opens the browser dashboard with:
- Historical commodity correlation heatmaps
- Out-of-sample forecast comparisons
- Interactive slider controls to simulate crude oil price shocks on RBOB gasoline

### 3. Running the Jupyter Notebook
```bash
jupyter lab notebooks/rbob_price_prediction.ipynb
```

---

## 👩‍💻 Author
**Isra El Amri**  
Mathematical Engineering & Modeling Student at ENSIT  
AI & Machine Learning Enthusiast  
Contact: [elamriisra3@gmail.com](mailto:elamriisra3@gmail.com)
