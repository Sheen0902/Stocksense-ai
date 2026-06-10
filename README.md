# ☀️ StockSense AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-FFB800?style=flat-square&logo=python&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Data-Yahoo%20Finance-6001D2?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square"/>
</p>

<p align="center">
  <b>AI-powered stock analysis · Global markets · Next-day forecasting · Solar Gold theme</b>
</p>

---

## Overview

StockSense AI is a full-stack financial analytics web app built with Python and Streamlit. It combines real-time market data, machine learning price forecasting, and interactive visualisations into a single elegant interface — styled in a custom Solar Gold dark theme.

Designed as an educational tool for anyone interested in understanding stock market trends, technical indicators, and AI-driven predictions.

---

## Features

| Category | Details |
|---|---|
| 🔐 **User Session** | Sign-in with name & email, personalised greeting, session-based watchlist |
| 📊 **10 Chart Tabs** | Candlestick · Predictions · Moving Averages · Volume · RSI · Bollinger Bands · Compare · Portfolio · Sectors · News |
| ⚖️ **Stock Comparison** | Side-by-side AI predictions, normalised 2-year performance chart, return correlation |
| 🌍 **Global Markets** | US · India (NSE) · UK (LSE) · Germany (XETRA) · Japan · China · Brazil · Global ETFs |
| 📈 **10 Indices** | S&P 500 · NASDAQ · DOW · NIFTY 50 · FTSE 100 · DAX · Nikkei · Hang Seng · CAC 40 · VIX |
| ⭐ **Watchlist** | Add any ticker, live price tracking, one-click load |
| 🔮 **AI Forecasting** | 14-feature ML pipeline — Linear Regression or Random Forest |
| ⚡ **Performance** | Batched API calls, full model pipeline cached via `st.cache_data` |
| 📰 **News Feed** | Latest headlines per ticker, two-column layout, always visible |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit + custom CSS (Solar Gold theme) |
| Charts | Plotly |
| Market Data | yfinance (Yahoo Finance) |
| Machine Learning | scikit-learn — LinearRegression, RandomForestRegressor |
| Data Processing | pandas, numpy |
| Timezone handling | pytz |

---

## Machine Learning Model

- **Features (14):** MA5, MA20, MA50, EMA12, EMA26, MACD, Daily Return, Lag1, Lag2, Lag5, 20-day Volatility, Volume Ratio, RSI (14), Bollinger %B
- **Target:** Next trading day's closing price
- **Train / Test split:** 80% / 20% chronological holdout
- **Metrics reported:** MAE, RMSE, average % error

> **Disclaimer:** All predictions are back-tests on historical data. This is not financial advice.

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/stocksense-ai.git
cd stocksense-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run stocksense_final.py
```

Opens at **http://localhost:8501**

---

## Deploy to Streamlit Cloud (free)

1. Fork or push this repo to your GitHub account
2. Visit **https://share.streamlit.io** and sign in with GitHub
3. Click **New app**
4. Set: repository → `stocksense-ai` · branch → `main` · file → `stocksense_final.py`
5. Click **Deploy** — your public URL is live in ~2 minutes

---

## Project Structure

```
stocksense-ai/
├── stocksense_final.py   # Main application
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## Data & Accuracy Notes

- **Data source:** Yahoo Finance via `yfinance` — free, no API key required
- **US equities:** ~15 minutes delayed during market hours
- **International / NSE / LSE:** End-of-day data
- **Model accuracy:** Reported MAE and RMSE are in-sample back-test figures on the 20% holdout set — not forward-looking guarantees

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

<p align="center">Built with ☀️ · Powered by Python & Streamlit</p>
