# ☀️ StockSense AI — Solar Gold

> AI-powered stock analysis · Global markets · Next-day forecasting · Solar Gold theme

![Python](https://img.shields.io/badge/Python-3.10+-FFB800?style=flat&logo=python&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- 🔐 **User sign-in** — personalised greeting, session-based watchlist
- 📊 **10 analysis tabs** — Candlestick, Predictions, MA, Volume, RSI, Bollinger Bands, Compare, Portfolio, Sectors, News
- ⚖️ **Stock comparison** — side-by-side AI predictions, normalised chart, return correlation
- 🌍 **Global markets** — US, India (NSE), UK, Germany, Japan, China, Brazil, ETFs
- 📈 **10 market indices** — S&P 500, NASDAQ, NIFTY 50, FTSE 100, DAX, Nikkei, Hang Seng, CAC 40, DOW, VIX
- ⭐ **Watchlist** — add any ticker, live prices, load instantly
- 🔮 **AI forecast** — 14-feature ML model (Linear Regression or Random Forest)
- ⚡ **Fast** — batched data fetching, full pipeline cached with `st.cache_data`

## Disclaimer

> StockSense AI is for **educational purposes only**. Predictions are ML back-tests on historical data — not financial advice. Data via Yahoo Finance (yfinance). US equities ~15 min delayed; most others end-of-day.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stocksense-ai.git
cd stocksense-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run stocksense_final.py
```

App opens at **http://localhost:8501**

---

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub (see below)
2. Go to **https://share.streamlit.io**
3. Click **New app**
4. Select your repo · branch: `main` · file: `stocksense_final.py`
5. Click **Deploy** — live URL in ~2 minutes

---

## Project Structure

```
stocksense-ai/
├── stocksense_final.py   # Main app
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | yfinance (Yahoo Finance) |
| Charts | Plotly |
| ML | scikit-learn (LinearRegression / RandomForest) |
| Data processing | pandas, numpy |
