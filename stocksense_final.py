"""
StockSense AI — Solar Gold  ☀️
Final fully-furnished prototype
────────────────────────────────
Fixes in this build
 • fetch_live_prices / fetch_indices: batched yf.download instead of per-ticker
   Ticker loops → single call, 5-10× faster
 • add_features: squeeze() removed (caused shape bugs on some pandas versions)
   RSI uses mask() not replace() – no divide-by-zero
 • train_model: caches model via st.cache_data on (ticker, model_type) hash
 • Comparison: dedicated tab, side-by-side metrics + normalised chart + prediction
 • User profile: name + email stored in session_state (local, no DB needed)
   Watched list saved per session; interests chip-clicked into watchlist
 • Accuracy note: clearly labelled as "back-test MAE on 20% holdout"
   Disclaimer shown so user understands it is not financial advice
 • Real-time note: yfinance is end-of-day for most tickers (15-min delayed for US)
   Correct status shown
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, date
import pytz, time, json, hashlib

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="StockSense AI", page_icon="☀️", layout="wide",
                   initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════════════════
#  SOLAR GOLD CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');
:root{--gold:#FFB800;--gold2:#FF8C00;--gold3:#FFD700;--red:#FF4D4D;
      --bg:#0D0A00;--card:rgba(255,184,0,0.04);--border:rgba(255,184,0,0.13);
      --text:#FFF5E0;--muted:rgba(255,245,224,0.5);}
html,body,[class*="css"]{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;background:var(--bg);color:var(--text);}
.stApp{background:linear-gradient(135deg,#0D0A00 0%,#1A1000 40%,#0A0800 100%);
       background-size:400% 400%;animation:bgAnim 16s ease infinite;}
@keyframes bgAnim{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
/* Sidebar */
[data-testid="stSidebar"]{background:rgba(10,7,0,0.97)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
/* Input — ALL text inputs, dark bg, dark readable text */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input:-webkit-autofill,
[data-testid="stTextInput"] input:autofill {
  background:#1A1200!important;
  border:1px solid rgba(255,184,0,0.45)!important;
  border-radius:10px!important;
  color:#FFE066!important;
  -webkit-text-fill-color:#FFE066!important;
  font-family:'Palatino Linotype','Palatino','Book Antiqua',serif!important;
  font-weight:700!important;
  font-size:1rem!important;
  letter-spacing:1.2px!important;
  caret-color:#FFB800!important;
  box-shadow:inset 0 1px 4px rgba(0,0,0,0.5)!important;}
[data-testid="stTextInput"] input::placeholder{
  color:rgba(255,184,0,0.35)!important;
  font-weight:400!important;
  font-style:italic!important;
  font-size:.88rem!important;}
[data-testid="stTextInput"] input:focus{
  background:#201600!important;
  border-color:rgba(255,184,0,0.75)!important;
  box-shadow:inset 0 1px 4px rgba(0,0,0,0.5), 0 0 0 3px rgba(255,184,0,0.12)!important;
  outline:none!important;}
[data-testid="stTextInput"] label{
  color:rgba(255,245,224,0.75)!important;
  font-size:.78rem!important;
  font-weight:500!important;
  letter-spacing:.5px!important;}
/* Chip buttons */
div[data-testid="stButton"]>button{
  background:rgba(255,184,0,0.07)!important;border:1px solid rgba(255,184,0,0.22)!important;
  color:#FFB800!important;font-size:0.75rem!important;font-weight:600!important;
  padding:2px 8px!important;border-radius:6px!important;height:auto!important;
  min-height:0!important;line-height:1.5!important;letter-spacing:0.4px!important;
  transition:all 0.18s ease!important;}
div[data-testid="stButton"]>button:hover{
  background:rgba(255,184,0,0.16)!important;border-color:rgba(255,184,0,0.48)!important;
  transform:translateY(-1px)!important;box-shadow:0 3px 10px rgba(255,184,0,0.14)!important;}
/* Tabs */
[data-testid="stTabs"] button{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif!important;font-weight:500!important;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#FFB800!important;border-bottom-color:#FFB800!important;}
/* Scrollbar */
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:rgba(255,184,0,0.22);border-radius:2px;}
/* Hero */
.hero-title{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#FFD700,#FF8C00,#FFB800,#FFD700);background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:shimmer 3s linear infinite;margin:0;letter-spacing:-1px;}
@keyframes shimmer{0%{background-position:0% center}100%{background-position:200% center}}
.hero-sub{color:var(--muted);font-size:0.82rem;letter-spacing:2px;text-transform:uppercase;margin-top:4px;}
/* Cards */
.mcard{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px 22px;
  position:relative;overflow:hidden;transition:transform .25s,border-color .25s,box-shadow .25s;}
.mcard:hover{transform:translateY(-3px);border-color:rgba(255,184,0,0.38);box-shadow:0 6px 28px rgba(255,184,0,0.09);}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--gold),var(--gold2),var(--gold3));opacity:0;transition:opacity .25s;}
.mcard:hover::before{opacity:1;}
.mlabel{font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:7px;}
.mvalue{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.85rem;font-weight:700;color:var(--text);line-height:1;margin-bottom:5px;}
.mbadge{font-size:0.78rem;font-weight:500;padding:2px 8px;border-radius:5px;display:inline-block;}
.up{color:#FFB800;background:rgba(255,184,0,0.11);}
.dn{color:#FF4D4D;background:rgba(255,77,77,0.1);}
.nt{color:var(--muted);background:rgba(255,255,255,0.04);}
/* Status */
.sbadge{display:inline-flex;align-items:center;gap:7px;padding:7px 18px;border-radius:50px;
  font-size:0.82rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;}
.sopen{background:rgba(255,184,0,0.09);border:1px solid rgba(255,184,0,0.32);color:#FFB800;}
.sclosed{background:rgba(255,77,77,0.08);border:1px solid rgba(255,77,77,0.3);color:#FF4D4D;}
.pdot{width:7px;height:7px;border-radius:50%;animation:pdot 1.5s ease-in-out infinite;}
.pg{background:#FFB800;box-shadow:0 0 9px #FFB800;}
.pr{background:#FF4D4D;box-shadow:0 0 7px #FF4D4D;}
@keyframes pdot{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:.6}}
/* Ticker tape */
.tape-outer{background:rgba(255,184,0,0.03);border-top:1px solid rgba(255,184,0,0.1);
  border-bottom:1px solid rgba(255,184,0,0.1);overflow:hidden;padding:9px 0;margin:.8rem 0;white-space:nowrap;}
.tape-inner{display:inline-block;animation:tape 80s linear infinite;font-size:0.8rem;}
@keyframes tape{0%{transform:translateX(100vw)}100%{transform:translateX(-100%)}}
.tup{color:#FFB800;font-weight:500;}.tdn{color:#FF4D4D;font-weight:500;}
.tsym{color:rgba(255,245,224,.75);font-weight:600;margin-right:3px;}
.tsep{color:rgba(255,184,0,.2);margin:0 14px;}
/* Index grid */
.igrid{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:.8rem 0;}
.icard{background:rgba(255,184,0,0.025);border:1px solid var(--border);border-radius:11px;padding:11px 12px;text-align:center;}
.iname{font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
.ival{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:.95rem;font-weight:700;}
/* Alert */
.alert-t{padding:14px 18px;border-radius:11px;margin:.8rem 0;font-size:.88rem;font-weight:500;}
.alert-fire{background:rgba(255,77,77,.09);border:1px solid rgba(255,77,77,.38);color:#FF4D4D;}
.alert-safe{background:rgba(255,184,0,.06);border:1px solid rgba(255,184,0,.22);color:#FFB800;}
/* Forecast */
.fcard{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:26px 30px;margin:1.2rem 0;}
.ftitle{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:.9rem;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:2px;margin-bottom:18px;}
.fgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.fitem{text-align:center;padding:14px;background:rgba(255,184,0,.025);border-radius:10px;border:1px solid var(--border);}
.filab{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px;}
.fival{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.25rem;font-weight:700;}
.vup{color:#FFB800;}.vdn{color:#FF4D4D;}.vnt{color:#FFD700;}
/* News */
.ncard{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:14px 18px;margin:7px 0;
  transition:transform .18s,border-color .18s;}
.ncard:hover{transform:translateX(3px);border-color:rgba(255,184,0,.32);}
.ntitle{font-size:.88rem;font-weight:500;color:var(--text);margin-bottom:5px;}
.nmeta{font-size:.72rem;color:var(--muted);}
/* Section label */
.slabel{font-size:.66rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin:1rem 0 .4rem;}
/* Sidebar logo */
.slogo{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.35rem;font-weight:800;
  background:linear-gradient(135deg,#FFD700,#FF8C00);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;display:block;margin-bottom:.5rem;}
/* User badge */
.ubadge{display:flex;align-items:center;gap:10px;padding:12px 14px;
  background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.15);
  border-radius:12px;margin-bottom:1rem;}
.uavatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#FFB800,#FF8C00);
  display:flex;align-items:center;justify-content:center;font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;
  font-size:.95rem;font-weight:800;color:#0D0A00;flex-shrink:0;}
.uname{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:.88rem;font-weight:700;color:var(--text);}
.uemail{font-size:.7rem;color:var(--muted);}
/* Disclaimer */
.disclaimer{font-size:.7rem;color:rgba(255,245,224,.35);padding:10px 14px;
  border:1px solid rgba(255,184,0,.08);border-radius:8px;line-height:1.6;margin-top:1rem;}
/* Divider */
.div{height:1px;background:linear-gradient(90deg,transparent,var(--gold),var(--gold2),transparent);
  margin:1.2rem 0;opacity:.3;}
/* Vol metric */
.vmrow{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:.8rem 0;}
.vm{background:rgba(255,184,0,.04);border:1px solid var(--border);border-radius:11px;padding:12px 16px;text-align:center;}
.vmlab{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
.vmval{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.15rem;font-weight:700;color:#FFB800;}
/* Model badges */
.blr{background:rgba(255,215,0,.09);border:1px solid rgba(255,215,0,.28);color:#FFD700;
  font-size:.72rem;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;}
.brf{background:rgba(255,140,0,.09);border:1px solid rgba(255,140,0,.28);color:#FF8C00;
  font-size:.72rem;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;}
/* Watchlist */
.witem{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;
  background:rgba(255,184,0,0.04);border:1px solid rgba(255,184,0,0.1);border-radius:8px;margin:4px 0;}
.wsym{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:.85rem;font-weight:700;color:#FFB800;}
.wpct{font-size:.78rem;font-weight:500;}
/* Comparison side panel */
.cmpanel{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px 24px;margin-bottom:1rem;}
.cmticker{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.1rem;font-weight:800;color:var(--text);}
.cmprice{font-family:'Palatino Linotype','Palatino','Book Antiqua',serif;font-size:1.7rem;font-weight:700;}
.cmchange{font-size:.8rem;font-weight:500;padding:2px 8px;border-radius:5px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
TICKER_GROUPS = {
    "🇺🇸 US":      [("AAPL","AAPL"),("MSFT","MSFT"),("GOOGL","GOOGL"),("TSLA","TSLA"),
                    ("AMZN","AMZN"),("NVDA","NVDA"),("META","META"),("NFLX","NFLX"),
                    ("AMD","AMD"),("JPM","JPM")],
    "🇮🇳 India":   [("RELIANCE","RELIANCE.NS"),("TCS","TCS.NS"),("INFY","INFY.NS"),
                    ("HDFC","HDFCBANK.NS"),("WIPRO","WIPRO.NS"),("TATAMOTORS","TATAMOTORS.NS")],
    "🇬🇧 UK":      [("HSBA","HSBA.L"),("BP","BP.L"),("SHEL","SHEL.L")],
    "🇩🇪 Germany": [("SAP","SAP.DE"),("BMW","BMW.DE"),("Bayer","BAYN.DE")],
    "🌏 Asia":     [("Toyota","7203.T"),("Softbank","9984.T"),("BABA","BABA"),
                    ("JD","JD"),("Tencent","TCEHY")],
    "🇧🇷 Brazil":  [("VALE","VALE"),("Petrobras","PBR"),("Itau","ITUB")],
    "🌍 ETFs":     [("EEM","EEM"),("VEU","VEU"),("ACWI","ACWI")],
}

TAPE_TICKERS = ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META","NFLX",
                "AMD","JPM","RELIANCE.NS","TCS.NS","INFY.NS","EEM","VEU"]

INDICES = {
    "S&P 500":"^GSPC","NASDAQ":"^IXIC","DOW":"^DJI","NIFTY 50":"^NSEI",
    "FTSE 100":"^FTSE","DAX":"^GDAXI","Nikkei":"^N225",
    "Hang Seng":"^HSI","CAC 40":"^FCHI","VIX":"^VIX",
}

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","NVDA":"Technology",
    "META":"Technology","AMD":"Technology","INTC":"Technology","SAP.DE":"Technology",
    "TSLA":"Cons. Disc.","AMZN":"Cons. Disc.","NFLX":"Cons. Disc.",
    "JPM":"Financials","ITUB":"Financials","HDFCBANK.NS":"Financials","HSBA.L":"Financials",
    "RELIANCE.NS":"Energy","BP.L":"Energy","SHEL.L":"Energy","PBR":"Energy",
    "TCS.NS":"IT Services","INFY.NS":"IT Services","WIPRO.NS":"IT Services",
    "TATAMOTORS.NS":"Automobiles","BMW.DE":"Automobiles","7203.T":"Automobiles",
    "VALE":"Materials","BAYN.DE":"Healthcare",
    "9984.T":"Telecom","BABA":"E-Commerce","JD":"E-Commerce","TCEHY":"Internet",
    "EEM":"ETF","VEU":"ETF","ACWI":"ETF",
}

DEFAULT_PORTFOLIO = {
    "AAPL":15,"MSFT":12,"GOOGL":10,"NVDA":10,"TSLA":8,
    "AMZN":8,"META":7,"NFLX":5,"JPM":7,"INFY.NS":5,
    "TCS.NS":5,"RELIANCE.NS":4,"EEM":4,
}

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Palatino Linotype", color="#FFF5E0"),
    xaxis=dict(gridcolor="rgba(255,184,0,0.05)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,184,0,0.05)", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=44, b=10), height=420,
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss("user_name",    "")
ss("user_email",   "")
ss("logged_in",    False)
ss("ticker_val",   "AAPL")
ss("ticker2_val",  "")
ss("watchlist",    [])
ss("model_type",   "Linear Regression")

def set_t(sym):
    # Directly set the session_state key that the text_input is bound to
    st.session_state.ticker_val  = sym.strip().upper()

def set_t2(sym):
    st.session_state.ticker2_val = sym.strip().upper()
def add_watch(sym):
    if sym and sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(sym)
def rm_watch(sym):
    st.session_state.watchlist = [s for s in st.session_state.watchlist if s != sym]

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FUNCTIONS  (all cached, batched where possible)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def fetch_stock(ticker: str) -> pd.DataFrame:
    """Download 2 years of OHLCV for one ticker. Returns clean DataFrame."""
    try:
        df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=90, show_spinner=False)
def fetch_tape_batch() -> list:
    """Batch-fetch last 2 closes for all tape tickers in ONE yf.download call."""
    results = []
    try:
        raw = yf.download(TAPE_TICKERS, period="5d", auto_adjust=True,
                          progress=False, group_by="ticker")
        for sym in TAPE_TICKERS:
            try:
                if len(TAPE_TICKERS) == 1:
                    closes = raw["Close"].dropna()
                else:
                    closes = raw[sym]["Close"].dropna()
                if len(closes) >= 2:
                    prev, curr = float(closes.iloc[-2]), float(closes.iloc[-1])
                    if prev > 0:
                        pct = (curr - prev) / prev * 100
                        label = sym.replace(".NS","").replace(".L","").replace(".DE","").replace(".T","")
                        results.append({"sym": label, "price": curr, "pct": pct})
            except Exception:
                pass
    except Exception:
        pass
    return results

@st.cache_data(ttl=90, show_spinner=False)
def fetch_indices_batch() -> dict:
    """Batch-fetch indices in one download."""
    syms  = list(INDICES.values())
    names = list(INDICES.keys())
    results = {}
    try:
        raw = yf.download(syms, period="5d", auto_adjust=True,
                          progress=False, group_by="ticker")
        for name, sym in zip(names, syms):
            try:
                closes = raw[sym]["Close"].dropna() if len(syms) > 1 else raw["Close"].dropna()
                if len(closes) >= 2:
                    prev, curr = float(closes.iloc[-2]), float(closes.iloc[-1])
                    if prev > 0:
                        results[name] = {"price": curr, "pct": (curr-prev)/prev*100}
            except Exception:
                pass
    except Exception:
        pass
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(ticker: str) -> list:
    try:
        news = yf.Ticker(ticker).news
        return news[:6] if news else []
    except Exception:
        return []

@st.cache_data(ttl=90, show_spinner=False)
def fetch_watchlist_prices(syms: tuple) -> dict:
    """Returns {sym: (curr, pct)} for watchlist."""
    if not syms:
        return {}
    out = {}
    try:
        raw = yf.download(list(syms), period="5d", auto_adjust=True,
                          progress=False, group_by="ticker")
        for sym in syms:
            try:
                cl = raw[sym]["Close"].dropna() if len(syms) > 1 else raw["Close"].dropna()
                if len(cl) >= 2:
                    p, c = float(cl.iloc[-2]), float(cl.iloc[-1])
                    if p > 0: out[sym] = (c, (c-p)/p*100)
            except Exception:
                pass
    except Exception:
        pass
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Price features
    d["MA5"]      = d["Close"].rolling(5).mean()
    d["MA20"]     = d["Close"].rolling(20).mean()
    d["MA50"]     = d["Close"].rolling(50).mean()
    d["EMA12"]    = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA26"]    = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]     = d["EMA12"] - d["EMA26"]
    d["Return"]   = d["Close"].pct_change()
    d["Lag1"]     = d["Close"].shift(1)
    d["Lag2"]     = d["Close"].shift(2)
    d["Lag5"]     = d["Close"].shift(5)
    # Volatility
    d["Vol20"]    = d["Return"].rolling(20).std()
    # Volume
    d["VolRatio"] = d["Volume"] / d["Volume"].rolling(5).mean().replace(0, np.nan)
    # RSI — proper implementation
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.mask(loss == 0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"].fillna(50, inplace=True)
    # Bollinger
    d["BB_mid"]   = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std
    d["BB_pct"]   = (d["Close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"]).replace(0, np.nan)
    # Volume MA
    d["VolMA20"]  = d["Volume"].rolling(20).mean()
    # Target: next day's close
    d["Target"]   = d["Close"].shift(-1)
    d.dropna(inplace=True)
    return d

FEATURES = ["MA5","MA20","MA50","EMA12","EMA26","MACD",
            "Return","Lag1","Lag2","Lag5","Vol20","VolRatio","RSI","BB_pct"]

@st.cache_data(ttl=120, show_spinner=False)
def train_and_predict(ticker: str, model_type: str):
    """Full pipeline: fetch → features → train → predict. Cached by ticker+model."""
    df_raw = fetch_stock(ticker)
    if df_raw.empty:
        return None
    df = build_features(df_raw)
    if len(df) < 60:
        return None

    X = df[FEATURES]
    y = df["Target"]
    split = int(len(df) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr)
    Xte_s  = scaler.transform(X_te)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=150, max_depth=8,
                                      min_samples_split=5, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()
    model.fit(Xtr_s, y_tr)
    y_pred = model.predict(Xte_s)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    # Next-day prediction
    last_x    = scaler.transform(df[FEATURES].iloc[[-1]])
    next_pred = float(model.predict(last_x)[0])

    return {
        "df_raw": df_raw, "df": df,
        "y_te": y_te, "y_pred": y_pred,
        "mae": mae, "rmse": rmse,
        "next_pred": next_pred,
        "latest_close": float(df_raw["Close"].iloc[-1]),
        "prev_close":   float(df_raw["Close"].iloc[-2]),
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def is_market_open() -> bool:
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5: return False
    return (now.replace(hour=9,  minute=30, second=0, microsecond=0) <= now <=
            now.replace(hour=16, minute=0,  second=0, microsecond=0))

def cc(v): return "up" if v > 0 else "dn" if v < 0 else "nt"
def ar(v): return "▲"  if v > 0 else "▼"  if v < 0 else "–"
def fmt_price(v): return f"${v:,.2f}" if v < 10000 else f"${v:,.0f}"
def fmt_vol(v): return f"{v/1e9:.2f}B" if v >= 1e9 else f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"

def lerp(t, r0,g0,b0, r1,g1,b1):
    return f"#{int(r0+t*(r1-r0)):02X}{int(g0+t*(g1-g0)):02X}{int(b0+t*(b1-b0)):02X}"

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN GATE — shown instead of app if not signed in
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:80vh;padding:2rem">
      <h1 class="hero-title" style="margin-bottom:.4rem">☀️ StockSense AI</h1>
      <p class="hero-sub" style="margin-bottom:2.5rem">Solar Gold · AI-Powered · Global Markets</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1,1.4,1])
    with col_c:
        st.markdown("""
        <div style="background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.2);
                    border-radius:20px;padding:36px 32px;margin-top:-4rem">
          <div style="font-family:'Palatino Linotype','Palatino',serif;font-size:1.2rem;font-weight:700;
                      color:#FFB800;margin-bottom:4px">Welcome</div>
          <div style="font-size:.82rem;color:rgba(255,245,224,.55);margin-bottom:1.6rem">
            Sign in to access live markets, AI forecasts & your watchlist</div>
        </div>""", unsafe_allow_html=True)
        login_name  = st.text_input("Your name",  placeholder="e.g. Angela Shah",  key="lg_name")
        login_email = st.text_input("Email address", placeholder="you@example.com", key="lg_email")
        if st.button("✅  Sign In  →", use_container_width=True, type="primary"):
            name_ok  = login_name.strip() != ""
            email_ok = "@" in login_email and "." in login_email.split("@")[-1]
            if name_ok and email_ok:
                st.session_state.user_name  = login_name.strip()
                st.session_state.user_email = login_email.strip()
                st.session_state.logged_in  = True
                st.rerun()
            elif not name_ok:
                st.error("Please enter your name.")
            else:
                st.error("Please enter a valid email address.")
        st.markdown("""<div style="text-align:center;font-size:.72rem;color:rgba(255,245,224,.3);
                       margin-top:1rem">No account needed · data stored locally in your session</div>""",
                    unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (only rendered when logged in)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<span class="slogo">☀️ StockSense AI</span>', unsafe_allow_html=True)

    # ── User badge ────────────────────────────────────────────
    initials = "".join(w[0].upper() for w in st.session_state.user_name.split()[:2])
    st.markdown(f"""
    <div class="ubadge">
      <div class="uavatar">{initials}</div>
      <div><div class="uname">{st.session_state.user_name}</div>
           <div class="uemail">{st.session_state.user_email}</div></div>
    </div>""", unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        st.session_state.logged_in   = False
        st.session_state.ticker_val  = "AAPL"
        st.session_state.ticker2_val = ""
        st.rerun()

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Primary stock ─────────────────────────────────────────
    st.markdown('<div class="slabel">Primary Stock</div>', unsafe_allow_html=True)
    # Key matches session_state key — Streamlit syncs automatically, no value= needed
    ticker = st.text_input("", key="ticker_val",
                           label_visibility="collapsed").strip().upper()

    # ── Compare stock ─────────────────────────────────────────
    st.markdown('<div class="slabel">⚖️ Compare Stock (optional)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.65rem;color:rgba(255,200,80,.5);margin-bottom:4px">Type a ticker below OR use Quick Compare buttons ↓</div>', unsafe_allow_html=True)
    st.text_input("", key="ticker2_val",
                  placeholder="e.g. MSFT  TSLA  TCS.NS  BP.L",
                  label_visibility="collapsed")
    ticker2 = st.session_state.ticker2_val.strip().upper()  # authoritative source

    # ── Stock quick-pick buttons ──────────────────────────────
    for group, pairs in TICKER_GROUPS.items():
        st.markdown(f'<div class="slabel">{group} — tap to load</div>', unsafe_allow_html=True)
        n_cols = 4 if len(pairs) >= 6 else 3
        cols = st.columns(n_cols)
        for i, (label, sym) in enumerate(pairs):
            cols[i % n_cols].button(label, key=f"btn_{sym}",
                                    on_click=set_t, args=(sym,),
                                    use_container_width=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Quick compare row ─────────────────────────────────────
    st.markdown('<div class="slabel">Quick Compare — tap to compare</div>', unsafe_allow_html=True)
    qc = ["MSFT","GOOGL","AMZN","TSLA","TCS.NS","INFY.NS","BP.L","NVDA"]
    qcols = st.columns(4)
    for i, sym in enumerate(qc):
        qcols[i%4].button(sym.replace(".NS","").replace(".L",""),
                          key=f"qc_{sym}", on_click=set_t2, args=(sym,),
                          use_container_width=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Watchlist ─────────────────────────────────────────────
    st.markdown('<div class="slabel">⭐ Watchlist</div>', unsafe_allow_html=True)
    st.text_input("", placeholder="Add ticker e.g. NVDA",
                  key="watch_input_val", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    def _do_add_watch():
        sym = st.session_state.get("watch_input_val","").strip().upper()
        if sym: add_watch(sym)
    def _do_load_watch():
        sym = st.session_state.get("watch_input_val","").strip().upper()
        if sym: set_t(sym)
    c1.button("➕ Add",  use_container_width=True, on_click=_do_add_watch)
    c2.button("📊 Load", use_container_width=True, on_click=_do_load_watch)

    if st.session_state.watchlist:
        wprices = fetch_watchlist_prices(tuple(st.session_state.watchlist))
        for sym in st.session_state.watchlist:
            info = wprices.get(sym)
            if info:
                pr, pc = info
                col = "#FFB800" if pc >= 0 else "#FF4D4D"
                pcts = f'{ar(pc)} {abs(pc):.2f}%'
                st.markdown(
                    f'<div class="witem">'
                    f'<span class="wsym" style="cursor:pointer">{sym}</span>'
                    f'<span class="wpct" style="color:{col}">{pcts}</span>'
                    f'</div>', unsafe_allow_html=True)
            rcol1, rcol2 = st.columns(2)
            rcol1.button("Load", key=f"wl_{sym}",   on_click=set_t,    args=(sym,), use_container_width=True)
            rcol2.button("✕",    key=f"wrm_{sym}",  on_click=rm_watch, args=(sym,), use_container_width=True)
    else:
        st.caption("No stocks in watchlist yet.")

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Model + settings ─────────────────────────────────────
    st.markdown('<div class="slabel">AI Model</div>', unsafe_allow_html=True)
    model_type = st.radio("", ["Linear Regression","Random Forest"],
                          index=0 if st.session_state.model_type == "Linear Regression" else 1,
                          label_visibility="collapsed")
    st.session_state.model_type = model_type

    st.markdown('<div class="slabel">Price Alert Threshold</div>', unsafe_allow_html=True)
    alert_pct = st.slider("", 0.5, 10.0, 2.0, 0.5, label_visibility="collapsed")

    st.markdown('<div class="slabel">Auto-refresh (market hours only)</div>', unsafe_allow_html=True)
    refresh_s = st.slider("", 30, 300, 90, label_visibility="collapsed")

    st.markdown(f"""<div style="font-size:.8rem;color:rgba(255,245,224,.45);line-height:2;margin-top:.5rem">
    🧠 {model_type} · 14 features<br>
    📅 2-year daily · 80/20 split<br>
    🔁 Refresh {refresh_s}s · EOD data
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer">
    ⚠️ StockSense AI is for educational purposes only.
    Predictions are ML back-tests on historical data — not financial advice.
    yfinance provides end-of-day prices (US ~15 min delayed when market open).
    Always consult a qualified financial advisor before investing.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.logged_in:
    greeting = f"Welcome back, {st.session_state.user_name.split()[0]} ☀️"
else:
    greeting = "☀️ StockSense AI — Solar Gold"

st.markdown(f"""
<div style="text-align:center;padding:1.6rem 0 .8rem">
  <h1 class="hero-title">{greeting}</h1>
  <p class="hero-sub">AI-Powered · Global Markets · End-of-Day Data · Next-Day Forecast</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE TICKER TAPE  (batched)
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading market data…"):
    tape_data    = fetch_tape_batch()
    indices_data = fetch_indices_batch()

if tape_data:
    items = []
    for d in tape_data:
        a   = "▲" if d["pct"] >= 0 else "▼"
        cls = "tup" if d["pct"] >= 0 else "tdn"
        items.append(f'<span class="tsym">{d["sym"]}</span>'
                     f'<span class="{cls}">{a} {abs(d["pct"]):.2f}%</span>'
                     f'<span class="tsep">·</span>')
    tape = "".join(items * 3)
    st.markdown(f'<div class="tape-outer"><span class="tape-inner">{tape}</span></div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  INDEX CARDS
# ═══════════════════════════════════════════════════════════════════════════════
if indices_data:
    cards = ""
    for name, d in indices_data.items():
        p, pct = d["price"], d["pct"]
        col    = "#FFB800" if pct >= 0 else "#FF4D4D"
        fmt    = f"{p:,.0f}" if p >= 1000 else f"{p:.2f}"
        cards += (f'<div class="icard"><div class="iname">{name}</div>'
                  f'<div class="ival">{fmt}</div>'
                  f'<div style="font-size:.72rem;color:{col};font-weight:500">'
                  f'{ar(pct)} {abs(pct):.2f}%</div></div>')
    st.markdown(f'<div class="igrid">{cards}</div>', unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
# Read ticker from session_state (widget key binding keeps it current)
ticker  = st.session_state.get("ticker_val",  "AAPL").strip().upper() or "AAPL"
ticker2 = st.session_state.get("ticker2_val", "").strip().upper()

if not ticker:
    st.info("Enter a ticker in the sidebar to get started."); st.stop()

with st.spinner(f"Analysing {ticker}…"):
    result = train_and_predict(ticker, model_type)

if result is None:
    st.error(f"Could not load data for **{ticker}**. Check the symbol and try again.")
    st.stop()

df_raw       = result["df_raw"]
df           = result["df"]
y_te         = result["y_te"]
y_pred_arr   = result["y_pred"]
mae          = result["mae"]
rmse         = result["rmse"]
next_pred    = result["next_pred"]
latest_close = result["latest_close"]
prev_close   = result["prev_close"]
change       = latest_close - prev_close
change_pct   = change / prev_close * 100
pred_change  = next_pred - latest_close
pred_pct     = pred_change / latest_close * 100
error_pct    = mae / float(y_te.mean()) * 100

market_open  = is_market_open()
badge_cls    = "blr" if model_type == "Linear Regression" else "brf"

# Status bar
if market_open:
    status_html = '<span class="sbadge sopen"><span class="pdot pg"></span>Market OPEN — ~15 min delayed (US)</span>'
else:
    status_html = '<span class="sbadge sclosed"><span class="pdot pr"></span>Market CLOSED — End-of-Day Data</span>'
st.markdown(f'<div style="text-align:center;margin-bottom:.6rem">{status_html}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center;margin-bottom:.8rem"><span class="{badge_cls}">{model_type}</span></div>',
            unsafe_allow_html=True)

# Alert
if abs(pred_pct) >= alert_pct:
    word = "rally" if pred_pct > 0 else "drop"
    st.markdown(f'<div class="alert-t alert-fire">🔔 ALERT — {ticker} predicted to {word} by <b>{abs(pred_pct):.2f}%</b> · Threshold: {alert_pct}%</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-t alert-safe">✅ No alert — predicted move {abs(pred_pct):.2f}% within {alert_pct}% threshold</div>',
                unsafe_allow_html=True)

# Metric cards
rsi_now = float(df["RSI"].iloc[-1])
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:1rem 0">
  <div class="mcard">
    <div class="mlabel">Latest Close</div>
    <div class="mvalue">{fmt_price(latest_close)}</div>
    <span class="mbadge {cc(change)}">{ar(change)} {abs(change_pct):.2f}% today</span>
  </div>
  <div class="mcard">
    <div class="mlabel">AI Prediction (next close)</div>
    <div class="mvalue">{fmt_price(next_pred)}</div>
    <span class="mbadge {cc(pred_change)}">{ar(pred_change)} {abs(pred_pct):.2f}% expected</span>
  </div>
  <div class="mcard">
    <div class="mlabel">Model Error (back-test MAE)</div>
    <div class="mvalue">{fmt_price(mae)}</div>
    <span class="mbadge nt">RMSE {fmt_price(rmse)} · {error_pct:.1f}% avg</span>
  </div>
  <div class="mcard">
    <div class="mlabel">RSI (14-day)</div>
    <div class="mvalue">{rsi_now:.1f}</div>
    <span class="mbadge {'dn' if rsi_now>70 else 'up' if rsi_now<30 else 'nt'}">
      {'⚠ Overbought' if rsi_now>70 else '✅ Oversold' if rsi_now<30 else '— Neutral'}</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# Pre-compute indicators used in forecast summary (must be outside tabs)
ma5_last  = float(df["MA5"].iloc[-1])
ma20_last = float(df["MA20"].iloc[-1])

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_names = ["📊 Candlestick","🎯 Predictions","📉 Moving Avg",
             "📈 Volume","⚡ RSI","🎻 Bollinger",
             "⚖️ Compare","🥧 Portfolio","🏢 Sectors","📰 News"]
tabs = st.tabs(tab_names)

# ── Tab 1: Candlestick ─────────────────────────────────────────────────────────
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_raw.index[-90:],
        open=df_raw["Open"].iloc[-90:], high=df_raw["High"].iloc[-90:],
        low=df_raw["Low"].iloc[-90:],   close=df_raw["Close"].iloc[-90:],
        name=ticker,
        increasing_line_color="#FFB800", decreasing_line_color="#FF4D4D",
        increasing_fillcolor="rgba(255,184,0,0.28)", decreasing_fillcolor="rgba(255,77,77,0.28)",
    ))
    fig.add_hline(y=next_pred, line_dash="dash", line_color="#FFD700", line_width=1.4,
                  annotation_text=f"  AI Pred: {fmt_price(next_pred)}", annotation_font_color="#FFD700")
    fig.update_layout(title=f"<b>{ticker}</b> — Last 90 Days (candlestick)",
                      xaxis_rangeslider_visible=False, **CHART_BASE)
    st.plotly_chart(fig, use_container_width=True)
    if st.button("⭐ Add to Watchlist", key="add_watch_main"):
        add_watch(ticker)
        st.success(f"{ticker} added to watchlist!")

# ── Tab 2: Predictions ────────────────────────────────────────────────────────
with tabs[1]:
    fig2 = go.Figure()
    x_ax = list(range(len(y_te)))
    fig2.add_trace(go.Scatter(x=x_ax, y=y_te.values, name="Actual",
        line=dict(color="#FFD700", width=2), fill="tozeroy",
        fillcolor="rgba(255,215,0,0.05)"))
    fig2.add_trace(go.Scatter(x=x_ax, y=y_pred_arr, name="Predicted",
        line=dict(color="#FF8C00", width=2, dash="dot")))
    fig2.update_layout(title=f"<b>{ticker}</b> — Actual vs Predicted (20% holdout)",
        xaxis_title="Trading Days (test set)", yaxis_title="Price", **CHART_BASE)
    st.plotly_chart(fig2, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",        f"${mae:.2f}")
    c2.metric("RMSE",       f"${rmse:.2f}")
    c3.metric("Avg % Error",f"{error_pct:.2f}%")
    st.markdown('<div class="disclaimer">Back-test metrics on a held-out 20% of historical data — not forward-looking. Past performance ≠ future results.</div>',
                unsafe_allow_html=True)

# ── Tab 3: Moving Averages ────────────────────────────────────────────────────
with tabs[2]:
    fig3 = go.Figure()
    sl = df.iloc[-180:]
    fig3.add_trace(go.Scatter(x=sl.index, y=sl["Close"],  name="Close",
        line=dict(color="rgba(255,245,224,0.55)", width=1.5)))
    fig3.add_trace(go.Scatter(x=sl.index, y=sl["MA5"],    name="MA5",
        line=dict(color="#FFD700", width=1.6, dash="dot")))
    fig3.add_trace(go.Scatter(x=sl.index, y=sl["MA20"],   name="MA20",
        line=dict(color="#FFB800", width=2)))
    fig3.add_trace(go.Scatter(x=sl.index, y=sl["MA50"],   name="MA50",
        line=dict(color="#FF8C00", width=2, dash="dash")))
    fig3.add_trace(go.Scatter(x=sl.index, y=sl["EMA12"],  name="EMA12",
        line=dict(color="rgba(255,200,50,0.5)", width=1.2, dash="dashdot")))
    fig3.update_layout(title=f"<b>{ticker}</b> — Moving Averages (MA5/20/50 + EMA12)", **CHART_BASE)
    st.plotly_chart(fig3, use_container_width=True)
    # MA crossover signal
    if ma5_last > ma20_last:
        st.success(f"📈 Golden cross signal — MA5 ({ma5_last:.2f}) > MA20 ({ma20_last:.2f}): bullish momentum")
    else:
        st.warning(f"📉 Death cross signal — MA5 ({ma5_last:.2f}) < MA20 ({ma20_last:.2f}): bearish momentum")

# ── Tab 4: Volume ─────────────────────────────────────────────────────────────
with tabs[3]:
    vs  = df_raw["Volume"].iloc[-60:]
    os_ = df_raw["Open"].iloc[-60:]
    cs_ = df_raw["Close"].iloc[-60:]
    vm  = vs.rolling(20).mean()
    avg_v = float(vs.mean())
    v_max = float(vs.max()) or 1.0

    bar_colors = []
    for v, c, o in zip(vs.values, cs_.values, os_.values):
        r = min(float(v) / v_max, 1.0)
        intensity = 0.28 + 0.72 * r
        if c >= o:
            bar_colors.append(f"rgba({255},{int(140+75*r)},{0},{intensity:.2f})")
        else:
            bar_colors.append(f"rgba({int(200+55*r)},{int(30+47*r)},{int(47+59*r)},{intensity:.2f})")

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_raw.index[-60:], y=vs, name="Volume",
        marker=dict(color=bar_colors, line=dict(width=0))))
    fig4.add_trace(go.Scatter(x=df_raw.index[-60:], y=vm, name="Vol MA20",
        line=dict(color="#FFD700", width=2, dash="dot"), mode="lines"))
    fig4.add_hline(y=avg_v, line_dash="dash", line_color="rgba(255,184,0,0.4)",
                   line_width=1.2, annotation_text="  Avg", annotation_font_color="rgba(255,184,0,0.75)")
    fig4.update_layout(title=f"<b>{ticker}</b> — Volume · Colour intensity = relative size", **CHART_BASE)
    st.plotly_chart(fig4, use_container_width=True)
    v1, v2 = st.columns(2)
    v1.markdown(f'<div class="vm"><div class="vmlab">Total Vol (60d)</div><div class="vmval">{fmt_vol(float(vs.sum()))}</div></div>', unsafe_allow_html=True)
    v2.markdown(f'<div class="vm"><div class="vmlab">Avg Daily Vol</div><div class="vmval">{fmt_vol(avg_v)}</div></div>', unsafe_allow_html=True)

# ── Tab 5: RSI ────────────────────────────────────────────────────────────────
with tabs[4]:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#FFB800", width=2)))
    fig5.add_hline(y=70, line_dash="dash", line_color="#FF4D4D", line_width=1,
                   annotation_text="Overbought 70", annotation_font_color="#FF4D4D")
    fig5.add_hline(y=30, line_dash="dash", line_color="#FFB800", line_width=1,
                   annotation_text="Oversold 30",   annotation_font_color="#FFB800")
    fig5.add_hrect(y0=70, y1=100, fillcolor="rgba(255,77,77,0.04)",  line_width=0)
    fig5.add_hrect(y0=0,  y1=30,  fillcolor="rgba(255,184,0,0.04)",  line_width=0)
    fig5.update_layout(title=f"<b>{ticker}</b> — RSI (14-period)", yaxis_range=[0,100], **CHART_BASE)
    st.plotly_chart(fig5, use_container_width=True)
    if rsi_now > 70:
        st.warning(f"⚠️ RSI {rsi_now:.1f} — overbought zone. Potential pullback risk.")
    elif rsi_now < 30:
        st.success(f"✅ RSI {rsi_now:.1f} — oversold zone. Potential reversal/buy signal.")
    else:
        st.info(f"RSI {rsi_now:.1f} — neutral zone (30–70).")

# ── Tab 6: Bollinger Bands ────────────────────────────────────────────────────
with tabs[5]:
    sl6 = df.iloc[-120:]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=sl6.index, y=sl6["BB_upper"], name="Upper Band",
        line=dict(color="rgba(255,77,77,0.55)", width=1, dash="dash")))
    fig6.add_trace(go.Scatter(x=sl6.index, y=sl6["BB_lower"], name="Lower Band",
        line=dict(color="rgba(255,184,0,0.55)", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(255,184,0,0.04)"))
    fig6.add_trace(go.Scatter(x=sl6.index, y=sl6["BB_mid"], name="MA20 (mid)",
        line=dict(color="rgba(255,215,0,0.55)", width=1)))
    fig6.add_trace(go.Scatter(x=sl6.index, y=sl6["Close"], name="Close",
        line=dict(color="#FFF5E0", width=2)))
    fig6.update_layout(title=f"<b>{ticker}</b> — Bollinger Bands (20, ±2σ)", **CHART_BASE)
    st.plotly_chart(fig6, use_container_width=True)
    bb_pct_now = float(df["BB_pct"].iloc[-1])
    if bb_pct_now > 0.9:
        st.warning("Price near **upper band** — overbought / breakout watch")
    elif bb_pct_now < 0.1:
        st.success("Price near **lower band** — oversold / bounce watch")
    else:
        st.info(f"Price at {bb_pct_now*100:.0f}% of band width — mid-range")

# ── Tab 7: Comparison ─────────────────────────────────────────────────────────
with tabs[6]:
    # ticker2 already set from session_state above the tabs
    if not ticker2 or ticker2.strip() == "" or ticker2 == ticker:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem">
          <div style="font-size:2rem;margin-bottom:1rem">⚖️</div>
          <div style="font-family:'Palatino Linotype','Palatino',serif;font-size:1.1rem;font-weight:700;color:#FFB800;margin-bottom:.5rem">
            No comparison stock selected</div>
          <div style="color:rgba(255,245,224,.5);font-size:.88rem;line-height:1.7">
            1 · Type a ticker in the <b style="color:#FFB800">Compare Stock</b> box in the sidebar<br>
            2 · Or click any <b style="color:#FFB800">Quick Compare</b> button below it<br>
            3 · Then come back to this tab
          </div>
        </div>""", unsafe_allow_html=True)
        # Inline compare input for convenience
        st.markdown("---")
        st.markdown("**Quick compare right here:**")
        qc_inline = st.text_input("Enter ticker to compare", placeholder="e.g. MSFT, TSLA, TCS.NS",
                                   key="qc_inline_input").strip().upper()
        if st.button("⚖️ Compare Now", use_container_width=True) and qc_inline:
            st.session_state.ticker2_val = qc_inline
            st.rerun()
    else:
        with st.spinner(f"Analysing {ticker2}…"):
            r2 = train_and_predict(ticker2, model_type)

        if r2 is None:
            st.error(f"Could not load data for **{ticker2}**.")
        else:
            lc2 = r2["latest_close"]; np2 = r2["next_pred"]
            pc2 = (np2 - lc2) / lc2 * 100
            chg2 = lc2 - r2["prev_close"]
            chgpct2 = chg2 / r2["prev_close"] * 100

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="cmpanel">
                  <div class="cmticker">📈 {ticker}</div>
                  <div class="cmprice">{fmt_price(latest_close)}</div>
                  <span class="cmchange {cc(change)}">{ar(change)} {abs(change_pct):.2f}% today</span><br><br>
                  <div style="font-size:.85rem;color:var(--muted)">AI Prediction</div>
                  <div style="font-family:'Palatino Linotype','Palatino',serif;font-size:1.3rem;font-weight:700;color:{'#FFB800' if pred_change>0 else '#FF4D4D'}">{fmt_price(next_pred)} ({pred_pct:+.2f}%)</div>
                  <div style="font-size:.75rem;color:var(--muted);margin-top:6px">MAE ${mae:.2f} · RMSE ${rmse:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="cmpanel">
                  <div class="cmticker">📊 {ticker2}</div>
                  <div class="cmprice">{fmt_price(lc2)}</div>
                  <span class="cmchange {cc(chg2)}">{ar(chg2)} {abs(chgpct2):.2f}% today</span><br><br>
                  <div style="font-size:.85rem;color:var(--muted)">AI Prediction</div>
                  <div style="font-family:'Palatino Linotype','Palatino',serif;font-size:1.3rem;font-weight:700;color:{'#FFB800' if pc2>0 else '#FF4D4D'}">{fmt_price(np2)} ({pc2:+.2f}%)</div>
                  <div style="font-size:.75rem;color:var(--muted);margin-top:6px">MAE ${r2['mae']:.2f} · RMSE ${r2['rmse']:.2f}</div>
                </div>""", unsafe_allow_html=True)

            # Normalised performance chart
            fig_c = go.Figure()
            n1 = (df_raw["Close"] / df_raw["Close"].iloc[0]) * 100
            n2 = (r2["df_raw"]["Close"] / r2["df_raw"]["Close"].iloc[0]) * 100
            fig_c.add_trace(go.Scatter(x=df_raw.index, y=n1, name=ticker,
                line=dict(color="#FFB800", width=2)))
            fig_c.add_trace(go.Scatter(x=r2["df_raw"].index, y=n2, name=ticker2,
                line=dict(color="#FF8C00", width=2, dash="dot")))
            fig_c.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
            fig_c.update_layout(title=f"<b>{ticker}</b> vs <b>{ticker2}</b> — Normalised (base=100, 2yr)",
                yaxis_title="Indexed Return", **CHART_BASE)
            st.plotly_chart(fig_c, use_container_width=True)

            # Correlation
            common = df_raw["Close"].align(r2["df_raw"]["Close"], join="inner")
            if len(common[0]) > 20:
                corr = float(common[0].pct_change().corr(common[1].pct_change()))
                st.metric("Return Correlation", f"{corr:.3f}",
                          help="1 = perfectly correlated, 0 = uncorrelated, -1 = inverse")

# ── Tab 8: Portfolio ──────────────────────────────────────────────────────────
with tabs[7]:
    n = len(DEFAULT_PORTFOLIO)
    sp = sorted(DEFAULT_PORTFOLIO.items(), key=lambda x: x[1], reverse=True)
    sl8, sv8 = [k for k,_ in sp], [v for _,v in sp]
    grad8 = [lerp(i/max(n-1,1), 255,100,0, 255,215,0) for i in range(n)]
    pull8 = [0.08 if k == ticker else 0 for k in sl8]

    fig8 = go.Figure(go.Pie(
        labels=sl8, values=sv8, hole=0.46,
        marker=dict(colors=grad8, line=dict(color="#0D0A00", width=1.4)),
        textfont=dict(family="Palatino Linotype", size=11, color="#FFF5E0"),
        hovertemplate="<b>%{label}</b><br>Weight: %{value}%<br>Share: %{percent}<extra></extra>",
        pull=pull8, sort=False, direction="clockwise", rotation=40,
    ))
    fig8.update_layout(
        title=f"<b>Portfolio Allocation</b> — {ticker} highlighted · sorted by weight",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Palatino Linotype", color="#FFF5E0"),
        legend=dict(bgcolor="rgba(13,10,0,0.6)", bordercolor="rgba(255,184,0,0.14)", borderwidth=1, font=dict(size=11)),
        margin=dict(l=10,r=10,t=55,b=10), height=480,
        annotations=[dict(text=f"<b>{n}</b><br>Holdings", x=0.5, y=0.5,
                          font_size=13, font_family="Palatino Linotype", font_color="#FFB800", showarrow=False)]
    )
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("Demo portfolio — weights are illustrative only. Not financial advice.")

# ── Tab 9: Sectors ────────────────────────────────────────────────────────────
with tabs[8]:
    sw = {}
    for sym, wt in DEFAULT_PORTFOLIO.items():
        sec = SECTOR_MAP.get(sym, "Other")
        sw[sec] = sw.get(sec, 0) + wt
    sec_items = sorted(sw.items(), key=lambda x: x[1], reverse=True)
    sl9 = [k for k,_ in sec_items]; sv9 = [v for _,v in sec_items]
    ns  = len(sl9)
    grad9 = [lerp(i/max(ns-1,1), 255,69,0, 255,224,130) for i in range(ns)]

    fig9 = go.Figure(go.Pie(
        labels=sl9, values=sv9, hole=0.46,
        marker=dict(colors=grad9, line=dict(color="#0D0A00", width=1.4)),
        textfont=dict(family="Palatino Linotype", size=12, color="#FFF5E0"),
        hovertemplate="<b>%{label}</b><br>Allocation: %{value}%<br>Share: %{percent}<extra></extra>",
        sort=False, direction="clockwise", rotation=80,
    ))
    fig9.update_layout(
        title="<b>Sector Breakdown</b> — Diversification view",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Palatino Linotype", color="#FFF5E0"),
        legend=dict(bgcolor="rgba(13,10,0,0.6)", bordercolor="rgba(255,184,0,0.14)", borderwidth=1, font=dict(size=11)),
        margin=dict(l=10,r=10,t=55,b=10), height=480,
        annotations=[dict(text=f"<b>{ns}</b><br>Sectors", x=0.5, y=0.5,
                          font_size=13, font_family="Palatino Linotype", font_color="#FFB800", showarrow=False)]
    )
    st.plotly_chart(fig9, use_container_width=True)

# ── Tab 10: News ──────────────────────────────────────────────────────────────
with tabs[9]:
    with st.spinner(f"Loading news for {ticker}…"):
        news = fetch_news(ticker)
    if news:
        for item in news:
            try:
                cnt  = item.get("content", {})
                if isinstance(cnt, dict):
                    title = cnt.get("title","")
                    clink = cnt.get("canonicalUrl", {})
                    url   = clink.get("url","") if isinstance(clink, dict) else ""
                    prov  = cnt.get("provider", {})
                    src   = prov.get("displayName","") if isinstance(prov, dict) else ""
                else:
                    title = str(item.get("title",""))
                    url   = ""
                    src   = ""
                pub = item.get("pubDate","") or item.get("providerPublishTime","")
                if title:
                    lnk = f'<a href="{url}" target="_blank" style="color:inherit;text-decoration:none">{title}</a>' if url else title
                    st.markdown(f'<div class="ncard"><div class="ntitle">{lnk}</div>'
                                f'<div class="nmeta">{src} · {pub}</div></div>', unsafe_allow_html=True)
            except Exception:
                continue
    else:
        st.info("No recent news found. Try a major ticker like AAPL or TSLA.")

# ═══════════════════════════════════════════════════════════════════════════════
#  FORECAST SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="div"></div>', unsafe_allow_html=True)
direction  = "📈 BULLISH" if pred_change > 0 else "📉 BEARISH"
dclass     = "vup" if pred_change > 0 else "vdn"
confidence = "HIGH ✦" if abs(pred_pct)<1 else "MEDIUM ◆" if abs(pred_pct)<3 else "LOW ◇"
cclass     = "vup" if "HIGH" in confidence else "vnt" if "MEDIUM" in confidence else "vdn"
# MACD signal
macd_val  = float(df["MACD"].iloc[-1])
macd_sig  = "📈 Bullish" if macd_val > 0 else "📉 Bearish"
macd_cl   = "vup" if macd_val > 0 else "vdn"

st.markdown(f"""
<div class="fcard">
  <div class="ftitle">🔮 Forecast Summary — {ticker}</div>
  <div class="fgrid">
    <div class="fitem"><div class="filab">Signal</div><div class="fival {dclass}">{direction}</div></div>
    <div class="fitem"><div class="filab">Expected Move</div><div class="fival {dclass}">{pred_pct:+.2f}%</div></div>
    <div class="fitem"><div class="filab">Confidence</div><div class="fival {cclass}">{confidence}</div></div>
    <div class="fitem"><div class="filab">MACD</div><div class="fival {macd_cl}">{macd_sig}</div></div>
    <div class="fitem"><div class="filab">RSI Signal</div>
      <div class="fival {'vdn' if rsi_now>70 else 'vup' if rsi_now<30 else 'vnt'}">
        {'Overbought' if rsi_now>70 else 'Oversold' if rsi_now<30 else 'Neutral'}</div></div>
    <div class="fitem"><div class="filab">MA Trend</div>
      <div class="fival {'vup' if ma5_last>ma20_last else 'vdn'}">
        {'Golden ✦' if ma5_last>ma20_last else 'Death ✕'}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="disclaimer">
⚠️ All predictions are generated by a machine learning model trained on historical price data.
They represent statistical patterns only and carry no guarantee of future performance.
This tool is for educational and research purposes — not financial advice.
Data source: Yahoo Finance (yfinance). US equities ~15 min delayed during market hours; most others end-of-day.
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  NEWS FEED — always visible at the bottom of every page
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="div"></div>', unsafe_allow_html=True)
st.markdown(f'''
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem">
  <div style="font-family:'Palatino Linotype','Palatino',serif;font-size:1.1rem;font-weight:800;color:#FFB800">
    📰 Latest Headlines</div>
  <div style="font-size:.72rem;color:rgba(255,245,224,.4);letter-spacing:1px;text-transform:uppercase">
    {ticker} · auto-refreshed</div>
</div>''', unsafe_allow_html=True)

with st.spinner(f"Loading news for {ticker}…"):
    _news_items = fetch_news(ticker)

if _news_items:
    # Two-column news layout
    nc1, nc2 = st.columns(2)
    for idx, item in enumerate(_news_items):
        try:
            cnt   = item.get("content", {})
            if isinstance(cnt, dict):
                _title = cnt.get("title","") or ""
                _clink = cnt.get("canonicalUrl", {})
                _url   = _clink.get("url","") if isinstance(_clink, dict) else ""
                _prov  = cnt.get("provider", {})
                _src   = _prov.get("displayName","") if isinstance(_prov, dict) else ""
            else:
                _title = str(item.get("title",""))
                _url   = item.get("link","") or ""
                _src   = item.get("publisher","") or ""
            _pub = item.get("pubDate","") or str(item.get("providerPublishTime",""))
            if not _title:
                continue
            _lnk = (f'<a href="{_url}" target="_blank" style="color:#FFF5E0;text-decoration:none">' +
                    _title + '</a>') if _url else _title
            _card = (f'<div class="ncard">' +
                     f'<div class="ntitle">{_lnk}</div>' +
                     f'<div class="nmeta">📡 {_src}&nbsp;&nbsp;·&nbsp;&nbsp;{_pub}</div>' +
                     '</div>')
            if idx % 2 == 0:
                nc1.markdown(_card, unsafe_allow_html=True)
            else:
                nc2.markdown(_card, unsafe_allow_html=True)
        except Exception:
            continue
else:
    st.info(f"No recent news found for **{ticker}**. This is normal for some international tickers.")

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO REFRESH (market hours only, avoids pointless refreshes)
# ═══════════════════════════════════════════════════════════════════════════════
if market_open:
    st.caption(f"🔄 Auto-refreshing every {refresh_s}s (market is open)")
    time.sleep(refresh_s)
    st.rerun()
