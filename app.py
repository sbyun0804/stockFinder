# app.py
# -------------------------------------------------------------
# Fairvalue Scout (FMP + Yahoo fallback)
# Prices/profiles/peers/news from FMP; TTM financials use FMP when
# available, else Yahoo Finance fallback. Handles negative earnings (no P/E).
# UI: Streamlit (+ optional AgGrid) | Storage: local SQLite watchlist
# -------------------------------------------------------------

import os
import io
import json
import math
import time
import datetime as dt
import altair as alt
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor, as_completed


# Optional hover/selection UI for peers
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False

# -------------------------
# ---- CONFIG & SETUP -----
# -------------------------
st.set_page_config(page_title="Fairvalue Scout", layout="wide")

CONFIG_PATH = ".uvsf_config.json"

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg: Dict[str, Any]):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save config: {e}")

cfg = load_config()

api_key = st.secrets.get("FMP_API_KEY")


if "watchlist_engine" not in st.session_state:
    st.session_state.watchlist_engine = create_engine("sqlite:///watchlist.db", echo=False)

# Create table if not exists
with st.session_state.watchlist_engine.begin() as conn:
    conn.exec_driver_sql(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at TEXT,
            added_price REAL,
            note TEXT
        );
        """
    )

# -------------------------
# ---- HELPER UTILS -------
# -------------------------

def fmt_intlike(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return "—"

def fmt_float3(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    try:
        return f"{float(v):,.3f}"
    except Exception:
        return "—"

@st.cache_data(show_spinner=False)
def fmp_get(url: str, params: Dict[str, Any]) -> Any:
    """GET with retries for transient errors."""
    last_exc = None
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code in (429, 502, 503):
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(1.0 * (i + 1))
    raise last_exc if last_exc else RuntimeError("fmp_get failed")

# -------------------------
# ---- DATA PROVIDERS -----
# -------------------------
FMP_BASE = "https://financialmodelingprep.com/stable"

@st.cache_data(show_spinner=False)
def get_quote_fmp(symbol: str) -> Optional[float]:
    try:
        data = fmp_get(f"{FMP_BASE}/quote", {"symbol": symbol.upper(), "apikey": api_key})
        if isinstance(data, list) and data:
            return data[0].get("price") or data[0].get("previousClose")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def get_profile_fmp(symbol: str) -> Dict[str, Any]:
    # FMP first
    item = {}
    try:
        data = fmp_get(f"{FMP_BASE}/profile", {"symbol": symbol.upper(), "apikey": api_key})
        item = data[0] if isinstance(data, list) and data else {}
    except Exception:
        item = {}

    prof = {
        "ticker": symbol.upper(),
        "name": item.get("companyName") or item.get("companyNameLong") or item.get("symbol"),
        "market_cap": item.get("mktCap"),
        "share_class_shares_outstanding": item.get("sharesOutstanding"),
        "homepage_url": item.get("website"),
        "description": item.get("description"),
        "exchange": item.get("exchange"),
        "industry": item.get("industry"),
        "sector": item.get("sector"),
        "country": item.get("country"),
    }

    # Yahoo fast-info fallback if market cap / shares missing
    if (prof["market_cap"] is None) or (prof["share_class_shares_outstanding"] is None):
        try:
            t = yf.Ticker(symbol)
            fi = getattr(t, "fast_info", {}) or {}
            mc = fi.get("market_cap") or fi.get("marketCap")
            sh = fi.get("shares") or fi.get("shares_outstanding") or fi.get("sharesOutstanding")
            prof["market_cap"] = prof["market_cap"] or (float(mc) if mc is not None else None)
            prof["share_class_shares_outstanding"] = prof["share_class_shares_outstanding"] or (float(sh) if sh is not None else None)
        except Exception:
            pass

    return prof

@st.cache_data(show_spinner=True)
def get_price_light(symbol: str, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "apikey": api_key}
    if date_from: params["from"] = date_from
    if date_to:   params["to"]   = date_to
    data = fmp_get(f"{FMP_BASE}/historical-price-eod/light", params)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    return df

@st.cache_data(show_spinner=True)
def get_dividends(symbol: str, limit: int = 100) -> pd.DataFrame:
    data = fmp_get(f"{FMP_BASE}/dividends", {"symbol": symbol.upper(), "limit": limit, "apikey": api_key})
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    # make naive datetimes for safe comparisons
    for c in ["date", "recordDate", "paymentDate", "declarationDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert(None)
    return df

@st.cache_data(show_spinner=True)
def get_earnings(symbol: str, limit: int = 100) -> pd.DataFrame:
    data = fmp_get(f"{FMP_BASE}/earnings", {"symbol": symbol.upper(), "limit": limit, "apikey": api_key})
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    return df

# ---------- Yahoo Finance TTM (robust fallback) ----------
@st.cache_data(show_spinner=True)
def get_ttm_financials_yf(symbol: str) -> Dict[str, Any]:
    """
    Robust TTM from Yahoo Finance quarterly statements.
    """
    t = yf.Ticker(symbol)

    def _sum_rows(df, names, n=4):
        if df is None or df.empty: return None
        for nm in names:
            try:
                s = df.loc[nm].dropna().head(n)
                if not s.empty:
                    return float(s.sum())
            except Exception:
                continue
        return None

    def _point(df, names):
        if df is None or df.empty: return None
        for nm in names:
            try:
                v = df.loc[nm].dropna()
                if not v.empty:
                    return float(v.iloc[0])
            except Exception:
                continue
        return None

    try:
        q_is = t.quarterly_financials
        q_bs = t.quarterly_balance_sheet

        rev = _sum_rows(q_is, ["Total Revenue", "TotalRevenue", "Revenue"])
        net = _sum_rows(q_is, ["Net Income", "NetIncome"])
        cogs = _sum_rows(q_is, ["Cost Of Revenue", "CostOfRevenue"])

        assets = _point(q_bs, ["Total Assets", "TotalAssets"])
        liab   = _point(q_bs, ["Total Liab", "TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"])
        equity = _point(q_bs, ["Total Stockholder Equity", "TotalEquityGrossMinorityInterest", "TotalStockholdersEquity"])
        sld    = _point(q_bs, ["Short Long Term Debt", "ShortLongTermDebt", "ShortTermDebt"])
        ltd    = _point(q_bs, ["Long Term Debt", "LongTermDebt"])
        total_debt = ((sld or 0.0) + (ltd or 0.0)) if (sld is not None or ltd is not None) else None

        expenses = (rev - net) if (rev is not None and net is not None) else None

        return {
            "revenues_ttm": rev,
            "cogs_ttm": cogs,
            "net_income_ttm": net,
            "total_assets": assets,
            "total_liabilities": liab,
            "total_equity": equity,
            "total_debt": total_debt,
            "expenses_ttm": expenses,
            "source_quarters": 4,
        }
    except Exception:
        return {k: None for k in [
            "revenues_ttm","cogs_ttm","net_income_ttm","total_assets","total_liabilities",
            "total_equity","total_debt","expenses_ttm","source_quarters"
        ]}

# ---------- Unified TTM: FMP first, then Yahoo fallback ----------
@st.cache_data(show_spinner=True)
def get_ttm_financials(symbol: str) -> Dict[str, Any]:
    """
    Try FMP TTM endpoints; on 401/402/403/404 or any error, fall back to Yahoo Finance TTM.
    """
    try:
        inc = fmp_get(f"{FMP_BASE}/income-statement-ttm", {"symbol": symbol.upper(), "apikey": api_key})
        bal = fmp_get(f"{FMP_BASE}/balance-sheet-statement-ttm", {"symbol": symbol.upper(), "apikey": api_key})
        inc0 = inc[0] if isinstance(inc, list) and inc else {}
        bal0 = bal[0] if isinstance(bal, list) and bal else {}

        revenues = inc0.get("revenue")
        net_income = inc0.get("netIncome") or inc0.get("bottomLineNetIncome")
        total_assets = bal0.get("totalAssets")
        total_liabilities = bal0.get("totalLiabilities")
        total_debt = bal0.get("totalDebt") or ((bal0.get("shortTermDebt") or 0) + (bal0.get("longTermDebt") or 0))
        total_equity = bal0.get("totalStockholdersEquity") or bal0.get("totalEquity")

        expenses = inc0.get("costAndExpenses")
        if expenses is None:
            cogs = inc0.get("costOfRevenue") or 0
            opex = inc0.get("operatingExpenses") or 0
            expenses = (cogs or 0) + (opex or 0)

        # Equity fallback if missing
        if (total_equity in (None, 0)) and (total_assets is not None and total_debt is not None):
            try:
                total_equity = float(total_assets) - float(total_debt)
            except Exception:
                pass

        return {
            "revenues_ttm": revenues,
            "cogs_ttm": inc0.get("costOfRevenue"),
            "net_income_ttm": net_income,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "total_equity": total_equity,
            "total_debt": total_debt,
            "expenses_ttm": expenses,
            "source_quarters": None,
        }
    except Exception:
        return get_ttm_financials_yf(symbol)

# ---------- FMP quarterly rev/exp for chart ----------
@st.cache_data(show_spinner=True)
def get_income_quarterly_fmp(symbol: str,limit: int = 20, shares_out: float | None = None) -> pd.DataFrame:
    """Quarterly Revenue/Expenses/NetIncome/EPS from FMP. EPS falls back to netIncome/shares_out if needed."""
    try:
        data = fmp_get(
            f"{FMP_BASE}/income-statement",
            {"symbol": symbol.upper(), "period": "quarter", "limit": limit, "apikey": api_key},
        )
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").tail(limit)

        # Revenue
        rev = df.set_index("date")["revenue"] if "revenue" in df.columns else pd.Series(dtype=float)
        # Expenses: prefer explicit, else COGS + OpEx
        if "costAndExpenses" in df.columns:
            exp = df.set_index("date")["costAndExpenses"]
        else:
            cogs = df.set_index("date")["costOfRevenue"] if "costOfRevenue" in df.columns else 0
            opex = df.set_index("date")["operatingExpenses"] if "operatingExpenses" in df.columns else 0
            exp = pd.Series(cogs, index=rev.index).add(pd.Series(opex, index=rev.index), fill_value=0)

        # Net Income
        ni = df.set_index("date")["netIncome"] if "netIncome" in df.columns else pd.Series(dtype=float)

        # EPS: from payload if present; else compute approx via shares_out
        if "eps" in df.columns:
            eps = df.set_index("date")["eps"]
        else:
            eps = (ni / float(shares_out)) if (shares_out and not pd.isna(shares_out)) else pd.Series(dtype=float)

        out = pd.DataFrame({"Revenue": rev, "Expenses": exp})
        if not ni.empty:
            out["Net Income"] = ni
        if isinstance(eps, pd.Series) and not eps.empty:
            out["EPS"] = eps
        return out
    except Exception:
        return pd.DataFrame()

# --------- FMP peers ----------
@st.cache_data(show_spinner=True)
def get_peers_fmp(symbol: str) -> List[str]:
    try:
        url = f"{FMP_BASE}/stock-peers"
        r = requests.get(url, params={"symbol": symbol.upper(), "apikey": api_key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "peersList" in data:
            return [s.upper() for s in (data.get("peersList") or [])]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "peersList" in data[0]:
                return [s.upper() for s in data[0].get("peersList", [])]
            if all("symbol" in x for x in data):
                return [x["symbol"].upper() for x in data]
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=True)
def get_peers_fmp_bulk(symbol: str) -> List[str]:
    try:
        url = f"{FMP_BASE}/peers-bulk"
        r = requests.get(url, params={"apikey": api_key}, timeout=30)
        r.raise_for_status()
        import io as _io
        df = pd.read_csv(_io.StringIO(r.text))
        row = df.loc[df["symbol"].str.upper() == symbol.upper()]
        if not row.empty:
            peers_str = str(row.iloc[0].get("peers", ""))
            return [p.strip().upper() for p in peers_str.split(",") if p.strip()]
    except Exception:
        pass
    return []

# -------------------------
# ---- METRICS & LOGIC ----
# -------------------------
def compute_multiples_row(
    ticker: str,
    price: Optional[float],
    shares_out: Optional[float],
    ttm: Dict[str, Any],
) -> Dict[str, Any]:
    mc = price * shares_out if (price is not None and shares_out) else None

    revenues = ttm.get("revenues_ttm") if ttm else None
    net_inc = ttm.get("net_income_ttm") if ttm else None
    equity = ttm.get("total_equity") if ttm else None
    assets = ttm.get("total_assets") if ttm else None
    debt = ttm.get("total_debt") if ttm else None

    # Equity fallback: Assets − Total Debt (guarded)
    if equity is None or (isinstance(equity, float) and math.isnan(equity)):
        try:
            if (
                assets is not None and not (isinstance(assets, float) and math.isnan(assets))
                and debt is not None and not (isinstance(debt, float) and math.isnan(debt))
            ):
                equity = float(assets) - float(debt)
        except Exception:
            pass

    # EPS (allow negative) and P/E (only when EPS > 0)
    eps_ttm = (net_inc / shares_out) if (net_inc is not None and shares_out and isinstance(net_inc, (int, float))) else None
    pe = (price / eps_ttm) if (price is not None and eps_ttm and eps_ttm > 0) else None
    if net_inc is not None and shares_out and isinstance(net_inc, (int, float)) and net_inc > 0:
        eps_ttm = net_inc / shares_out
        if price is not None and eps_ttm != 0:
            pe = price / eps_ttm

    # PSR and P/B
    ps = mc / revenues if (mc is not None and revenues and revenues > 0) else None
    pb = mc / equity if (mc is not None and equity and equity > 0) else None

    # Debt/Equity
    d_e = None
    try:
        if debt is not None and equity and float(equity) != 0:
            d_e = float(debt) / float(equity)
    except Exception:
        d_e = None

    return {
        "Ticker": ticker.upper(),
        "Price": price,
        "Shares Out": shares_out,
        "Market Cap": mc,
        "Revenue (TTM)": revenues,
        "Net Income (TTM)": net_inc,
        "COGS (TTM)": ttm.get("cogs_ttm") if ttm else None,
        "Total Debt": debt,
        "Total Equity": equity,
        "P/E": pe,
        "PSR": ps,
        "P/B": pb,
        "Debt/Equity": d_e,
        "EPS (TTM)": eps_ttm,
    }

def what_if_price_from_multiple(row: Dict[str, Any], multiple: float, kind: str) -> Optional[float]:
    shares = row.get("Shares Out")
    if kind == "P/E":
        eps = row.get("EPS (TTM)")
        if eps is None or eps <= 0:
            return None
        return multiple * eps
    if kind == "PSR":
        rev = row.get("Revenue (TTM)")
        if shares and rev and rev > 0:
            mc_target = multiple * rev
            return mc_target / shares
    return None

# -------------------------
# ---- UI: SIDEBAR --------
# -------------------------
st.sidebar.title("Settings")

# fmp_key = st.sidebar.text_input("FMP API Key", type="password", value=cfg.get("fmp_key", ""))

st.sidebar.markdown("---")
base_ticker = st.sidebar.text_input("Base Ticker", value=cfg.get("base_ticker", "UNH")).upper().strip()
# Reset peer-related state when the base ticker changes so suggestions/selection refresh
if base_ticker != st.session_state.get("_last_base_ticker"):
    st.session_state["_last_base_ticker"] = base_ticker
    for k in ["peer_set", "_last_peer_list", "peer_rows", "_last_comp_sel", "peer_rows_comp"]:
        if k in st.session_state:
            del st.session_state[k]
cap_band_pct = st.sidebar.slider("Peer Market Cap Band (±%)", 10, 200, int(cfg.get("cap_band_pct", 50)))
st.sidebar.caption("Peers can be filtered to companies within ± this % of the base company's market cap.")
allow_international = st.sidebar.checkbox("Allow international peers if no US comps", value=bool(cfg.get("allow_international", False)))

st.sidebar.markdown("---")
peer_input = st.sidebar.text_area(
    "Peer tickers (comma-separated, optional)",
    value=cfg.get("peer_input", ""),
    help="Add/remove peers here or use the suggestions list below."
)

st.sidebar.markdown("---")
metric_choices = [
    "Market Cap", "Revenue (TTM)", "Net Income (TTM)", "COGS (TTM)",
    "Total Debt", "Total Equity", "P/E", "PSR", "P/B", "Debt/Equity", "EPS (TTM)",
]
selected_metrics = st.sidebar.multiselect(
    "Columns to show",
    metric_choices,
    default=cfg.get("selected_metrics", [
        "Market Cap", "Revenue (TTM)", "Net Income (TTM)", "COGS (TTM)",
        "P/E", "PSR", "P/B", "Debt/Equity", "EPS (TTM)",
    ]),
)

if st.sidebar.button("💾 Save settings locally"):
    save_config({
        "base_ticker": base_ticker,
        "cap_band_pct": cap_band_pct,
        "allow_international": allow_international,
        "peer_input": peer_input,
        "selected_metrics": selected_metrics,
    })
    st.sidebar.success("Saved to .uvsf_config.json (stored locally in this folder).")
# -------------------------

# ---- MAIN: HEADER --------
# -------------------------
st.title("📉 Fairvalue Scout")
st.caption("Prices/profiles/peers/news from FMP; TTM financials use FMP when available, else Yahoo Finance fallback. Handles negative earnings (no P/E).")
st.info(
    "This app was created by **Sehoon Byun**. "
    "If you want to use it, please reach me at "
    "[sbyun0804@gmail.com](mailto:sbyun0804@gmail.com)."
)
# if not fmp_key:
#     st.info("Enter your FMP API key in the sidebar to begin.")
#     st.stop()

# -------------------------
# ---- STEP 1: BASE STOCK --
# -------------------------
colA, colB = st.columns([2, 3])
with colA:
    st.subheader(f"Base: {base_ticker}")
    try:
        profile = get_profile_fmp(base_ticker)
    except Exception as e:
        st.error(f"Failed to load profile for {base_ticker}: {e}")
        st.stop()

    price = get_quote_fmp(base_ticker)

    # Backfill Market Cap & Shares Outstanding (Yahoo fast_info)
    market_cap = profile.get("market_cap")
    shares_out = profile.get("share_class_shares_outstanding")

    if (market_cap is None) or (shares_out is None):
        try:
            t = yf.Ticker(base_ticker)
            fi = getattr(t, "fast_info", {}) or {}
            mc = fi.get("market_cap") or fi.get("marketCap")
            sh = fi.get("shares") or fi.get("shares_outstanding") or fi.get("sharesOutstanding")
            market_cap = market_cap or (float(mc) if mc is not None else None)
            shares_out = shares_out or (float(sh) if sh is not None else None)
        except Exception:
            pass

    # If still missing shares but we have price & market cap, compute
    if (shares_out is None) and (market_cap is not None) and (price is not None):
        try:
            shares_out = float(market_cap) / float(price)
        except Exception:
            pass

    # TTM (FMP -> Yahoo fallback)
    ttm = get_ttm_financials(base_ticker)
    base_row = compute_multiples_row(base_ticker, price, shares_out, ttm)

    st.metric("Last Price", f"{base_row['Price'] if base_row['Price'] is not None else 'N/A':}")
    if ttm.get("revenues_ttm") is None or ttm.get("net_income_ttm") is None:
        st.caption("⚠️ Some TTM fields unavailable for this ticker (provider didn’t report enough quarterly data).")

    # Snapshot (pretty-printed)
       # Snapshot (pretty-printed)  ----------------------------------------------
    # Compute Dividend (TTM): sum of dividends in the last 12 months
    div_ttm = None
    try:
        _div = get_dividends(base_ticker)
        if not _div.empty:
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=1)
            mask = pd.Series(False, index=_div.index)
            for col in ["paymentDate", "recordDate", "date"]:
                if col in _div.columns and pd.api.types.is_datetime64_any_dtype(_div[col]):
                    mask = mask | (_div[col] >= cutoff)
            if "dividend" in _div.columns:
                vals = _div.loc[mask, "dividend"].dropna()
                if not vals.empty:
                    div_ttm = float(vals.sum())
    except Exception:
        div_ttm = None

    snap_cols = [
        "Revenue (TTM)", "Net Income (TTM)", "COGS (TTM)",
        "Total Debt", "Total Equity", "P/E", "PSR", "P/B",
        "Debt/Equity", "EPS (TTM)"
    ]
    snap_raw = {k: base_row.get(k) for k in snap_cols}
    # add Dividend (TTM) as an extra field
    snap_raw["Dividend (TTM)"] = div_ttm

    # integer-like big numbers vs small ratios/currency
    zero_cols = {"Revenue (TTM)", "Net Income (TTM)", "COGS (TTM)", "Total Debt", "Total Equity"}
    snap_fmt = {
        k: (fmt_intlike(v) if k in zero_cols else fmt_float3(v))
        for k, v in snap_raw.items()
    }

    # show in a neat two-column table
    st.table(pd.DataFrame(list(snap_fmt.items()), columns=["Metric", "Value"]))


with colB:
    st.subheader("Description")
    st.write(profile.get("description") or "—")
    if profile.get("homepage_url"):
        st.write(f"**Website**: {profile.get('homepage_url')}")
    st.write(f"**Market Cap (provider or fallback)**: {fmt_intlike(market_cap)}")
    st.write(f"**Shares Out**: {fmt_intlike(profile.get('share_class_shares_outstanding') or shares_out)}")
    st.write(f"**Sector/Industry**: {profile.get('sector')} / {profile.get('industry')}")

# -------------------------
# ---- STEP 2: PEERS -------
# -------------------------
st.markdown("---")
with st.expander("Peers (suggestions, manual add, selection & hover)", expanded=True):
    st.subheader("Peer Suggestions")
    suggested_peers: List[str] = get_peers_fmp(base_ticker) or get_peers_fmp_bulk(base_ticker)
    if suggested_peers:
        st.success(f"FMP suggested peers: {', '.join(suggested_peers[:12])}{'...' if len(suggested_peers)>12 else ''}")
    else:
        st.info("No peers returned from FMP for this ticker. You can add peers manually below.")

    # Persist and merge peers rather than clear on manual input
    if "peer_set" not in st.session_state:
        st.session_state.peer_set = set()

    if suggested_peers and not st.session_state.peer_set:
        st.session_state.peer_set.update(p for p in suggested_peers if p != base_ticker)

    manual_peers = {p.strip().upper() for p in (peer_input.split(",") if peer_input else []) if p.strip()}
    st.session_state.peer_set.update(manual_peers)

    # Build list for loading
    peer_list = sorted({p for p in st.session_state.peer_set if p != base_ticker})

    @st.cache_data(show_spinner=True)
    def load_peer_rows(peers: List[str]) -> List[Dict[str, Any]]:
        def _fetch_one(tkr: str) -> Dict[str, Any]:
            try:
                prof = get_profile_fmp(tkr)
            except Exception:
                prof = {"name": None, "description": None, "ticker": tkr}
            try:
                pr = get_quote_fmp(tkr)
            except Exception:
                pr = None

            # back-calc shares if missing
            shares = prof.get("share_class_shares_outstanding")
            if not shares and prof.get("market_cap") and pr:
                try:
                    shares = float(prof["market_cap"]) / float(pr)
                except Exception:
                    pass

            try:
                fin = get_ttm_financials(tkr)
            except Exception:
                fin = {
                    "revenues_ttm": np.nan, "cogs_ttm": np.nan, "net_income_ttm": np.nan,
                    "total_debt": np.nan, "total_equity": np.nan, "total_assets": np.nan, "total_liabilities": np.nan
                }
            row = compute_multiples_row(tkr, pr, shares, fin)
            row["Name"] = prof.get("name")
            row["Description"] = (prof.get("description") or "")
            return row

        rows: List[Dict[str, Any]] = []
        if not peers:
            return rows
        # Fetch in parallel
        with ThreadPoolExecutor(max_workers=min(8, len(peers))) as ex:
            future_map = {ex.submit(_fetch_one, p): p for p in peers}
            for fut in as_completed(future_map):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    st.warning(f"{future_map[fut]}: failed to load ({e})")
        # keep input order
        rows.sort(key=lambda r: peers.index(r.get("Ticker", "")) if r.get("Ticker", "") in peers else 1e9)
        return rows

    # Load peer rows only when peer_list changes
    if peer_list != st.session_state.get("_last_peer_list", []):
        st.session_state["_last_peer_list"] = peer_list
        st.session_state["peer_rows"] = load_peer_rows(peer_list) if peer_list else []

    peer_rows = st.session_state.get("peer_rows", [])

    # ---- Selection table integrated with description ----
    if peer_rows:
        info_df = pd.DataFrame([{
            "Ticker": base_row["Ticker"], "Name": profile.get("name"), "Description": profile.get("description") or ""
        }] + peer_rows)

        if HAS_AGGRID:
            grid_df = info_df[["Ticker", "Name", "Description"]].copy()
            grid_df["Preview"] = grid_df["Description"].fillna("").apply(lambda x: (x[:120] + "…") if len(x) > 120 else x)

            gb = GridOptionsBuilder.from_dataframe(grid_df[["Ticker","Name","Preview","Description"]])
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_grid_options(domLayout="autoHeight")
            gb.configure_column("Preview", header_name="Description", tooltipField="Description", autoHeight=True, wrapText=True, flex=2)
            gb.configure_column("Ticker", flex=0)
            gb.configure_column("Name", flex=1)
            gb.configure_column("Description", hide=True)

            grid_options = gb.build()
            grid = AgGrid(
                grid_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                theme="balham",
            )
            selected = grid.get("selected_rows", [])
            selected_tickers = [row["Ticker"] for row in selected if "Ticker" in row and row["Ticker"] != base_ticker]
        else:
            tmp = info_df[["Ticker","Name","Description"]].copy()
            tmp["Select"] = tmp["Ticker"].ne(base_ticker)  # default select peers
            tmp = st.data_editor(
                tmp,
                use_container_width=True,
                column_config={
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Select": st.column_config.CheckboxColumn("Select"),
                },
                disabled=["Ticker","Name","Description"],
                key="peer_select_editor",
            )
            selected_tickers = tmp.loc[tmp["Select"] & tmp["Ticker"].ne(base_ticker), "Ticker"].tolist()
    else:
        selected_tickers = []

# -------------------------
# ---- STEP 3: COMP TABLE --
# -------------------------
st.subheader("Comparison Table")

# Only use selected peers for the comparison table; if none selected, use all loaded peers (excluding base)
if not selected_tickers and st.session_state.get("peer_rows"):
    selected_tickers = [r["Ticker"] for r in st.session_state["peer_rows"]]

# Cache rows for selected comparison set
if selected_tickers != st.session_state.get("_last_comp_sel", []):
    st.session_state["_last_comp_sel"] = selected_tickers
    st.session_state["peer_rows_comp"] = [r for r in st.session_state.get("peer_rows", []) if r["Ticker"] in selected_tickers]

peer_rows_for_table = st.session_state.get("peer_rows_comp", [])
all_rows = [base_row] + peer_rows_for_table
if not all_rows:
    st.stop()

df = pd.DataFrame(all_rows)

# Honor "Columns to show"
show_cols = ["Ticker", "Price", *selected_metrics]
show_cols = [c for c in show_cols if c in df.columns]
if not show_cols:
    show_cols = ["Ticker", "Price", "Market Cap", "Revenue (TTM)", "Net Income (TTM)", "P/E", "PSR"]

view_df = df[show_cols].copy()

# Pretty formatting per column (big aggregates -> 0 decimals; ratios/prices -> 3)
intlike_cols = {
    "Market Cap", "Revenue (TTM)", "Net Income (TTM)", "COGS (TTM)",
    "Total Debt", "Total Equity", "Shares Out"
}
float_cols = set(view_df.columns) - {"Ticker"} - intlike_cols

for c in (intlike_cols & set(view_df.columns)):
    view_df[c] = pd.to_numeric(view_df[c], errors="coerce").map(lambda v: None if pd.isna(v) else f"{int(round(v)):,}")

for c in float_cols:
    view_df[c] = pd.to_numeric(view_df[c], errors="coerce").map(lambda v: None if pd.isna(v) else f"{float(v):,.3f}")

st.dataframe(view_df, use_container_width=True)


# ---- STEP 4: TRENDS ------
# -------------------------
st.subheader("Trends (5y): Revenue vs Expenses, Net Income & EPS")

hist_df = get_income_quarterly_fmp(base_ticker, limit=20, shares_out=shares_out)  # ~5y
if not hist_df.empty:
    plot_df = hist_df.reset_index(names="date")

    # Left axis series
    left_metrics = [c for c in ["Revenue","Expenses","Net Income"] if c in plot_df.columns]
    base_long = plot_df.melt("date", value_vars=left_metrics, var_name="Metric", value_name="Value")

    # --- FIX 1: lock left-axis domain ---
    lmin = float(base_long["Value"].min()) if not base_long.empty else 0.0
    lmax = float(base_long["Value"].max()) if not base_long.empty else 1.0
    lpad = (lmax - lmin) * 0.05 or 1.0
    left_domain = [lmin - lpad, lmax + lpad]

    chart_left = alt.Chart(base_long).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Value:Q", title="USD", scale=alt.Scale(domain=left_domain)),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(domain=["Revenue","Expenses","Net Income"],
                            range=["#4c78a8", "#f58518", "#54a24b"])
        ),
        tooltip=[
            alt.Tooltip("date:T"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Value:Q", format=",.0f"),
        ],
    )

    # EPS on right axis (optional layer)
    if "EPS" in plot_df.columns and not plot_df["EPS"].isna().all():
        eps_long = plot_df[["date","EPS"]].dropna()

        # --- FIX 2: lock right-axis (EPS) domain ---
        emin = float(eps_long["EPS"].min())
        emax = float(eps_long["EPS"].max())
        epad = (emax - emin) * 0.05 or 0.1
        eps_domain = [emin - epad, emax + epad]

        chart_eps = alt.Chart(eps_long).mark_line(strokeDash=[4, 3]).encode(
            x="date:T",
            y=alt.Y("EPS:Q", title="EPS",
                    axis=alt.Axis(titleColor="#e45756"),
                    scale=alt.Scale(domain=eps_domain)),
            color=alt.value("#e45756"),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("EPS:Q", format=",.3f")],
        )
        chart = alt.layer(chart_left, chart_eps).resolve_scale(y='independent')
    else:
        chart = chart_left

    # --- FIX 3: x-only zoom/pan (no y autoscale) ---
    zoomx = alt.selection_interval(bind='scales', encodings=['x'])
    try:
        chart = chart.add_params(zoomx)       # Altair v5
    except Exception:
        chart = chart.add_selection(zoomx)    # Altair v4 fallback

    # --- FIX 4: ensure axes aren’t clipped ---
    chart = chart.properties(height=380, padding={'left': 72, 'right': 72})

    st.altair_chart(chart, use_container_width=True)
else:
    st.caption("(No quarterly breakdown available from FMP for trend chart.)")

# -------------------------
# ---- PRICE & DATES -------
# -------------------------
with st.expander("Price & Volume + Upcoming Dates", expanded=True):
    today = dt.date.today()
    default_from = today - dt.timedelta(days=365)
    _, colp2, colp3 = st.columns([2,1,1])
    with colp2:
        d_from = st.date_input("From", value=default_from, help="Light EOD endpoint supports optional from/to")
    with colp3:
        d_to = st.date_input("To", value=today)
    d_from_str = d_from.isoformat() if d_from else None
    d_to_str = d_to.isoformat() if d_to else None

    px = get_price_light(base_ticker, d_from_str, d_to_str)
    if not px.empty and {"date","price","volume"}.issubset(px.columns):
        price_chart = alt.Chart(px).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price:Q", title="Price"),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=",.2f"), alt.Tooltip("volume:Q", format=",.0f")],
            color=alt.value("#4c78a8"),
        )
        vol_chart = alt.Chart(px).mark_bar(opacity=0.3).encode(
            x="date:T",
            y=alt.Y("volume:Q", title="Volume"),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("volume:Q", format=",.0f")],
            color=alt.value("#9ecae9"),
        )
        st.altair_chart(alt.layer(price_chart, vol_chart).resolve_scale(y='independent').properties(height=320), use_container_width=True)
    else:
        st.caption("No price data from FMP light endpoint for the selected range.")

    # Upcoming dates (dividends & earnings)
    div = get_dividends(base_ticker)
    ear = get_earnings(base_ticker)
    upcoming_bits = []

    now = pd.Timestamp.utcnow().tz_localize(None).normalize()

    if not div.empty:
        # Build a combined future mask safely
        masks = []
        for col in ["paymentDate", "recordDate", "date"]:
            if col in div.columns:
                s = div[col]
                if pd.api.types.is_datetime64_any_dtype(s):
                    masks.append(s.notna() & (s >= now))
        if masks:
            future_mask = masks[0]
            for m in masks[1:]:
                future_mask = future_mask | m
            next_div_rows = div[future_mask]
            if not next_div_rows.empty:
                r = next_div_rows.sort_values(["paymentDate","recordDate","date"]).iloc[0]
                pay = r.get("paymentDate")
                rec = r.get("recordDate")
                amt = r.get("dividend")
                freq = r.get("frequency","")
                upcoming_bits.append(
                    f"💸 Dividend: pay **{pay.date() if pd.notna(pay) else 'TBA'}**, "
                    f"record **{rec.date() if pd.notna(rec) else 'TBA'}**, "
                    f"amount **{fmt_float3(amt)}** ({freq})."
                )

    if not ear.empty and "date" in ear.columns and pd.api.types.is_datetime64_any_dtype(ear["date"]):
        next_ear = ear[ear["date"] >= now]
        if not next_ear.empty:
            r = next_ear.sort_values("date").iloc[0]
            upcoming_bits.append(
                f"📣 Earnings: **{r['date'].date()}** "
                f"(EPS est: {fmt_float3(r.get('epsEstimated'))}; "
                f"rev est: {fmt_intlike(r.get('revenueEstimated'))})."
            )

    if upcoming_bits:
        st.markdown("**Upcoming dates:**  \n" + "  \n".join(upcoming_bits))
    else:
        st.caption("No upcoming dividend/earnings dates found.")

# -------------------------
# ---- STEP 5: WHAT-IF ----
# -------------------------
st.markdown("---")
st.subheader("What-if Multiples")

kind = st.radio("Multiple type", ["P/E", "PSR"], horizontal=True)

peer_df_for_stats = pd.DataFrame(peer_rows_for_table)
valid_peer_vals = (
    peer_df_for_stats[kind].replace([np.inf, -np.inf], np.nan).dropna()
    if (not peer_df_for_stats.empty and kind in peer_df_for_stats.columns)
    else pd.Series(dtype=float)
)

p25 = valid_peer_vals.quantile(0.25) if not valid_peer_vals.empty else None
p50 = valid_peer_vals.quantile(0.50) if not valid_peer_vals.empty else None
p75 = valid_peer_vals.quantile(0.75) if not valid_peer_vals.empty else None
pavg = valid_peer_vals.mean()         if not valid_peer_vals.empty else None

col1, col2, col3, col4, _ = st.columns(5)
with col1:
    st.metric(f"Peer 25th {kind}", f"{p25:.2f}" if p25 is not None else "N/A")
with col2:
    st.metric(f"Peer Median {kind}", f"{p50:.2f}" if p50 is not None else "N/A")
with col3:
    st.metric(f"Peer 75th {kind}", f"{p75:.2f}" if p75 is not None else "N/A")
with col4:
    st.metric(f"Peer Avg {kind}", f"{pavg:.2f}" if pavg is not None else "N/A")

# Default the custom input to the median if available
custom_mult = st.number_input(
    f"Custom {kind} (optional)",
    value=float(p50) if p50 is not None else 0.0,
    step=0.1
)

targets = {
    "P25":    what_if_price_from_multiple(base_row, p25, kind)  if p25  else None,
    "Median": what_if_price_from_multiple(base_row, p50, kind)  if p50  else None,
    "P75":    what_if_price_from_multiple(base_row, p75, kind)  if p75  else None,
    "Avg":    what_if_price_from_multiple(base_row, pavg, kind) if pavg else None,
    "Custom": what_if_price_from_multiple(base_row, custom_mult, kind) if custom_mult else None,
}

st.write({k: (round(v, 2) if v is not None else None) for k, v in targets.items()})

# -------------------------
# ---- STEP 6: WATCHLIST ---
# -------------------------
st.markdown("---")
st.subheader("Watchlist")

wl_col1, wl_col2, wl_col3 = st.columns([2, 1, 2])
with wl_col1:
    add_note = st.text_input("Note (optional)")
with wl_col2:
    if st.button("➕ Add base ticker to watchlist"):
        now_ts = dt.datetime.utcnow().isoformat()
        with st.session_state.watchlist_engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT OR REPLACE INTO watchlist (ticker, added_at, added_price, note) VALUES (?, ?, ?, ?)",
                (base_ticker, now_ts, base_row.get("Price") or None, add_note or None),
            )
        st.success(f"Added {base_ticker} at {base_row.get('Price')} on {now_ts} UTC")

with wl_col3:
    if st.button("🗑️ Remove base ticker"):
        with st.session_state.watchlist_engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM watchlist WHERE ticker = ?", (base_ticker,))
        st.warning(f"Removed {base_ticker} from watchlist")

with st.session_state.watchlist_engine.connect() as conn:
    wl = pd.read_sql("SELECT * FROM watchlist", conn)

if not wl.empty:
    prices = []
    for t in wl["ticker"].tolist():
        try:
            p = get_quote_fmp(t)
        except Exception:
            p = None
        prices.append(p)
    wl["current_price"] = prices
    wl["pl_abs"] = wl["current_price"] - wl["added_price"]
    wl["pl_pct"] = (wl["pl_abs"] / wl["added_price"]) * 100.0

    # Pretty print watchlist numbers
    for col in ["added_price", "current_price", "pl_abs"]:
        if col in wl.columns:
            wl[col] = wl[col].apply(lambda v: None if pd.isna(v) else (f"{v:,.2f}" if isinstance(v, (float, int)) else v))
    if "pl_pct" in wl.columns:
        wl["pl_pct"] = wl["pl_pct"].apply(lambda v: None if pd.isna(v) else f"{v:.2f}%")

    st.dataframe(wl, use_container_width=True)
else:
    st.info("Your watchlist is empty.")

# -------------------------
# ---- FORECASTS & NEWS ----
# -------------------------
with st.expander("Analyst Forecasts (FMP)"):
    try:
        est = fmp_get(
            f"{FMP_BASE}/analyst-estimates",
            {"symbol": base_ticker, "period": "annual", "limit": 10, "apikey": api_key},
        )
        if isinstance(est, list) and est:
            df_est = pd.DataFrame(est)
            cols = [c for c in [
                "date","revenueAvg","epsAvg","revenueHigh","revenueLow",
                "epsHigh","epsLow","numAnalystsRevenue","numAnalystsEps"
            ] if c in df_est.columns]
            df_est = df_est[cols].copy()
            style_map = {}
            for c in ["revenueAvg","revenueHigh","revenueLow"]:
                if c in df_est.columns: style_map[c] = "{:,.0f}"
            for c in ["epsAvg","epsHigh","epsLow"]:
                if c in df_est.columns: style_map[c] = "{:,.3f}"
            st.dataframe(df_est.style.format(style_map), use_container_width=True)
        else:
            st.caption("No analyst estimates available.")
    except Exception as e:
        st.caption(f"Estimates unavailable: {e}")

with st.expander("Latest News (FMP)"):
    try:
        news = fmp_get(f"{FMP_BASE}/news/stock", {"symbols": base_ticker, "limit": 10, "apikey": api_key})
        if isinstance(news, list) and news:
            nd = pd.DataFrame(news)
            cols = [c for c in ["publishedDate","publisher","title","url"] if c in nd.columns]
            st.dataframe(nd[cols].sort_values("publishedDate", ascending=False), use_container_width=True)
        else:
            st.caption("No recent news from FMP.")
    except Exception as e:
        st.caption(f"News unavailable: {e}")

# -------------------------
# ---- FOOTER --------------
# -------------------------
st.markdown(
    """
    <small>
    Notes: Multiples use TTM. P/E omitted for non-positive earnings. PSR = Market Cap / TTM Revenue.
    Debt/Equity = Total Debt / Total Equity (with Equity fallback = Total Assets − Total Debt).
    Data quality depends on provider disclosures.
    </small>
    """,
    unsafe_allow_html=True,
)
