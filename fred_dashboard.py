"""
US Macro Dashboard v2 — Recession & Bear Market Risk Monitor
============================================================
Indicator set (expanded):

0. 🚨 Composite Risk Score
   - Aggregates yield curve, Sahm, EBP prob, NFCI, SLOOS, HY OAS into 0-100 score
   - Real-time alert banner with crossed thresholds

1. Credit & Risk
   - High Yield OAS              → BAMLH0A0HYM2
   - Excess Bond Premium         → Fed Board CSV (ebp_csv.csv)
   - Chicago Fed NFCI            → NFCI
   - Baa - 10Y spread            → BAA10Y (equity risk premium proxy)
   - VIX                         → VIXCLS

2. Yield Curve & Lending
   - 10Y - 2Y / 10Y - 3M / 10Y - FF
   - SLOOS C&I tightening        → DRTSCILM

3. Labor Market
   - UNRATE, ICSA, CCSA, PAYEMS
   - Sahm Rule                   → SAHMREALTIME  (NEW — best coincident recession signal)

4. Inflation
   - CPI / Core CPI / PCE / Core PCE

5. Activity & Markets
   - CFNAI + State LEI (USSLIND) + PMI proxy (fixed-reference normalized)
   - Buffett Indicator with recalibrated Z.1 valuation bands

6. Housing (NEW)
   - PERMIT, HOUST, MORTGAGE30US

7. Consumer (NEW)
   - UMCSENT, RSAFS YoY, PSAVERT

8. Liquidity & Policy (NEW)
   - M2SL YoY, WALCL (Fed balance sheet), FEDFUNDS, GDP / GDPC1

Setup:
    pip install streamlit pandas plotly fredapi python-dotenv numpy requests

Run:
    python -m streamlit run fred_dashboard.py
"""

import io
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from fredapi import Fred
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

# Try Streamlit secrets first (cloud), fall back to .env (local)
FRED_API_KEY = ""
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

if not FRED_API_KEY:
    st.error("⚠️ FRED API key not found. Set it in `.env` locally or in Streamlit Secrets when deployed.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

st.set_page_config(
    page_title="US Macro Risk Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    st.error("⚠️ FRED API key not found. Create `.env` with `FRED_API_KEY=your_key_here`.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

DEFAULT_LOOKBACK_YEARS = 10
FETCH_BUFFER_DAYS = 800   # bigger buffer so YoY / rolling stats are valid at start
REF_START = "1990-01-01"  # fixed reference period start
REF_END   = "2019-12-31"  # fixed reference period end (pre-COVID)

EBP_CSV_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"


# ---------------------------------------------------------------------------
# DATA FETCH HELPERS
# ---------------------------------------------------------------------------
def _to_date_str(d) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        return d
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def _buffered_start(start_date, extra_days: int = FETCH_BUFFER_DAYS):
    if start_date is None:
        return None
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date).to_pydatetime()
    elif isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()
    return start_date - timedelta(days=extra_days)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series(series_id: str, start_date=None, end_date=None,
                 max_retries: int = 3, silent: bool = False) -> pd.Series:
    start_str = _to_date_str(start_date)
    end_str = _to_date_str(end_date)
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = {}
            if start_str: kwargs["observation_start"] = start_str
            if end_str:   kwargs["observation_end"] = end_str
            s = fred.get_series(series_id, **kwargs)
            s.name = series_id
            return s
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "does not exist" in msg or "bad request" in msg:
                break
            if any(t in msg for t in ("internal server error", "500", "502",
                                      "503", "504", "timeout", "timed out", "connection")):
                time.sleep(0.6 * (2 ** attempt))
                continue
            break
    try:
        s = fred.get_series(series_id)
        s.name = series_id
        if start_str: s = s[s.index >= pd.Timestamp(start_str)]
        if end_str:   s = s[s.index <= pd.Timestamp(end_str)]
        return s
    except Exception as e:
        last_err = e
    if not silent:
        st.warning(f"Could not fetch **{series_id}**: {last_err}")
    return pd.Series(dtype=float, name=series_id)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ebp_csv() -> pd.DataFrame:
    try:
        r = requests.get(EBP_CSV_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if c.startswith("date")), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        prob_candidates = [c for c in df.columns
                           if "prob" in c or c in ("est_prob", "recession_prob", "rec_prob")]
        if prob_candidates:
            df = df.rename(columns={prob_candidates[0]: "est_prob"})
        keep = [c for c in ("ebp", "gz_spread", "est_prob") if c in df.columns]
        return df[keep].astype(float)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def clip_range(s, start, end):
    if s is None: return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        if s.empty: return s
        if start is not None: s = s[s.index >= pd.Timestamp(_to_date_str(start))]
        if end is not None:   s = s[s.index <= pd.Timestamp(_to_date_str(end))]
        return s
    if s.empty: return s
    if start is not None: s = s[s.index >= pd.Timestamp(_to_date_str(start))]
    if end is not None:   s = s[s.index <= pd.Timestamp(_to_date_str(end))]
    return s


def latest_value(s: pd.Series, fmt: str = "{:.2f}", suffix: str = "") -> str:
    s = s.dropna()
    if s.empty: return "N/A"
    return fmt.format(s.iloc[-1]) + suffix


def latest_scalar(s: pd.Series):
    s = s.dropna() if s is not None else pd.Series(dtype=float)
    return None if s.empty else float(s.iloc[-1])


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    if s.empty: return s
    return s.pct_change(periods) * 100


def add_recession_shading(fig, start, end, row=None, col=None):
    rec = fetch_series("USREC", silent=True).dropna()
    if rec.empty: return
    rec = rec.astype(int)
    in_rec = rec == 1
    starts, ends = [], []
    prev = 0
    for dt, v in in_rec.items():
        if v and not prev: starts.append(dt)
        elif not v and prev: ends.append(dt)
        prev = v
    if len(ends) < len(starts): ends.append(rec.index[-1])
    start_ts = pd.Timestamp(_to_date_str(start)) if start is not None else None
    end_ts = pd.Timestamp(_to_date_str(end)) if end is not None else None
    for s_dt, e_dt in zip(starts, ends):
        if start_ts is not None and e_dt < start_ts: continue
        if end_ts is not None and s_dt > end_ts: continue
        s_clip = max(s_dt, start_ts) if start_ts is not None else s_dt
        e_clip = min(e_dt, end_ts) if end_ts is not None else e_dt
        kwargs = dict(fillcolor="LightGray", opacity=0.25, line_width=0, layer="below")
        if row is not None and col is not None:
            fig.add_vrect(x0=s_clip, x1=e_clip, row=row, col=col, **kwargs)
        else:
            fig.add_vrect(x0=s_clip, x1=e_clip, **kwargs)


def apply_xrange(fig, start, end):
    start_ts = pd.Timestamp(_to_date_str(start))
    end_ts = pd.Timestamp(_to_date_str(end))
    for key in fig.layout:
        if key.startswith("xaxis"):
            fig.layout[key].range = [start_ts, end_ts]
    fig.update_xaxes(range=[start_ts, end_ts])


def fixed_ref_zscore(s: pd.Series, ref_start: str = REF_START,
                     ref_end: str = REF_END) -> pd.Series:
    """Z-score a series against a FIXED reference window (default 1990-2019)."""
    if s is None or s.dropna().empty:
        return pd.Series(dtype=float)
    ref = s.loc[ref_start:ref_end].dropna()
    if ref.empty or ref.std() == 0:
        ref = s.dropna()
        if ref.empty or ref.std() == 0:
            return pd.Series(dtype=float, index=s.index)
    return (s - ref.mean()) / ref.std()


# ---------------------------------------------------------------------------
# BUFFETT INDICATOR (Z.1 version)
# ---------------------------------------------------------------------------
def compute_buffett_indicator(start, end, fetch_start) -> pd.Series:
    eq = fetch_series("NCBEILQ027S", "1945-01-01", end, silent=True).dropna()
    if eq.empty:
        eq = fetch_series("MVEONWMVBSNNCB", "1945-01-01", end, silent=True).dropna()
    if eq.empty:
        return pd.Series(dtype=float, name="Buffett")
    eq_b = eq / 1000.0  # millions -> billions
    gdp = fetch_series("GDP", "1945-01-01", end, silent=True).dropna()
    if gdp.empty:
        return pd.Series(dtype=float, name="Buffett")
    monthly_idx = pd.date_range(
        start=min(eq_b.index.min(), gdp.index.min()),
        end=max(eq_b.index.max(), gdp.index.max()),
        freq="ME",
    )
    eq_m  = eq_b.reindex(eq_b.index.union(monthly_idx)).sort_index().ffill().reindex(monthly_idx)
    gdp_m = gdp.reindex(gdp.index.union(monthly_idx)).sort_index().ffill().reindex(monthly_idx)
    df = pd.concat([eq_m.rename("EQ_B"), gdp_m.rename("GDP_B")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="Buffett")
    return (df["EQ_B"] / df["GDP_B"] * 100).rename("Buffett")


def buffett_percentile_bands(buf_full: pd.Series):
    """Compute valuation bands from the post-1990 distribution of Z.1 Buffett."""
    ref = buf_full.loc[REF_START:].dropna()
    if ref.empty:
        return None
    return {
        "p20": float(np.percentile(ref, 20)),
        "p50": float(np.percentile(ref, 50)),
        "p80": float(np.percentile(ref, 80)),
        "p95": float(np.percentile(ref, 95)),
    }


# ---------------------------------------------------------------------------
# COMPOSITE RECESSION RISK SCORE (0-100)
# ---------------------------------------------------------------------------
def compute_risk_score():
    """
    Build a composite recession risk score from:
      - Yield curve (10Y-3M)        20%
      - Sahm Rule                   20%
      - EBP recession probability   20%
      - NFCI                        15%
      - SLOOS C&I tightening        15%
      - HY OAS deviation            10%
    Each component scored 0-100; weighted average returned.
    Also returns dict of component scores and triggered alerts.
    """
    components = {}
    alerts = []

    # 1. Yield curve 10Y-3M (inverted ⇒ high risk)
    t10y3m = fetch_series("T10Y3M", silent=True).dropna()
    yc_val = latest_scalar(t10y3m)
    if yc_val is not None:
        # -1.5% ⇒ 100, 0% ⇒ 60, +2% ⇒ 0
        score = float(np.clip(60 - yc_val * 30, 0, 100))
        components["Yield Curve (10Y-3M)"] = (score, f"{yc_val:+.2f}%")
        if yc_val < 0:
            alerts.append(f"⚠️ Yield curve inverted ({yc_val:+.2f}%)")

    # 2. Sahm Rule
    sahm = fetch_series("SAHMREALTIME", silent=True).dropna()
    sahm_val = latest_scalar(sahm)
    if sahm_val is not None:
        # 0 ⇒ 0, 0.5 (trigger) ⇒ 90, 1.0+ ⇒ 100
        score = float(np.clip(sahm_val * 180, 0, 100))
        components["Sahm Rule"] = (score, f"{sahm_val:+.2f}")
        if sahm_val >= 0.5:
            alerts.append(f"🚨 Sahm Rule TRIGGERED ({sahm_val:+.2f} ≥ 0.50)")
        elif sahm_val >= 0.3:
            alerts.append(f"⚠️ Sahm Rule approaching trigger ({sahm_val:+.2f})")

    # 3. EBP-based recession probability
    ebp_df = fetch_ebp_csv()
    ebp_prob_val = None
    if not ebp_df.empty and "est_prob" in ebp_df.columns:
        prob_series = ebp_df["est_prob"].dropna()
        if not prob_series.empty:
            raw = float(prob_series.iloc[-1])
            ebp_prob_val = raw * 100 if raw <= 1.5 else raw
            components["EBP Recession Prob"] = (float(np.clip(ebp_prob_val, 0, 100)),
                                                f"{ebp_prob_val:.1f}%")
            if ebp_prob_val >= 50:
                alerts.append(f"🚨 EBP model prob ≥ 50% ({ebp_prob_val:.1f}%)")
            elif ebp_prob_val >= 30:
                alerts.append(f"⚠️ EBP model prob elevated ({ebp_prob_val:.1f}%)")

    # 4. NFCI
    nfci = fetch_series("NFCI", silent=True).dropna()
    nfci_val = latest_scalar(nfci)
    if nfci_val is not None:
        # -1 ⇒ 0, 0 ⇒ 50, +1 ⇒ 90, +2 ⇒ 100
        score = float(np.clip(50 + nfci_val * 40, 0, 100))
        components["NFCI"] = (score, f"{nfci_val:+.2f}")
        if nfci_val > 0:
            alerts.append(f"⚠️ Financial conditions tightening (NFCI {nfci_val:+.2f})")

    # 5. SLOOS C&I tightening
    sloos = fetch_series("DRTSCILM", silent=True).dropna()
    sloos_val = latest_scalar(sloos)
    if sloos_val is not None:
        # -20 ⇒ 0, 0 ⇒ 40, +20 ⇒ 75, +50+ ⇒ 100
        score = float(np.clip(40 + sloos_val * 1.5, 0, 100))
        components["SLOOS Tightening"] = (score, f"{sloos_val:+.1f}%")
        if sloos_val >= 30:
            alerts.append(f"⚠️ Banks tightening sharply (SLOOS {sloos_val:+.1f}%)")

    # 6. HY OAS deviation from 1Y mean
    hy = fetch_series("BAMLH0A0HYM2", silent=True).dropna()
    if not hy.empty and len(hy) > 252:
        hy_now = float(hy.iloc[-1])
        hy_mean = float(hy.iloc[-252:].mean())
        hy_std = float(hy.iloc[-252:].std())
        z = (hy_now - hy_mean) / hy_std if hy_std > 0 else 0
        score = float(np.clip(50 + z * 25, 0, 100))
        components["HY OAS (1Y z-score)"] = (score, f"{hy_now:.2f}% (z={z:+.1f})")
        if z >= 1.5:
            alerts.append(f"🚨 HY OAS spike (z={z:+.1f}σ above 1Y mean)")
        elif z >= 1.0:
            alerts.append(f"⚠️ HY OAS elevated (z={z:+.1f}σ)")

    # Weighted composite
    weights = {
        "Yield Curve (10Y-3M)": 0.20,
        "Sahm Rule":            0.20,
        "EBP Recession Prob":   0.20,
        "NFCI":                 0.15,
        "SLOOS Tightening":     0.15,
        "HY OAS (1Y z-score)":  0.10,
    }
    total_w, total_s = 0.0, 0.0
    for k, w in weights.items():
        if k in components:
            total_s += components[k][0] * w
            total_w += w
    composite = total_s / total_w if total_w > 0 else None

    return composite, components, alerts


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("🚨 Risk Monitor Controls")

today = datetime.today()
default_start = today - timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)

st.sidebar.markdown("**Quick range**")
preset_cols = st.sidebar.columns(4)
preset_years = None
if preset_cols[0].button("1Y"):  preset_years = 1
if preset_cols[1].button("5Y"):  preset_years = 5
if preset_cols[2].button("10Y"): preset_years = 10
if preset_cols[3].button("Max"): preset_years = 75

if preset_years is not None:
    st.session_state["start_date"] = today - timedelta(days=365 * preset_years)
    st.session_state["end_date"] = today

start_date = st.sidebar.date_input(
    "Start date",
    value=st.session_state.get("start_date", default_start),
    min_value=datetime(1950, 1, 1), max_value=today, key="start_date",
)
end_date = st.sidebar.date_input(
    "End date",
    value=st.session_state.get("end_date", today),
    min_value=start_date, max_value=today, key="end_date",
)

if st.sidebar.button("🔄 Clear cache & refresh"):
    st.cache_data.clear()
    st.rerun()

show_recessions = st.sidebar.checkbox("Show NBER recession shading", value=True)

section = st.sidebar.radio(
    "Section",
    [
        "🚨 Risk Score & Alerts",
        "🏠 Overview",
        "⚠️ Credit & Risk",
        "📈 Yield Curve & Lending",
        "👷 Labor Market",
        "💰 Inflation",
        "🏭 Activity & Markets",
        "🏘️ Housing",
        "🛍️ Consumer",
        "💵 Liquidity & Money",
        "🇺🇸 GDP & Fed Funds",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Default range: **{DEFAULT_LOOKBACK_YEARS} years**  \n"
    f"Reference period for z-scores: **{REF_START[:4]}–{REF_END[:4]}**  \n"
    "Sources: [FRED](https://fred.stlouisfed.org) • "
    "[Fed Board EBP](https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html)"
)

fetch_start = _buffered_start(start_date)


def maybe_shade(fig, row=None, col=None):
    if show_recessions:
        add_recession_shading(fig, start_date, end_date, row=row, col=col)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
st.title("🇺🇸 US Macro Risk Monitor")
st.caption(f"Range: **{start_date}** → **{end_date}** • "
           f"Reference window for normalization: **{REF_START[:4]}–{REF_END[:4]}**")


# =========================================================================
# 0. RISK SCORE & ALERTS  (NEW — landing page)
# =========================================================================
if section == "🚨 Risk Score & Alerts":
    st.subheader("Composite Recession Risk Score")

    composite, components, alerts = compute_risk_score()

    # ---- Top row: gauge + alerts ----
    col_g, col_a = st.columns([1, 2])

    with col_g:
        if composite is not None:
            color = ("#2ecc71" if composite < 30
                     else "#f39c12" if composite < 55
                     else "#e74c3c" if composite < 75
                     else "#8e44ad")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=composite,
                number={"suffix": "/100", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 30],   "color": "#d5f5e3"},  # low
                        {"range": [30, 55],  "color": "#fcf3cf"},  # moderate
                        {"range": [55, 75],  "color": "#f5b7b1"},  # elevated
                        {"range": [75, 100], "color": "#d2b4de"},  # extreme
                    ],
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.75, "value": composite},
                },
                title={"text": "Composite Risk", "font": {"size": 18}},
            ))
            gauge.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(gauge, width='stretch')

            label = ("LOW"      if composite < 30
                     else "MODERATE" if composite < 55
                     else "ELEVATED" if composite < 75
                     else "EXTREME")
            st.markdown(f"### Regime: **{label}**")
        else:
            st.warning("Composite score unavailable — data fetch failed.")

    with col_a:
        st.markdown("### 🔔 Active Alerts")
        if alerts:
            for a in alerts:
                if a.startswith("🚨"):
                    st.error(a)
                else:
                    st.warning(a)
        else:
            st.success("✅ No threshold breaches detected. Macro conditions benign.")

        st.markdown("### Component Scores")
        if components:
            comp_df = pd.DataFrame(
                [(k, v[0], v[1]) for k, v in components.items()],
                columns=["Indicator", "Score (0-100)", "Latest"],
            )
            comp_df["Score (0-100)"] = comp_df["Score (0-100)"].round(1)
            st.dataframe(comp_df, width='stretch', hide_index=True)

    st.markdown("---")

    # ---- Component bar chart ----
    if components:
        st.subheader("Component Risk Contributions")
        names = list(components.keys())
        scores = [components[n][0] for n in names]
        weights = {
            "Yield Curve (10Y-3M)": 0.20, "Sahm Rule": 0.20,
            "EBP Recession Prob": 0.20, "NFCI": 0.15,
            "SLOOS Tightening": 0.15, "HY OAS (1Y z-score)": 0.10,
        }
        contribs = [scores[i] * weights.get(names[i], 0) for i in range(len(names))]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Raw Component Scores",
                                            "Weighted Contribution to Composite"])
        fig.add_trace(go.Bar(x=names, y=scores,
                             marker_color=["#e74c3c" if s >= 60 else
                                           "#f39c12" if s >= 40 else "#2ecc71"
                                           for s in scores],
                             text=[f"{s:.0f}" for s in scores], textposition="outside"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=contribs,
                             marker_color="#3498db",
                             text=[f"{c:.1f}" for c in contribs], textposition="outside"),
                      row=1, col=2)
        fig.update_layout(height=420, showlegend=False, margin=dict(t=60, b=120))
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, width='stretch')

    st.info(
        "**How to read this score:**  \n"
        "• **0–30** = Low risk: expansion, no recession signals.  \n"
        "• **30–55** = Moderate: some warning signs, monitor.  \n"
        "• **55–75** = Elevated: multiple indicators flashing — defensive positioning warranted.  \n"
        "• **75–100** = Extreme: recession likely within 6–12 months historically.  \n\n"
        "Weights: Yield Curve 20% • Sahm 20% • EBP Prob 20% • NFCI 15% • SLOOS 15% • HY OAS 10%."
    )


# =========================================================================
# OVERVIEW
# =========================================================================
elif section == "🏠 Overview":
    st.subheader("Headline Snapshot")

    series_quick = {
        "Unemployment":  ("UNRATE",       "{:.2f}", "%"),
        "Sahm Rule":     ("SAHMREALTIME", "{:+.2f}", ""),
        "CPI YoY":       ("CPIAUCSL",     "{:.2f}", "%"),
        "Core PCE YoY":  ("PCEPILFE",     "{:.2f}", "%"),
        "Fed Funds":     ("FEDFUNDS",     "{:.2f}", "%"),
        "10Y-3M":        ("T10Y3M",       "{:+.2f}", "%"),
        "HY OAS":        ("BAMLH0A0HYM2", "{:.2f}", "%"),
        "NFCI":          ("NFCI",         "{:+.2f}", ""),
    }

    cols = st.columns(4)
    for i, (label, (sid, fmt, suf)) in enumerate(series_quick.items()):
        s = fetch_series(sid).dropna()
        if "YoY" in label and not s.empty:
            s = yoy(s).dropna()
        with cols[i % 4]:
            st.metric(label, latest_value(s, fmt, suf))

    st.markdown("---")
    st.subheader("Snapshot Charts")
    layout = [
        ("UNRATE",       "Unemployment Rate (%)",        False),
        ("CPIAUCSL",     "CPI YoY (%)",                  True),
        ("T10Y3M",       "10Y-3M Treasury Spread (%)",   False),
        ("BAMLH0A0HYM2", "High Yield OAS (%)",           False),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t, _ in layout])
    for i, (sid, _, is_yoy) in enumerate(layout):
        r, c = i // 2 + 1, i % 2 + 1
        s_full = fetch_series(sid, fetch_start, end_date)
        if is_yoy: s_full = yoy(s_full)
        s = clip_range(s_full, start_date, end_date)
        if not s.dropna().empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=sid),
                          row=r, col=c)
            maybe_shade(fig, row=r, col=c)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40, b=20))
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 1. CREDIT & RISK
# =========================================================================
elif section == "⚠️ Credit & Risk":
    st.subheader("Credit Spreads, Financial Conditions & Volatility")

    hy_oas = clip_range(fetch_series("BAMLH0A0HYM2", fetch_start, end_date), start_date, end_date)
    nfci   = clip_range(fetch_series("NFCI", fetch_start, end_date), start_date, end_date)
    baa10  = clip_range(fetch_series("BAA10Y", fetch_start, end_date), start_date, end_date)
    vix    = clip_range(fetch_series("VIXCLS", fetch_start, end_date), start_date, end_date)

    ebp_df = fetch_ebp_csv()
    ebp_df_clip = clip_range(ebp_df, start_date, end_date)
    ebp = ebp_df_clip["ebp"] if (not ebp_df_clip.empty and "ebp" in ebp_df_clip.columns) else pd.Series(dtype=float)
    prob = ebp_df_clip["est_prob"] if (not ebp_df_clip.empty and "est_prob" in ebp_df_clip.columns) else pd.Series(dtype=float)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("HY OAS", latest_value(hy_oas, "{:.2f}", "%"))
    with c2: st.metric("Baa-10Y", latest_value(baa10, "{:.2f}", "%"))
    with c3: st.metric("EBP", latest_value(ebp, "{:+.2f}"))
    with c4: st.metric("NFCI", latest_value(nfci, "{:+.2f}"))
    with c5: st.metric("VIX", latest_value(vix, "{:.1f}"))

    st.caption(
        "🔎 HY OAS / Baa-10Y ↑ = credit stress. EBP > 0 = bonds pricing risk above fundamentals. "
        "NFCI > 0 = tighter than average financial conditions. VIX > 25 = equity stress regime."
    )

    n_panels = 4 + (1 if not prob.empty else 0)
    titles = ["High Yield OAS (%)", "Baa - 10Y Spread (%)",
              "Excess Bond Premium",
              "NFCI (red=tight) & VIX (right axis)"]
    if not prob.empty:
        titles.append("EBP Model — Recession Probability (%)")

    fig = make_subplots(rows=n_panels, cols=1, subplot_titles=titles,
                        vertical_spacing=0.06, shared_xaxes=True,
                        specs=[[{}], [{}], [{}], [{"secondary_y": True}]] +
                              ([[{}]] if not prob.empty else []))

    if not hy_oas.empty:
        fig.add_trace(go.Scatter(x=hy_oas.index, y=hy_oas.values,
                                 line=dict(color="crimson")), row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not baa10.empty:
        fig.add_trace(go.Scatter(x=baa10.index, y=baa10.values,
                                 line=dict(color="darkred")), row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    if not ebp.empty:
        fig.add_trace(go.Scatter(x=ebp.index, y=ebp.values,
                                 line=dict(color="darkorange")), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        maybe_shade(fig, row=3, col=1)
    if not nfci.empty:
        fig.add_trace(go.Scatter(x=nfci.index, y=nfci.values,
                                 line=dict(color="steelblue"), name="NFCI"),
                      row=4, col=1, secondary_y=False)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    if not vix.empty:
        fig.add_trace(go.Scatter(x=vix.index, y=vix.values,
                                 line=dict(color="purple", width=1), name="VIX"),
                      row=4, col=1, secondary_y=True)
    maybe_shade(fig, row=4, col=1)

    if not prob.empty:
        prob_pct = prob * 100 if prob.max() <= 1.5 else prob
        fig.add_trace(go.Scatter(x=prob_pct.index, y=prob_pct.values,
                                 line=dict(color="purple", width=2),
                                 fill="tozeroy", fillcolor="rgba(128,0,128,0.15)"),
                      row=5, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="red",
                      annotation_text="50% threshold", row=5, col=1)
        maybe_shade(fig, row=5, col=1)

    fig.update_layout(height=240 * n_panels, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 2. YIELD CURVE & LENDING
# =========================================================================
elif section == "📈 Yield Curve & Lending":
    st.subheader("Yield Curve & Bank Lending Standards")

    sp_10_2  = clip_range(fetch_series("T10Y2Y", fetch_start, end_date), start_date, end_date)
    sp_10_3m = clip_range(fetch_series("T10Y3M", fetch_start, end_date), start_date, end_date)
    dgs10    = clip_range(fetch_series("DGS10", fetch_start, end_date), start_date, end_date)
    fedfn    = clip_range(fetch_series("FEDFUNDS", fetch_start, end_date), start_date, end_date)
    sloos    = clip_range(fetch_series("DRTSCILM", fetch_start, end_date), start_date, end_date)

    if not dgs10.empty and not fedfn.empty:
        ten_minus_ff = (dgs10.resample("ME").last() - fedfn.resample("ME").last()).dropna()
        ten_minus_ff.name = "10Y - FF"
    else:
        ten_minus_ff = pd.Series(dtype=float, name="10Y - FF")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("10Y − 2Y",         latest_value(sp_10_2,      "{:+.2f}", "%"))
    with c2: st.metric("10Y − 3M",         latest_value(sp_10_3m,     "{:+.2f}", "%"))
    with c3: st.metric("10Y − Fed Funds",  latest_value(ten_minus_ff, "{:+.2f}", "%"))
    with c4: st.metric("SLOOS Tightening", latest_value(sloos,        "{:+.1f}", "%"))

    st.caption("🔎 Negative spreads = inversion. SLOOS > 0 = banks tightening C&I lending standards (leads cycle 9-12 mo).")

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Yield Curve Spreads (%)",
                                        "SLOOS — Net % Banks Tightening C&I Lending"],
                        vertical_spacing=0.12)
    for s, name, color in [(sp_10_2, "10Y - 2Y", "royalblue"),
                           (sp_10_3m, "10Y - 3M", "seagreen"),
                           (ten_minus_ff, "10Y - FF", "darkviolet")]:
        if not s.dropna().empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     name=name, line=dict(color=color)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    maybe_shade(fig, row=1, col=1)

    if not sloos.dropna().empty:
        fig.add_trace(go.Bar(x=sloos.index, y=sloos.values, name="SLOOS",
                             marker=dict(color=np.where(sloos.values >= 0,
                                                        "indianred", "mediumseagreen"))),
                      row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    fig.update_layout(height=700, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 3. LABOR MARKET (with Sahm Rule)
# =========================================================================
elif section == "👷 Labor Market":
    st.subheader("Labor Market — incl. Sahm Rule")

    unrate = clip_range(fetch_series("UNRATE", fetch_start, end_date), start_date, end_date)
    icsa   = clip_range(fetch_series("ICSA",   fetch_start, end_date), start_date, end_date)
    ccsa   = clip_range(fetch_series("CCSA",   fetch_start, end_date), start_date, end_date)
    sahm   = clip_range(fetch_series("SAHMREALTIME", fetch_start, end_date), start_date, end_date)

    payems_full = fetch_series("PAYEMS", fetch_start, end_date)
    payems = clip_range(payems_full, start_date, end_date)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Unemployment", latest_value(unrate, "{:.2f}", "%"))
    with c2:
        sahm_v = latest_scalar(sahm)
        delta = None
        if sahm_v is not None:
            delta = "🚨 TRIGGERED" if sahm_v >= 0.5 else ("⚠️ Near trigger" if sahm_v >= 0.3 else "OK")
        st.metric("Sahm Rule", latest_value(sahm, "{:+.2f}"), delta=delta)
    with c3: st.metric("Initial Claims", latest_value(icsa, "{:,.0f}"))
    with c4: st.metric("Continued Claims", latest_value(ccsa, "{:,.0f}"))
    with c5:
        if not payems.empty and len(payems) > 1:
            chg = payems.iloc[-1] - payems.iloc[-2]
            st.metric("Payrolls Δ (1mo)", f"{chg:+,.0f}K")
        else:
            st.metric("Payrolls Δ", "N/A")

    st.caption("🚨 **Sahm Rule**: triggers recession when 3-mo avg UR rises ≥ 0.50 above its 12-mo low. "
               "Near-perfect historical record post-1970.")

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=["Unemployment Rate (%)",
                                        "Sahm Rule (trigger = 0.50)",
                                        "Nonfarm Payrolls — Monthly Δ (K)",
                                        "Initial Claims (Weekly)",
                                        "Continued Claims",
                                        ""],
                        vertical_spacing=0.10, horizontal_spacing=0.08)

    if not unrate.empty:
        fig.add_trace(go.Scatter(x=unrate.index, y=unrate.values,
                                 line=dict(color="firebrick")), row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not sahm.empty:
        colors = np.where(sahm.values >= 0.5, "crimson",
                  np.where(sahm.values >= 0.3, "orange", "steelblue"))
        fig.add_trace(go.Bar(x=sahm.index, y=sahm.values, marker_color=colors), row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="Trigger 0.50", row=1, col=2)
        maybe_shade(fig, row=1, col=2)
    if not payems_full.empty:
        chg = clip_range(payems_full.diff(), start_date, end_date)
        colors = np.where(chg.values >= 0, "seagreen", "indianred")
        fig.add_trace(go.Bar(x=chg.index, y=chg.values, marker_color=colors), row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    if not icsa.empty:
        fig.add_trace(go.Scatter(x=icsa.index, y=icsa.values,
                                 line=dict(color="darkorange")), row=2, col=2)
        maybe_shade(fig, row=2, col=2)
    if not ccsa.empty:
        fig.add_trace(go.Scatter(x=ccsa.index, y=ccsa.values,
                                 line=dict(color="steelblue")), row=3, col=1)
        maybe_shade(fig, row=3, col=1)

    fig.update_layout(height=850, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 4. INFLATION
# =========================================================================
elif section == "💰 Inflation":
    st.subheader("Inflation: CPI & PCE")

    cpi_full      = fetch_series("CPIAUCSL", fetch_start, end_date)
    core_cpi_full = fetch_series("CPILFESL", fetch_start, end_date)
    pce_full      = fetch_series("PCEPI",    fetch_start, end_date)
    core_pce_full = fetch_series("PCEPILFE", fetch_start, end_date)

    cpi_y      = clip_range(yoy(cpi_full),      start_date, end_date)
    core_cpi_y = clip_range(yoy(core_cpi_full), start_date, end_date)
    pce_y      = clip_range(yoy(pce_full),      start_date, end_date)
    core_pce_y = clip_range(yoy(core_pce_full), start_date, end_date)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CPI YoY",      latest_value(cpi_y,      "{:.2f}", "%"))
    with c2: st.metric("Core CPI YoY", latest_value(core_cpi_y, "{:.2f}", "%"))
    with c3: st.metric("PCE YoY",      latest_value(pce_y,      "{:.2f}", "%"))
    with c4: st.metric("Core PCE YoY", latest_value(core_pce_y, "{:.2f}", "%"))

    fig = go.Figure()
    for s, name, color in [(cpi_y, "CPI YoY", "crimson"),
                           (core_cpi_y, "Core CPI YoY", "darkorange"),
                           (pce_y, "PCE YoY", "steelblue"),
                           (core_pce_y, "Core PCE YoY", "darkgreen")]:
        if not s.dropna().empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     name=name, line=dict(color=color, width=2)))
    fig.add_hline(y=2, line_dash="dash", line_color="red",
                  annotation_text="Fed 2% target", annotation_position="right")
    fig.update_layout(title="Inflation Measures (YoY %)", yaxis_title="% YoY",
                      height=550, hovermode="x unified")
    maybe_shade(fig)
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 5. ACTIVITY & MARKETS — CFNAI, USSLIND, fixed-ref PMI proxy, Buffett
# =========================================================================
elif section == "🏭 Activity & Markets":
    st.subheader("Activity & Markets")

    st.info(
        "**Buffett Indicator** = `NCBEILQ027S` (Z.1 Corporate Equities, quarterly) ÷ "
        "`GDP` × 100%, ffilled to monthly. Valuation bands recalibrated to "
        f"post-{REF_START[:4]} percentile distribution of the Z.1 series."
    )

    cfnai = clip_range(fetch_series("CFNAI", fetch_start, end_date), start_date, end_date)
    usslind = clip_range(fetch_series("USSLIND", fetch_start, end_date), start_date, end_date)

    # PMI proxy with FIXED reference normalization (1990-2019)
    indpro_full = fetch_series("INDPRO", "1985-01-01", end_date)
    pmi_proxy = pd.Series(dtype=float)
    if not indpro_full.empty:
        chg3 = indpro_full.pct_change(3) * 100
        z = fixed_ref_zscore(chg3, REF_START, REF_END)
        pmi_full = (50 + 5 * z).rename("PMI-style proxy")
        pmi_proxy = clip_range(pmi_full, start_date, end_date)

    # Buffett indicator
    buffett_full = compute_buffett_indicator(start_date, end_date, fetch_start)
    buffett = clip_range(buffett_full, start_date, end_date)
    bands = buffett_percentile_bands(buffett_full)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CFNAI", latest_value(cfnai, "{:+.2f}"))
    with c2: st.metric("State LEI", latest_value(usslind, "{:+.2f}"))
    with c3: st.metric("PMI proxy (fixed-ref)", latest_value(pmi_proxy, "{:.1f}"))
    with c4: st.metric("Buffett (Z.1)", latest_value(buffett, "{:.1f}", "%"))

    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=["CFNAI — Chicago Fed National Activity Index",
                                        "USSLIND — State Leading Index",
                                        f"PMI-style Proxy (z-scored vs {REF_START[:4]}-{REF_END[:4]})",
                                        "Buffett Indicator — Z.1 Equities ÷ GDP"],
                        vertical_spacing=0.08)
    if not cfnai.empty:
        fig.add_trace(go.Scatter(x=cfnai.index, y=cfnai.values,
                                 line=dict(color="steelblue")), row=1, col=1)
        fig.add_hline(y=-0.7, line_dash="dot", line_color="red",
                      annotation_text="Recession threshold", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not usslind.empty:
        fig.add_trace(go.Scatter(x=usslind.index, y=usslind.values,
                                 line=dict(color="teal")), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    if not pmi_proxy.dropna().empty:
        fig.add_trace(go.Scatter(x=pmi_proxy.index, y=pmi_proxy.values,
                                 line=dict(color="darkorange")), row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="Neutral", row=3, col=1)
        maybe_shade(fig, row=3, col=1)
    if not buffett.dropna().empty:
        fig.add_trace(go.Scatter(x=buffett.index, y=buffett.values,
                                 line=dict(color="darkgreen", width=2)), row=4, col=1)
        if bands:
            for level, label, color in [
                (bands["p20"], f"P20 ({bands['p20']:.0f}%)", "green"),
                (bands["p50"], f"Median ({bands['p50']:.0f}%)", "gray"),
                (bands["p80"], f"P80 ({bands['p80']:.0f}%)", "orange"),
                (bands["p95"], f"P95 ({bands['p95']:.0f}%)", "red"),
            ]:
                fig.add_hline(y=level, line_dash="dot", line_color=color,
                              annotation_text=label, annotation_position="right",
                              row=4, col=1)
        maybe_shade(fig, row=4, col=1)

    fig.update_layout(height=1100, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')

    if bands and not buffett.dropna().empty:
        cur = buffett.dropna().iloc[-1]
        pct = (buffett_full.loc[REF_START:].dropna() <= cur).mean() * 100
        st.success(
            f"**Current Buffett Indicator: {cur:.1f}%** — that's the "
            f"**{pct:.0f}th percentile** of post-{REF_START[:4]} readings. "
            f"At extreme readings (>P95), 10-year forward S&P returns have historically "
            f"averaged ~0–3% annualized vs ~10% baseline."
        )


# =========================================================================
# 6. HOUSING (NEW)
# =========================================================================
elif section == "🏘️ Housing":
    st.subheader("Housing — Leading Indicator of the Cycle")
    st.caption("Housing typically leads the broader economy by 12–18 months. "
               "Permits down + mortgage rates high = consumer-led slowdown ahead.")

    permit = clip_range(fetch_series("PERMIT", fetch_start, end_date), start_date, end_date)
    houst  = clip_range(fetch_series("HOUST",  fetch_start, end_date), start_date, end_date)
    mort   = clip_range(fetch_series("MORTGAGE30US", fetch_start, end_date), start_date, end_date)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Building Permits (K)", latest_value(permit, "{:,.0f}"))
    with c2: st.metric("Housing Starts (K)",   latest_value(houst,  "{:,.0f}"))
    with c3: st.metric("30Y Mortgage Rate",    latest_value(mort,   "{:.2f}", "%"))

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=["Building Permits (Thousands SAAR)",
                                        "Housing Starts (Thousands SAAR)",
                                        "30-Year Fixed Mortgage Rate (%)"],
                        vertical_spacing=0.10)
    if not permit.empty:
        fig.add_trace(go.Scatter(x=permit.index, y=permit.values,
                                 line=dict(color="steelblue")), row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not houst.empty:
        fig.add_trace(go.Scatter(x=houst.index, y=houst.values,
                                 line=dict(color="teal")), row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    if not mort.empty:
        fig.add_trace(go.Scatter(x=mort.index, y=mort.values,
                                 line=dict(color="crimson")), row=3, col=1)
        maybe_shade(fig, row=3, col=1)

    fig.update_layout(height=850, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 7. CONSUMER (NEW)
# =========================================================================
elif section == "🛍️ Consumer":
    st.subheader("Consumer — 70% of US GDP")
    st.caption("Sentiment leads consumption. Retail sales YoY tracks the cycle. "
               "Personal savings rate flags consumer fragility.")

    umc = clip_range(fetch_series("UMCSENT", fetch_start, end_date), start_date, end_date)
    rsafs_full = fetch_series("RSAFS", fetch_start, end_date)
    rsafs_yoy = clip_range(yoy(rsafs_full), start_date, end_date)
    psavert = clip_range(fetch_series("PSAVERT", fetch_start, end_date), start_date, end_date)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Michigan Sentiment", latest_value(umc, "{:.1f}"))
    with c2: st.metric("Retail Sales YoY",    latest_value(rsafs_yoy, "{:+.2f}", "%"))
    with c3: st.metric("Personal Savings",   latest_value(psavert, "{:.2f}", "%"))

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=["University of Michigan Consumer Sentiment",
                                        "Retail Sales — YoY %",
                                        "Personal Savings Rate (%)"],
                        vertical_spacing=0.10)
    if not umc.empty:
        fig.add_trace(go.Scatter(x=umc.index, y=umc.values,
                                 line=dict(color="purple")), row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not rsafs_yoy.empty:
        fig.add_trace(go.Bar(x=rsafs_yoy.index, y=rsafs_yoy.values,
                             marker_color=np.where(rsafs_yoy.values >= 0,
                                                   "seagreen", "indianred")), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        maybe_shade(fig, row=2, col=1)
    if not psavert.empty:
        fig.add_trace(go.Scatter(x=psavert.index, y=psavert.values,
                                 line=dict(color="darkblue")), row=3, col=1)
        maybe_shade(fig, row=3, col=1)

    fig.update_layout(height=850, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 8. LIQUIDITY & MONEY (NEW)
# =========================================================================
elif section == "💵 Liquidity & Money":
    st.subheader("Money Supply & Fed Balance Sheet")
    st.caption("M2 YoY drives medium-term inflation and asset prices. "
               "Fed balance sheet (WALCL) signals QE/QT regime.")

    m2_full = fetch_series("M2SL", fetch_start, end_date)
    m2_yoy = clip_range(yoy(m2_full), start_date, end_date)
    walcl = clip_range(fetch_series("WALCL", fetch_start, end_date), start_date, end_date)

    c1, c2 = st.columns(2)
    with c1: st.metric("M2 YoY", latest_value(m2_yoy, "{:+.2f}", "%"))
    with c2:
        if not walcl.empty:
            st.metric("Fed Balance Sheet", f"${walcl.iloc[-1]/1e6:,.2f}T")
        else:
            st.metric("Fed Balance Sheet", "N/A")

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["M2 Money Supply — YoY %",
                                        "Fed Balance Sheet (WALCL, $ Millions)"],
                        vertical_spacing=0.12)
    if not m2_yoy.empty:
        fig.add_trace(go.Bar(x=m2_yoy.index, y=m2_yoy.values,
                             marker_color=np.where(m2_yoy.values >= 0,
                                                   "seagreen", "indianred")), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        maybe_shade(fig, row=1, col=1)
    if not walcl.empty:
        fig.add_trace(go.Scatter(x=walcl.index, y=walcl.values,
                                 line=dict(color="darkblue", width=2),
                                 fill="tozeroy",
                                 fillcolor="rgba(0,0,139,0.1)"), row=2, col=1)
        maybe_shade(fig, row=2, col=1)

    fig.update_layout(height=700, showlegend=False, hovermode="x unified")
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# =========================================================================
# 9. GDP & FED FUNDS
# =========================================================================
elif section == "🇺🇸 GDP & Fed Funds":
    st.subheader("GDP & Federal Funds Rate")

    gdp_n_full = fetch_series("GDP", fetch_start, end_date)
    gdp_r_full = fetch_series("GDPC1", fetch_start, end_date)
    fedfn_full = fetch_series("FEDFUNDS", fetch_start, end_date)

    gdp_n = clip_range(gdp_n_full, start_date, end_date)
    gdp_r = clip_range(gdp_r_full, start_date, end_date)
    fedfn = clip_range(fedfn_full, start_date, end_date)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Nominal GDP", f"${gdp_n.iloc[-1]/1000:,.2f}T" if not gdp_n.empty else "N/A")
    with c2: st.metric("Real GDP",    f"${gdp_r.iloc[-1]/1000:,.2f}T" if not gdp_r.empty else "N/A")
    with c3:
        if not gdp_r.empty and len(gdp_r) > 4:
            yoy_r = (gdp_r.iloc[-1] / gdp_r.iloc[-5] - 1) * 100
            st.metric("Real GDP YoY", f"{yoy_r:+.2f}%")
        else:
            st.metric("Real GDP YoY", "N/A")
    with c4: st.metric("Fed Funds Rate", latest_value(fedfn, "{:.2f}", "%"))

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=["GDP Levels (Billions $)",
                                        "Real GDP YoY Growth (%)",
                                        "Federal Funds Effective Rate (%)"],
                        vertical_spacing=0.10)
    if not gdp_n.empty:
        fig.add_trace(go.Scatter(x=gdp_n.index, y=gdp_n.values, name="Nominal GDP",
                                 line=dict(color="steelblue", width=2)), row=1, col=1)
    if not gdp_r.empty:
        fig.add_trace(go.Scatter(x=gdp_r.index, y=gdp_r.values, name="Real GDP",
                                 line=dict(color="darkgreen", width=2, dash="dot")), row=1, col=1)
    maybe_shade(fig, row=1, col=1)

    if not gdp_r_full.empty:
        gdp_r_yoy = clip_range(gdp_r_full.pct_change(4) * 100, start_date, end_date)
        fig.add_trace(go.Bar(x=gdp_r_yoy.index, y=gdp_r_yoy.values, name="Real GDP YoY",
                             marker_color=np.where(gdp_r_yoy.values >= 0, "seagreen", "indianred")),
                      row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        maybe_shade(fig, row=2, col=1)

    if not fedfn.empty:
        fig.add_trace(go.Scatter(x=fedfn.index, y=fedfn.values, name="Fed Funds",
                                 line=dict(color="darkred", width=2),
                                 fill="tozeroy", fillcolor="rgba(139,0,0,0.10)"), row=3, col=1)
        maybe_shade(fig, row=3, col=1)

    fig.update_layout(height=900, hovermode="x unified", showlegend=True)
    apply_xrange(fig, start_date, end_date)
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit • Data from FRED & Federal Reserve Board • "
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
    "⚠️ This tool is a **risk monitor**, not a forecaster. No dashboard predicts "
    "exogenous shocks (pandemics, wars, banking crises). Use as one input among many."
)