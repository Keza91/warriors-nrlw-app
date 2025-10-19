import os
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import statsmodels.api as sm
from utils.positions import apply_position_overrides

def get_gsheet_client():
    key = "_gsheet_client"
    if key not in st.session_state:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        st.session_state[key] = gspread.authorize(creds)
    return st.session_state[key]

@st.cache_data(show_spinner=True)
def fetch_sheet_df(sheet_id: str, worksheet_name: str, range_a1: str) -> pd.DataFrame:
    gc = get_gsheet_client()
    ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    values = ws.get(range_a1)
    if not values:
        raise ValueError("No data returned from Google Sheets.")
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    # convert numeric columns and compute Aerobic (m)
    for col in [
        COLS["minutes"], COLS["hsr"], COLS["vhsr"],
        COLS["accel_eff"], COLS["accel_dist"],
        COLS["decel_eff"], COLS["decel_dist"], COLS["target"]
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = apply_position_overrides(df, position_col=COLS["position"])
    return df.dropna(subset=[COLS["position"], COLS["target"], COLS["minutes"]])

# =========================
# CONFIG — NRLW Data
# =========================
SHEET_ID = "1JqDkmbXDCWyNjpsMffbQyi6pSXQWOEm4lAf6cvmEhM4"
WORKSHEET_NAME = "NRLW_ALL"
RANGE_A1 = "B2:AZ"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

COLS = {
    "position": "Position",
    "minutes": "Match Minutes",   
    "hsr": "High Speed Running",  
    "vhsr": "VHSR",
    "accel_eff": "Accel Efforts",
    "accel_dist": "Accel Distance",
    "decel_eff": "Decel Efforts",
    "decel_dist": "Decel Distance",
    "target": "N/m",              
}



METRICS = [
    "High Speed Running",
    "VHSR",
    "Accel Efforts",
    "Accel Distance",
    "Decel Efforts",
    "Decel Distance",
]

METRIC_TO_COL = {
    "High Speed Running": COLS["hsr"],
    "VHSR": COLS["vhsr"],
    "Accel Efforts": COLS["accel_eff"],
    "Accel Distance": COLS["accel_dist"],
    "Decel Efforts": COLS["decel_eff"],
    "Decel Distance": COLS["decel_dist"],
}

MINUTES_FILTER = 35

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
# =========================
# PAGE STYLE
# =========================

# TABLE BORDERS #
st.markdown("""
<style>
/* --- Rounded outer edges only --- */
table {
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;         /* soft corner radius */
    overflow: hidden;
}

/* Keep internal borders intact */
table td, table th {
    border: 1px solid rgba(0,0,0,0.1);
}

/* Soft header background matching theme */
table thead th {
    background-color: var(--secondary-background-color, #e5e5f3);
}

/* Subtle zebra striping for readability */
table tbody tr:nth-child(even) td {
    background-color: rgba(0,0,0,0.05);
}

/* Scrollable and full-width container */
div[data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    display: block;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 70vh;
    table-layout: fixed;
}

/* Wider column spacing */
table th, table td {
    padding: 10px 20px;
    min-width: 160px;
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# TITLE #
st.markdown("""
<style>
.block-container {
    padding-top: 3rem !important;
    max-width: 1180px;
    margin: auto;
}
h1, h2, h3 {
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header Layout ----------
col1, col2, col3 = st.columns([0.5, 4, 0.5])
with col1:
    st.image("Wahs.png", width=85)
with col2:
    st.markdown("<h1 style='text-align:center; color:#262C68;'>Positional Top Ups</h1>", unsafe_allow_html=True)
with col3:
    st.image("NRLW Logo.png", width=85)

st.markdown("---")

# =========================
# HELPERS
# =========================
def render_table(df: pd.DataFrame, index=False):
    html = df.to_html(index=index, justify="center", classes="results-table", escape=False)
    st.markdown(html, unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def compute_univariate_slopes(df: pd.DataFrame) -> dict:
    slopes_by_pos = {}
    for pos, d in df.groupby(COLS["position"]):
        slopes = {}
        for m in METRICS:
            col = METRIC_TO_COL[m]
            d_valid = d.dropna(subset=[col, COLS["target"]])
            if len(d_valid) > 2:
                X = sm.add_constant(d_valid[col], has_constant="add")
                y = d_valid[COLS["target"]]
                model = sm.OLS(y, X).fit()
                slopes[m] = float(model.params.get(col, 0.0))
            else:
                slopes[m] = 0.0
        slopes_by_pos[pos] = slopes
    return slopes_by_pos

# =========================
# MAIN
# =========================
raw_df = fetch_sheet_df(SHEET_ID, WORKSHEET_NAME, RANGE_A1)

# Apply ≥35 min filter
df = raw_df[raw_df[COLS["minutes"]] >= MINUTES_FILTER].copy()
if df.empty:
    st.error(f"No rows with {COLS['minutes']} ≥ {MINUTES_FILTER}.")
    st.stop()

slopes_by_position = compute_univariate_slopes(df)
if not slopes_by_position:
    st.error("No slopes computed. Check your sheet headers and data.")
    st.stop()

# ---------- Step 1 ----------
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 1 · Choose Position")

position = st.selectbox(
    "Position",
    sorted(slopes_by_position.keys()),
    help="Pick the player’s position. Each position has its own N/m per-unit slopes."
)
SLOPES = slopes_by_position[position]

slopes_df = pd.DataFrame([SLOPES], index=[position])[METRICS].reset_index()
slopes_df.columns = ["Position"] + METRICS
render_table(slopes_df)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Step 2 ----------
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 2 · Target & Inputs")

st.markdown('<div class="target-input-box">', unsafe_allow_html=True)
target_nm = st.number_input(
    "Target N/m",
    min_value=0.0, value=1.0, step=0.1, format="%.1f",
    help="Set the N/m load you want to reach. Example: enter 1.0 to plan a +1 N/m top-up session.",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

pos_slopes = {m: max(0.0, SLOPES[m]) for m in METRICS}
sum_pos = sum(pos_slopes.values())
default_weights = {m: round(pos_slopes[m] / sum_pos, 2) for m in METRICS} if sum_pos > 0 else {m: round(1.0 / len(METRICS), 2) for m in METRICS}

st.markdown("#### Weights & Current Metrics")

cols = st.columns([3, 2, 2])
with cols[0]:
    st.markdown("**Metric**")
with cols[1]:
    st.markdown("**Weight**", help="Controls how much of the top-up is assigned to each metric. Defaults are proportional to slope efficiency.")
with cols[2]:
    st.markdown("**Current**", help="Enter the player’s current total for each metric.")

weight_inputs, current_inputs = {}, {}
for m in METRICS:
    c = st.columns([3, 2, 2])
    with c[0]:
        st.markdown(m)
    with c[1]:
        weight_inputs[m] = st.number_input(
            f"Weight {m}", min_value=0.0, max_value=1.0, step=0.01,
            value=float(default_weights[m]), label_visibility="collapsed", key=f"w_{m}"
        )
    with c[2]:
        current_inputs[m] = st.number_input(
            f"Current {m}", min_value=0.0, step=1.0,
            value=0.0, label_visibility="collapsed", key=f"c_{m}"
        )
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Step 3 ----------
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 3 · Results")

selected_metrics = st.multiselect(
    "Select metrics to use for top-up",
    METRICS,
    default=[m for m in METRICS if SLOPES[m] > 0],
    help="Choose which metrics you want to use for topping-up."
)

current_contrib = sum(current_inputs[m] * SLOPES[m] for m in METRICS)
need = target_nm - current_contrib

valid_metrics = [m for m in selected_metrics if SLOPES[m] > 0]
sum_w = sum(weight_inputs[m] for m in valid_metrics) if valid_metrics else 0.0
alloc_weights = {m: (weight_inputs[m] / sum_w if sum_w > 0 else 0.0) for m in valid_metrics}

top_up = {m: 0.0 for m in METRICS}
if need > 0 and valid_metrics:
    for m in valid_metrics:
        nm_piece = need * alloc_weights[m]
        top_up[m] = nm_piece / SLOPES[m]

rows = []
for m in valid_metrics:
    rows.append({
        "Metric": m,
        "Current": int(round(current_inputs[m])),
        "Top-Up Needed": int(round(top_up[m])),
        "New Total": int(round(current_inputs[m] + top_up[m])),
        "N/m Gain from Top-Up": round(top_up[m] * SLOPES[m], 1),
    })
results_df = pd.DataFrame(rows, columns=["Metric", "Current", "Top-Up Needed", "New Total", "N/m Gain from Top-Up"])

c1, c2 = st.columns([3, 1])
with c1:
    st.metric("Current Contribution", f"{current_contrib:.1f} N/m")
with c2:
    st.metric("Remaining to Target", f"{need:.1f} N/m")

render_table(results_df)

achieved_total = current_contrib + sum(top_up[m] * SLOPES[m] for m in valid_metrics)
st.markdown('<div class="achieved-metric">', unsafe_allow_html=True)
st.metric("Achieved N/m after Top-Up", f"{achieved_total:.1f} N/m")
st.markdown('</div>', unsafe_allow_html=True)

if not valid_metrics:
    st.warning("No valid metrics selected. Nothing to allocate.")
else:
    bad = [m for m in selected_metrics if SLOPES[m] <= 0]
    if bad:
        st.info("These selected metrics have ≤ 0 slopes and were ignored: " + ", ".join(bad))

st.markdown("</div>", unsafe_allow_html=True)
