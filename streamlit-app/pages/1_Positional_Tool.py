import os
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import statsmodels.api as sm
from utils.positions import apply_position_overrides
from utils.ui import inject_responsive_layout, render_page_header

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
# CONFIG - NRLW Data
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

DISPLAY_LABELS = {
    "High Speed Running": "HSR",
}


def display_metric(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


PRIORITY = {
    "High Speed Running": 0.3,
    "VHSR": 0.6,
    "Accel Efforts": 1.0,
    "Accel Distance": 0.9,
    "Decel Efforts": 1.25,
    "Decel Distance": 1.1
}

MINUTES_FILTER = 35

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="small")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
inject_responsive_layout()

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
    min-width: 120px;
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
    padding-top: clamp(2rem, 4vw, 3rem) !important;
    max-width: min(1180px, 100%) !important;
    margin: 0 auto;
}
h1, h2, h3 {
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header Layout ----------
render_page_header("Positional Top Ups", "Wahs.png", "NRLW Logo.png", heading="h1")

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

# Apply =35 min filter
df = raw_df[raw_df[COLS["minutes"]] >= MINUTES_FILTER].copy()
if df.empty:
    st.error(f"No rows with {COLS['minutes']} = {MINUTES_FILTER}.")
    st.stop()

slopes_by_position = compute_univariate_slopes(df)
if not slopes_by_position:
    st.error("No slopes computed. Check your sheet headers and data.")
    st.stop()

# ---------- Step 1 ----------
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 1 - Choose Position")

pos_select_col, _ = st.columns([1, 3])
with pos_select_col:
    position = st.selectbox(
        "Position",
        sorted(slopes_by_position.keys()),
        help="Pick the player's position. Each position has its own N/m per-unit slopes."
    )
SLOPES = slopes_by_position[position]

slopes_df = pd.DataFrame([SLOPES], index=[position])[METRICS].reset_index()
slopes_df.columns = ["Position"] + [display_metric(m) for m in METRICS]
render_table(slopes_df)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('---')

# ---------- Step 2 ----------
st.markdown('<div class="step-box step2-box">', unsafe_allow_html=True)
st.markdown("### Step 2 - Target & Inputs")

target_col, _ = st.columns([1, 3])
with target_col:
    target_nm = st.number_input(
        "Target N/m",
        min_value=0.0, value=1.0, step=0.1, format="%.1f",
        help="Set the N/m load you want to reach. Example: enter 1.0 to plan a +1 N/m top-up session.",
        label_visibility="collapsed"
    )

pos_slopes = {m: max(0.0, SLOPES[m]) for m in METRICS}
sum_pos = sum(pos_slopes.values())
default_weights = {m: round(pos_slopes[m] / sum_pos, 2) for m in METRICS} if sum_pos > 0 else {m: round(1.0 / len(METRICS), 2) for m in METRICS}

st.markdown("#### Weights & Current Metrics")

st.markdown(
    """
<style>
.step2-box [data-testid=\"stDataEditor\"] table thead th,
.step2-box [data-testid=\"stDataEditor\"] table tbody td {
    text-align: center !important;
    vertical-align: middle !important;
}
.step2-box [data-testid=\"stDataEditor\"] table colgroup col:first-child {
    width: 120px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

initial_rows = [
    {"Metric": display_metric(m), "Weight": float(default_weights[m]), "Current": 0.0}
    for m in METRICS
]
cache_key = f"step2_editor_{position}"
if "_step2_editor_state" not in st.session_state:
    st.session_state._step2_editor_state = {}
if cache_key not in st.session_state._step2_editor_state:
    st.session_state._step2_editor_state[cache_key] = pd.DataFrame(initial_rows)
else:
    stored = st.session_state._step2_editor_state[cache_key]
    aligned = pd.DataFrame(initial_rows)
    for label in aligned["Metric"]:
        if label in stored["Metric"].values:
            row = stored.loc[stored["Metric"] == label].iloc[0]
            aligned.loc[aligned["Metric"] == label, "Weight"] = float(row.get("Weight", 0.0))
            aligned.loc[aligned["Metric"] == label, "Current"] = float(row.get("Current", 0.0))
    st.session_state._step2_editor_state[cache_key] = aligned

data_editor_df = st.data_editor(
    st.session_state._step2_editor_state[cache_key],
    column_config={
        "Metric": st.column_config.TextColumn("Metric", width="small", help="Metric name (read-only)."),
        "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", help="Controls how much of the top-up is assigned to each metric. Defaults are proportional to slope efficiency."),
        "Current": st.column_config.NumberColumn("Current", min_value=0.0, step=1.0, format="%.0f", help="Enter the player's current total for each metric.")
    },
    disabled=["Metric"],
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    key=f"weights_current_editor_{position}"
)
# Persist edits for this position
st.session_state._step2_editor_state[cache_key] = data_editor_df.copy()

weight_inputs, current_inputs = {}, {}
label_to_metric = {display_metric(m): m for m in METRICS}
for _, row in data_editor_df.iterrows():
    metric = label_to_metric[row["Metric"]]
    weight_inputs[metric] = max(0.0, min(1.0, float(row.get("Weight", 0.0))))
    current_inputs[metric] = max(0.0, float(row.get("Current", 0.0)))

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('---')

# ---------- Step 3 ----------
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 3 - Results")

display_options = [display_metric(m) for m in METRICS]
preselected = [display_metric(m) for m in METRICS if SLOPES[m] > 0]
selected_display = st.multiselect(
    "Select metrics to use for top-up",
    display_options,
    default=preselected,
    help="Choose which metrics you want to use for topping-up."
)
selected_metrics = [METRICS[display_options.index(label)] for label in selected_display]

current_contrib = sum(current_inputs[m] * SLOPES[m] for m in METRICS)
need = target_nm - current_contrib

valid_metrics = [m for m in selected_metrics if SLOPES[m] > 0]
sum_w = sum(weight_inputs[m] for m in valid_metrics) if valid_metrics else 0.0
alloc_weights = {m: (weight_inputs[m] / sum_w if sum_w > 0 else 0.0) for m in valid_metrics}

MIN_SLOPE = 0.01  # or tweak (try 0.005 or 0.02)

safe_slopes = {m: max(SLOPES[m], MIN_SLOPE) for m in METRICS}

top_up = {m: 0.0 for m in METRICS}
if need > 0 and valid_metrics:
    for m in valid_metrics:
        nm_piece = need * alloc_weights[m]
        top_up[m] = nm_piece / safe_slopes[m]

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
if not results_df.empty:
    results_df["Metric"] = results_df["Metric"].map(display_metric)

c1, c2 = st.columns([2, 1])
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
        ignored = ", ".join(display_metric(m) for m in bad)
        st.info("These selected metrics have <= 0 slopes and were ignored: " + ignored)

st.markdown("</div>", unsafe_allow_html=True)














