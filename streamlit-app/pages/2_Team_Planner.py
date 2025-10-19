# app_team_planner_dual_anchor.py — Trial Version
# - Step 2 now uses % targets (Max Speed, Total Distance, N/m) relative to exemplar
# - Tooltips added to Step 2 inputs
# - Exemplar row fully italicized
# - Includes Aerobic (m) metric
# - Renames High Speed Running → HSR for display

import pandas as pd
import numpy as np
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
    num_cols = [
        COLS["minutes"], COLS["max_speed"], COLS["total_dist"], COLS["zone_28"], COLS["hmld"],
        COLS["hsr"], COLS["vhsr"], COLS["accel_eff"], COLS["accel_dist"],
        COLS["decel_eff"], COLS["decel_dist"], COLS["target"]
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if COLS["zone_28"] in df.columns and COLS["hsr"] in df.columns:
        df[COLS["aerobic"]] = df[COLS["zone_28"]] - df[COLS["hsr"]]
    df = apply_position_overrides(df, position_col=COLS["position"])
    return df.dropna(subset=[COLS["position"], COLS["minutes"]])

# =========================
# CONFIG — NRLW Data
# =========================
SHEET_ID = "1JqDkmbXDCWyNjpsMffbQyi6pSXQWOEm4lAf6cvmEhM4"
WORKSHEET_NAME = "NRLW_ALL"
RANGE_A1 = "B2:AZ"

COLS = {
    "position": "Position",
    "minutes": "Match Minutes",
    "max_speed": "Max Speed",
    "total_dist": "Total Distance",
    "aerobic": "Aerobic (m)",
    "zone_28": "Distance Zone 2-8 (m)",
    "hmld": "HMLD",
    "hsr": "High Speed Running",  # keep for sheet
    "vhsr": "VHSR",
    "accel_eff": "Accel Efforts",
    "accel_dist": "Accel Distance",
    "decel_eff": "Decel Efforts",
    "decel_dist": "Decel Distance",
    "target": "N/m",
    "venue": "Venue",
    "result": "Result",
}

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

MINUTES_FILTER = 35

LIT_METRICS = [
    COLS["max_speed"], COLS["total_dist"], COLS["aerobic"], COLS["zone_28"], COLS["hmld"]
]
HIT_METRICS = [
    COLS["hsr"], COLS["vhsr"], COLS["accel_eff"], COLS["accel_dist"],
    COLS["decel_eff"], COLS["decel_dist"]
]

DISPLAY_NAMES = {
    COLS["hsr"]: "HSR",
    COLS["max_speed"]: "Max Speed",
    COLS["total_dist"]: "Total Distance",
    COLS["aerobic"]: "Aerobic (m)",
    COLS["zone_28"]: "Distance Zone 2-8 (m)",
    COLS["hmld"]: "HMLD",
    COLS["vhsr"]: "VHSR",
    COLS["accel_eff"]: "Accel Efforts",
    COLS["accel_dist"]: "Accel Distance",
    COLS["decel_eff"]: "Decel Efforts",
    COLS["decel_dist"]: "Decel Distance",
    COLS["target"]: "N/m",
}

EXPORT_COLUMN_MAP = {
    COLS["max_speed"]: "Maximum Velocity",
    COLS["total_dist"]: "Total Distance",
    COLS["hmld"]: "HMLD (Gen 2)",
    COLS["hsr"]: "High Speed Distance",
    COLS["vhsr"]: "Very High Speed Distance",
    COLS["accel_eff"]: "Acceleration Bands 1-3 Efforts",
    COLS["decel_eff"]: "Deceleration Bands 1-3 Efforts",
}

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    # Place logo at the top with centered alignment
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
  #  st.logo(link="https://upload.wikimedia.org/wikipedia/en/thumb/5/5b/Warriors_%28NRL%29_Logo.svg/1200px-Warriors_%28NRL%29_Logo.svg.png",size="large")
  #  st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/ff/OneNZ_2023.svg/1200px-OneNZ_2023.svg.png", use_container_width=True)
# =========================
# PAGE STYLE
# =========================

st.set_page_config(page_title="Team Planner", layout="wide")

# TABLE BOARDERS # 
st.markdown("""
<style>
/* ===== Rounded Edges + Scrollable + Full-Width Tables ===== */

/* Main table styling */
table {
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.1);    /* existing subtle outline */
    border-radius: 12px;                  /* soft corner radius */
    overflow: hidden;
    width: 100% !important;               /* full width alignment */
}

/* Keep internal borders intact */
table td, table th {
    border: 1px solid rgba(0,0,0,0.1);
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;                  /* maintain layout for scrolling */
}

/* Soft header background matching theme */
table thead th {
    background-color: var(--secondary-background-color, #e5e5f3);
}

/* Subtle zebra striping for readability */
table tbody tr:nth-child(even) td {
    background-color: rgba(0,0,0,0.05);
}

/* Scrollable container for wide/tall tables */
div[data-testid="stMarkdownContainer"] table {
    display: block;                       /* enables scrollbars */
    overflow-x: auto;                     /* horizontal scroll */
    overflow-y: auto;                     /* vertical scroll */
    max-height: 70vh;                     /* vertical scroll window */
}

/* Exclude Season Set Up data editor from these rules */
[data-testid="stDataEditorContainer"] table {
    width: auto !important;
    display: table !important;
    overflow: visible !important;
    max-height: none !important;
}
</style>
""", unsafe_allow_html=True)


# TITLE #
st.markdown("""
<style>
/* Align Season Set Up page with others */
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
    st.markdown(
        "<h1 style='text-align:center; color:#262C68;'>Team Planner</h1>",
        unsafe_allow_html=True
    )
with col3:
    st.image("NRLW Logo.png", width=85)


st.markdown("---")

def render_table(df: pd.DataFrame, index=False):
    df_disp = df.rename(columns=DISPLAY_NAMES)
    return st.markdown(
        df_disp.to_html(index=index, justify="center", classes="compact-table", escape=False),
        unsafe_allow_html=True
    )

raw_df = fetch_sheet_df(SHEET_ID, WORKSHEET_NAME, RANGE_A1)
df = raw_df[raw_df[COLS["minutes"]] >= MINUTES_FILTER].copy()

# =========================
# EXEMPLAR & CONTEXT FILTERS
# =========================

st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 1 · Exemplar & Relative Scaling")

left, mid, right = st.columns(3)
with left:
    exemplar_type = st.radio(
        "Exemplar type",
        options=["Typical (P50)", "High-Day (P75)"],
        index=1
    )
with mid:
    chosen_venue = st.selectbox("Venue", options=["All"] + sorted(df[COLS["venue"]].dropna().unique()), index=0)
with right:
    chosen_result = st.selectbox("Result", options=["All"] + sorted(df[COLS["result"]].dropna().unique()), index=0)

df_ctx = df.copy()
if chosen_venue != "All":
    df_ctx = df_ctx[df_ctx[COLS["venue"]] == chosen_venue]
if chosen_result != "All":
    df_ctx = df_ctx[df_ctx[COLS["result"]] == chosen_result]

def get_exemplar_val(series: pd.Series) -> float:
    return series.median(skipna=True) if exemplar_type.startswith("Typical") else series.quantile(0.75)

exemplar = {m: get_exemplar_val(df_ctx[m]) for m in LIT_METRICS + HIT_METRICS + [COLS["target"]]}

# Relative % scaling
pos_list = sorted(p for p in df[COLS["position"]].dropna().unique() if str(p).strip())
rel_scaling = {}
for pos in pos_list:
    pos_df = df_ctx[df_ctx[COLS["position"]] == pos]
    if pos_df.empty: 
        continue
    rel_scaling[pos] = {m: (pos_df[m].mean() / exemplar[m] * 100 if exemplar[m] > 0 else np.nan)
                        for m in LIT_METRICS + HIT_METRICS + [COLS["target"]]}

# Step 1 table
rel_rows = []
for pos in pos_list:
    row = {"Position": pos}
    for m in LIT_METRICS + HIT_METRICS + [COLS["target"]]:
        val = rel_scaling.get(pos, {}).get(m, np.nan)
        row[m] = f"{val:.0f}%" if pd.notnull(val) else "—"
    rel_rows.append(row)

rel_df = pd.DataFrame(rel_rows, columns=["Position"] + LIT_METRICS + HIT_METRICS + [COLS["target"]])
render_table(rel_df)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# TARGET INPUTS (PERCENTAGES WITH TOOLTIP)
# ==========================
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 2 · Set Team Targets (% of Exemplar)")

c1, c2, c3 = st.columns(3)
with c1:
    pct_ms = st.number_input(
        "Max Speed %",
        min_value=0.0,
        value=100.0,
        step=5.0,
        format="%.0f",
        help="Set the target maximum speed as a % of exemplar. \
E.g. 110% means players should aim to exceed exemplar sprint speeds by 10%."
    )
with c2:
    pct_dist = st.number_input(
        "Total Distance %",
        min_value=0.0,
        value=100.0,
        step=5.0,
        format="%.0f",
        help="Set the overall running load for the team as a % of exemplar total distance. \
This scales all low-intensity metrics (Aerobic, Zone 2–8, HMLD)."
    )
with c3:
    pct_nm = st.number_input(
        "N/m %",
        min_value=0.0,
        value=100.0,
        step=5.0,
        format="%.0f",
        help="Set the mechanical load per minute (N/m) as a % of exemplar. \
This scales all high-intensity metrics (HSR, VHSR, accel/decel)."
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
# Convert % to absolute anchors
team_target_ms = exemplar[COLS["max_speed"]] * (pct_ms / 100.0)
team_target_dist = exemplar[COLS["total_dist"]] * (pct_dist / 100.0)
team_target_nm = exemplar[COLS["target"]] * (pct_nm / 100.0)

ms_scale = pct_ms / 100.0 if pct_ms is not None else 1.0

# ===========================
# STEP 3 Team Session Targets
# ============================
st.markdown('<div class="step-box">', unsafe_allow_html=True)
st.markdown("### Step 3 · Set Team Session Targets")

ms_col = COLS["max_speed"]
max_speed_ctx = (
    df_ctx.groupby(COLS["position"])[ms_col]
    .max()
    .dropna()
    .to_dict()
)
max_speed_all = (
    df.groupby(COLS["position"])[ms_col]
    .max()
    .dropna()
    .to_dict()
)

rows_out = []
for pos in pos_list:
    row = {"Position": pos}

    # Max Speed
    ms_value = max_speed_ctx.get(pos)
    if ms_value is None or pd.isna(ms_value):
        ms_value = max_speed_all.get(pos)
    if pd.notnull(ms_value):
        row[ms_col] = round(float(ms_value) * ms_scale, 1)
    else:
        row[ms_col] = np.nan

    # Other LIT
    dist_ratio = (team_target_dist / exemplar[COLS["total_dist"]]) if exemplar[COLS["total_dist"]] > 0 else np.nan
    for m in [c for c in LIT_METRICS if c != ms_col]:
        pos_scale = rel_scaling.get(pos, {}).get(m, np.nan)
        if pd.notnull(pos_scale) and pd.notnull(exemplar[m]) and pd.notnull(dist_ratio):
            row[m] = int(round(exemplar[m] * dist_ratio * (pos_scale / 100.0)))
        else:
            row[m] = np.nan

    # HIT
    nm_ratio = (team_target_nm / exemplar[COLS["target"]]) if exemplar[COLS["target"]] > 0 else np.nan
    for m in HIT_METRICS:
        pos_scale = rel_scaling.get(pos, {}).get(m, np.nan)
        if pd.notnull(pos_scale) and pd.notnull(exemplar[m]) and pd.notnull(nm_ratio):
            row[m] = int(round(exemplar[m] * nm_ratio * (pos_scale / 100.0)))
        else:
            row[m] = np.nan

    target_scale = rel_scaling.get(pos, {}).get(COLS["target"], np.nan)
    row["N/m"] = round(team_target_nm * (target_scale / 100.0), 1) if pd.notnull(target_scale) else np.nan

    rows_out.append(row)

col_order = ["Position"] + LIT_METRICS + HIT_METRICS + ["N/m"]
position_targets_df = pd.DataFrame(rows_out, columns=col_order)

# Exemplar row italics
exp_row = {"Position": "<i>Exemplar</i>"}
exp_row[COLS["max_speed"]] = f"<i>{round(exemplar[COLS['max_speed']] * ms_scale, 1)}</i>"
for m in [c for c in LIT_METRICS if c != COLS["max_speed"]] + HIT_METRICS:
    exp_row[m] = f"<i>{int(round(exemplar[m]))}</i>"
exp_row["N/m"] = f"<i>{round(exemplar[COLS['target']], 1)}</i>"

display_df = position_targets_df.copy()
out_df = pd.concat([display_df, pd.DataFrame([exp_row], columns=col_order)], ignore_index=True)
render_table(out_df)
st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# STEP 4 Export Template
# ============================
st.markdown("---")
st.markdown("### Step 4 · Export Targets to Athlete Template")

uploaded_template = st.file_uploader(
    "Upload athlete template (.csv)",
    type="csv",
    help="Set up Athlete Thresholds in OpenField and include **Max Vel, Dist, HMLD, HSR, VHSR, Acc1-3 Eff** and **Decel1-3 Eff**. """
         "Export the CSV and upload it here. "

         "Then import back into OpenField to set Athlete Thresholds for the session"
)

processed_template = None
template_df = None
if uploaded_template is not None:
    try:
        template_df = pd.read_csv(uploaded_template)
        processed_template = apply_position_overrides(template_df.copy(), position_col="Position")
    except Exception as exc:
        st.error(f"Unable to read template: {exc}")
        processed_template = None

if processed_template is not None:
    if position_targets_df.empty:
        st.warning("No positional targets available to apply.")
    elif "Position" not in processed_template.columns:
        st.warning("Uploaded template must include a Position column.")
    else:
        lookup = (
            position_targets_df.set_index("Position")[list(EXPORT_COLUMN_MAP.keys())]
            .apply(pd.to_numeric, errors="coerce")
            .to_dict("index")
        )
        updated_template = template_df.copy()
        filled_rows = 0

        normalized_positions = processed_template["Position"].astype(str).str.strip().str.lower()
        position_map = {p.lower(): p for p in lookup.keys()}

        for idx, pos_value in normalized_positions.items():
            if not pos_value:
                continue
            original_key = position_map.get(pos_value)
            if original_key is None:
                continue
            metrics = lookup.get(original_key, {})
            row_updated = False
            for metric_col, template_col in EXPORT_COLUMN_MAP.items():
                if template_col not in updated_template.columns:
                    continue
                value = metrics.get(metric_col)
                if pd.isna(value):
                    continue
                if metric_col == COLS["max_speed"]:
                    formatted = round(float(value), 1)
                else:
                    formatted = int(round(float(value)))
                updated_template.at[idx, template_col] = formatted
                row_updated = True
            if row_updated:
                filled_rows += 1

        if filled_rows == 0:
            st.info("No rows in the template matched the positions above.")
        else:
            st.success(f"Applied team targets to {filled_rows} athletes.")
            preview_columns = [
                c for c in ["Athlete Name", "Position"] + list(EXPORT_COLUMN_MAP.values())
                if c in updated_template.columns
            ]
            if preview_columns:
                st.dataframe(updated_template[preview_columns].head(20))

            download_bytes = updated_template.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download updated template",
                data=download_bytes,
                file_name="team_session_targets_updated.csv",
                mime="text/csv"
            )

