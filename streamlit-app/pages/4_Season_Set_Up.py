import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from google.oauth2.service_account import Credentials
import gspread
from collections import Counter
from utils.ui import inject_responsive_layout, render_page_header

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    # Place logo at the top with centered alignment
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Season Set Up", layout="wide")

inject_responsive_layout()

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
render_page_header("Season Set Up", "Wahs.png", "NRLW Logo.png", heading="h1")

st.markdown("---")
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
    if not values or len(values) < 2:
        raise ValueError("No data returned from Google Sheets.")
    header, rows = values[0], values[1:]
    df = pd.DataFrame(rows, columns=header)
    # convert numeric columns and compute Aerobic (m)
    return df

# ---------------------- CONSTANTS ------------------------
SHEET_ID = "1JqDkmbXDCWyNjpsMffbQyi6pSXQWOEm4lAf6cvmEhM4"
WORKSHEET_NAME = "NRLW_ALL"
RANGE_A1 = "B2:AZ"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

PLAN_CSV = Path("season_plan.csv")

PHASE_OPTIONS = ["Nil", "Accumulation", "Intensification", "Realisation", "Deload"]
DAY_TYPES = ["Off", "D1", "D2", "D3", "D4", "D5 (Captain's Run)", "Match", "NON"]
NON_REPEATABLE = {"D1", "D2", "D3", "D4", "D5 (Captain's Run)", "Match"}
WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TRAINING_DAY_CANON = ["D1", "D2", "D3", "D4", "D5 (Captain's Run)"]

FIRST_COL_WIDTH = 150
METRIC_COL_WIDTH = 130

# ---------------------- HELPERS --------------------------
def fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")

def monday_series(start: date, count: int):
    return [start + timedelta(weeks=i) for i in range(count)]

@st.cache_data(ttl=3600)
def load_sheet_df() -> pd.DataFrame:
    """Load the fixed Google Sheet range into a DataFrame."""
    try:
        df = fetch_sheet_df(SHEET_ID, WORKSHEET_NAME, RANGE_A1)
        df.replace("", np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return pd.DataFrame()

def opposition_unique(df: pd.DataFrame):
    """Unique Opposition values (non-empty, first-seen order)."""
    if df.empty or "Opposition" not in df.columns:
        return []
    vals = []
    for v in df["Opposition"]:
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "nil") and s not in vals:
            vals.append(s)
    return vals

def build_weekly_plan(pre_start, pre_end, in_start, opp_seq):
    """Preseason derived from dates; In-season fixed R1–R11 + QF, SF, GF. Defaults to Off and Nil."""
    delta_days = (pre_end - pre_start).days
    pre_weeks = max(1, (delta_days // 7) + 1)
    pre_mons = monday_series(pre_start, pre_weeks)

    in_labels = [f"R{i}" for i in range(1, 12)] + ["QF", "SF", "GF"]
    in_mons = monday_series(in_start, len(in_labels))
    rows = []

    # Preseason rows
    for i, s in enumerate(pre_mons, 1):
        row = {"Week": f"PS{i}", "Start": fmt(s), "Segment": "Preseason",
               "Phase": "Nil", "Opposition": "Nil",
               "TD Multiplier (%)": 100, "N/m Multiplier (%)": 100}
        for d in WEEK_DAYS:
            row[d] = "Off"
        rows.append(row)

    # In-season rows
    for idx, (lbl, s) in enumerate(zip(in_labels, in_mons)):
        opp = "TBC" if lbl in ["QF", "SF", "GF"] else (opp_seq[idx] if idx < len(opp_seq) else "Nil")
        row = {"Week": lbl, "Start": fmt(s), "Segment": "In-Season",
               "Phase": "Nil", "Opposition": opp,
               "TD Multiplier (%)": 100, "N/m Multiplier (%)": 100}
        for d in WEEK_DAYS:
            row[d] = "Off"
        rows.append(row)

    cols = ["Week", "Start", "Segment", "Phase", "Opposition",
            "TD Multiplier (%)", "N/m Multiplier (%)"] + WEEK_DAYS
    return pd.DataFrame(rows, columns=cols)

def enforce_unique_day_types(df):
    """Within each row (week), ensure D1..D5/Match are unique. Duplicates reset to Off."""
    out = df.copy()
    for i in range(len(out)):
        used = set()
        for d in WEEK_DAYS:
            val = str(out.at[i, d])
            if val in NON_REPEATABLE:
                if val in used:
                    out.at[i, d] = "Off"
                else:
                    used.add(val)
    return out

def parse_duration_to_minutes(s):
    """Parse 'HH:MM:SS' to minutes (float)."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    try:
        h, m, sec = [float(x) for x in s.split(":")]
        return h * 60 + m + sec / 60.0
    except Exception:
        return np.nan

def canonical_week_sequence(days_list):
    """Return ordered unique training-days sequence (e.g., 'D1–D2–D4–D5 (Captain's Run)')."""
    unique = []
    for d in days_list:
        d = str(d).strip()
        if d in TRAINING_DAY_CANON and d not in unique:
            unique.append(d)
    order_map = {d: i for i, d in enumerate(TRAINING_DAY_CANON)}
    unique.sort(key=lambda x: order_map[x])
    return "–".join(unique) if unique else "—"

def render_fixed_width_table(df: pd.DataFrame, first_col_name: str) -> str:
    """Render a centered HTML table with fixed column widths, matching across tables."""
    if first_col_name not in df.columns:
        raise ValueError(f"{first_col_name} not in DataFrame columns.")
    colgroup = [f'<col style="width:{FIRST_COL_WIDTH}px;">']
    colgroup += [f'<col style="width:{METRIC_COL_WIDTH}px;">' for _ in range(len(df.columns) - 1)]
    colgroup_html = "<colgroup>" + "".join(colgroup) + "</colgroup>"
    style = """
    <style>
      table.fixedw { border-collapse: collapse; width: max-content; }
      table.fixedw th, table.fixedw td {
        border: 1px solid #e5e7eb; padding: 6px 8px; text-align: center; vertical-align: middle;
      }
      table.fixedw thead th { background: #f9fafb; font-weight: 600; }
    </style>
    """
    thead = "<thead><tr>" + "".join([f"<th>{c}</th>" for c in df.columns]) + "</tr></thead>"
    rows = []
    for _, r in df.iterrows():
        tds = "".join([f"<td>{'' if pd.isna(v) else v}</td>" for v in r.values])
        rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(rows) + "</tbody>"
    return f'{style}<table class="fixedw">{colgroup_html}{thead}{tbody}</table>'

def apply_row_colours_by_winrate(html: str, winrate_map: dict, suppress: bool) -> str:
    """Shade rows green/red by win rate (suppressed for Venue)."""
    if suppress:
        return html
    rows = html.split("<tbody>")[1].split("</tbody>")[0].split("</tr>")
    header_and_start = html.split("<tbody>")[0] + "<tbody>"
    end = "</tbody>" + html.split("</tbody>")[1]
    new_rows = []
    for row in rows:
        if not row.strip():
            continue
        try:
            first_td_start = row.index("<td>") + 4
            first_td_end = row.index("</td>", first_td_start)
            grp = row[first_td_start:first_td_end]
        except Exception:
            grp = None
        if grp in winrate_map:
            wr = winrate_map[grp]
            if pd.notna(wr):
                colour = "#d1fae5" if wr > 0.5 else "#fee2e2"
                if "<tr>" in row:
                    row = row.replace("<tr>", f'<tr style="background-color:{colour};">', 1)
        new_rows.append(row + "</tr>")
    return header_and_start + "".join(new_rows) + end

# ---------------------- SEASON SETUP ----------------------
st.subheader("Season dates")
c1, c2, c3 = st.columns(3)
with c1:
    pre_start = st.date_input("Preseason start", date(2026, 1, 12), format="DD/MM/YYYY")
with c2:
    pre_end = st.date_input("Preseason end", date(2026, 3, 8), format="DD/MM/YYYY")
with c3:
    in_start = st.date_input("In-season start", date(2026, 3, 15), format="DD/MM/YYYY")

sheet_df = load_sheet_df()
oppo_seq = opposition_unique(sheet_df)
OPPO_OPTIONS = ["Nil"] + oppo_seq

if "season_df" not in st.session_state:
    st.session_state.season_df = build_weekly_plan(pre_start, pre_end, in_start, oppo_seq)
st.markdown("---")
st.subheader("Macro Cycle")
plan_df = st.session_state.season_df.copy()

col_cfg = {
    "Week": st.column_config.TextColumn("Week", width="small"),
    "Start": st.column_config.TextColumn("Start", width="small"),
    "Segment": st.column_config.TextColumn("Segment", width="small"),
    "Phase": st.column_config.SelectboxColumn("Phase", options=PHASE_OPTIONS, width="small"),
    "Opposition": st.column_config.SelectboxColumn("Opposition", options=OPPO_OPTIONS, width="small"),
    "TD Multiplier (%)": st.column_config.NumberColumn("TD Multiplier (%)", min_value=0, max_value=300, step=1, format="%.0f%%", width="small"),
    "N/m Multiplier (%)": st.column_config.NumberColumn("N/m Multiplier (%)", min_value=0, max_value=300, step=1, format="%.0f%%", width="small"),
}
for d in WEEK_DAYS:
    col_cfg[d] = st.column_config.SelectboxColumn(d, options=DAY_TYPES)

# Centered editing UI
st.markdown("""
<style>
[data-testid="stDataEditor"] table thead th,
[data-testid="stDataEditor"] table tbody td {
  text-align:center !important; vertical-align:middle !important;
}
[data-testid="stDataEditor"] select,[data-testid="stDataEditor"] input {
  text-align-last:center; text-align:center;
}
</style>
""", unsafe_allow_html=True)

edited = st.data_editor(
    plan_df,
    column_config=col_cfg,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=["Week", "Start", "Segment"],
    key="weekly_plan_editor"
)
san = enforce_unique_day_types(edited)
if not san.equals(edited):
    edited = san.copy()
st.session_state.season_df = edited.copy()

if st.button("Save Plan", use_container_width=True):
    st.session_state.season_df.to_csv(PLAN_CSV, index=False)
    st.success("Season plan saved.")
