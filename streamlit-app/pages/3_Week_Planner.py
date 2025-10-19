import pandas as pd
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

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
    df_sheet = pd.DataFrame(rows, columns=header)
    # convert numeric columns and compute Aerobic (m)
    numeric_cols = [
        COLS["minutes"], COLS["max_speed"], COLS["total_dist"], COLS["zone_28"],
        COLS["hmld"], COLS["hsr"], COLS["vhsr"], COLS["accel_eff"],
        COLS["accel_dist"], COLS["decel_eff"], COLS["decel_dist"], COLS["target"]
    ]
    for col in numeric_cols:
        if col in df_sheet.columns:
            df_sheet[col] = pd.to_numeric(df_sheet[col], errors="coerce")
    if COLS["zone_28"] in df_sheet.columns and COLS["hsr"] in df_sheet.columns:
        df_sheet[COLS["aerobic"]] = df_sheet[COLS["zone_28"]] - df_sheet[COLS["hsr"]]
    df_sheet = df_sheet.dropna(subset=[COLS["position"]])
    return df_sheet

# =========================
# CONFIG
# ========================
SHEET_ID = "1JqDkmbXDCWyNjpsMffbQyi6pSXQWOEm4lAf6cvmEhM4"
WORKSHEET_NAME = "NRLW_ALL"
RANGE_A1 = "B2:AY"

COLS = {
    "week": "Week",
    "day": "Day",
    "position": "Position",
    "minutes": "Match Minutes",
    "max_speed": "Max Speed",
    "total_dist": "Total Distance",
    "aerobic": "Aerobic (m)",
    "zone_28": "Distance Zone 2-8 (m)",
    "hmld": "HMLD",
    "hsr": "High Speed Running",
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
ALL_METRICS = LIT_METRICS + HIT_METRICS + [COLS["target"]]

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

DAY_TYPES = ["Off", "D1", "D2", "D3", "D4", "D5 (Captain's Run)", "Match", "NON"]
WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# =========================
# SIDEBAR LOGO
# =========================
with st.sidebar:
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(page_title="Practice Planner", layout="wide")

st.markdown("""
<style>
table {
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    overflow: hidden;
    width: 100% !important;
}
.compact-table {
    width: 100%;
    table-layout: fixed;
}
table td, table th {
    border: 1px solid rgba(0,0,0,0.1);
    text-align: center;
    vertical-align: middle;
    white-space: normal;
    padding: 10px 18px;
}
table thead th { background-color: var(--secondary-background-color, #e5e5f3); text-align: center; }
table tbody tr:nth-child(even) td { background-color: rgba(0,0,0,0.05); }
div[data-testid="stMarkdownContainer"] table {
    display: block; overflow-x: auto; overflow-y: auto; max-height: 70vh;
}
.off-day { background-color: #e5e5e5 !important; color: #888 !important; }
.compact-table th { text-align: center !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col1, col2, col3 = st.columns([0.5, 4, 0.5])
with col1: st.image("Wahs.png", width=85)
with col2: st.markdown("<h1 style='text-align:center; color:#262C68;'>Practice Planner</h1>", unsafe_allow_html=True)
with col3: st.image("NRLW Logo.png", width=85)
st.markdown("---")

# ============================================================
# HELPERS
# ============================================================
def norm_day(label: str) -> str:
    if not isinstance(label, str):
        return "Off"
    s = label.strip().upper()
    if s.startswith("D1"):
        return "D1"
    if s.startswith("D2"):
        return "D2"
    if s.startswith("D3"):
        return "D3"
    if s.startswith("D4"):
        return "D4"
    if s.startswith("D5") or s.startswith("CR") or "CAPTAIN" in s:
        return "D5"
    if "MATCH" in s:
        return "Match"
    if "NON" in s:
        return "NON"
    if "OFF" in s:
        return "Off"
    return label.strip()

def exemplar_stat(series: pd.Series, percentile: int) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    if percentile == 50:
        return float(np.nanmedian(cleaned))
    return float(np.nanpercentile(cleaned, percentile))

def format_metric_value(metric: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if metric in (COLS["max_speed"], COLS["target"]):
        return f"{value:.1f}"
    return f"{int(round(value))}"

def render_html_table(df: pd.DataFrame, off_cols=None) -> None:
    off_cols = off_cols or []
    cols = df.columns.tolist()
    if cols:
        metric_w = 18
        total_w = 18
        remaining = 100.0 - metric_w - total_w
        mid_count = max(len(cols) - 2, 1)
        mid_w = remaining / mid_count
        widths = [metric_w] + [mid_w] * (len(cols) - 2) + [total_w]
        colgroup = "<colgroup>" + "".join([f"<col style='width:{w:.2f}%'>" for w in widths]) + "</colgroup>"
    else:
        colgroup = ""
    thead = "<thead><tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr></thead>"
    rows_html = []
    for _, row in df.iterrows():
        tds = []
        for col in cols:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                display = ""
            else:
                display = str(val)
            if col in off_cols and col not in ("Metric", "Total"):
                tds.append(f"<td class='off-day'>{display}</td>")
            else:
                tds.append(f"<td>{display}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
    html = f"<table class='compact-table'>{colgroup}{thead}{tbody}</table>"
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# STEP 1 · DEFINE WEEK STRUCTURE
# ============================================================
st.markdown("### Step 1 · Define Week Structure")

defaults = {
    "Mon": "D1",
    "Tue": "D2",
    "Wed": "Off",
    "Thu": "D4",
    "Fri": "D5 (Captain's Run)",
    "Sat": "Match",
    "Sun": "Off",
}

used_types = set()
day_config = {}
cols = st.columns(len(WEEK_DAYS))
for idx, wd in enumerate(WEEK_DAYS):
    with cols[idx]:
        st.markdown(f"**{wd}**")
        available = ["Off"] + [d for d in DAY_TYPES if d != "Off" and d not in used_types]
        selected_default = defaults.get(wd, "Off")
        selected = st.selectbox(
            "Type",
            options=available,
            index=available.index(selected_default) if selected_default in available else 0,
            key=f"day_type_{wd}"
        )
        day_config[wd] = {"type": selected}
        if selected != "Off":
            used_types.add(selected)

selected_day_types = [day_config[wd]["type"] for wd in WEEK_DAYS]
day_cols = list(dict.fromkeys(selected_day_types))
off_cols = [d for d in day_cols if norm_day(d) == "Off"]

# ============================================================
# DATA LOADING
# ============================================================
raw_df = fetch_sheet_df(SHEET_ID, WORKSHEET_NAME, RANGE_A1)
df = raw_df.copy()
df["DayNorm"] = df[COLS["day"]].apply(norm_day)

# ============================================================
# STEP 2 · EXEMPLAR BASIS
# ============================================================
st.markdown("---")
st.markdown("### Step 2 · Week Totals")

col_left, col_right = st.columns([2, 1])
with col_left:
    exemplar_type = st.radio(
        "Exemplar type",
        options=["Typical (P50)", "High-Day (P75)"],
        index=1,
        help="Select the percentile used to build the exemplar load profile."
    )
with col_right:
    apply_minutes_filter = st.checkbox(
        "Only include match minutes ≥ 35",
        value=True,
        help="Apply the minutes filter before calculating exemplar values."
    )

if apply_minutes_filter and COLS["minutes"] in df.columns:
    mins = df[COLS["minutes"]]
    df_ctx = df[(mins.isna()) | (mins >= MINUTES_FILTER)].copy()
else:
    df_ctx = df.copy()

percentile = 50 if exemplar_type.startswith("Typical") else 75

overall_exemplar = {}
for metric in ALL_METRICS:
    overall_exemplar[metric] = exemplar_stat(df_ctx[metric], percentile)

# Day-level exemplar values for the selected structure
exemplar_day_values = {}
for metric in ALL_METRICS:
    grouped = df_ctx.groupby("DayNorm")[metric].apply(lambda s: exemplar_stat(s, percentile))
    exemplar_day_values[metric] = {k: (0.0 if pd.isna(v) else float(v)) for k, v in grouped.items()}

# Weekly totals and day shares for the structure
exemplar_totals = {}
day_shares = {}
base_day_lookup = {}
for metric in ALL_METRICS:
    day_map = exemplar_day_values.get(metric, {})
    raw_values = {}
    total = 0.0
    for day_label in day_cols:
        day_key = norm_day(day_label)
        if day_key in ("Off", ""):
            raw_values[day_label] = 0.0
            continue
        val = float(day_map.get(day_key, 0.0) or 0.0)
        raw_values[day_label] = max(val, 0.0)
        total += raw_values[day_label]
    exemplar_totals[metric] = total
    base_day_lookup[metric] = raw_values
    if total > 0:
        day_shares[metric] = {d: (raw_values[d] / total if raw_values[d] > 0 else 0.0) for d in raw_values}
    else:
        day_shares[metric] = {d: 0.0 for d in raw_values}

# Relative scaling ratios per position (kept hidden)
positions = [p for p in sorted(df_ctx[COLS["position"]].dropna().unique()) if p]
rel_scaling = {}
for pos in positions:
    pos_df = df_ctx[df_ctx[COLS["position"]] == pos]
    rel_scaling[pos] = {}
    for metric in ALL_METRICS:
        baseline = overall_exemplar.get(metric)
        if baseline is None or np.isnan(baseline) or baseline <= 0:
            rel_scaling[pos][metric] = np.nan
            continue
        pos_val = exemplar_stat(pos_df[metric], percentile)
        if pos_val is None or np.isnan(pos_val):
            rel_scaling[pos][metric] = np.nan
        else:
            rel_scaling[pos][metric] = pos_val / baseline * 100



if not positions:
    st.warning("No positional data available after filtering.")
    st.stop()

pos_index = positions.index("Centre") if "Centre" in positions else 0
selected_pos = st.selectbox("Select position", positions, index=pos_index)

rows = []
for metric in ALL_METRICS:
    if metric == COLS["max_speed"]:
        pos_df_filtered = df_ctx[df_ctx[COLS["position"]] == selected_pos]
        pos_df_full = raw_df[raw_df[COLS["position"]] == selected_pos]
        if pos_df_filtered.empty or pos_df_full.empty:
            continue
        pos_max = pos_df_full[metric].max(skipna=True)
        if pd.isna(pos_max) or pos_max <= 0:
            continue
        day_means = pos_df_filtered.groupby("DayNorm")[metric].mean()
        share_row = {"Metric": f"{DISPLAY_NAMES[metric]} (%)"}
        values_row = {"Metric": f"{DISPLAY_NAMES[metric]} (value)"}
        for day_label in day_cols:
            day_key = norm_day(day_label)
            if day_key in ("Off", ""):
                share_row[day_label] = "0.0%"
                values_row[day_label] = "0"
                continue
            day_mean = float(day_means.get(day_key, 0.0) or 0.0)
            ratio = (day_mean / pos_max) if pos_max > 0 else 0.0
            share_row[day_label] = f"{ratio * 100:.1f}%" if ratio > 0 else "0.0%"
            values_row[day_label] = format_metric_value(metric, day_mean) if day_mean > 0 else "0"
        share_row["Total"] = ""
        original_max = day_means.max(skipna=True) if not day_means.empty else 0.0
        values_row["Total"] = format_metric_value(metric, pos_max)
        rows.append(share_row)
        rows.append(values_row)
        continue

    base_total = exemplar_totals.get(metric, 0.0)
    if base_total <= 0:
        continue
    ratio = rel_scaling.get(selected_pos, {}).get(metric, np.nan)
    if ratio is None or np.isnan(ratio):
        continue
    ratio_factor = ratio / 100.0
    scaled_total = base_total * ratio_factor
    shares = day_shares.get(metric, {})

    share_row = {"Metric": f"{DISPLAY_NAMES[metric]} (%)"}
    values_row = {"Metric": f"{DISPLAY_NAMES[metric]} (value)"}

    share_sum = sum(shares.values())
    for day_label in day_cols:
        share_pct = shares.get(day_label, 0.0)
        share_row[day_label] = f"{share_pct * 100:.1f}%" if share_pct > 0 else "0.0%"
        day_value = scaled_total * share_pct
        if share_pct > 0 and day_value > 0:
            values_row[day_label] = format_metric_value(metric, day_value)
        elif norm_day(day_label) == "Off":
            values_row[day_label] = "0"
        else:
            values_row[day_label] = "0"

    share_row["Total"] = f"{max(share_sum, 0.0) * 100:.1f}%" if share_sum > 0 else "0%"
    scaled_total_formatted = format_metric_value(metric, scaled_total)
    base_total_formatted = format_metric_value(metric, base_total)
    if base_total_formatted:
        values_row["Total"] = f"{base_total_formatted} (<i>Exemplar {scaled_total_formatted}</i>)"
    else:
        values_row["Total"] = base_total_formatted

    rows.append(share_row)
    rows.append(values_row)

if not rows:
    st.info("No metrics available for the current selections.")
else:
    table_df = pd.DataFrame(rows, columns=["Metric"] + day_cols + ["Total"])
    display_cols = ["Metric"] + [col.replace(" (Captain's Run)", "") for col in day_cols] + ["Total"]
    table_df.columns = display_cols
    display_off_cols = []
    for col in off_cols:
        if col == "D5 (Captain's Run)":
            display_off_cols.append("D5")
        else:
            display_off_cols.append(col)
    render_html_table(table_df, off_cols=display_off_cols)
    st.caption("Each metric distributes the exemplar total across the selected structure, then scales by the chosen position's ratio. Day percentages always sum to 100%.")

