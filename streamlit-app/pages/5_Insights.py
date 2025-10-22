import json
from datetime import date, datetime
from pathlib import Path

import gspread
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

from utils.positions import apply_position_overrides
from utils.ui import inject_responsive_layout, render_page_header

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    # Place logo at the top with centered alignment
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="small")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Insights", layout="wide")

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
render_page_header("Insights", "Wahs.png", "NRLW Logo.png", heading="h1")

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

TRAINING_DAY_CANON = ["D1", "D2", "D3", "D4", "D5 (Captain's Run)"]
FIRST_COL_WIDTH = 180
METRIC_COL_WIDTH = 150

PROFILE_PATH = Path("context_profiles.json")
DELTA_SUFFIX = "_\u0394%"

DISPLAY_NAMES = {
    "Max Speed": "Max Speed",
    "Total Distance": "Total Distance",
    "Aerobic (m)": "Aerobic (m)",
    "Distance Zone 2-8 (m)": "Dist Zone 2-8 (m)",
    "HMLD": "HMLD",
    "High Speed Running": "HSR",
    "VHSR": "VHSR",
    "Accel Efforts": "Accel Efforts",
    "Accel Distance": "Accel Distance",
    "Decel Efforts": "Decel Efforts",
    "Decel Distance": "Decel Distance",
    "N/m": "N/m",
    "Total Duration (min)": "Total Duration (min)",
}

WINNING_METRIC_LABELS = {
    "Total Duration (min)": ("Total Duration", "min"),
    "Max Speed": ("Max Speed", "m/s"),
    "Total Distance": ("Total Distance", "m"),
    "Aerobic (m)": ("Aerobic", "m"),
    "Distance Zone 2-8 (m)": ("Dist Zone 2-8", "m"),
    "HMLD": ("HMLD", "m"),
    "High Speed Running": ("HSR", "m"),
    "VHSR": ("VHSR", "m"),
    "Accel Efforts": ("Accel Efforts", ""),
    "Accel Distance": ("Accel Distance", "m"),
    "Decel Efforts": ("Decel Efforts", ""),
    "Decel Distance": ("Decel Distance", "m"),
    "N/m": ("N/m", "N/m"),
}

DAY_ORDER = ["D1", "D2", "D3", "D4", "D5 (Captain's Run)", "Match"]
DAY_ALIASES = {
    "D5": "D5 (Captain's Run)",
    "D5 (CAPTAIN'S RUN)": "D5 (Captain's Run)",
    "D5 CAPTAINS RUN": "D5 (Captain's Run)",
    "CAPTAIN'S RUN": "D5 (Captain's Run)",
    "CAPTAINS RUN": "D5 (Captain's Run)",
    "MATCH": "Match",
}

# ---------------------- PAGE STYLE ------------------------
st.markdown("""
<style>
.block-container { padding-top: .3rem; max-width: 1180px; margin: auto; }
h1, h2, h3 { text-align: center; margin: .25rem 0 .85rem; font-weight: 700; }
table.fixedw { border-collapse: collapse; }
table.fixedw th, table.fixedw td {
  border: 1px solid #e5e7eb; padding: 6px 8px; text-align: center; vertical-align: middle;
  white-space: nowrap;
}
table.fixedw thead th { background: #f9fafb; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.insight-box {
  background: rgba(38, 44, 104, 0.06);
  border-radius: 12px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.75rem;
}
.insight-box ul {
  margin: 0;
  padding-left: 1.1rem;
}
.insight-box li {
  margin-bottom: 0.35rem;
}
.win-pill {
  color: #047857;
  font-weight: 600;
}
.loss-pill {
  color: #b91c1c;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HELPERS ---------------------------
def fmt_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")
def load_saved_profile():
    if not PROFILE_PATH.exists():
        return None
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.warning("Saved insight snapshot could not be parsed and was ignored.")
    except Exception as exc:
        st.warning(f"Unable to read saved insight snapshot: {exc}")
    return None


def save_profile(payload: dict) -> bool:
    try:
        PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        st.error(f"Unable to save snapshot: {exc}")
        return False


def normalise_scalar(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def records_from_frame(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    out = df.reset_index()
    if out.columns[0] == "index":
        out = out.rename(columns={"index": "Day"})
    elif out.columns[0] != "Day":
        out = out.rename(columns={out.columns[0]: "Day"})
    records = out.to_dict(orient="records")
    return [{k: normalise_scalar(v) for k, v in row.items()} for row in records]


def prepare_profile_payload(group_by, include_losses, positions_filter, gdf, season_mean, grp_win_rate, win_means, loss_means, contrib_diff, win_std=None, loss_std=None):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "group_by": group_by,
        "include_losses": bool(include_losses),
        "positions": sorted(list(positions_filter)) if positions_filter else None,
        "season_mean": {k: normalise_scalar(v) for k, v in season_mean.items()},
        "context_table": [{k: normalise_scalar(v) for k, v in row.items()} for row in gdf.to_dict(orient="records")],
        "win_rate": {k: normalise_scalar(v) for k, v in (grp_win_rate or {}).items()},
        "win_means": records_from_frame(win_means),
        "loss_means": records_from_frame(loss_means),
        "contrib_diff": records_from_frame(contrib_diff),
        "win_std": records_from_frame(win_std),
        "loss_std": records_from_frame(loss_std),
    }
    return payload


def profile_to_result(profile):
    if not profile:
        return None
    context_records = profile.get("context_table") or profile.get("data")
    if not context_records:
        return None
    gdf = pd.DataFrame(context_records)
    season_mean = pd.Series(profile.get("season_mean", {}))
    grp_win_rate = profile.get("win_rate", {})
    win_means_records = profile.get("win_means")
    loss_means_records = profile.get("loss_means")
    contrib_diff_records = profile.get("contrib_diff")
    win_std_records = profile.get("win_std")
    if win_std_records is None:
        win_std_records = profile.get("win_ci")
    loss_std_records = profile.get("loss_std")
    if loss_std_records is None:
        loss_std_records = profile.get("loss_ci")
    win_means = pd.DataFrame(win_means_records).set_index("Day") if win_means_records else None
    loss_means = pd.DataFrame(loss_means_records).set_index("Day") if loss_means_records else None
    contrib_diff = pd.DataFrame(contrib_diff_records) if contrib_diff_records else None
    win_std = pd.DataFrame(win_std_records).set_index("Day") if win_std_records else None
    loss_std = pd.DataFrame(loss_std_records).set_index("Day") if loss_std_records else None
    return gdf, season_mean, grp_win_rate, win_means, loss_means, contrib_diff, win_std, loss_std

@st.cache_data(ttl=3600)
def load_sheet_df() -> pd.DataFrame:
    """Load Google Sheet directly."""
    try:
        df = fetch_sheet_df(SHEET_ID, WORKSHEET_NAME, RANGE_A1)
        df.replace("", np.nan, inplace=True)
        df = apply_position_overrides(df, position_col="Position")
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return pd.DataFrame()

def parse_duration_to_minutes(s):
    """Convert HH:MM:SS strings to minutes."""
    if pd.isna(s): return np.nan
    try:
        h, m, sec = [float(x) for x in str(s).split(":")]
        return h * 60 + m + sec / 60.0
    except Exception:
        return np.nan

def render_fixed_width_table(df: pd.DataFrame, first_col_name: str, wrap: bool = False) -> str:
    """Render a fixed-width HTML table."""
    if first_col_name not in df.columns:
        raise ValueError(f"{first_col_name} not in DataFrame columns.")
    colgroup = [f'<col style="width:{FIRST_COL_WIDTH}px;">']
    colgroup += [f'<col style="width:{METRIC_COL_WIDTH}px;">' for _ in range(len(df.columns) - 1)]
    colgroup_html = "<colgroup>" + "".join(colgroup) + "</colgroup>"
    thead = "<thead><tr>" + "".join([f"<th>{c}</th>" for c in df.columns]) + "</tr></thead>"
    rows = []
    for _, r in df.iterrows():
        tds = "".join([f"<td>{'' if pd.isna(v) else v}</td>" for v in r.values])
        rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(rows) + "</tbody>"
    table_class = "fixedw fixedw-wrap" if wrap else "fixedw"
    table_width = FIRST_COL_WIDTH + METRIC_COL_WIDTH * (len(df.columns) - 1)
    return f'<table class="{table_class}" style="table-layout:fixed; width:{table_width}px;">{colgroup_html}{thead}{tbody}</table>'

def apply_row_colours_by_winrate(html: str, winrate_map: dict, suppress: bool) -> str:
    """Apply green/red shading by win rate."""
    if suppress: return html
    rows = html.split("<tbody>")[1].split("</tbody>")[0].split("</tr>")
    header_and_start = html.split("<tbody>")[0] + "<tbody>"
    end = "</tbody>" + html.split("</tbody>")[1]
    new_rows = []
    for row in rows:
        if not row.strip(): continue
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

# ---------------------- INSIGHTS SECTION -------------------
st.markdown("### In-Season Intelligence")
st.caption("Grouped load profiles, Δ% differences, and actionable winning vs losing insights derived from match-linked training data.")

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
with filter_col1:
    group_by = st.selectbox("Group by:", ["Venue", "Opposition", "Opposition Seed"])
with filter_col2:
    include_losses = st.checkbox("Include losses", value=False)

raw_positions_df = load_sheet_df()
if raw_positions_df.empty:
    st.info("No data available to display insights.")
    st.stop()

position_series = raw_positions_df.get("Position")
if position_series is not None:
    pos_options = sorted(position_series.dropna().astype(str).str.strip().unique())
else:
    pos_options = []

with filter_col3:
    if pos_options:
        selected_positions = st.multiselect("Include positions", pos_options, default=pos_options, help="Choose the positions to include in all calculations.")
    else:
        selected_positions = []
        st.info("No position data found; defaulting to all records.")

@st.cache_data(ttl=1800)
def build_context_data(group_by, include_losses, positions_filter):
    df = load_sheet_df()
    if df.empty:
        return None

    req = [
        "Venue", "Result", "Match Minutes", "Opposition", "Opposition Seed", "Date", "Day", "Week", "Position",
        "Total Distance", "Distance Zone 2-8 (m)", "High Speed Running", "HMLD",
        "VHSR", "Max Speed", "Accel Efforts", "Accel Distance",
        "Decel Efforts", "Decel Distance", "N/m", "Total Duration"
    ]
    if any(c not in df.columns for c in req):
        return None

    df["Result"] = df["Result"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    for c in [x for x in req if x not in ("Venue", "Result", "Opposition", "Date", "Day", "Week", "Total Duration", "Position")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Total Duration (min)"] = df["Total Duration"].apply(parse_duration_to_minutes)
    if "Distance Zone 2-8 (m)" in df.columns and "High Speed Running" in df.columns:
        df["Aerobic (m)"] = df["Distance Zone 2-8 (m)"] - df["High Speed Running"]
    if "Day" in df.columns:
        df["Day"] = df["Day"].apply(lambda v: v if pd.isna(v) else str(v).strip())
        df["Day"] = df["Day"].apply(lambda v: DAY_ALIASES.get(v.upper(), v) if isinstance(v, str) else v)
    if "Position" in df.columns:
        df["Position"] = df["Position"].astype(str).str.strip()
    if positions_filter:
        df = df[df["Position"].isin(positions_filter)]
        if df.empty:
            return None

    df_tables = df[df["Match Minutes"] >= 35].copy()
    if not include_losses:
        df_tables = df_tables[df_tables["Result"] == "Won"]
    if df_tables.empty:
        return None

    df_tables["Opposition Seed"] = pd.to_numeric(df_tables["Opposition Seed"], errors="coerce").round(0)
    df_tables["Aerobic (m)"] = df_tables["Distance Zone 2-8 (m)"] - df_tables["High Speed Running"]

    metrics = [
        "Max Speed", "Total Distance", "Aerobic (m)", "Distance Zone 2-8 (m)",
        "HMLD", "High Speed Running", "VHSR", "Accel Efforts", "Accel Distance",
        "Decel Efforts", "Decel Distance", "N/m"
    ]

    gdf = df_tables.groupby(group_by, as_index=False)[metrics].mean().round(1)
    if group_by == "Opposition Seed" and group_by in gdf.columns:
        gdf[group_by] = gdf[group_by].round().astype('Int64')
    season_mean = df_tables[metrics].mean().round(1)
    for m in metrics:
        gdf[f"{m}_Δ%"] = ((gdf[m] - season_mean[m]) / season_mean[m] * 100).round(1)

    df_match = df[(df["Match Minutes"] >= 35) & (df["Result"].isin(["Won", "Loss"]))].copy()
    if "Day" in df_match.columns:
        df_match = df_match[df_match["Day"] == "Match"]
    grp_win_rate = df_match.groupby(group_by)["Result"].apply(lambda s: (s == "Won").mean()).to_dict()

    df_week_map = df[(df["Match Minutes"] >= 35) & (df["Result"].isin(["Won", "Loss"]))].copy()
    if "Day" in df_week_map.columns:
        df_week_map = df_week_map[df_week_map["Day"] == "Match"]
    wk_result_map = df_week_map.dropna(subset=["Week"]).groupby("Week")["Result"].agg(lambda s: s.iloc[0]).to_dict()

    df["WeekResult"] = df["Week"].map(wk_result_map)
    match_inclusive_days = TRAINING_DAY_CANON + ["Match"]
    df_train = df[
        df["Day"].isin(match_inclusive_days)
        & df["Week"].notna()
        & df["Week"].str.upper().str.startswith("R")
        & df["WeekResult"].isin(["Won", "Loss"])
    ].copy()

    t_metrics = [col for col in dict.fromkeys(metrics + ["Total Duration (min)"]) if col in df_train.columns]

    def compute_means_std(df_subset):
        if df_subset.empty:
            return None, None
        grouped = df_subset.groupby("Day")[t_metrics].agg(['mean', 'std'])
        means = grouped.xs('mean', axis=1, level=1).reindex(columns=t_metrics)
        stds = grouped.xs('std', axis=1, level=1).reindex(columns=t_metrics)
        stds = stds.replace([np.inf, -np.inf], np.nan)
        return means, stds

    win_subset = df_train[df_train["WeekResult"] == "Won"]
    loss_subset = df_train[df_train["WeekResult"] == "Loss"]
    win_means, win_std = compute_means_std(win_subset)
    loss_means, loss_std = compute_means_std(loss_subset)

    contrib_diff = None
    if not df_train.empty:
        per_week = df_train.groupby(["Week", "WeekResult", "Day"])["Total Distance"].sum().reset_index()
        totals = per_week.groupby(["Week", "WeekResult"])["Total Distance"].transform("sum")
        per_week["share_td"] = per_week["Total Distance"] / totals.replace(0, np.nan)
        win_share = per_week[per_week["WeekResult"] == "Won"].groupby("Day")["share_td"].mean()
        loss_share = per_week[per_week["WeekResult"] == "Loss"].groupby("Day")["share_td"].mean()
        common = win_share.index.intersection(loss_share.index)
        if len(common) > 0:
            contrib_diff = pd.DataFrame({
                "Winning %": (win_share.loc[common] * 100).round(1),
                "Losing %": (loss_share.loc[common] * 100).round(1),
                "Δ pp": ((win_share.loc[common] - loss_share.loc[common]) * 100).round(1)
            }).reset_index().rename(columns={"index": "Day"})

    return gdf, season_mean, grp_win_rate, win_means, loss_means, contrib_diff

positions_filter = tuple(selected_positions) if selected_positions else tuple()
if pos_options and not selected_positions:
    st.info("Select at least one position to display insights.")
    result = None
else:
    result = build_context_data(group_by, include_losses, positions_filter)

if result:
    if isinstance(result, tuple) and len(result) == 8:
        gdf, season_mean, grp_win_rate, win_means, loss_means, contrib_diff, win_std, loss_std = result
    elif isinstance(result, tuple) and len(result) == 6:
        gdf, season_mean, grp_win_rate, win_means, loss_means, contrib_diff = result
        win_std = loss_std = None
    else:
        raise ValueError("Unexpected result payload length")

    st.markdown("#### Context Table")
    context_df = gdf.drop(columns=[c for c in gdf.columns if "_Δ%" in c]).copy()
    context_cols = [f"  {col}  " for col in context_df.columns]
    context_first_col = context_cols[0]
    context_df.columns = context_cols
    ctx_html = render_fixed_width_table(context_df, first_col_name=context_first_col)
    st.markdown(ctx_html, unsafe_allow_html=True)

    st.markdown("#### Difference Table (Δ%)")
    diff_cols = [c for c in gdf.columns if "_Δ%" in c]
    disp_key = "Opposition" if group_by == "Opposition" else group_by
    diff_df = gdf[[disp_key] + diff_cols].copy()
    for c in diff_cols:
        diff_df[c] = diff_df[c].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    diff_df.columns = [diff_df.columns[0]] + [col.replace("_Δ%", " Δ%").replace("_", " ") for col in diff_df.columns[1:]]
    diff_html = render_fixed_width_table(diff_df, first_col_name=diff_df.columns[0])
    diff_html = apply_row_colours_by_winrate(diff_html, grp_win_rate, suppress=(group_by == "Venue"))
    st.markdown(diff_html, unsafe_allow_html=True)
    st.caption("Green = Won, Red = Loss. Δ% = difference from season mean.")

    # --- CONTEXT SNAPSHOT ---
    st.markdown("---")
    st.subheader("Context Snapshot", help="""
Highlights which metrics sit above or below season averages,
calculated from Δ% differences across the selected grouping.
""")
    diffs = []
    for _, row in gdf.iterrows():
        for m in [c for c in gdf.columns if "_Δ%" in c]:
            diffs.append((row[disp_key], m.replace("_Δ%", ""), row[m]))
    if diffs:
        ddf = pd.DataFrame(diffs, columns=["Group", "Metric", "Δ%"])
        top_pos = ddf.sort_values("Δ%", ascending=False).head(3)
        top_neg = ddf.sort_values("Δ%", ascending=True).head(2)
        lines = []
        for _, r in top_pos.iterrows():
            lines.append(f"Higher {r['Metric']} ({r['Δ%']:.1f}%)")
        for _, r in top_neg.iterrows():
            lines.append(f"Lower {r['Metric']} ({r['Δ%']:.1f}%)")
        st.info(f"**Summary:** {', '.join(lines)}")

    # --- WINNING VS LOSING INSIGHTS ---
    st.markdown("---")
    st.subheader("Winning vs Losing Week Snapshot", help="""
Compares training and match-week characteristics between wins and losses.
Training days (D1–D5) are linked to their match result using the Week column.
Shows real values (m, min, N/m) to illustrate what winning weeks look like.
""")

    if win_means is not None and loss_means is not None and not win_means.empty and not loss_means.empty:
        metrics_label_units = WINNING_METRIC_LABELS
        allowed_days = TRAINING_DAY_CANON
        common_days = [day for day in allowed_days if day in win_means.index and day in loss_means.index]

        metrics_by_day = {}
        select_label_map = {}
        for day in common_days:
            entries = []
            display_day = day
            short_label = day.split(" ")[0]
            select_label_map[day] = short_label

            for m_key, (label, unit) in metrics_label_units.items():
                if m_key not in win_means.columns or m_key not in loss_means.columns:
                    continue
                wv = win_means.at[day, m_key]
                lv = loss_means.at[day, m_key]
                if pd.isna(wv) or pd.isna(lv) or lv == 0:
                    continue

                win_margin = None
                loss_margin = None
                if win_std is not None and not win_std.empty and day in win_std.index and m_key in win_std.columns:
                    win_margin = win_std.at[day, m_key]
                if loss_std is not None and not loss_std.empty and day in loss_std.index and m_key in loss_std.columns:
                    loss_margin = loss_std.at[day, m_key]

                diff_pct = ((wv - lv) / lv) * 100.0
                if abs(diff_pct) < 10:
                    continue

                if unit == "m":
                    w_disp = f"{wv:.0f}"
                    l_disp = f"{lv:.0f}"
                    w_margin_disp = f"{win_margin:.0f}" if win_margin is not None and pd.notna(win_margin) else None
                    l_margin_disp = f"{loss_margin:.0f}" if loss_margin is not None and pd.notna(loss_margin) else None
                elif unit == "N/m":
                    w_disp = f"{wv:.1f}"
                    l_disp = f"{lv:.1f}"
                    w_margin_disp = f"{win_margin:.1f}" if win_margin is not None and pd.notna(win_margin) else None
                    l_margin_disp = f"{loss_margin:.1f}" if loss_margin is not None and pd.notna(loss_margin) else None
                else:
                    w_disp = f"{wv:.1f}"
                    l_disp = f"{lv:.1f}"
                    if unit == "min":
                        w_margin_disp = f"{win_margin:.0f}" if win_margin is not None and pd.notna(win_margin) else None
                        l_margin_disp = f"{loss_margin:.0f}" if loss_margin is not None and pd.notna(loss_margin) else None
                    else:
                        w_margin_disp = f"{win_margin:.1f}" if win_margin is not None and pd.notna(win_margin) else None
                        l_margin_disp = f"{loss_margin:.1f}" if loss_margin is not None and pd.notna(loss_margin) else None

                win_text = f"{w_disp}{unit}" if unit else w_disp
                loss_text = f"{l_disp}{unit}" if unit else l_disp
                if w_margin_disp is not None:
                    margin_text = f"{w_margin_disp}{unit}" if unit else w_margin_disp
                    win_text = f"{win_text} (±{margin_text})"
                if l_margin_disp is not None:
                    margin_text = f"{l_margin_disp}{unit}" if unit else l_margin_disp
                    loss_text = f"{loss_text} (±{margin_text})"

                win_text = win_text.replace('&plusmn;', '±')
                loss_text = loss_text.replace('&plusmn;', '±')

                entries.append({
                    "label": label,
                    "win_text": win_text,
                    "loss_text": loss_text,
                    "diff_pct": diff_pct,
                })

            if entries:
                metrics_by_day[day] = {
                    "display": display_day,
                    "entries": entries,
                }

        if not metrics_by_day:
            st.info("Insufficient linked data to generate insights.")
        else:
            select_options = ["All Days"] + [select_label_map[day] for day in metrics_by_day.keys()]
            day_select_col, _ = st.columns([1, 3])
            with day_select_col:
                day_selection = st.selectbox("Select Day", select_options, key="insights_day_select")

            if day_selection == "All Days":
                days_to_render = list(metrics_by_day.keys())
            else:
                reverse_map = {v: k for k, v in select_label_map.items()}
                selected_day = reverse_map.get(day_selection)
                if not selected_day:
                    st.info("No metrics available for the selected day.")
                    days_to_render = []
                else:
                    days_to_render = [selected_day]

            for day in days_to_render:
                info = metrics_by_day.get(day)
                if not info:
                    continue
                display_day = "D5 (Captain's Run)" if "Captain" in info["display"] else info["display"]
                st.markdown(f"**{display_day}**")

                entries = info["entries"]
                if not entries:
                    st.caption("No metrics cleared the display threshold for this day.")
                    continue

                columns = None
                for idx, entry in enumerate(entries):
                    if idx % 3 == 0:
                        columns = st.columns(3)
                    col = columns[idx % 3]
                    delta_value = entry["diff_pct"]
                    with col:
                        col.metric(
                            label=entry["label"],
                            value=entry["win_text"],
                            delta=f"{delta_value:+.1f}%"
                        )
                        col.caption(f"Loss: {entry['loss_text']}")

                st.markdown("")
    else:
        st.info("Insufficient linked training-week data to compare winning vs losing weeks.")
else:
    st.info("No data available for the selected filters.")

