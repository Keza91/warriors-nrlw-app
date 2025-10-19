import streamlit as st

st.set_page_config(page_title="Warriors NRLW Loading Tool", layout="centered")

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    # Place logo at the top with centered alignment
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("assets/Wahs.png", size="large")
    st.sidebar.image("assets/OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN PAGE STYLE
# =========================

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

# =========================
# HEADER (Main Content)
# =========================
col1, col2, col3 = st.columns([0.5, 4, 0.5])

with col1:
    st.image("assets/Wahs.png", width=85)

with col2:
    st.markdown(
        "<h3 style='text-align:center; color:#262C68;'>One NZ Warriors NRLW Prognostic Loading Tool</h3>",
        unsafe_allow_html=True
    )

with col3:
    st.image("assets/NRLW Logo.png", width=85)

st.markdown("---")
st.markdown("Welcome! Use the sidebar to navigate:")
st.markdown("- **Positional Top ups** → Player-level High-Intensity top-ups.")

st.markdown("- **Team Session Planner** → Full-team session planning.")

st.markdown("- **Team Week Planner** → Full-team week planning.")
st.markdown("-- Set Athlete Thresholds for OpenField .")
st.markdown("- **Season Set Up** → Macro-cycle planning (Weekly structure).")

st.markdown("- **Insights** → Contextual intelligence & winning patterns.")

st.markdown("- **Information** → Explains key metrics, rationale, and methodology.")

st.markdown("---")

