import streamlit as st
from utils.ui import inject_responsive_layout, render_page_header

st.set_page_config(page_title="Warriors NRLW Loading Tool", layout="centered")

with st.sidebar:
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

inject_responsive_layout()

st.markdown(
    """
<style>
table {
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    overflow: hidden;
    width: 100% !important;
}

table td, table th {
    border: 1px solid rgba(0,0,0,0.1);
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
}

table thead th {
    background-color: var(--secondary-background-color, #e5e5f3);
}

table tbody tr:nth-child(even) td {
    background-color: rgba(0,0,0,0.05);
}

div[data-testid="stMarkdownContainer"] table {
    display: block;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 70vh;
}

[data-testid="stDataEditorContainer"] table {
    width: auto !important;
    display: table !important;
    overflow: visible !important;
    max-height: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

render_page_header("One NZ Warriors NRLW Prognostic Loading Tool", "Wahs.png", "NRLW Logo.png", heading="h1")

st.markdown("---")

st.markdown(
    """
Welcome! Use the sidebar to navigate:

- **Positional Top Ups** – Player-level high-intensity top-ups.
- **Team Session Planner** – Full-team session planning.
- **Team Week Planner** – Full-week planning across the squad.
- **Season Set Up** – Macro-cycle planning and weekly structure.
- **Insights** – Contextual intelligence and winning patterns.
- **Information** – Key metrics, rationale, and methodology.
"""
)

st.markdown("---")


