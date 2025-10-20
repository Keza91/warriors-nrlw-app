import base64
from pathlib import Path
from typing import Iterable

import streamlit as st


_BASE_DIR = Path(__file__).resolve().parent.parent


def inject_responsive_layout(max_width: int = 1180) -> None:
    """Inject responsive CSS so pages adapt on smaller screens."""
    st.markdown(
        f"""
<style>
:root {{
    --page-max-width: {max_width}px;
}}

.block-container {{
    max-width: min(var(--page-max-width), 100%) !important;
    padding-left: clamp(0.75rem, 3vw, 2.5rem) !important;
    padding-right: clamp(0.75rem, 3vw, 2.5rem) !important;
    padding-top: clamp(2rem, 4vw, 3rem) !important;
    padding-bottom: clamp(2rem, 4vw, 3rem) !important;
}}

section.main > div:first-child {{
    width: 100%;
}}

.responsive-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(0.75rem, 4vw, 2.5rem);
    margin: clamp(1rem, 4vw, 2.5rem) auto;
    max-width: min(var(--page-max-width), 100%);
}}

.responsive-header .header-logo-wrap {{
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: clamp(45px, 10vw, 90px);
}}

.responsive-header .header-logo {{
    width: clamp(45px, 10vw, 90px);
    height: auto;
    object-fit: contain;
}}

.responsive-header .header-title {{
    flex: 1 1 auto;
    margin: 0;
    text-align: center;
    font-size: clamp(1.4rem, 3vw, 2.6rem);
    color: #262C68;
}}

@media (max-width: 992px) {{
    section.main div[data-testid="column"] {{
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding: 1rem !important;
        max-width: 100% !important;
    }}

    section.main div[data-testid="column"] {{
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        flex: 1 1 100% !important;
    }}

    section.main div[data-testid="stHorizontalBlock"] {{
        gap: 0.75rem !important;
    }}

    .stColumns {{
        display: flex;
        flex-direction: column !important;
        gap: 0.75rem !important;
    }}

    div[data-testid="stMetric"] {{
        width: 100% !important;
        min-width: 0 !important;
    }}

    div[data-testid="stMarkdownContainer"] table {{
        display: block;
        overflow-x: auto;
        overflow-y: auto;
        width: 100% !important;
        font-size: 0.85rem;
        max-height: none !important;
    }}

    .responsive-header {{
        justify-content: space-between;
        gap: clamp(0.5rem, 5vw, 1.5rem);
    }}

    .responsive-header .header-logo-wrap {{
        min-width: clamp(40px, 16vw, 70px);
    }}

    .responsive-header .header-logo {{
        width: clamp(40px, 16vw, 70px);
    }}

    .responsive-header .header-title {{
        font-size: clamp(1.2rem, 5vw, 1.6rem);
    }}

    input[type="number"], select, textarea {{
        width: 100% !important;
        min-height: 45px;
        font-size: 1rem;
    }}

    h1, h2, h3 {{
        text-align: center !important;
    }}

    button[kind="primary"] {{
        width: 100% !important;
        font-size: 1rem;
    }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _candidate_paths(image_path: Path) -> Iterable[Path]:
    if image_path.is_absolute():
        yield image_path
        yield image_path.expanduser()
        return

    search_roots = [
        _BASE_DIR,
        _BASE_DIR / "assets",
        _BASE_DIR.parent,
        _BASE_DIR.parent / "assets",
    ]

    for root in search_roots:
        yield (root / image_path).resolve()


def _image_to_data_uri(image_path: str) -> str:
    rel_path = Path(image_path)
    seen: set[Path] = set()
    for candidate in _candidate_paths(rel_path):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".gif": "image/gif",
            }
            mime = mime_map.get(candidate.suffix.lower(), "image/png")
            encoded = base64.b64encode(candidate.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    return ""


def render_page_header(
    title: str,
    left_image: str,
    right_image: str,
    *,
    heading: str = "h1",
    title_class: str | None = None,
) -> None:
    """Render a responsive header with flanking logos that stays horizontal on phones."""
    tag = heading.lower()
    if tag not in {"h1", "h2", "h3", "h4"}:
        tag = "h1"

    left_src = _image_to_data_uri(left_image)
    right_src = _image_to_data_uri(right_image)

    left_html = (
        f'<img src="{left_src}" alt="Left logo" class="header-logo" />'
        if left_src
        else ""
    )
    right_html = (
        f'<img src="{right_src}" alt="Right logo" class="header-logo" />'
        if right_src
        else ""
    )

    title_cls = f"header-title {title_class}" if title_class else "header-title"

    st.markdown(
        f"""
<div class="responsive-header">
  <div class="header-logo-wrap">{left_html}</div>
  <{tag} class="{title_cls}">{title}</{tag}>
  <div class="header-logo-wrap">{right_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
