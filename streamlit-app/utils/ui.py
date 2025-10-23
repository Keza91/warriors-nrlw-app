import base64
from pathlib import Path
from typing import Iterable

import streamlit as st


_BASE_DIR = Path(__file__).resolve().parent.parent



def inject_responsive_layout(max_width: int = 1180) -> None:
    """Inject responsive CSS so pages adapt on smaller screens."""
    css = f"""
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

.app-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(0.5rem, 3vw, 1.5rem);
    margin: clamp(1rem, 4vw, 2.5rem) auto;
    max-width: min(var(--page-max-width), 100%);
}}

.app-header-logo-wrap {{
    flex: 0 0 auto;
}}

.app-header-logo {{
    width: clamp(34px, 9vw, 58px);
    height: clamp(34px, 9vw, 58px);
    object-fit: contain;
    display: block;
}}

.app-header-title {{
    margin: 0;
    text-align: center;
    font-size: clamp(1.05rem, 2.2vw, 1.9rem);
    color: #262C68;
}}

@media (max-width: 768px) {{
    .block-container {{
        padding: 1rem !important;
        max-width: 100% !important;
    }}

    .app-header {{
        gap: clamp(0.4rem, 4vw, 1rem);
    }}

    .app-header-logo {{
        width: clamp(26px, 12vw, 44px);
        height: clamp(26px, 12vw, 44px);
    }}

    .app-header-title {{
        font-size: clamp(0.95rem, 3.4vw, 1.3rem);
    }}
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)



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


def _resolve_image_path(image_path: str) -> str | None:
    rel_path = Path(image_path)
    for candidate in _candidate_paths(rel_path):
        if candidate.exists():
            return str(candidate)
    return None




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

    left_html = f"<img src='{left_src}' class='app-header-logo' alt='Left logo'/>" if left_src else ""
    right_html = f"<img src='{right_src}' class='app-header-logo' alt='Right logo'/>" if right_src else ""

    title_cls = f"app-header-title {title_class}" if title_class else "app-header-title"

    header_html = f"""
<div class='app-header'>
  <div class='app-header-logo-wrap'>{left_html}</div>
  <{tag} class='{title_cls}'>{title}</{tag}>
  <div class='app-header-logo-wrap'>{right_html}</div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)











