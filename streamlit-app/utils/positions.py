"""Utilities for normalising positional labels across the app."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Map lower-case variants to canonical labels
POSITION_REPLACEMENTS_LOWER = {
    "half": "Half Back",
    "halfback": "Half Back",
    "half back": "Half Back",
    "half-back": "Half Back",
    "5/8": "Half Back",
    "five-eighth": "Half Back",
    "fullback": "Full Back",
    "full back": "Full Back",
    "full-back": "Full Back",
    "fullback/wing": "Full Back",
    "wing/fullback": "Full Back",
    "fb": "Full Back",
}

# Labels that should now be removed and replaced via player overrides
REMOVED_LABELS_LOWER = {
    "half/hooker",
    "half & hooker",
    "half hooker",
    "hooker/half",
}

PLAYER_NAME_CANDIDATES = [
    "Player",
    "Player Name",
    "Athlete",
    "Athlete Name",
    "Name",
]

DATE_CANDIDATES = ["Date", "Session Date", "Match Date"]


def _flip_name(name: str) -> str:
    """Convert "Surname, First" into "First Surname" for consistent keys."""
    if not isinstance(name, str):
        return name
    if "," in name:
        family, given = name.split(",", 1)
        return f"{given.strip()} {family.strip()}".strip()
    return name.strip()


def _extract_player_key(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a normalised player key suitable for mapping overrides."""
    for col in PLAYER_NAME_CANDIDATES:
        if col not in df.columns:
            continue
        series = df[col].astype("string").str.strip()
        if col == "Name":
            series = series.str.split(" - ").str[0]
        if col in {"Athlete Name", "Player Name"}:
            series = series.apply(_flip_name)
        series = series.str.replace(r"\s+", " ", regex=True)
        series = series.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        if series.notna().any():
            return series.str.lower()
    return None


def _extract_date(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().any():
                return parsed
    return None


def apply_position_overrides(df: pd.DataFrame, position_col: str = "Position") -> pd.DataFrame:
    """Return a copy of *df* with positional labels normalised.

    1. Harmonises legacy labels (e.g. "Half" -> "Half Back").
    2. Drops deprecated labels such as "Half/Hooker" so they can be
       re-populated from player-level overrides.
    3. Uses the latest known position per player (based on available
       date information) to back-fill any missing labels.
    """

    if position_col not in df.columns:
        lowered = position_col.lower()
        fallback = next(
            (c for c in df.columns if c.strip().lower() == lowered),
            None,
        )
        if fallback is None:
            fallback = next(
                (c for c in df.columns if c.strip().lower() in {"position", "position name"}),
                None,
            )
        if fallback is None:
            return df
        position_col = fallback

    out = df.copy()
    positions = out[position_col].astype("string").str.strip()
    positions = positions.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "nil": pd.NA})

    lower = positions.str.lower()
    positions = positions.mask(lower.isin(REMOVED_LABELS_LOWER), pd.NA)

    mapped = lower.map(POSITION_REPLACEMENTS_LOWER)
    replace_mask = lower.isin(POSITION_REPLACEMENTS_LOWER)
    positions = positions.where(~replace_mask, mapped)

    out[position_col] = positions.astype("string")

    player_keys = _extract_player_key(out)
    if player_keys is not None:
        out["_player_key"] = player_keys
        out["_row_order"] = np.arange(len(out), dtype=float)
        date_series = _extract_date(out)
        if date_series is not None:
            out["_player_date"] = date_series
            sort_cols = ["_player_key", "_player_date", "_row_order"]
        else:
            sort_cols = ["_player_key", "_row_order"]

        latest_positions = (
            out.dropna(subset=["_player_key"])
               .dropna(subset=[position_col])
               .sort_values(sort_cols)
               .groupby("_player_key", dropna=False)[position_col]
               .last()
        )
        filled = out["_player_key"].map(latest_positions)
        out[position_col] = out[position_col].fillna(filled)
        out.drop(columns=[c for c in ["_player_key", "_player_date", "_row_order"] if c in out.columns], inplace=True)

    out[position_col] = out[position_col].astype("string").str.strip()
    out[position_col] = out[position_col].replace({"": pd.NA})
    return out
