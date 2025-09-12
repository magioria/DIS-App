import pandas as pd
import numpy as np
import streamlit as st

# Categories and thresholds
BINS = [
    (-18, -5,  "Poor Defender",          "#c62828"),  # red
    (-5,   0,  "Below Average",          "#ef6c00"),  # orange
    (0,    3,  "Average Defender",       "#ffef8a"),  # light yellow
    (3,    7,  "Solid Contributor",      "#fdd835"),  # yellow
    (7,   13,  "Strong Defender",        "#9ccc65"),  # light green
    (13,  20,  "Elite Defender",         "#43a047"),  # green
    (20,  35,  "Generational / DPOY",    "#1b5e20"),  # dark green
]
BOUNDARIES = [-18, -5, 0, 3, 7, 13, 20, 35]

TEAM_BINS = [
    (-5, -1.3, "Poor Defensive Team", "#c62828"),   # dark red
    (-1.3, 0.7, "Below Average", "#ef6c00"),            # orange
    (0.7, 2.8, "Average", "#ffef8a"),                   # light yellow
    (2.8, 4.3, "Solid Defensive Team", "#fdd835"),      # yellow
    (4.3, 6.0, "Strong Defensive Team", "#9ccc65"),     # light green
    (6.0, 10, "Elite / Championship Defense", "#1b5e20") # dark green
]

def style_table(df: pd.DataFrame) -> str:
    """Return HTML for a table where DIS cells are full-width colored pills."""
    BINS = [
        (-18, -5,  "#c62828"),  # red
        (-5,   0,  "#ef6c00"),  # orange
        (0,    3,  "#ffef8a"),  # light yellow 
        (3,    7,  "#fdd835"),  # yellow      
        (7,   13,  "#9ccc65"),  # light green
        (13,  20,  "#43a047"),  # green
        (20,  35,  "#1b5e20"),  # dark green
    ]

    def dis_pill(v: float) -> str:
        color = "#e0e0e0"; txt = "black"
        for lo, hi, col in BINS:
            if lo <= v < hi:
                color = col
                if col in ("#fdd835", "#ffef8a"):
                    txt = "black"
                else:
                    txt = "white"
                break
        return (
            f"<div style='width:100%; display:block; box-sizing:border-box; "
            f"background:{color}; color:{txt}; font-weight:600; "
            f"padding:4px 10px; border-radius:8px; text-align:left'>{v:.2f}</div>"
        )

    html = df.to_html(index=False, escape=False,
                      formatters={"DIS": dis_pill})

    # Small CSS to make table span container and align nicely
    css = """
    <style>
      table { width:100% !important; border-collapse:separate; border-spacing:0 6px; }
      th, td { padding:6px 8px; vertical-align:middle; }
    </style>
    """
    return css + html

def _dis_cell_html(v: float) -> str:
    """Return HTML for a colored DIS cell."""
    color = "#e0e0e0"; txt = "black"
    for lo, hi, _, col in BINS:
        if lo <= v < hi:
            color = col
            if col in ("#fdd835", "#ffef8a"):
                txt = "black"
            else:
                txt = "white"
            break
    return (f"<div style='background:{color};color:{txt};font-weight:600;"
            f"padding:2px 8px;border-radius:6px;text-align:left'>{v:.2f}</div>")

def _team_dis_cell_html(v: float) -> str:
    """Return HTML for a colored Team DIS cell."""
    color = "#e0e0e0"; txt = "black"
    for lo, hi, _, col in TEAM_BINS:
        if lo <= v < hi:
            if col in ("#fdd835", "#ffef8a"):
                txt = "black"
            else:
                txt = "white"
            color = col
            break
    return (f"<div style='background:{color};color:{txt};font-weight:600;"
            f"padding:2px 8px;border-radius:6px;text-align:left'>{v:.2f}</div>")

def _slice_to_html_team(df_slice: pd.DataFrame) -> str:
    formatters = {}
    if "DIS" in df_slice.columns:
        formatters["DIS"] = _team_dis_cell_html
    if "Team DIS" in df_slice.columns:
        formatters["Team DIS"] = _team_dis_cell_html

    html = df_slice.to_html(index=False, escape=False, formatters=formatters)
    html = html.replace("<th>", "<th style='text-align:left;'>")
    return html

@st.cache_data(ttl=120)
def _slice_to_html(df_slice: pd.DataFrame) -> str:
    """Cache the HTML of a page slice to improve paging speed."""
    html = df_slice.to_html(
        index=False,
        escape=False,
        formatters={"DIS": _dis_cell_html}
    )

    html = html.replace("<th>", "<th style='text-align:left;'>")

    return html


def _dis_category(dis: float):
    for lo, hi, name, color in BINS:
        if lo <= dis < hi:
            if color in ("#fdd835", "#ffef8a"):
                txt = "black"
            else:
                txt = "white"
            return name, color, txt
    return "—", "#e0e0e0", "black"

def _team_dis_category(v: float):
    for lo, hi, name, color in TEAM_BINS:
        if lo <= v < hi:
            txt = "black" if color in ("#fdd835", "#ffef8a") else "white"
            return name, color, txt
    return "—", "#e0e0e0", "black"

def _percentile_of_value(series: pd.Series, value: float) -> float:
    arr = np.asarray(series.dropna(), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr <= value).mean() * 100.0)
