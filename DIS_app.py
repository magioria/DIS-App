import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import math

BASE_DIR = Path(__file__).parent

OUTPUTS_DIR = BASE_DIR / "outputs"
CORRELATION_DIR = BASE_DIR / "correlation"

@st.cache_data
def load_all_seasons():
    files = sorted(OUTPUTS_DIR.glob("DIS_*.csv"))
    dfs = []
    for p in files:
        season = p.stem.split("DIS_")[-1]
        df = pd.read_csv(p)
        df["Season"] = season
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["Player", "Team", "Pos", "G", "MP", "DIS", "Season"])

def season_order_key(s: str):
    try:
        start = int(s.split("-")[0])
        end = int("20" + s.split("-")[1]) if len(s.split("-")[1]) == 2 else int(s.split("-")[1])
        return (start, end)
    except:
        return (0, 0)

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

def plot_dis_scale_with_steps():
    fig, ax = plt.subplots(figsize=(12, 2.8))

    # draw bands
    for low, high, name, color in BINS:
        ax.barh(0, high - low, left=low, color=color,
                edgecolor="black", height=0.66)

    # category labels — all black text, some on two lines
    for low, high, name, color in BINS:
        x = (low + high) / 2

        if name == "Below Average":
            label = "Below\nAverage"
        elif name == "Average Defender":
            label = "Average\nDefender"
        elif name == "Solid Contributor":
            label = "Solid\nContributor"
        elif name == "Strong Defender":
            label = "Strong\nDefender"
        else:
            label = name

        ax.text(x, 0, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="black")

    # boundary ticks + numbers
    for x in BOUNDARIES:
        ax.vlines(x, -0.5, -0.08, color="black", linewidth=1)
        ax.text(x, -0.65, f"{x:g}", ha="center", va="top", fontsize=9)

    # cosmetics
    ax.set_xlim(min(BOUNDARIES), max(BOUNDARIES))
    ax.set_ylim(-0.8, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("DIS")
    ax.set_title("DIS Interpretation Scale (green = best, red = worst)",
                 fontsize=12, fontweight="bold")
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    return fig

def plot_team_dis_scale():
    fig, ax = plt.subplots(figsize=(12, 2.8))

    for low, high, name, color in TEAM_BINS:
        ax.barh(0, high - low, left=low, color=color, edgecolor="black", height=0.6)
        x = (low + high) / 2
        ax.text(x, 0, name.replace(" ", "\n"), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")

    # Draw boundary ticks
    for lo, hi, _, _ in TEAM_BINS:
        ax.vlines(lo, -0.4, -0.1, color="black", linewidth=1)
        ax.text(lo, -0.55, f"{lo:.1f}", ha="center", va="top", fontsize=9)
    ax.vlines(TEAM_BINS[-1][1], -0.4, -0.1, color="black", linewidth=1)
    ax.text(TEAM_BINS[-1][1], -0.55, f"{TEAM_BINS[-1][1]:.1f}", ha="center", va="top", fontsize=9)

    ax.set_xlim(min(lo for lo, _, _, _ in TEAM_BINS), max(hi for _, hi, _, _ in TEAM_BINS))
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Minutes-weighted Team DIS")
    ax.set_title("Team DIS Interpretation Scale", fontsize=12, fontweight="bold")

    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    return fig

def style_table(df: pd.DataFrame) -> str:
    """Return HTML for a table where DIS cells are full-width colored pills."""
    BINS = [
        (-18, -5,  "#c62828"),  # red
        (-5,   0,  "#ef6c00"),  # orange
        (0,    3,  "#ffef8a"),  # light yellow (Average)
        (3,    7,  "#fdd835"),  # yellow       (Solid)
        (7,   13,  "#9ccc65"),  # light green
        (13,  20,  "#43a047"),  # green
        (20,  35,  "#1b5e20"),  # dark green
    ]

    def dis_pill(v: float) -> str:
        color = "#e0e0e0"; txt = "black"
        for lo, hi, col in BINS:
            if lo <= v < hi:
                color = col
                # Always black text for yellows (Average + Solid)
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

    # Render as HTML; use formatter only for DIS
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
    """Return HTML for a colored DIS cell (fast, no Styler)."""
    color = "#e0e0e0"; txt = "black"
    for lo, hi, _, col in BINS:
        if lo <= v < hi:
            color = col
            # Always black text for yellows (Average + Solid)
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
            if col in ("#fdd835", "#ffef8a"):  # yellows
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

    # Only adjust headers, leave table style intact
    html = html.replace("<th>", "<th style='text-align:left;'>")

    return html

# ----- Callbacks (single source of truth) -----
def _set_page(key: str, n_pages: int, set_to: int | None = None, delta: int | None = None):
    """Single source of truth: update page in session_state (0-based)."""
    page_key = f"{key}_page"
    page = int(st.session_state.get(page_key, 0))
    if set_to is not None:
        page = int(set_to)
    elif delta is not None:
        page = page + int(delta)
    page = max(0, min(page, n_pages - 1))
    st.session_state[page_key] = page  # <-- do NOT touch jump here

def _jump_changed(key: str, n_pages: int):
    """When the jump number_input changes, set page accordingly."""
    jump_key = f"{key}_jump_val"
    val = int(st.session_state.get(jump_key, 1))
    val = max(1, min(val, n_pages))
    _set_page(key, n_pages, set_to=val - 1)

def _ps_changed(key: str):
    """When rows-per-page changes, reset to first page."""
    st.session_state[f"{key}_page"] = 0

def render_leaderboard(df: pd.DataFrame, key: str = "lb",
                       page_size_options=(10, 25, 50, 100),
                       default_page_size=25,
                       sort_by="DIS", descending=True):
    """
    Stable, mobile-friendly leaderboard:
      - 'per page' selector (top-left)
      - Rank column (global)
      - Footer row: Prev • Next • Page [jump]
      - Pagination controlled via callbacks (no double-advance)
    """
    # Sort once (defines Rank)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)

    # ---- keys ----
    page_key = f"{key}_page"
    ps_key   = f"{key}_ps"
    jump_key = f"{key}_jump_val"

    # init only page (do NOT pre-seed ps/jump)
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    # ---- TOP: rows-per-page selector ----
    top = st.columns([2, 8])
    with top[0]:
        current_ps = st.session_state.get(ps_key, default_page_size)
        try:
            idx = page_size_options.index(current_ps)
        except ValueError:
            idx = page_size_options.index(default_page_size)

        st.selectbox(
            "per page",
            page_size_options,
            index=idx,                # widget provides default
            key=ps_key,               # widget owns its value
            on_change=_ps_changed,    # reset to page 0
            args=(key,),
        )
    with top[1]:
        st.write("")

    page_size = int(st.session_state.get(ps_key, page_size_options[idx]))

    # ---- pagination math ----
    n_rows  = len(df)
    n_pages = max(1, math.ceil(n_rows / page_size))
    page    = int(st.session_state.get(page_key, 0))
    page    = max(0, min(page, n_pages - 1))
    st.session_state[page_key] = page

    start, end = page * page_size, min((page + 1) * page_size, n_rows)

    # ---- slice + Rank ----
    slice_df = df.iloc[start:end].copy()
    slice_df.insert(0, "Rank", np.arange(start + 1, end + 1))
    base_cols = ["Rank", "Player", "Team", "Pos", "G", "MP", "DIS"]
    visible = [c for c in base_cols if c in slice_df.columns]
    slice_df = slice_df[visible]

    # ---- table (fast HTML, no nested scroll) ----
    st.markdown(_slice_to_html(slice_df), unsafe_allow_html=True)

    # ---- FOOTER: info left, controls right (Prev • Next • Page [jump]) ----
    bottom = st.columns([6, 6])
    with bottom[0]:
        st.caption(f"{start + 1} to {end} of {n_rows:,}")

    with bottom[1]:
        prev_col, jump_col, next_col = st.columns([2, 3, 2])

        with prev_col:
            st.button(
                "◀ Prev", use_container_width=True,
                disabled=(page == 0),
                on_click=_set_page, args=(key, n_pages),
                kwargs={"delta": -1},
                key=f"{key}_prev_btn",
            )

        with next_col:
            st.button(
                "Next ▶", use_container_width=True,
                disabled=(page >= n_pages - 1),
                on_click=_set_page, args=(key, n_pages),
                kwargs={"delta": +1},
                key=f"{key}_next_btn",
            )

        with jump_col:
            st.number_input(
                "Page",
                min_value=1, max_value=n_pages, step=1,
                value=page + 1,            # <- supply the current value here
                key=jump_key,
                on_change=_jump_changed,   # <- callback sets page
                args=(key, n_pages),
            )

def _dis_category(dis: float):
    for lo, hi, name, color in BINS:
        if lo <= dis < hi:
            # Always black for yellow shades (Solid + Average)
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

def _pct_color(pct: float) -> str:
    """
    Color by league percentile, matched to actual DIS distribution
    (cumulative cutoffs from your data).
    """
    if pct < 24.0:
        return "#c62828"   # Poor
    elif pct < 56.0:
        return "#ef6c00"   # Below Avg
    elif pct < 71.0:
        return "#ffef8a"   # Average
    elif pct < 85.0:
        return "#fdd835"   # Solid
    elif pct < 95.5:
        return "#9ccc65"   # Strong
    elif pct < 99.3:
        return "#43a047"   # Elite
    else:
        return "#1b5e20"   # Generational / DPOY

def _pct_bar(label: str, pct: float):
    fig, ax = plt.subplots(figsize=(5.6, 0.5))
    left = np.clip(pct, 0, 100)
    ax.barh([0], [left], color=_pct_color(left))
    ax.barh([0], [100 - left], left=[left], color="#e0e0e0")
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel(label)
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_visible(False)
    ax.text(left, 0, f" {left:.0f}%", va="center", ha="left", fontsize=11, color="black")
    st.pyplot(fig, use_container_width=False)

def _league_hist_with_marker(df_season: pd.DataFrame, player_dis: float, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(df_season["DIS"], bins=30, color="#90caf9", edgecolor="black", alpha=0.75)
    ax.axvline(player_dis, color="red", linewidth=2, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("DIS"); ax.set_ylabel("Number of players")
    st.pyplot(fig, use_container_width=True)

def show_player_profile(player_row: pd.Series, df_season: pd.DataFrame):
    """Render a player profile card + optional histogram.
       - player_row: a single row (for the selected player) from df_season
       - df_season: the full season dataframe (visible, public CSV)
    """
    player_name = str(player_row["Player"])
    player_pos  = str(player_row.get("Pos", "—"))
    player_dis  = float(player_row["DIS"])
    cat, color, txt = _dis_category(player_dis)

    # Header
    st.subheader(f"{player_name} — {player_pos}")

    st.markdown("""
    Percentiles show how a player ranks compared to others. For example, 90th percentile = better than 90% of the league.            
    """)

    # DIS badge
    st.markdown(
        f"<div style='display:inline-block;padding:6px 10px;border-radius:10px;"
        f"background:{color};color:{txt};font-weight:700'>"
        f"DIS {player_dis:.2f} · {cat}</div>",
        unsafe_allow_html=True
    )

    # Percentiles
    league_pct = _percentile_of_value(df_season["DIS"], player_dis)
    _pct_bar("League percentile (this season)", league_pct)

    if "Pos" in df_season.columns:
        pos_series = df_season.loc[df_season["Pos"] == player_pos, "DIS"]
        if len(pos_series) >= 5:
            pos_pct = _percentile_of_value(pos_series, player_dis)
            _pct_bar(f"Percentile among {player_pos} (this season)", pos_pct)

    # Optional: distribution context
    with st.expander("Show player in league distribution"):
        _league_hist_with_marker(
            df_season, player_dis,
            title="League Distribution of DIS (player highlighted)"
        )

def show_team_profile(team: str, df_season: pd.DataFrame):
    """Render a team profile card with current row, DIS badge, history line chart, and history table."""
    # Current-season weighted DIS for the team
    team_df = df_season[df_season["Team"] == team]
    team_dis = (team_df["DIS"] * team_df["MP"]).sum() / team_df["MP"].sum()
    cat, color, txt = _team_dis_category(team_dis)

    st.subheader(f"{team} — Team Profile")

    # Current season row (team leaderboard style)
    team_season = (
        df_season.groupby("Team")
        .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
        .reset_index(name="DIS")
        .sort_values("DIS", ascending=False)
        .reset_index(drop=True)
    )

    # Add rank
    team_season.insert(0, "Rank", range(1, len(team_season) + 1))

    # Add player count
    players_count = df_season.groupby("Team")["Player"].count().reset_index(name="Players")
    team_season = team_season.merge(players_count, on="Team")

    # Filter to just this team
    team_row = team_season[team_season["Team"] == team][["Rank", "Team", "Players", "DIS"]]

    # Rename columns for display
    team_row = team_row.rename(columns={"DIS": "Team DIS"})

    team_row["Team DIS"] = team_row["Team DIS"].round(2)

    # Show styled row (colored DIS pill)
    st.markdown(_slice_to_html_team(team_row), unsafe_allow_html=True)

    # DIS badge
    st.markdown(
        f"<div style='display:inline-block;padding:6px 10px;border-radius:10px;"
        f"background:{color};color:{txt};font-weight:700'>"
        f"Team DIS {team_dis:.2f} · {cat}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # History across all seasons
    all_dis = load_all_seasons()
    team_hist = (
        all_dis[all_dis["Team"].str.lower() == team.lower()]
        .groupby("Season")
        .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
        .reset_index(name="Team DIS")
        .sort_values("Season", key=lambda s: s.map(season_order_key)))

    if not team_hist.empty:
        st.subheader(f"{team} — DIS History")

        # Line chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(team_hist["Season"], team_hist["Team DIS"], marker="o", linewidth=2)
        ax.set_xlabel("Season"); ax.set_ylabel("Team DIS")
        ax.set_title("DIS over seasons", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # History table (colored DIS cells, no Rank column)
        hist_tbl = team_hist.sort_values("Season", ascending=False).reset_index(drop=True)
        st.markdown(_slice_to_html_team(hist_tbl), unsafe_allow_html=True)
    else:
        st.info("No multi-season history found for this team.")

page = st.sidebar.radio("Navigate", ["What is DIS?", "Player Leaderboard", "Team Leaderboard"])

if page == "What is DIS?":
    
    st.title("Defensive Impact Score (DIS) 🏀")
    st.header("**What is DIS?**")
    st.markdown("""
    The **Defensive Impact Score**, abbreviated DIS, is a custom stat that estimates how impactful each player is on defense.

    Unlike traditional defensive stats that rely heavily on steals, blocks, or team ratings, DIS blends multiple layers of data, including box score performance, matchup quality, and hustle stats, to assess how **consistently** and **effectively** a player influences defensive outcomes.

    DIS is designed to be **scale-consistent** across seasons, to evaluate players fairly regardless of their position, and to minimize the distortion caused by overall team performance, offering a robust tool for identifying both **elite** and **underrated defenders** who may not show up in highlight reels.
    """)     

    st.divider()       
    
    st.header("**DIS Interpretation Scale**")

    st.markdown("""
    Values are standardized so that 0 = league average. Positive = better than average, negative = worse.
                
    - 20 or more → Generational / DPOY-level season
    - 13–19.9 → Elite Defender 
    - 7–12.9 → Strong defender
    - 3–6.9 → Solid contributor
    - 0–2.9 → Average defender
    - -5 to -0.1 → Below average
    - Less than -5 → Poor Defender
    """)

    st.pyplot(plot_dis_scale_with_steps())

    st.divider()

    st.header("**How reliable is DIS?**")
                
    st.markdown("""  
    To test the credibility of DIS, we compared **all Defensive Player of the Year nominees** and **All-Defensive Team selections** with the **Top 25 DIS players** from each season.  

    - ✅ **78%** of the time, those official award players were also Top 25 in DIS, confirming strong alignment.  
    - ❌ **22%** of the time, official selections had a lower DIS than expected, while higher-DIS players were overlooked.  

    This validation shows that DIS is highly consistent with how defense is recognized in the NBA, while also uncovering **underrated defenders** who may not receive the same level of media coverage or voting recognition. 
    
    To check its reliability, DIS was also compared with established defensive metrics like **D-LEBRON**, Defensive Win Shares (**DWS**), and Defensive Box Plus Minus (**DBPM**). 
    The correlations are **strong**, meaning DIS captures many of the same defensive signals these trusted stats recognize. 
                
    But the **differences matter**: DIS also highlights players whose defensive value isn’t fully reflected in box score production or plus-minus models, adding **new layers of insight** into what makes a player impactful on defense.                       
    """)

    st.divider()

    st.header("**Why it matters**")
    
    st.markdown("""
    DIS brings together the best parts of existing defensive stats while fixing their biggest weaknesses:

    - **More consistent** than box score stats, because it looks beyond steals and blocks.
    - **More stable** than plus-minus models, reducing noise from teammates and lineup context.
    - **More fair** across positions, letting rim protectors, wings, and guards be compared on the same scale.
    - Able to spot **hidden gems**, highlighting defenders who play key roles but often go unnoticed.

    In short, DIS gives you a clearer, more complete picture of **who really changes the game** on defense.         
    """)
    
    st.divider()

    st.markdown("""
    Want to see how players rank by DIS? 👉 Check out the **Player Leaderboard** page to explore the top and bottom defenders.             
    """)

    st.divider()

    st.header("**How Team DIS is computed**")

    st.markdown("""
    For teams, we don’t simply average the DIS of all players.  
    Instead, we use a **minutes-weighted average**:

    $$
    \text{Team DIS} = \frac{\sum_i (DIS_i \times Minutes_i)}{\sum_i Minutes_i}
    $$

    This ensures that players who spend more time on the court have more influence on their team’s DIS,
    while players with very few minutes don’t distort the average.
    """)

    st.divider()

    st.header("**Team DIS Interpretation Scale**")

    st.markdown("""
    Just like with players, teams can be grouped into categories:

    - 6.0 or more → **Elite / Championship Defense**
    - 4.3 to 6.0 → **Strong Defensive Team**
    - 2.8 to 4.3 → **Solid Defensive Team**
    - 0.7 to 2.8 → **Average**
    - -1.3 to 0.7 → **Below Average**
    - Less than -1.3 → **Poor Defensive Team**

    """)

    st.pyplot(plot_team_dis_scale())

    st.markdown("""
    Want to see how teams rank by DIS? 👉 Check out the **Team Leaderboard** page to explore the best and worst defensive teams.             
    """)

elif page == "Player Leaderboard":

    # Map of season names to file paths (built from /data contents)
    season_files = {p.stem.split("DIS_")[1]: p for p in OUTPUTS_DIR.glob("DIS_*.csv")}
    seasons_sorted = sorted(season_files.keys(), reverse=True)

    # Sidebar selector
    st.sidebar.title("Filters")
    selected_season = st.sidebar.selectbox("Select season", seasons_sorted, index=0)

    # Load the selected season's data
    df = pd.read_csv(season_files[selected_season])

    # Select and clean columns
    columns_to_display = ["Player", "Team", "Pos", "G", "MP", "DIS"]
    df_display = df.copy()

    # Player search
    all_players = sorted(df_display["Player"].unique())
    player = st.sidebar.selectbox("🔍 Search a Player to view full Profile & History", options=[""] + all_players, index=0)
    
    # Games filter
    min_games = st.sidebar.number_input("Minimum Games Played", min_value=1, max_value=int(df_display["G"].max()), value=1, step=1, format="%d",)

    # Minutes filter
    min_mp = st.sidebar.number_input("Minimum Minutes Played", min_value=1, max_value=int(df_display["MP"].max()), value=1, step=50, format="%d",)

    # --- NBA award eligibility toggle ---
    eligible_only = st.sidebar.checkbox("Only show award-eligible players (NBA 65-game rule)", value=False)
    if eligible_only:
        min_games = max(min_games, 65)
        min_mp = max(min_mp, 1300)

    # 65-game rule note
    pop = st.sidebar.popover("ℹ️ What is the 65-game rule?")
    with pop:
        st.markdown("""
        *From the 2023-24 season, the NBA implemented the 65-game rule: players must appear in at least 65 games to be eligible for awards like DPOY and All-Defensive teams.  
        In addition, players must play at least 20 minutes in all but two of those 65 games (≈ **1300 minutes**).*

        *Keep this in mind if you want to filter for the eligible players ;)*
        """)

    dis_min = st.sidebar.number_input("Minimum DIS", min_value=float(df_display["DIS"].min()), max_value=float(df_display["DIS"].max()), value=float(df_display["DIS"].min()), step=0.5, format="%.1f",)

    # Team filter
    teams = df_display["Team"].unique()
    selected_team = st.sidebar.selectbox("Filter by team", options=["All"] + sorted(teams))

    # Position filter
    positions = df["Pos"].dropna().unique()
    selected_pos = st.sidebar.selectbox("Filter by position", options=["All"] + sorted(positions))

    # Apply filters
    filtered_df = df_display.copy()
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["Team"] == selected_team]
    if selected_pos != "All":
        filtered_df = filtered_df[filtered_df["Pos"] == selected_pos]
    filtered_df = filtered_df[filtered_df["MP"] >= min_mp]
    filtered_df = filtered_df[filtered_df["G"] >= min_games]
    filtered_df = filtered_df[filtered_df["DIS"] >= dis_min]

    # Compare multiple players
    player_options = filtered_df["Player"].unique()
    players_to_compare = st.sidebar.multiselect("Compare players", options=sorted(player_options))

    if player:
        result = filtered_df[filtered_df["Player"].str.contains(player, case=False, na=False)]
        if result.empty:
            st.warning("Player not found.")
        else:
            result_tbl = result[columns_to_display].reset_index(drop=True).copy()
            # compute global rank for each player row based on their DIS value
            result_tbl.insert(0,"Rank", result_tbl["DIS"].apply(lambda x: int((df["DIS"] > x).sum() + 1)))
            st.markdown(_slice_to_html(result_tbl), unsafe_allow_html=True)


        if len(result) == 1:
            player_row = result.iloc[0]

            show_player_profile(player_row, df)             
            # df_season = current season dataframe
            # player_row = the row for the player user selected (e.g., via st.dataframe selection or a selectbox)

            st.divider()
         
            # Player DIS History
            all_dis = load_all_seasons()
            player_name = player_row["Player"]
            player_hist = all_dis[all_dis["Player"].str.lower() == player_name.lower()].copy()
            if not player_hist.empty:
                player_hist = player_hist.sort_values("Season", key=lambda s: s.map(season_order_key))
                st.subheader(f"{player_name} — DIS History")
                # Static (non-interactive) line plot
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(player_hist["Season"], player_hist["DIS"], marker="o", linewidth=2)

                ax.set_xlabel("Season")
                ax.set_ylabel("DIS")
                ax.set_title("DIS over seasons", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)

                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                hist_tbl = (player_hist.sort_values("Season", ascending=False)[["Season", "Team", "Pos", "G", "MP", "DIS"]].reset_index(drop=True).copy())
                st.markdown(_slice_to_html(hist_tbl), unsafe_allow_html=True)

                # ✅ Average DIS across all seasons this player actually played
                avg_dis_pl = round(player_hist["DIS"].astype(float).mean(), 2)
                st.markdown(f"**Average DIS for {player_name}:** {avg_dis_pl}")

                csv_bytes = player_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Player DIS History (CSV)",
                    data=csv_bytes,
                    file_name=f"{player_name.replace(' ', '_')}_DIS_history.csv",
                    mime="text/csv"
                )
            else:
                st.info("No multi-season history found for this player.")

    elif players_to_compare:
        comparison_df = filtered_df[filtered_df["Player"].isin(players_to_compare)]
        st.subheader("Player Comparison")
        tbl = comparison_df[columns_to_display].reset_index(drop=True).copy()
        tbl.insert(0, "Rank", tbl["DIS"].apply(lambda x: int((df["DIS"] > x).sum() + 1)))
        st.markdown(_slice_to_html(tbl), unsafe_allow_html=True)

        # ── Multi-season line chart (one line per selected player)
        st.subheader("DIS History Comparison(multi-season)")

        all_dis = load_all_seasons()
        hist = all_dis[all_dis["Player"].isin(players_to_compare)].copy()

        if hist.empty:
            st.info("No multi-season history available for the selected players.")
        else:
            # --- 1. build global chronological season order
            all_seasons = sorted(hist["Season"].unique(), key=season_order_key)

            fig, ax = plt.subplots(figsize=(10, 4))

            # --- 2. plot each player aligned on same x-axis
            for name, g in hist.groupby("Player"):
                g = g.set_index("Season").reindex(all_seasons)   # align to global order
                ax.plot(all_seasons, g["DIS"], marker="o", linewidth=2, label=name)

            ax.set_xlabel("Season")
            ax.set_ylabel("DIS")
            ax.set_title("DIS over seasons", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            plt.xticks(rotation=45, ha="right")
            ax.legend(title="Player", fontsize=9, frameon=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    else:
        if filtered_df.empty:
            st.warning("No players match the selected filters.")
        else:
            st.caption(
                f"Filters • Season: {selected_season} • G ≥ {min_games} • "
                f"MP ≥ {min_mp} • DIS ≥ {dis_min:.1f} "
                + (f"• Team: {selected_team}" if selected_team != 'All' else "")
                + (f" • Pos: {selected_pos}" if selected_pos != 'All' else ""))
            st.subheader(f"Top Defensive Players — {selected_season} Season")
            render_leaderboard(filtered_df, key="main_lb",
                   page_size_options=(10,25,50,100),
                   default_page_size=25,
                   sort_by="DIS", descending=True)

        avg_dis = round(filtered_df["DIS"].mean(), 3)
        st.metric(label="Average DIS", value=avg_dis)
        st.markdown("""
        **Note**: DIS values are standardized so that 0 always represents the league average across the dataset. Positive values = better than average, negative = worse than average.            
        """)
        st.metric(label="Players shown", value=len(filtered_df))

        # Top Defenders by Position
        st.subheader("Top Defenders by Position")
        positions_to_show = ["PG", "SG", "SF", "PF", "C"]
        for pos in positions_to_show:
            pos_df = filtered_df[filtered_df["Pos"] == pos].sort_values(by="DIS", ascending=False)
            if not pos_df.empty:
                top10 = (
                    pos_df[["Player","Team","G", "MP","DIS"]]
                    .head(10)
                    .reset_index(drop=True)
                    .rename_axis("Rank")
                    .rename(index=lambda x: x + 1))
                top10.insert(0, "Rank", np.arange(1, len(top10) + 1))
                avg_pos_dis = round(df[df["Pos"] == pos]["DIS"].mean(), 2)
                avg_pos_filt_dis = round(pos_df["DIS"].mean(), 2)
                with st.expander(f"Top 10 {pos}s"):
                    st.markdown(_slice_to_html(top10), unsafe_allow_html=True)
                    st.markdown(f"**Average DIS for all {pos}s:** {avg_pos_dis}")
                    st.markdown(f"**Average DIS for all {pos}s (only filtered players):** {avg_pos_filt_dis}")

elif page == "Team Leaderboard":
    # Map of season names to file paths (built from /data contents)
    season_files = {p.stem.split("DIS_")[1]: p for p in OUTPUTS_DIR.glob("DIS_*.csv")}
    seasons_sorted = sorted(season_files.keys(), reverse=True)

    # Sidebar selector
    st.sidebar.title("Filters")
    selected_season = st.sidebar.selectbox("Select season", seasons_sorted, index=0)

    # Load the selected season's data
    df_display = pd.read_csv(season_files[selected_season])

    # Team search
    all_teams = sorted(df_display["Team"].unique())
    team = st.sidebar.selectbox("🔍 Search a Team to view Profile & History", options=[""] + all_teams, index=0)
        
    #Compare Teams
    teams_to_compare = st.sidebar.multiselect("Compare Teams", options=sorted(df_display["Team"].unique()), default=[])

    if team:
        show_team_profile(team, df_display)
        st.stop()

    elif teams_to_compare:
        comparison_df = (
            df_display.groupby("Team")
            .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
            .reset_index(name="Team DIS")
            .sort_values("Team DIS", ascending=False)
            .reset_index(drop=True))

        # Add ranks
        comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

        # Add players count
        players_count = df_display.groupby("Team")["Player"].count().reset_index(name="Players")
        comparison_df = comparison_df.merge(players_count, on="Team")

        # Filter to selected teams
        comparison_df = comparison_df[comparison_df["Team"].isin(teams_to_compare)]

        st.subheader("Team Comparison")
        st.markdown(_slice_to_html_team(comparison_df), unsafe_allow_html=True)

        # --- Multi-season line chart ---
        st.subheader("Team DIS History Comparison (multi-season)")

        all_dis = load_all_seasons()
        hist = (
            all_dis[all_dis["Team"].isin(teams_to_compare)]
            .groupby(["Season", "Team"])
            .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
            .reset_index(name="Team DIS"))

        if hist.empty:
            st.info("No multi-season history available for the selected teams.")
        else:
            all_seasons = sorted(hist["Season"].unique(), key=season_order_key)

            fig, ax = plt.subplots(figsize=(10, 4))
            for name, g in hist.groupby("Team"):
                g = g.set_index("Season").reindex(all_seasons)
                ax.plot(all_seasons, g["Team DIS"], marker="o", linewidth=2, label=name)

            ax.set_xlabel("Season")
            ax.set_ylabel("Team DIS")
            ax.set_title("Team DIS over seasons", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            plt.xticks(rotation=45, ha="right")
            ax.legend(title="Team", fontsize=9, frameon=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    else:
        st.subheader(f"Team DIS Leaderboard (Minutes-Weighted) — {selected_season}")

        # Minutes-weighted average DIS
        team_weighted = (
            df_display.groupby("Team")
            .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
            .reset_index(name="DIS")
            .sort_values("DIS", ascending=False))

        # Optional: also show how many players per team contributed
        team_weighted["Players considered"] = df_display.groupby("Team")["Player"].count().values

        team_weighted["DIS"] = team_weighted["DIS"].round(2)
        team_weighted = team_weighted.rename(columns={"DIS": "Team DIS"})

        # Add rank column
        team_weighted.insert(0, "Rank", range(1, len(team_weighted) + 1))

        st.markdown(_slice_to_html_team(team_weighted), unsafe_allow_html=True)

        st.divider()
        st.pyplot(plot_team_dis_scale()) 
  