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
            f"padding:4px 10px; border-radius:8px; text-align:right'>{v:.6f}</div>"
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
            f"padding:2px 8px;border-radius:6px;text-align:right'>{v:.6f}</div>")

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

page = st.sidebar.radio("Navigate", ["What is DIS?", "Leaderboard"])

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

    st.header("**Why it matters")
    
    st.markdown("""
    DIS brings together the best parts of existing defensive stats while fixing their biggest weaknesses:

    - More consistent than box score stats, because it looks beyond steals and blocks.
    - More stable than plus-minus models, reducing noise from teammates and lineup context.
    - More fair across positions, letting rim protectors, wings, and guards be compared on the same scale.
    - Able to spot hidden gems, highlighting defenders who play key roles but often go unnoticed.

    In short, DIS gives you a clearer, more complete picture of who really changes the game on defense.         
    """)

    st.divider()

    st.markdown("""
    Want to see how players rank by DIS? 👉 Check out the **Leaderboard** page to explore the top and bottom defenders.             
    """)

elif page == "Leaderboard":

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
    player = st.sidebar.text_input("🔍 **Search a Player to view full Profile & History**")
    
    # Games filter
    min_games = st.sidebar.slider("Minimum Games Played", min_value=1, max_value=int(df_display["G"].max()), value=1, step=1)

    # Minutes filter
    max_minutes = int(df_display["MP"].max())
    minute_options = [1] + list(range(50, max_minutes + 1, 50))
    min_mp = st.sidebar.select_slider("Minimum Minutes Played", options=minute_options, value=1)

    # --- NBA award eligibility toggle ---
    eligible_only = st.sidebar.checkbox("Only show award-eligible players (NBA 65-game rule)", value=False)
    if eligible_only:
        # Approximate NBA requirement: 65 games & 1,300 total minutes
        min_games = max(min_games, 65)
        min_mp = max(min_mp, 1300)

    # 65-game rule note
    st.sidebar.markdown("""
    *From the 2023-24 season, the NBA implemented the 65-game rule: players must appear in at least 65 games to be eligible for awards like DPOY and All-Defensive teams.  
    In addition, players must play at least 20 minutes in all but two of those 65 games (≈ **1300 minutes**).*

    *Keep this in mind if you want to filter for the eligible players ;)*
    """)

    dis_min = st.sidebar.slider(
        "Minimum DIS (threshold)",
        min_value=float(df_display["DIS"].min()),
        max_value=float(df_display["DIS"].max()),
        value=float(df_display["DIS"].min()),  # default = minimum DIS
        step=0.1
    )

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
                st.line_chart(player_hist.set_index("Season")["DIS"])
                hist_tbl = player_hist[["Season","Team","Pos","G","MP","DIS"]].reset_index(drop=True).copy()
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

        # Bar chart
        st.subheader("DIS Comparison Chart")
        plt.figure(figsize=(10, 5))
        bars = plt.bar(comparison_df["Player"], comparison_df["DIS"])
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h/2, f"{h:.2f}",
                     ha='center', va='center', fontsize=15)
        plt.ylabel("Defensive Impact Score")
        plt.xticks(rotation=45, ha='right')
        plt.title("DIS for Selected Players")
        plt.tight_layout()
        st.pyplot(plt)

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