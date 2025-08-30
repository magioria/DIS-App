import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import glob
from pathlib import Path
import re
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
    (0,    7,  "Solid Contributor",      "#fdd835"),  # yellow
    (7,   13,  "Strong Defender",        "#9ccc65"),  # light green
    (13,  20,  "Elite Defender",         "#43a047"),  # green
    (20,  35,  "Generational / DPOY",    "#1b5e20"),  # dark green
]

BOUNDARIES = [-18, -5, 0, 7, 13, 20, 35]

def plot_dis_scale_with_steps():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    # Draw each colored band
    for low, high, name, color in BINS:
        ax.barh(0, high - low, left=low, color=color, edgecolor="black", height=0.6)
        ax.text((low + high) / 2, 0, name,
                ha="center", va="center", fontsize=10,
                color="black", fontweight="bold")
    # Vertical dashed boundary lines with labels
    for x in BOUNDARIES:
        ax.vlines(x, -0.5, 0.5, linestyles="dashed", linewidth=1.2, color="black")
        ax.text(x, 0.55, f"{x:g}", ha="center", va="bottom", fontsize=10)
    # Axis formatting
    ax.set_xlim(min(BOUNDARIES), max(BOUNDARIES))
    ax.set_ylim(-0.8, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("DIS")
    ax.set_title("DIS Interpretation Scale (green = best, red = worst)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig

def style_table(df: pd.DataFrame):
    styler = df.style
    try:
        styler = styler.hide(axis="index")
    except Exception:
        styler = styler.hide_index()

    def dis_color(val: float) -> str:
        for low, high, _, color in BINS:
            if low <= val < high:
                txt = "white" if color not in ("#fdd835",) else "black"
                return f"background-color: {color}; color: {txt}; font-weight: 600;"
        return ""

    if "DIS" in df.columns:
        styler = styler.apply(
            lambda s: [dis_color(v) for v in s] if s.name == "DIS" else ["" for _ in s],
            axis=0,
        )
    return styler

def _dis_cell_html(v: float) -> str:
    """Return HTML for a colored DIS cell (fast, no Styler)."""
    color = "#e0e0e0"; txt = "black"
    for lo, hi, _, col in BINS:
        if lo <= v < hi:
            color = col
            txt = "black" if col == "#fdd835" else "white"
            break
    return (f"<div style='background:{color};color:{txt};font-weight:600;"
            f"padding:2px 8px;border-radius:6px;text-align:right'>{v:.6f}</div>")

@st.cache_data(ttl=120)
def _slice_to_html(df_slice: pd.DataFrame) -> str:
    """Cache the HTML of a page slice to improve paging speed."""
    html = df_slice.to_html(index=False, escape=False,
                            formatters={"DIS": _dis_cell_html})
    return "<style>table{width:100%!important}</style>" + html

def _page_window(current: int, total: int, window: int = 7) -> list[int]:
    """Return a compact list of page numbers to show (1-based)."""
    if total <= window:
        return list(range(1, total + 1))
    half = window // 2
    start = max(1, current - half)
    end = min(total, start + window - 1)
    start = max(1, end - window + 1)
    return list(range(start, end + 1))

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
            txt = "black" if color == "#fdd835" else "white"
            return name, color, txt
    return "—", "#e0e0e0", "black"

def _percentile_of_value(series: pd.Series, value: float) -> float:
    arr = np.asarray(series.dropna(), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr <= value).mean() * 100.0)

def _pct_color(pct: float) -> str:
    """Return color based on percentile, aligned with DIS categories."""
    if pct < 20:
        return "#c62828"   # red (Poor)
    elif pct < 40:
        return "#ef6c00"   # orange (Below Avg)
    elif pct < 60:
        return "#fdd835"   # yellow (Solid)
    elif pct < 80:
        return "#9ccc65"   # light green (Strong)
    elif pct < 95:
        return "#43a047"   # green (Elite)
    else:
        return "#1b5e20"   # dark green (Generational / DPOY)

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

page = st.sidebar.radio("Navigate", ["What is DIS?", "Leaderboard", "Correlation Analysis"])

if page == "What is DIS?":
    
    st.title("Defensive Impact Score (DIS) 🏀")
    st.header("**What is DIS?**")
    st.markdown("""
    The **Defensive Impact Score**, abbreviated DIS, is a custom stat that estimates how impactful each player is on defense.

    Unlike traditional defensive stats that rely heavily on steals, blocks, or team ratings, DIS blends multiple layers of data, including box score performance, matchup quality, and hustle stats, to assess how **consistently** and **effectively** a player influences defensive outcomes.

    DIS is designed to be **scale-consistent** across seasons, to evaluate players fairly regardless of their position, and to minimize the distortion caused by overall team performance, offering a robust tool for identifying both **elite** and **underrated defenders** who may not show up in highlight reels.

    DIS Interpretation Scale (color legend for tables and charts):

    - 20 or more → Generational / DPOY-level season
    - 13–19.9 → Elite Defender 
    - 7–12.9 → Strong / Above-average defender
    - 0–6.9 → Solid contributor (average to good defense, reliable)
    - -5 to -0.1 → Below average (some defensive weaknesses)
    - Less than -5 → Poor Defender (significant negative impact)
    """)

    st.pyplot(plot_dis_scale_with_steps())

    st.markdown("""
    **Note**: DIS values are standardized so that 0 always represents the league average across the dataset. Positive values = better than average, negative = worse than average.            
    """)
                
    st.markdown("""
    ### How reliable is DIS?

    To test the credibility of DIS, we compared the **Top 20 DIS players each season** with the official **All-Defensive Teams** and **Defensive Player of the Year nominees**.  

    - ✅ **60%** of the time, DIS perfectly matched the official awards selections.  
    - ⚪ **12%** were good matches (players recognized as strong defenders but not officially awarded).  
    - ❌ **28%** were mismatches (strong DIS players overlooked in awards).  

    This validation shows that DIS aligns strongly with how defense is recognized in the NBA, while also highlighting **underrated defenders** who might not receive enough media or voting attention.
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

    # Games filter
    min_games = st.sidebar.slider("Minimum Games Played", min_value=1, max_value=int(df_display["G"].max()), value=1, step=1)

    # Minutes filter
    min_mp = st.sidebar.slider("Minimum Minutes Played", min_value=1, max_value=int(df_display["MP"].max()), value=1, step=50)

    # 65-game rule note
    st.sidebar.markdown("""
    *From the 2023-24 season, the NBA implemented the 65-game rule: players must appear in at least 65 games to be eligible for awards like DPOY and All-Defensive teams.  
    In addition, players must play at least 20 minutes in all but two of those 65 games (≈ **1300 minutes**).*

    *Keep this in mind if you want to filter for the eligible players ;)*
    """)

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

    # Player search
    player = st.sidebar.text_input("Search for a player")

    # Compare multiple players
    player_options = filtered_df["Player"].unique()
    players_to_compare = st.sidebar.multiselect("Compare players", options=sorted(player_options))

    if player:
        result = filtered_df[filtered_df["Player"].str.contains(player, case=False, na=False)]
        if result.empty:
            st.warning("Player not found.")
        else:
            st.dataframe(style_table(result[columns_to_display].rename_axis("Rank")), use_container_width=True)

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
                st.dataframe(
                    style_table(
                    player_hist[["Season","Team","Pos","G","MP","DIS"]]
                    .reset_index(drop=True)
                    .rename(index=lambda x: x + 1)),
                    use_container_width=True
                )
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
        st.dataframe(style_table(comparison_df[columns_to_display]), use_container_width=True)

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
                    pos_df[["Player","Team","MP","DIS"]]
                    .head(10)
                    .reset_index(drop=True)
                    .rename_axis("Rank")
                    .rename(index=lambda x: x + 1)
                )
                avg_pos_dis = round(df[df["Pos"] == pos]["DIS"].mean(), 2)
                avg_pos_filt_dis = round(pos_df["DIS"].mean(), 2)
                with st.expander(f"Top 10 {pos}s"):
                    st.dataframe(style_table(top10), use_container_width=True)
                    st.markdown(f"**Average DIS for all {pos}s:** {avg_pos_dis}")
                    st.markdown(f"**Average DIS for all {pos}s (only filtered players):** {avg_pos_filt_dis}")

elif page == "Correlation Analysis":

    st.header("Correlation Analysis 📊— DIS vs Public Defensive Metrics")

    st.markdown("""
    Why Correlation Analysis?

    The goal of this analysis is to test how well DIS aligns with established defensive metrics that are already used in public basketball analytics, like **D-LEBRON**, **DWS**(Defensive Win Shares) and **DBPM**(Defensive Box Plus Minus).

    If DIS shows a strong positive correlation with these stats, it means it captures many of the same defensive signals that experts and analysts already trust — giving **credibility** and **statistical backing** to the metric.

    At the same time, correlation is not expected to be perfect. DIS was designed to add **new layers of information** (like hustle data, matchup difficulty, contextual adjustments) that are not fully reflected in traditional metrics. So, where correlations are strong, DIS confirms its reliability; where they diverge, DIS provides **unique insights** into aspects of defense that are often overlooked.            
    """)

    # Add legend
    fig, ax = plt.subplots(figsize=(5, 0.4))
    cmap = plt.cm.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-1, vmax=1)
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation="horizontal")
    cb1.set_label("Correlation strength (blue = negative, red = positive)")
    st.pyplot(fig)

    # Load dataset that contains DIS + advanced stats
    df_corr = pd.read_csv(f"{CORRELATION_DIR}/dis_correlation_dataset.csv")

    # Let the user select season
    seasons = sorted(df_corr["Season"].unique(), reverse=True)
    selected_season = st.selectbox("Select season", seasons)

    season_df = df_corr[df_corr["Season"] == selected_season]

    st.subheader(f"Correlations for {selected_season}")

    # columns we care about
    metrics = ["DIS", "D-LEBRON", "DWS", "DBPM"]
    metrics = [c for c in metrics if c in season_df.columns]

    # force numeric and drop non-numeric/NaN rows for corr
    num_df = season_df[metrics].apply(pd.to_numeric, errors="coerce")

    pearson_corr  = num_df.corr(method="pearson")
    spearman_corr = num_df.corr(method="spearman")

    # Function to style correlations with a color gradient
    def style_corr(df):
        return (df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1, axis=None).format("{:.2f}"))

    st.write("**Pearson Correlation Matrix**")
    st.markdown("""
    Measures how strongly two variables move together in a straight-line (linear) way. Example: if DIS goes up when DWS goes up, that’s a high Pearson correlation.            
    """)
    st.dataframe(style_corr(pearson_corr))

    st.write("**Spearman Correlation Matrix**")
    st.markdown("""
    Looks at the rank order instead of exact values. Example: it checks if higher DIS tends to come with higher (or lower) DWS, even if the relationship isn’t perfectly straight-line.            
    """)
    st.dataframe(style_corr(spearman_corr))

    xvar = st.selectbox("X variable", [c for c in ["D-LEBRON", "DWS", "DBPM"] if c in num_df.columns])

    if xvar:
        plot_df = num_df[["DIS", xvar]].dropna()
        fig, ax = plt.subplots()
        ax.scatter(plot_df[xvar], plot_df["DIS"], alpha=0.6)
        r = plot_df["DIS"].corr(plot_df[xvar], method="pearson")
        ax.set_xlabel(xvar); ax.set_ylabel("DIS")
        ax.set_title(f"{xvar} vs DIS (Pearson r = {r:.2f})")
        st.pyplot(fig)