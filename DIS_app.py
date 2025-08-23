import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import glob
from pathlib import Path
import re

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

def render_intro():
    # Title + description visible on BOTH pages
    st.title("Defensive Impact Score (DIS) 🏀")
    st.header("**What is DIS?**")
    st.markdown("""
The **Defensive Impact Score**, abbreviated DIS, is a custom stat that estimates how impactful each player is on defense.

Unlike traditional defensive stats that rely heavily on steals, blocks, or team ratings, DIS blends multiple layers of data, including box score performance, matchup quality, and contextual adjustments, to assess how **consistently** and **effectively** a player influences defensive outcomes.

DIS is designed to be **scale-consistent** across seasons, position-agnostic, and resilient to team-level noise, offering a robust tool for identifying both **elite** and **underrated defenders** who may not show up in highlight reels.

It can be interpreted like this:

- 25 or more → DPOY Season
- 20–24.9 → Elite Defender
- 13–19.9 → Amazing Defender
- 7–12.9 → Good Defender
- -5 to 6.9 → Average Defender (0 = dataset average)
- Less than -5 → Poor Defender
                
### How reliable is DIS?

To test the credibility of DIS, we compared the **Top 20 DIS players each season** with the official **All-Defensive Teams** and **Defensive Player of the Year nominees**.  

- ✅ **60%** of the time, DIS perfectly matched the official awards selections.  
- ⚪ **12%** were good matches (players recognized as strong defenders but not officially awarded).  
- ❌ **28%** were mismatches (strong DIS players overlooked in awards).  

This validation shows that DIS aligns strongly with how defense is recognized in the NBA, while also highlighting **underrated defenders** who might not receive enough media or voting attention.
""")
    st.divider()

page = st.sidebar.radio("Navigate", ["Leaderboard", "Correlation Analysis"])

if page == "Correlation Analysis":
    render_intro()

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

else:
    render_intro()

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
    min_games = st.sidebar.slider("Minimum Games Played", min_value=1, max_value=int(df_display["G"].max()), value=40, step=1)

    # Minutes filter
    min_mp = st.sidebar.slider("Minimum Minutes Played", min_value=1, max_value=int(df_display["MP"].max()), value=1000, step=50)

    # 65-game rule note
    st.sidebar.markdown("""
*From the 2023-24 season, the NBA implemented the 65-game rule: players must appear in at least 65 games to be eligible for awards like DPOY and All-Defensive teams.  
In addition, players must play at least 20 minutes in all but two of those 65 games (≈ **1300 minutes**).

Keep this in mind if you want to filter for the eligible players ;)*
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
            st.dataframe(result[columns_to_display].rename_axis("Rank"))

        if len(result) == 1:
            player_row = result.iloc[0]
            z_keys = ["Z_Hustle", "Z_Defense", "Z_Difficulty", "Z_D_LEBRON"]
            labels = ["Hustle", "Defensive Effectiveness", "Matchup Difficulty", "D-LEBRON"]

            # Player values
            values = [player_row[col] for col in z_keys]; values += values[:1]

            # League average values
            league_avg = [df[col].mean() for col in z_keys]; league_avg += league_avg[:1]

            # Radar angles
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]

            # Plot radar
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2, color='royalblue', label=player_row["Player"])
            ax.fill(angles, values, alpha=0.25, color='skyblue')
            ax.plot(angles, league_avg, linewidth=2, color='gray', linestyle='dashed', label='League Average')
            ax.fill(angles, league_avg, alpha=0.1, color='gray')
            for i, val in enumerate(values[:-1]):
                ax.text(angles[i], val + 0.15, f"{val:.2f}", ha='center', va='top', fontsize=14, color='red')
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=12)
            ax.set_yticklabels([]); ax.set_ylim(-3, 5); ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            st.pyplot(fig)

            # Save chart to memory
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
            st.download_button(
                label="Download Radar Chart as PNG",
                data=buf,
                file_name=f"{player_row['Player'].replace(' ', '_')}_Radar.png",
                mime="image/png"
            )

            # Player DIS History
            all_dis = load_all_seasons()
            player_name = player_row["Player"]
            player_hist = all_dis[all_dis["Player"].str.lower() == player_name.lower()].copy()
            if not player_hist.empty:
                player_hist = player_hist.sort_values("Season", key=lambda s: s.map(season_order_key))
                st.subheader(f"{player_name} — DIS History")
                st.line_chart(player_hist.set_index("Season")["DIS"])
                st.dataframe(
                    player_hist[["Season","Team","Pos","G","MP","DIS"]]
                    .reset_index(drop=True)
                    .rename(index=lambda x: x + 1)
                )
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
        st.dataframe(comparison_df[columns_to_display])

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
            st.dataframe(
                filtered_df[columns_to_display]
                .reset_index(drop=True)
                .rename_axis("Rank")
                .rename(index=lambda x: x + 1)
            )

        avg_dis = round(filtered_df["DIS"].mean(), 3)
        st.metric(label="Average DIS", value=avg_dis)
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
                avg_pos_dis = round(pos_df["DIS"].mean(), 2)
                with st.expander(f"Top 10 {pos}s"):
                    st.dataframe(top10)
                    st.markdown(f"**Average DIS for all {pos}s:** {avg_pos_dis}")