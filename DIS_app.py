import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import math

from data_utils import load_all_seasons, season_order_key, OUTPUTS_DIR
from styling import _slice_to_html, _slice_to_html_team
from profiles import show_player_profile, show_team_profile
from plots import plot_dis_scale_with_steps, plot_team_dis_scale
from leaderboard import render_leaderboard

page = st.sidebar.radio("Navigate", ["What is DIS?", "Player Leaderboard", "Team Leaderboard"])

if page == "What is DIS?":
    
    st.title("Defensive Impact Score (DIS) 🏀")
    st.header("**What is DIS?**")
    st.markdown("""
    The **Defensive Impact Score**, abbreviated DIS, is a custom stat that estimates how impactful each player is on defense.

    Unlike traditional defensive stats that rely heavily on steals, blocks, or team ratings, DIS blends multiple layers of data, including hustle stats and matchup quality, to assess how **consistently** and **effectively** a player influences defensive outcomes.

    DIS is designed to be **scale-consistent** across seasons, to evaluate players fairly regardless of their position, and to minimize the distortion caused by overall team performance, offering a robust tool for identifying both **elite** and **underrated defenders** who may not show up in highlight reels.
    """)     

    st.divider()       
    
    st.header("**DIS Interpretation Scale**")

    st.markdown("""
    Values are standardized so that 0 = league average. Positive = better than average, negative = worse.
                
    - 20 or more → **Generational / DPOY-level season**
    - 13–19.9 → **Elite Defender**
    - 7–12.9 → **Strong defender**
    - 3–6.9 → **Solid contributor**
    - 0–2.9 → **Average defender**
    - -5 to -0.1 → **Below average**
    - Less than -5 → **Poor Defender**
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
    """)

    st.latex(r"""\text{Team DIS} = \frac{\sum_i \big(DIS_i \times Minutes_i\big)}{\sum_i Minutes_i}""")

    st.markdown("""
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

    season_files = {p.stem.split("DIS_")[1]: p for p in OUTPUTS_DIR.glob("DIS_*.csv")}
    seasons_sorted = sorted(season_files.keys(), reverse=True)

    st.sidebar.title("Filters")
    selected_season = st.sidebar.selectbox("Select season", seasons_sorted, index=0)

    df = pd.read_csv(season_files[selected_season])

    columns_to_display = ["Player", "Team", "Pos", "G", "MP", "DIS"]
    df_display = df.copy()

    all_players = sorted(df_display["Player"].unique())
    player = st.sidebar.selectbox("🔍 Search a Player to view full Profile & History", options=[""] + all_players, index=0)

    min_games = st.sidebar.number_input("Minimum Games Played", min_value=1, max_value=int(df_display["G"].max()), value=1, step=1, format="%d",)

    min_mp = st.sidebar.number_input("Minimum Minutes Played", min_value=1, max_value=int(df_display["MP"].max()), value=1, step=50, format="%d",)

    eligible_only = st.sidebar.checkbox("Only show award-eligible players (NBA 65-game rule)", value=False)
    if eligible_only:
        min_games = max(min_games, 65)
        min_mp = max(min_mp, 1300)

    pop = st.sidebar.popover("ℹ️ What is the 65-game rule?")
    with pop:
        st.markdown("""
        *From the 2023-24 season, the NBA implemented the 65-game rule: players must appear in at least 65 games to be eligible for awards like DPOY and All-Defensive teams.  
        In addition, players must play at least 20 minutes in all but two of those 65 games (≈ **1300 minutes**).*

        *Keep this in mind if you want to filter for the eligible players ;)*
        """)

    dis_min = st.sidebar.number_input("Minimum DIS", min_value=float(df_display["DIS"].min()), max_value=float(df_display["DIS"].max()), value=float(df_display["DIS"].min()), step=0.5, format="%.1f",)

    teams = df_display["Team"].unique()
    selected_team = st.sidebar.selectbox("Filter by team", options=["All"] + sorted(teams))

    positions = df["Pos"].dropna().unique()
    selected_pos = st.sidebar.selectbox("Filter by position", options=["All"] + sorted(positions))

    filtered_df = df_display.copy()
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["Team"] == selected_team]
    if selected_pos != "All":
        filtered_df = filtered_df[filtered_df["Pos"] == selected_pos]
    filtered_df = filtered_df[filtered_df["MP"] >= min_mp]
    filtered_df = filtered_df[filtered_df["G"] >= min_games]
    filtered_df = filtered_df[filtered_df["DIS"] >= dis_min]

    player_options = filtered_df["Player"].unique()
    players_to_compare = st.sidebar.multiselect("Compare players", options=sorted(player_options))

    if player:
        result = filtered_df[filtered_df["Player"].str.contains(player, case=False, na=False)]
        if result.empty:
            st.warning("Player not found.")
        else:
            result_tbl = result[columns_to_display].reset_index(drop=True).copy()
            result_tbl.insert(0,"Rank", result_tbl["DIS"].apply(lambda x: int((df["DIS"] > x).sum() + 1)))
            st.markdown(_slice_to_html(result_tbl), unsafe_allow_html=True)


        if len(result) == 1:
            player_row = result.iloc[0]

            show_player_profile(player_row, df)             

            st.divider()
         
            # Player DIS History
            all_dis = load_all_seasons()
            player_name = player_row["Player"]
            player_hist = all_dis[all_dis["Player"].str.lower() == player_name.lower()].copy()
            if not player_hist.empty:
                player_hist = player_hist.sort_values("Season", key=lambda s: s.map(season_order_key))
                st.subheader(f"{player_name} — DIS History")
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

        st.subheader("DIS History Comparison(multi-season)")

        all_dis = load_all_seasons()
        hist = all_dis[all_dis["Player"].isin(players_to_compare)].copy()

        if hist.empty:
            st.info("No multi-season history available for the selected players.")
        else:
            all_seasons = sorted(hist["Season"].unique(), key=season_order_key)

            fig, ax = plt.subplots(figsize=(10, 4))

            for name, g in hist.groupby("Player"):
                g = g.set_index("Season").reindex(all_seasons)
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
    season_files = {p.stem.split("DIS_")[1]: p for p in OUTPUTS_DIR.glob("DIS_*.csv")}
    seasons_sorted = sorted(season_files.keys(), reverse=True)

    st.sidebar.title("Filters")
    selected_season = st.sidebar.selectbox("Select season", seasons_sorted, index=0)

    df_display = pd.read_csv(season_files[selected_season])

    all_teams = sorted(df_display["Team"].unique())
    team = st.sidebar.selectbox("🔍 Search a Team to view Profile & History", options=[""] + all_teams, index=0)
        
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

        comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

        players_count = df_display.groupby("Team")["Player"].count().reset_index(name="Players considered")
        comparison_df = comparison_df.merge(players_count, on="Team")

        comparison_df = comparison_df[comparison_df["Team"].isin(teams_to_compare)]

        st.subheader("Team Comparison")
        st.markdown(_slice_to_html_team(comparison_df), unsafe_allow_html=True)

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

        team_weighted["Players considered"] = df_display.groupby("Team")["Player"].count().values

        team_weighted["DIS"] = team_weighted["DIS"].round(2)
        team_weighted = team_weighted.rename(columns={"DIS": "Team DIS"})

        team_weighted.insert(0, "Rank", range(1, len(team_weighted) + 1))

        st.markdown(_slice_to_html_team(team_weighted), unsafe_allow_html=True)
  