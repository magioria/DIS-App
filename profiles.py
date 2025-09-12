import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_all_seasons, season_order_key
from styling import _dis_category, _team_dis_category, _slice_to_html_team, _percentile_of_value
from plots import _pct_bar, _league_hist_with_marker

def show_player_profile(player_row: pd.Series, df_season: pd.DataFrame):
    """Render a player profile card + optional histogram.
       - player_row: a single row (for the selected player) from df_season
       - df_season: the full season dataframe
    """
    player_name = str(player_row["Player"])
    player_pos  = str(player_row.get("Pos", "—"))
    player_dis  = float(player_row["DIS"])
    cat, color, txt = _dis_category(player_dis)

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

    league_pct = _percentile_of_value(df_season["DIS"], player_dis)
    _pct_bar("League percentile (this season)", league_pct)

    if "Pos" in df_season.columns:
        pos_series = df_season.loc[df_season["Pos"] == player_pos, "DIS"]
        if len(pos_series) >= 5:
            pos_pct = _percentile_of_value(pos_series, player_dis)
            _pct_bar(f"Percentile among {player_pos} (this season)", pos_pct)

    with st.expander("Show player in league distribution"):
        _league_hist_with_marker(
            df_season, player_dis,
            title="League Distribution of DIS (player highlighted)"
        )

def show_team_profile(team: str, df_season: pd.DataFrame):
    """Render a team profile card with current row, DIS badge, history line chart, and history table."""
    # Current season minutes-weighted DIS for the team
    team_df = df_season[df_season["Team"] == team]
    team_dis = (team_df["DIS"] * team_df["MP"]).sum() / team_df["MP"].sum()
    cat, color, txt = _team_dis_category(team_dis)

    st.subheader(f"{team} — Team Profile")

    team_season = (
        df_season.groupby("Team")
        .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
        .reset_index(name="DIS")
        .sort_values("DIS", ascending=False)
        .reset_index(drop=True)
    )
   
    team_season.insert(0, "Rank", range(1, len(team_season) + 1))

    players_count = df_season.groupby("Team")["Player"].count().reset_index(name="Players considered")
    team_season = team_season.merge(players_count, on="Team")

    team_row = team_season[team_season["Team"] == team][["Rank", "Team", "DIS", "Players considered"]]

    team_row = team_row.rename(columns={"DIS": "Team DIS"})

    team_row["Team DIS"] = team_row["Team DIS"].round(2)

    st.markdown(_slice_to_html_team(team_row), unsafe_allow_html=True)

    # DIS badge
    st.markdown(
        f"<div style='display:inline-block;padding:6px 10px;border-radius:10px;"
        f"background:{color};color:{txt};font-weight:700'>"
        f"Team DIS {team_dis:.2f} · {cat}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    all_dis = load_all_seasons()
    team_hist = (
        all_dis[all_dis["Team"].str.lower() == team.lower()]
        .groupby("Season")
        .apply(lambda g: (g["DIS"] * g["MP"]).sum() / g["MP"].sum())
        .reset_index(name="Team DIS")
        .sort_values("Season", key=lambda s: s.map(season_order_key)))

    if not team_hist.empty:
        st.subheader(f"{team} — DIS History")

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

        hist_tbl = team_hist.sort_values("Season", ascending=False).reset_index(drop=True)
        st.markdown(_slice_to_html_team(hist_tbl), unsafe_allow_html=True)
    else:
        st.info("No multi-season history found for this team.")