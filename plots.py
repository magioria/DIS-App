import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

from styling import BINS, TEAM_BINS, BOUNDARIES

def plot_dis_scale_with_steps():
    fig, ax = plt.subplots(figsize=(12, 2.8))

    for low, high, name, color in BINS:
        ax.barh(0, high - low, left=low, color=color,
                edgecolor="black", height=0.66)

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

    for x in BOUNDARIES:
        ax.vlines(x, -0.5, -0.08, color="black", linewidth=1)
        ax.text(x, -0.65, f"{x:g}", ha="center", va="top", fontsize=9)

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

def _pct_color(pct: float) -> str:
    """
    Color by league percentile, matched to actual DIS distribution.
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