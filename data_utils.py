import pandas as pd
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

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