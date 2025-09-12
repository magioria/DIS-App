import math
import numpy as np
import streamlit as st
from styling import _slice_to_html


def _set_page(key: str, n_pages: int, set_to: int | None = None, delta: int | None = None):
    """Single source of truth: update page in session_state (0-based)."""
    page_key = f"{key}_page"
    page = int(st.session_state.get(page_key, 0))
    if set_to is not None:
        page = int(set_to)
    elif delta is not None:
        page = page + int(delta)
    page = max(0, min(page, n_pages - 1))
    st.session_state[page_key] = page 

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
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)

    page_key = f"{key}_page"
    ps_key   = f"{key}_ps"
    jump_key = f"{key}_jump_val"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

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
            index=idx,                
            key=ps_key,               
            on_change=_ps_changed,   
            args=(key,),
        )
    with top[1]:
        st.write("")

    page_size = int(st.session_state.get(ps_key, page_size_options[idx]))

    n_rows  = len(df)
    n_pages = max(1, math.ceil(n_rows / page_size))
    page    = int(st.session_state.get(page_key, 0))
    page    = max(0, min(page, n_pages - 1))
    st.session_state[page_key] = page

    start, end = page * page_size, min((page + 1) * page_size, n_rows)

    slice_df = df.iloc[start:end].copy()
    slice_df.insert(0, "Rank", np.arange(start + 1, end + 1))
    base_cols = ["Rank", "Player", "Team", "Pos", "G", "MP", "DIS"]
    visible = [c for c in base_cols if c in slice_df.columns]
    slice_df = slice_df[visible]

    st.markdown(_slice_to_html(slice_df), unsafe_allow_html=True)

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
                value=page + 1,            
                key=jump_key,
                on_change=_jump_changed,   
                args=(key, n_pages),
            )