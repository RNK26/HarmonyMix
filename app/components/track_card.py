"""
track_card.py — Streamlit component: renders a styled info card for a single track.
"""
import streamlit as st
import pandas as pd


def render_track_card(row: pd.Series, show_preview: bool = True) -> None:
    """Renders a compact info card for a track.

    Args:
        row:          A pandas Series representing one row of the tracks DataFrame.
        show_preview: If True and a 'spotify_preview_url' column is present,
                      renders an HTML audio player.
    """
    name = str(row.get('name', 'Unknown')).title()
    artist = str(row.get('artist', 'Unknown')).title()
    year = int(row.get('year', 0)) if row.get('year') else '—'
    # pd.isna() check matters here: a missing genre is a real NaN float, and
    # str(nan).title() == "Nan" (truthy), so a plain `or` fallback wouldn't catch it.
    raw_genre = row.get('genre')
    genre = (str(raw_genre).title() if pd.notna(raw_genre) and str(raw_genre).strip()
              else 'Genre unavailable')
    popularity = row.get('popularity', None)
    preview_url = row.get('spotify_preview_url', None)

    with st.container():
        st.markdown(f"### {name}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Artist", artist)
        col_b.metric("Year", year)
        col_c.metric("Genre", genre)

        if popularity is not None:
            st.progress(int(popularity), text=f"Popularity: {int(popularity)}/100")

        if show_preview:
            if preview_url and str(preview_url).startswith('http'):
                st.audio(preview_url, format="audio/mpeg")
            else:
                st.caption("Preview unavailable")
        st.divider()
