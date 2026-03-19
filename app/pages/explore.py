"""
explore.py — Streamlit multi-page: Dataset Explorer page.

Allows the user to browse, search, and filter the raw tracks DataFrame.
"""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.set_page_config(page_title="Explore — HarmonyMix", layout="wide")
st.title("Explore the Dataset")

@st.cache_data
def load_tracks():
    paths = [
        'data/processed/tracks_cleaned.csv',
        'data/raw/tracks.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

df = load_tracks()

if df.empty:
    st.warning("No track data available.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

if 'artist' in df.columns:
    all_artists = sorted(df['artist'].dropna().unique().tolist())
    selected_artists = st.sidebar.multiselect("Artist", all_artists, default=[])
    if selected_artists:
        df = df[df['artist'].isin(selected_artists)]

if 'year' in df.columns:
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

search_term = st.sidebar.text_input("Search by song name")
if search_term:
    df = df[df['name'].str.contains(search_term, case=False, na=False)]

# ── Dataset summary ───────────────────────────────────────────────────────────
# Reflects the currently filtered `df`, not the full raw catalogue, so it stays
# accurate as filters are applied.
st.subheader("Dataset Summary")
summary_cols = st.columns(5)
summary_cols[0].metric("Tracks", f"{len(df):,}")
if 'artist' in df.columns:
    summary_cols[1].metric("Artists", f"{df['artist'].nunique():,}")
if 'tempo' in df.columns and len(df):
    summary_cols[2].metric("Avg Tempo", f"{df['tempo'].mean():.0f} BPM")
if 'energy' in df.columns and len(df):
    summary_cols[3].metric("Avg Energy", f"{df['energy'].mean():.2f}")
if 'genre' in df.columns and len(df):
    genre_known_pct = df['genre'].notna().mean() * 100
    summary_cols[4].metric("Genre Known", f"{genre_known_pct:.0f}%")

st.caption("Summary reflects the filters applied in the sidebar.")

# ── Main table ────────────────────────────────────────────────────────────────

display_cols = [c for c in ['name', 'artist', 'year', 'genre', 'danceability',
                             'energy', 'valence', 'tempo', 'popularity'] if c in df.columns]
table_df = df[display_cols].reset_index(drop=True).copy()
if 'genre' in table_df.columns:
    table_df['genre'] = table_df['genre'].fillna('Genre unavailable')
if 'name' in table_df.columns:
    table_df['name'] = table_df['name'].str.title()
if 'artist' in table_df.columns:
    table_df['artist'] = table_df['artist'].str.title()
num_display_cols = [c for c in ['danceability', 'energy', 'valence', 'tempo'] if c in table_df.columns]
if num_display_cols:
    table_df[num_display_cols] = table_df[num_display_cols].round(3)

rename_map = {
    'name': 'Track Name', 'artist': 'Artist', 'year': 'Year', 'genre': 'Genre',
    'danceability': 'Danceability', 'energy': 'Energy', 'valence': 'Valence',
    'tempo': 'Tempo', 'popularity': 'Popularity',
}
table_df.rename(columns={k: v for k, v in rename_map.items() if k in table_df.columns}, inplace=True)

st.dataframe(
    table_df,
    width="stretch",
    height=500,
    hide_index=True,
)

# ── Quick stats ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Quick Stats")
num_cols = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
available = [c for c in num_cols if c in df.columns]
if available:
    st.dataframe(df[available].describe().round(3), width="stretch")
