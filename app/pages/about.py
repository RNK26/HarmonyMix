"""
about.py — Streamlit multi-page: About / Project Info page.

To activate multi-page routing, Streamlit requires pages to live in an
`app/pages/` folder. Each file becomes a sidebar navigation entry automatically.
"""
import streamlit as st

st.set_page_config(page_title="About — HarmonyMix", layout="wide")

st.title("About This Project")

st.markdown("""
HarmonyMix is a music recommender built on Spotify track and playlist data.
It combines three approaches:

| Module | Method |
|---|---|
| Content-Based | Cosine similarity over each track's audio features and metadata (artist, key, mode, time signature) |
| Collaborative | Playlist co-occurrence — tracks that show up together across playlists |
| **Hybrid** | A weighted average of both, tunable via the sidebar slider |

Artist is the single largest block of columns in the content feature matrix,
so "sounds similar" is driven at least as much by shared artist as by the
numeric audio features (danceability, energy, tempo, …).

### Dataset
- **Tracks** (~50k songs) — Spotify audio features and metadata
- **Playlists** (~9.7M listening events) — used for collaborative filtering
""")

st.caption("HarmonyMix — music recommendations from Spotify track and playlist data")
