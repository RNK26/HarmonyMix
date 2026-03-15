import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.explain import explain_recommendation
from src.utils.model_persistence import save_model, load_model
from components.track_card import render_track_card
from components.audio_radar import render_audio_radar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HarmonyMix",
    layout="wide",
    page_icon="🎵",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 30px; color: #1DB954; font-weight: 700;
        text-align: center; margin-bottom: 4px;
    }
    .subheader {
        font-size: 15px; text-align: center;
        color: #aaaaaa; margin-bottom: 24px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">HarmonyMix</div>', unsafe_allow_html=True)

# Falls back to a synthetic sample when the real dataset isn't present (e.g. a
# fresh clone, or the Docker image, which excludes data on purpose). The header
# must say so -- claiming "Real Spotify Data" while showing made-up songs would
# be misleading.
_HAS_REAL_TRACKS_DATA = os.path.exists('data/processed/tracks_cleaned.csv') or os.path.exists('data/raw/tracks.csv')
_subheader_text = (
    "Hybrid Content-Based &amp; Collaborative Filtering · 50 k tracks · Real Spotify Data"
    if _HAS_REAL_TRACKS_DATA else
    "Hybrid Content-Based &amp; Collaborative Filtering · Running in demo mode."
)
st.markdown(f'<div class="subheader">{_subheader_text}</div>', unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading track catalogue…")
def load_tracks() -> pd.DataFrame:
    for path in ['data/processed/tracks_cleaned.csv', 'data/raw/tracks.csv']:
        if os.path.exists(path):
            return pd.read_csv(path)
    # Fallback: synthetic mini-dataset so the app still demos without data files
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        'track_id':       [f'track_{i}' for i in range(n)],
        'name':           ['Blinding Lights', 'Shape of You', 'Dance Monkey', 'Someone You Loved',
                           'Rockstar', 'Sunflower', 'One Dance', 'Closer', 'Believer', 'Senorita']
                          + [f'Song {i}' for i in range(10, n)],
        'artist':         ['The Weeknd', 'Ed Sheeran', 'Tones and I', 'Lewis Capaldi', 'Post Malone',
                           'Post Malone', 'Drake', 'The Chainsmokers', 'Imagine Dragons', 'Shawn Mendes']
                          + [f'Artist {i}' for i in range(10, n)],
        'year':           rng.integers(1990, 2024, n),
        'duration_ms':    rng.integers(150000, 300000, n),
        'danceability':   rng.uniform(0.3, 0.9, n),
        'energy':         rng.uniform(0.3, 0.9, n),
        'key':            rng.integers(0, 12, n),
        'loudness':       rng.uniform(-15, -2, n),
        'mode':           rng.integers(0, 2, n),
        'time_signature': rng.integers(3, 5, n),
        'speechiness':    rng.uniform(0.01, 0.3, n),
        'acousticness':   rng.uniform(0.01, 0.9, n),
        'instrumentalness': rng.uniform(0.0, 0.5, n),
        'liveness':       rng.uniform(0.05, 0.5, n),
        'valence':        rng.uniform(0.1, 0.9, n),
        'tempo':          rng.uniform(70, 180, n),
        'tags':           [', '.join(rng.choice(['rock','pop','electronic','indie','alternative'], 2))
                           for _ in range(n)],
        'spotify_id':     [f'sp_{i}' for i in range(n)],
        'popularity':     rng.integers(10, 100, n),
    })


@st.cache_data(show_spinner="Loading playlist data…")
def load_playlists() -> pd.DataFrame:
    path = 'data/raw/playlists.csv'
    if os.path.exists(path):
        return pd.read_csv(path, usecols=lambda c: c in
                           ['playlist_id', 'user_id', 'track_id', 'song_id'])
    return pd.DataFrame(columns=['playlist_id', 'track_id'])


# ── Engine initialisation (with disk cache to avoid re-fitting every restart) ─
CONTENT_MODEL_PATH = 'models/content_engine.pkl'
COLLAB_MODEL_PATH  = 'models/collab_engine.pkl'

@st.cache_resource(show_spinner="Fitting recommendation engines… (first run only)")
def build_engines():
    tracks_df   = load_tracks()
    playlists_df = load_playlists()

    # Try loading pre-fitted content engine from disk
    content_engine = load_model(CONTENT_MODEL_PATH)

    # Re-fit if the data changed size (e.g. synthetic -> real data). Compares
    # the raw row count stored on the engine instead of re-cleaning tracks_df
    # just to get a length. getattr(..., None) treats an older pickle that
    # predates this attribute as "needs rebuild" instead of crashing.
    if content_engine is not None and getattr(content_engine, 'raw_row_count', None) != len(tracks_df):
        content_engine = None

    if content_engine is None:
        content_engine = ContentBasedRecommender(tracks_df)
        save_model(content_engine, CONTENT_MODEL_PATH)

    collab_engine = load_model(COLLAB_MODEL_PATH)
    # Same cache-invalidation check as the content engine above.
    if collab_engine is not None and getattr(collab_engine, 'raw_row_count', None) != len(playlists_df):
        collab_engine = None

    if collab_engine is None:
        collab_engine = CollaborativeFilteringRecommender(playlists_df)
        save_model(collab_engine, COLLAB_MODEL_PATH)

    hybrid_engine = HybridRecommender(content_engine, collab_engine)
    return tracks_df, content_engine, collab_engine, hybrid_engine

try:
    tracks_df, content_engine, collab_engine, hybrid_engine = build_engines()
except Exception as e:
    # Anything unexpected here (malformed CSV, out of memory, etc.) would
    # otherwise surface as a raw Streamlit traceback with no next step for
    # the user, so show a plain message and log the details server-side.
    logging.exception("Failed to start the recommendation engines")
    st.error(
        "Couldn't start the recommendation engines. Please check your dataset "
        "setup (see the Installation section in the README) and try again."
    )
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

st.sidebar.markdown("**Find a Song**")
search_query = st.sidebar.text_input("Type to search:", value="", placeholder="e.g. Cheap Thrills")


@st.cache_data(show_spinner=False)
def _build_song_list(_tracks_df: pd.DataFrame, search_query: str) -> list:
    """Filter/rank the track catalogue down to the <=100 names shown in the
    dropdown. Cached on `search_query` (the leading underscore on `_tracks_df`
    tells st.cache_data not to hash the dataframe) so this doesn't re-run on
    every rerun triggered by unrelated widgets like the top_n slider.
    """
    if search_query:
        mask = _tracks_df['name'].str.contains(search_query, case=False, na=False)
        return sorted(_tracks_df[mask]['name'].dropna().unique().tolist())[:100]
    # Default to top 100 most popular if available, else just first 100
    if 'popularity' in _tracks_df.columns:
        return _tracks_df.nlargest(100, 'popularity')['name'].dropna().unique().tolist()
    return sorted(_tracks_df['name'].dropna().unique().tolist())[:100]


song_list = _build_song_list(tracks_df, search_query)

if not song_list:
    st.sidebar.warning("No songs found matching that search.")
    st.stop()

selected_song_name = st.sidebar.selectbox("Select Target Song", song_list)

selected_row = tracks_df[tracks_df['name'] == selected_song_name].iloc[0]
selected_track_id   = selected_row['track_id']
selected_artist     = selected_row['artist']

st.sidebar.markdown("---")
st.sidebar.markdown("**Hybrid Weight**")
w_content = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.6, 0.05)
w_collab  = 1.0 - w_content
st.sidebar.caption(f"Collaborative Weight: **{w_collab:.2f}**")

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
show_explanations = st.sidebar.checkbox("Explain recommendations", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Re-fit engines (after new data)"):
    for p in [CONTENT_MODEL_PATH, COLLAB_MODEL_PATH]:
        if os.path.exists(p):
            os.remove(p)
    # Clear both caches: cache_resource alone would rebuild the engines from
    # the still-cached (stale) CSV data loaded by cache_data.
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Target Track")
    render_track_card(selected_row, show_preview=True)
    render_audio_radar(selected_row, title="Audio Feature Profile")

with col2:
    st.subheader("Recommendations")
    try:
        recs = hybrid_engine.get_recommendations(
            selected_track_id, w_content=w_content, top_n=top_n
        )

        if recs.empty:
            # Shouldn't normally happen (the hybrid engine falls back to
            # content-only for cold-start tracks), but say so explicitly
            # rather than rendering a blank table.
            st.warning("No recommendations available for this track.")
            st.stop()

        # Differs from the slider value only when the cold-start fallback
        # kicked in -- surface that up front so the score columns below
        # don't look unexplained.
        actual_w_content = recs['w_content_used'].iloc[0]
        if actual_w_content != w_content:
            st.info(
                f"**{selected_song_name}** has no playlist interaction data in this "
                f"dataset, so these recommendations rely on feature similarity only."
            )

        # Build display columns robustly
        display_cols = ['name', 'artist']
        if 'popularity' in recs.columns:
            display_cols.append('popularity')
        for score_col in ['content_score', 'collaborative_score', 'hybrid_score']:
            if score_col in recs.columns:
                display_cols.append(score_col)

        styled_df = recs[display_cols].copy()
        styled_df['name']   = styled_df['name'].str.title()
        styled_df['artist'] = styled_df['artist'].str.title()

        score_cols = [c for c in ['content_score', 'collaborative_score', 'hybrid_score']
                      if c in styled_df.columns]
        styled_df[score_cols] = styled_df[score_cols].round(3)

        rename_map = {
            'name': 'Track Name', 'artist': 'Artist',
            'popularity': 'Popularity',
            'content_score': 'Content Score',
            'collaborative_score': 'Collab Score',
            'hybrid_score': 'Hybrid Score',
        }
        styled_df.rename(columns={k: v for k, v in rename_map.items() if k in styled_df.columns},
                         inplace=True)

        gradient_col = 'Hybrid Score' if 'Hybrid Score' in styled_df.columns else styled_df.columns[-1]
        format_cols = {rename_map[c]: '{:.3f}' for c in score_cols}
        st.dataframe(
            styled_df.style.background_gradient(cmap='Greens', subset=[gradient_col])
                            .format(format_cols),
            width="stretch",
            height=420,
            hide_index=True,
        )

        # "Why these?" explanations, built from the same content/collaborative
        # scores already in `recs` — no extra model call, just formatting.
        if show_explanations and not recs.empty:
            with st.expander("Why these recommendations?", expanded=False):
                for _, row in recs.iterrows():
                    reason = explain_recommendation(row, w_content=w_content)
                    st.markdown(f"**{str(row.get('name', '')).title()}** — {reason}")

        # Audio radar comparison for top recommendation
        if not recs.empty:
            top_rec = recs.iloc[0]
            top_match = tracks_df[tracks_df['track_id'] == top_rec['track_id']]
            if not top_match.empty:
                st.markdown("---")
                st.markdown(f"#### Top Match: **{str(top_rec.get('name','')).title()}**")
                render_audio_radar(top_match.iloc[0], title="Top Match Audio Profile")

    except Exception as e:
        logging.exception("Failed to fetch recommendations")
        st.error("Something went wrong while fetching recommendations. Please try a different track.")
