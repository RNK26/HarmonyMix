import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.audio_radar import render_audio_radar
from components.track_card import render_track_card

from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.explain import explain_recommendation
from src.recommenders.hybrid_recommender import HybridRecommender
from src.utils.model_persistence import load_model, save_model

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

# Two ways to run. Locally the full dataset is on disk. The hosted version only
# has the trimmed bundle, because playlists.csv is 575 MB and GitHub refuses
# files over 100 MB. The header has to say which one you are looking at --
# claiming "50 k tracks" while serving 10 k would be misleading.
FULL_TRACKS_PATHS = ['data/processed/tracks_cleaned.csv', 'data/raw/tracks.csv']
FULL_PLAYLISTS_PATH = 'data/raw/playlists.csv'
DEMO_TRACKS_PATH = 'data/tracks_demo.csv'
DEMO_MATRIX_PATH = 'data/collab_matrix.npz'

USING_FULL_DATA = os.path.exists(FULL_PLAYLISTS_PATH)

_subheader_text = (
    "Hybrid Content-Based &amp; Collaborative Filtering · 50 k tracks · Real Spotify Data"
    if USING_FULL_DATA else
    "Hybrid Content-Based &amp; Collaborative Filtering · Demo bundle: "
    "the 10 k most-listened tracks, real Spotify data"
)
st.markdown(f'<div class="subheader">{_subheader_text}</div>', unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading track catalogue…")
def load_tracks() -> pd.DataFrame:
    """Full catalogue when it is on disk, otherwise the demo bundle.

    There used to be a synthetic fallback here that invented track names like
    "Song 42". It existed for the case where no data file was present at all,
    which is exactly what a deployment looked like, so the hosted app served
    made-up songs. The demo bundle is committed to the repo, so that case no
    longer happens and inventing data is worse than failing loudly.
    """
    for path in FULL_TRACKS_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    if os.path.exists(DEMO_TRACKS_PATH):
        return pd.read_csv(DEMO_TRACKS_PATH)
    raise FileNotFoundError(
        f"No track catalogue found. Expected one of {FULL_TRACKS_PATHS} for a "
        f"full local run, or {DEMO_TRACKS_PATH} from scripts/build_demo_bundle.py."
    )


@st.cache_data(show_spinner="Loading playlist data…")
def load_playlists() -> pd.DataFrame:
    return pd.read_csv(FULL_PLAYLISTS_PATH, usecols=lambda c: c in
                       ['playlist_id', 'user_id', 'track_id', 'song_id'])


def build_collab_engine(tracks_df: pd.DataFrame):
    """Counts co-occurrences locally, loads the precomputed matrix when hosted."""
    if USING_FULL_DATA:
        playlists_df = load_playlists()
        engine = load_model(COLLAB_MODEL_PATH)
        if engine is not None and getattr(engine, 'raw_row_count', None) != len(playlists_df):
            engine = None
        if engine is None:
            engine = CollaborativeFilteringRecommender(playlists_df)
            save_model(engine, COLLAB_MODEL_PATH)
        return engine

    matrix = sp.load_npz(DEMO_MATRIX_PATH)
    # Row i of the matrix is row i of tracks_demo.csv. build_demo_bundle.py
    # asserts that alignment when it writes the pair.
    return CollaborativeFilteringRecommender.from_matrix(matrix, tracks_df['track_id'].to_numpy())


# ── Engine initialisation (with disk cache to avoid re-fitting every restart) ─
CONTENT_MODEL_PATH = 'models/content_engine.pkl'
COLLAB_MODEL_PATH  = 'models/collab_engine.pkl'

@st.cache_resource(show_spinner="Fitting recommendation engines… (first run only)")
def build_engines():
    tracks_df = load_tracks()

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

    collab_engine = build_collab_engine(tracks_df)

    hybrid_engine = HybridRecommender(content_engine, collab_engine)
    return tracks_df, content_engine, collab_engine, hybrid_engine

try:
    tracks_df, content_engine, collab_engine, hybrid_engine = build_engines()
except Exception:
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
def get_listener_counts(_collab_engine) -> pd.Series:
    """Listeners per track, used as a real popularity signal.

    The raw tracks.csv has NO `popularity` column, so any code branching on one
    silently never fires. The collaborative engine already holds this number:
    the diagonal of the co-occurrence matrix is the count of users who listened
    to each track. Reusing it costs nothing and avoids re-reading a 600 MB CSV.
    """
    freq = np.asarray(_collab_engine.cooccurrence_matrix.diagonal()).flatten()
    return pd.Series(freq, index=_collab_engine.tracks)


listener_counts = get_listener_counts(collab_engine)


@st.cache_data(show_spinner=False)
def _build_song_list(_tracks_df: pd.DataFrame, _listener_counts: pd.Series,
                     search_query: str) -> list:
    """Filter/rank the track catalogue down to the <=100 names shown in the
    dropdown. Cached on `search_query` (the leading underscore on `_tracks_df`
    tells st.cache_data not to hash the dataframe) so this doesn't re-run on
    every rerun triggered by unrelated widgets like the top_n slider.

    Ordering is by LISTENER COUNT, not alphabetically.

    This was a real presentation bug. The previous version checked for a
    `popularity` column that does not exist in this dataset, so it fell through
    to `sorted(names)[:100]` -- the alphabetically first 100 tracks ('#1',
    '#1 Zero', '#10', ...). Around 44% of those have no listening history at
    all, so the demo defaulted to a cold-start track, the hybrid engine's
    fallback forced pure content-based scoring, and the whole app appeared to
    be content-only. The model was always correct; the default query was not.
    """
    if search_query:
        mask = _tracks_df['name'].str.contains(search_query, case=False, na=False)
        matches = _tracks_df[mask].copy()
    else:
        matches = _tracks_df.copy()

    matches['_listeners'] = matches['track_id'].map(_listener_counts).fillna(0)
    matches = matches.sort_values('_listeners', ascending=False)
    return matches['name'].dropna().unique().tolist()[:100]


song_list = _build_song_list(tracks_df, listener_counts, search_query)

if not song_list:
    st.sidebar.warning("No songs found matching that search.")
    st.stop()

# ── Demo presets ──────────────────────────────────────────────────────────────
# Two one-click examples so the two behaviours can be shown deliberately rather
# than hoped for: a heavily-listened track where collaborative filtering has
# plenty of signal, and a track with no listening history at all where the
# cold-start fallback fires. The contrast between them IS the design story.
@st.cache_data(show_spinner=False)
def get_preset_tracks(_tracks_df: pd.DataFrame, _listener_counts: pd.Series
                      ) -> tuple[str | None, str | None]:
    named = _tracks_df.dropna(subset=['name']).copy()
    named['_listeners'] = named['track_id'].map(_listener_counts).fillna(0)

    hot = named.nlargest(1, '_listeners')
    popular = hot['name'].iloc[0] if len(hot) else None

    # A cold-start track with no listening history. Prefer one with a
    # reasonably recognisable name over a random obscure row.
    cold_pool = named[named['_listeners'] == 0]
    cold = None
    if len(cold_pool):
        cold_pool = cold_pool.assign(_len=cold_pool['name'].str.len())
        cold = cold_pool.nsmallest(1, '_len')['name'].iloc[0]
    return popular, cold


preset_popular, preset_cold = get_preset_tracks(tracks_df, listener_counts)

st.sidebar.markdown("**Demo presets**")
_p1, _p2 = st.sidebar.columns(2)
if _p1.button("Popular", width="stretch",
              help="A heavily-listened track: collaborative filtering contributes strongly"):
    st.session_state['preset_song'] = preset_popular
if _p2.button("Cold start", width="stretch",
              help="A track with no listening history: the content-only fallback fires"):
    st.session_state['preset_song'] = preset_cold

# A preset may not be in the current (search-filtered) list, so put it at the
# front and select it. Clearing it after use means the dropdown stays free.
preset_song = st.session_state.pop('preset_song', None)
if preset_song and preset_song in tracks_df['name'].values:
    song_list = [preset_song] + [s for s in song_list if s != preset_song]

selected_song_name = st.sidebar.selectbox("Select Target Song", song_list)

selected_row = tracks_df[tracks_df['name'] == selected_song_name].iloc[0]
selected_track_id   = selected_row['track_id']
selected_artist     = selected_row['artist']

# ── Collaborative signal strength for the selected track ──────────────────────
# Makes the sparsity visible as a number instead of something to explain
# verbally. `nnz` on the track's co-occurrence row is how many other tracks
# ever appeared in the same user's history.
_n_listeners = int(listener_counts.get(selected_track_id, 0))
if selected_track_id in collab_engine.track_to_idx:
    _idx = collab_engine.track_to_idx[selected_track_id]
    _n_related = collab_engine.cooccurrence_matrix[_idx].nnz - 1
    _pct_related = _n_related / len(collab_engine.tracks) * 100
else:
    _n_related, _pct_related = 0, 0.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Collaborative signal**")
st.sidebar.caption(
    f"Listeners: **{_n_listeners:,}**  \n"
    f"Co-listened tracks: **{_n_related:,}** "
    f"({_pct_related:.1f}% of {len(collab_engine.tracks):,})"
)

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

        # ── Contribution breakdown ────────────────────────────────────────
        # Shows the blend as three numbers so the hybrid behaviour is visible
        # on screen rather than something that has to be described. Move the
        # weight slider and these move with it -- which is the point.
        if {'content_score', 'collaborative_score'}.issubset(recs.columns):
            _w_col = 1.0 - actual_w_content
            _content_part = (actual_w_content * recs['content_score']).mean()
            _collab_part = (_w_col * recs['collaborative_score']).mean()
            _with_signal = int((recs['collaborative_score'] > 0).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Content contribution", f"{_content_part:.3f}",
                      f"weight {actual_w_content:.2f}", delta_color="off")
            m2.metric("Collaborative contribution", f"{_collab_part:.3f}",
                      f"weight {_w_col:.2f}", delta_color="off")
            m3.metric("Recs with collab signal", f"{_with_signal}/{len(recs)}",
                      delta_color="off")

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

    except Exception:
        logging.exception("Failed to fetch recommendations")
        st.error("Something went wrong while fetching recommendations. Please try a different track.")
