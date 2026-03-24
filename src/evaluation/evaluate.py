"""
Offline evaluation for the recommenders.

We use a leave-one-out protocol on the user listening data: for each sampled
user we hide one of their tracks, ask the recommender for the top-K tracks given
another track the user listened to, and check whether the hidden track appears.

Metrics:
- Hit-Rate@K : fraction of users whose hidden track was in the top-K list.
- Coverage@K : fraction of the catalogue that appears across all recommendations.
- Diversity@K : how different the tracks *within one* recommendation list are
  from each other, in content-feature space (1 - mean pairwise similarity).
- Novelty@K : how non-obvious the recommended tracks are, using listening
  frequency as a popularity proxy (there's no `popularity` column in the raw
  data — see notebooks/EDA.ipynb, Section 10).

Hit-Rate/Coverage are computed only from playlist co-occurrence data, which is
the only ground-truth signal available. Content-based similarity has no such
labels, so it isn't scored the same way -- but Diversity/Novelty apply to the
output of *any* recommender (content,
collaborative, or hybrid), since they only look at what got recommended, not
whether it was "correct".
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender


def evaluate_collaborative(playlist_df: pd.DataFrame, k: int = 10,
                           sample_users: int = 500, seed: int = 42) -> dict:
    """Runs leave-one-out evaluation on the collaborative recommender.

    Args:
        playlist_df: Raw listening data (user_id/playlist_id, track_id).
        k: Number of recommendations to consider.
        sample_users: How many eligible users to evaluate.
        seed: Random seed for reproducible sampling.

    Returns:
        Dict with hit_rate, coverage, k and the number of users evaluated.
    """
    df = playlist_df.copy()
    if 'user_id' in df.columns:
        df = df.rename(columns={'user_id': 'playlist_id'})
    if 'song_id' in df.columns:
        df = df.rename(columns={'song_id': 'track_id'})

    recommender = CollaborativeFilteringRecommender(df)

    grouped = df.groupby('playlist_id')['track_id'].apply(list)
    eligible = grouped[grouped.apply(lambda t: len(set(t)) >= 2)]
    if eligible.empty:
        return {'hit_rate': 0.0, 'coverage': 0.0, 'k': k, 'users_evaluated': 0}

    rng = np.random.default_rng(seed)
    n = min(sample_users, len(eligible))
    chosen = rng.choice(eligible.index.to_numpy(), size=n, replace=False)

    hits = 0
    recommended_tracks = set()
    for pid in chosen:
        tracks = list(dict.fromkeys(eligible[pid]))  # unique, order-preserving
        held_out = tracks[-1]
        query = tracks[0]
        recs = recommender.get_recommendations(query, top_n=k)
        rec_ids = recs['track_id'].tolist()
        recommended_tracks.update(rec_ids)
        if held_out in rec_ids:
            hits += 1

    return {
        'hit_rate': hits / n,
        'coverage': len(recommended_tracks) / len(recommender.tracks),
        'k': k,
        'users_evaluated': n,
    }


def _diversity_at_k(feature_matrix, id_to_pos: dict, rec_ids: list) -> float:
    """1 - mean pairwise cosine similarity among a recommended list's feature
    vectors. 0.0 means every recommended track is (near-)identical to every
    other; higher means the list covers more varied ground.
    """
    positions = [id_to_pos[i] for i in rec_ids if i in id_to_pos]
    if len(positions) < 2:
        return 0.0
    sims = cosine_similarity(feature_matrix[positions])
    n = sims.shape[0]
    mean_pairwise_sim = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1 - mean_pairwise_sim)


def _novelty_at_k(rec_ids: list, interaction_counts: dict, total_interactions: int) -> float:
    """Mean self-information novelty: -log2((interactions + 1) / (total + 1))
    per recommended track, averaged over the list. A track with zero listening
    history scores the highest possible novelty; a track everyone has heard
    scores near zero. Standard proxy when there's no popularity label.
    """
    scores = [
        -np.log2((interaction_counts.get(tid, 0) + 1) / (total_interactions + 1))
        for tid in rec_ids
    ]
    return float(np.mean(scores)) if scores else 0.0


def evaluate_diversity_novelty(content_rec, get_recommendations_fn, query_track_ids,
                                interaction_counts: dict = None, top_n: int = 10) -> dict:
    """Scores Diversity@K (and, if given, Novelty@K) for any recommender's output.

    Diversity/novelty describe *what got recommended*, not whether it matches a
    held-out label, so unlike `evaluate_collaborative` this works for content,
    collaborative, or hybrid recommendations alike — pass whichever engine's
    `get_recommendations` (or a `functools.partial` fixing its other args, e.g.
    a specific `w_content`) as `get_recommendations_fn`.

    Args:
        content_rec: A fitted ContentBasedRecommender. Its feature matrix is
            what diversity is measured in, regardless of which engine actually
            produced the recommendation list.
        get_recommendations_fn: Callable `(track_id, top_n) -> DataFrame` with
            a `track_id` column — the recommender being scored.
        query_track_ids: Track ids to query, one recommendation list each.
        interaction_counts: Optional `{track_id: listen_count}` dict (e.g. from
            `playlists.csv`) used as the popularity proxy for novelty. If
            omitted, novelty is not computed.
        top_n: Recommendations requested per query.

    Returns:
        Dict with `diversity`, `queries_evaluated`, and `novelty` if
        `interaction_counts` was given.
    """
    id_to_pos = {tid: pos for pos, tid in enumerate(content_rec.df['track_id'])}
    total_interactions = sum(interaction_counts.values()) if interaction_counts else 0

    diversities, novelties = [], []
    for tid in query_track_ids:
        # top_n passed by keyword: a functools.partial fixing w_content (the
        # 2nd positional arg of HybridRecommender.get_recommendations) would
        # otherwise collide with top_n landing in that same position.
        recs = get_recommendations_fn(tid, top_n=top_n)
        rec_ids = recs['track_id'].tolist()
        if not rec_ids:
            continue
        diversities.append(_diversity_at_k(content_rec.features_matrix, id_to_pos, rec_ids))
        if interaction_counts is not None:
            novelties.append(_novelty_at_k(rec_ids, interaction_counts, total_interactions))

    result = {
        'diversity': float(np.mean(diversities)) if diversities else 0.0,
        'queries_evaluated': len(diversities),
    }
    if interaction_counts is not None:
        result['novelty'] = float(np.mean(novelties)) if novelties else 0.0
    return result
