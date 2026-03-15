import pandas as pd
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    """Combines Content-Based Filtering and Collaborative Filtering with custom weights."""
    
    def __init__(self, content_rec: ContentBasedRecommender, collab_rec: CollaborativeFilteringRecommender):
        """Initializes the hybrid recommendation engine.

        Args:
            content_rec: Instance of ContentBasedRecommender.
            collab_rec: Instance of CollaborativeFilteringRecommender.
        """
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self._base_score_cache = {}

    def _get_merged_base_scores(self, track_id: str) -> pd.DataFrame:
        """Returns content + collaborative scores for every track, merged with metadata.

        The full merge over all tracks is the expensive part, but it does not depend
        on the hybrid weight, so we cache it per track_id and reuse it across weight
        changes (e.g. when the user drags the slider).
        """
        if track_id in self._base_score_cache:
            return self._base_score_cache[track_id]

        # 1. Content scores
        content_recs = self.content_rec.get_recommendations(track_id, top_n=len(self.content_rec.df))
        content_scores = content_recs[['track_id', 'similarity_score']].rename(
            columns={'similarity_score': 'content_score'}
        )

        # 2. Collaborative scores
        collab_recs = self.collab_rec.get_recommendations(track_id, top_n=len(self.collab_rec.tracks))

        # 3. Merge base scores
        if collab_recs.empty:
            merged = content_scores.copy()
            merged['collaborative_score'] = 0.0
        else:
            merged = pd.merge(content_scores, collab_recs, on='track_id', how='outer').fillna(0.0)

        # 4. Join metadata so the app has names/artists to display
        meta_cols = ['track_id', 'name', 'artist']
        if 'popularity' in self.content_rec.df.columns:
            meta_cols.append('popularity')
        metadata_df = self.content_rec.df[meta_cols]
        result = pd.merge(merged, metadata_df, on='track_id', how='inner')

        self._base_score_cache[track_id] = result
        return result

    def get_recommendations(self, track_id: str, w_content: float = 0.5, top_n: int = 10) -> pd.DataFrame:
        """Generates hybrid predictions by calculating weighted average scores.

        Args:
            track_id: Target track ID.
            w_content: Requested weight for Content-Based predictions (0.0 to 1.0).
                Used as-is when the query track has collaborative data; see below
                for when it isn't.
            top_n: Number of final recommendations to return.

        Returns:
            DataFrame of recommended tracks with metadata, `content_score`,
            `collaborative_score`, `hybrid_score`, and `w_content_used` (the
            weight actually applied -- see below).
        """
        # Retrieve pre-merged 50k-row base scores (instant if cached)
        base_df = self._get_merged_base_scores(track_id).copy()

        # Adaptive weighting: if the query track has zero collaborative
        # history, collaborative_score is 0.0 for every candidate because
        # there's no data, not because anything scored "unrelated." Blending
        # that 0.0 in at weight (1 - w_content) would still shrink the final
        # score, silently penalizing cold-start tracks. Fall back to pure
        # content confidence for those queries instead.
        query_has_collab_signal = track_id in self.collab_rec.track_to_idx
        effective_w_content = w_content if query_has_collab_signal else 1.0

        w_collab = 1.0 - effective_w_content
        base_df['hybrid_score'] = (effective_w_content * base_df['content_score']) + \
                                  (w_collab * base_df['collaborative_score'])
        # Carried per-row (not on self) so explain.py can report the weight
        # actually used, which can differ from the requested w_content.
        base_df['w_content_used'] = effective_w_content

        # Use nlargest for a much faster partial sort instead of sorting all 50k rows
        return base_df.nlargest(top_n, 'hybrid_score')
