import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class CollaborativeFilteringRecommender:
    """Recommender engine based on user listening history co-occurrence interactions."""
    
    def __init__(self, playlist_df: pd.DataFrame):
        """Initializes the collaborative recommender.

        Args:
            playlist_df: DataFrame containing listening history mappings.
        """
        # Unify column naming: Kaggle dataset uses 'user_id' and 'song_id' or 'track_id'
        playlist_df = playlist_df.copy()
        if 'user_id' in playlist_df.columns:
            playlist_df.rename(columns={'user_id': 'playlist_id'}, inplace=True)
        if 'song_id' in playlist_df.columns:
            playlist_df.rename(columns={'song_id': 'track_id'}, inplace=True)

        # Stored so callers can cheaply check "has the source data changed?"
        # (e.g. app.py's cache-invalidation check) without keeping the full
        # raw interaction table as instance state.
        self.raw_row_count = len(playlist_df)

        self._build_cooccurrence_matrix(playlist_df)

    def _build_cooccurrence_matrix(self, playlist_df: pd.DataFrame):
        """Builds track-to-track co-occurrence tables to capture social context.

        Takes `playlist_df` as a parameter instead of reading `self.playlist_df`
        so the full raw interaction table isn't kept around as instance state
        (nothing after this method needs it).
        """
        # Find unique items and playlists
        self.tracks = playlist_df['track_id'].unique()
        self.track_to_idx = {track_id: idx for idx, track_id in enumerate(self.tracks)}

        playlists = playlist_df['playlist_id'].unique()
        playlist_to_idx = {pid: idx for idx, pid in enumerate(playlists)}

        # Construct interaction matrix mapping playlists containing tracks
        rows = playlist_df['playlist_id'].map(playlist_to_idx).values
        cols = playlist_df['track_id'].map(self.track_to_idx).values
        data = np.ones(len(playlist_df))

        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(playlists), len(self.tracks))
        )

        # Track-track similarity is computed via co-occurrence dot-product: X.T * X
        # Diagonal elements are count occurrences of single track.
        self.cooccurrence_matrix = self.interaction_matrix.T.dot(self.interaction_matrix)
        
    def get_recommendations(self, track_id: str, top_n: int = 10) -> pd.DataFrame:
        """Looks up track associations in playlist interaction spaces.
        
        Args:
            track_id: Target track ID.
            top_n: Max tracks returned.
            
        Returns:
            DataFrame containing recommended track_ids and score associations.
        """
        if track_id not in self.track_to_idx:
            # Reverts with empty dataframe if song lacks playlist occurrences (cold start)
            return pd.DataFrame(columns=['track_id', 'collaborative_score'])
            
        target_idx = self.track_to_idx[track_id]

        # Get target column/row representing counts of joint playlist inclusions
        co_counts = self.cooccurrence_matrix[target_idx].toarray().flatten()

        # Normalize by sqrt(freq_target * freq_candidate) -- cosine similarity
        # between the two tracks' binary playlist-membership vectors. Dividing
        # by only the target's frequency would not be symmetric: a track
        # played in exactly one playlist would trivially score 1.0 against
        # everything in that playlist. Dividing by both frequencies keeps
        # scores meaningful regardless of how popular the candidate is.
        frequencies = self.cooccurrence_matrix.diagonal()
        target_frequency = frequencies[target_idx]
        scores = co_counts / (np.sqrt(target_frequency * frequencies) + 1e-9)
        
        results = pd.DataFrame({
            'track_id': self.tracks,
            'collaborative_score': scores
        })
        
        # Filter target song and use nlargest for partial sorting
        recommendations = results[results['track_id'] != track_id]
        return recommendations.nlargest(top_n, 'collaborative_score')

