import pandas as pd
from scipy.sparse import issparse, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.preprocess import get_preprocessor_transformer, clean_and_prepare_dataset


class ContentBasedRecommender:
    """Recommender engine that ranks tracks by cosine similarity over a feature
    matrix built from audio features, tags, and artist/key/time_signature/mode
    metadata (see preprocess.get_preprocessor_transformer) -- not audio features
    alone. The one-hot artist block is the largest single block of columns in
    that matrix, so "sounds similar" is driven at least as much by shared
    artist and tags as by the numeric audio features.

    features_matrix is kept as a scipy CSR sparse matrix throughout, since
    cosine_similarity() accepts sparse input natively (avoids densifying a
    50k-row matrix).
    """

    def __init__(self, df: pd.DataFrame):
        """Initialises the content recommender with a fitted ColumnTransformer.

        Args:
            df: Raw DataFrame containing track metadata and audio features.
        """
        # Stored so callers can cheaply check "has the source data changed?"
        # (e.g. app.py's cache-invalidation check) via a plain int comparison
        # instead of re-running clean_and_prepare_dataset() on every startup
        # just to get a row count.
        self.raw_row_count = len(df)

        self.df = clean_and_prepare_dataset(df)
        self.df = self.df.reset_index(drop=True)   # ensure integer positional index

        # Columns that carry no audio-feature signal — drop before fitting
        cols_to_remove = ["track_id", "name", "spotify_id", "genre", "spotify_preview_url"]
        fit_df = self.df.drop(
            columns=[c for c in cols_to_remove if c in self.df.columns]
        )

        # Fit the transformer and store the result as a CSR sparse matrix.
        self.transformer = get_preprocessor_transformer(fit_df)
        raw_matrix = self.transformer.fit_transform(fit_df)
        if issparse(raw_matrix):
            self.features_matrix = raw_matrix.tocsr()
        else:
            self.features_matrix = csr_matrix(raw_matrix)

    def get_recommendations(self, track_id: str, top_n: int = 10) -> pd.DataFrame:
        """Finds top-N tracks similar to target track_id by cosine similarity.

        Args:
            track_id: Unique identifier for the query track.
            top_n:    Number of recommendations to return.

        Returns:
            DataFrame of recommended tracks with a 'similarity_score' column.
        """
        if 'track_id' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'track_id' column.")

        # Locate positional index of the target track
        mask = self.df['track_id'] == track_id
        if not mask.any():
            raise KeyError(f"Track ID '{track_id}' not found in database.")

        target_pos = self.df.index[mask][0]

        # CSR row slice → shape (1, n_features) sparse matrix — already 2D.
        # cosine_similarity handles sparse input natively; no toarray() needed.
        target_vector = self.features_matrix[target_pos]
        sim_scores = cosine_similarity(target_vector, self.features_matrix).flatten()

        # Build a tiny 2-column dataframe for speed (avoids copying all metadata columns)
        results_df = pd.DataFrame({
            'track_id': self.df['track_id'],
            'similarity_score': sim_scores
        })

        # Exclude the query track itself and return top-N sorted descending
        recommendations = results_df[results_df['track_id'] != track_id]
        recommendations = recommendations.sort_values(
            by='similarity_score', ascending=False
        )
        return recommendations.head(top_n)
