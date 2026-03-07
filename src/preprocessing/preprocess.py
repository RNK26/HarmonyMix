import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from category_encoders.count import CountEncoder
except ImportError:
    from sklearn.base import BaseEstimator, TransformerMixin

    class CountEncoder(BaseEstimator, TransformerMixin):
        """Fallback CountEncoder when category_encoders is not installed.
        Encodes categorical columns as their frequency (or normalized frequency).

        Note: mapping_ is a *fitted* attribute, NOT a constructor parameter.
        It must NOT be set in __init__ so that BaseEstimator.get_params() and
        sklearn.base.clone() can introspect constructor params cleanly.
        """
        def __init__(self, normalize=True):
            # Only store hyper-parameters here — no fitted state
            self.normalize = normalize

        def fit(self, X, y=None):
            # Initialize fitted state inside fit, not __init__
            self.mapping_ = {}
            if hasattr(X, 'columns'):
                cols = X.columns.tolist()
            else:
                cols = list(range(X.shape[1]))
            for col in cols:
                series = X[col] if hasattr(X, '__getitem__') else X[:, col]
                val_counts = pd.Series(series).value_counts(normalize=self.normalize)
                self.mapping_[col] = val_counts.to_dict()
            return self

        def transform(self, X):
            X_out = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
            if hasattr(X_out, 'columns'):
                cols = X_out.columns.tolist()
            else:
                cols = list(range(X_out.shape[1]))
            for col in cols:
                X_out[col] = X_out[col].map(self.mapping_.get(col, {})).fillna(0)
            return X_out

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV dataset containing track listings.
    
    Args:
        filepath: Path to the raw tracks CSV.
        
    Returns:
        Loaded Pandas DataFrame.
    """
    return pd.read_csv(filepath)

def get_preprocessor_transformer(df: pd.DataFrame) -> ColumnTransformer:
    """Builds the ColumnTransformer used to turn track features into a numeric matrix.

    Only columns that are actually present in `df` are wired into the transformer,
    so a dataset missing an optional column (e.g. 'mode') does not raise a
    ValueError at fit time.

    Design notes:
    - n_jobs=1 avoids joblib forking a dataframe copy per CPU core.
    - max_categories=200 on the artist OHE caps its width instead of creating one
      column per unique artist (~50k of them).
    - sparse_output=True keeps the combined matrix sparse (much less RAM).
    """
    frequency_encode_cols = ['year']
    # key, time_signature and mode are integers; they are cast to string before OHE.
    ohe_cols = ['artist', 'key', 'time_signature', 'mode']
    tfidf_col = 'tags'
    standard_scale_cols = ["duration_ms", "loudness", "tempo"]
    min_max_scale_cols = ["danceability", "energy", "speechiness",
                          "acousticness", "instrumentalness", "liveness", "valence"]

    present = set(df.columns)
    keep = lambda cols: [c for c in cols if c in present]

    transformers = []
    if keep(frequency_encode_cols):
        # category_encoders.CountEncoder auto-detects "categorical" columns by
        # dtype (object/category) and silently no-ops on anything else — 'year'
        # is int64, so without `cols=` it skipped encoding entirely and let raw
        # year values (~1900-2022) pass straight into the feature matrix. That
        # unscaled column then dominated cosine similarity (every pair of
        # tracks came out >0.9999 similar regardless of audio features).
        # `cols=` forces it to encode the named column regardless of dtype.
        transformers.append(("frequency_encode",
                              CountEncoder(normalize=True, cols=keep(frequency_encode_cols)),
                              keep(frequency_encode_cols)))
    if keep(ohe_cols):
        transformers.append(("ohe", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            max_categories=200,
        ), keep(ohe_cols)))
    if tfidf_col in present:
        transformers.append(("tfidf", TfidfVectorizer(max_features=85), tfidf_col))
    if keep(standard_scale_cols):
        transformers.append(("standard_scale", StandardScaler(), keep(standard_scale_cols)))
    if keep(min_max_scale_cols):
        transformers.append(("min_max_scale", MinMaxScaler(), keep(min_max_scale_cols)))

    return ColumnTransformer(transformers, remainder='drop', n_jobs=1)

def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans null values and lowercases categorical names.
    
    Args:
        df: Input raw tracks DataFrame.
        
    Returns:
        DataFrame clean tracks.
    """
    df_clean = df.copy()
    if 'spotify_id' in df_clean.columns:
        df_clean.drop_duplicates(subset=["spotify_id", "year", "duration_ms"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    if 'artist' in df_clean.columns:
        df_clean['artist'] = df_clean['artist'].astype(str).str.lower().str.strip()
    if 'tags' in df_clean.columns:
        df_clean['tags'] = df_clean['tags'].fillna("no_tags").astype(str).str.lower()
    
    # Cast integer categorical columns to string for OneHotEncoder compatibility
    for col in ['key', 'time_signature', 'mode']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-1).astype(int).astype(str)
    
    # Fill NaN for numeric columns to avoid transformer errors
    numeric_cols = ["duration_ms", "loudness", "tempo", "danceability", "energy",
                    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "year"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
    return df_clean

