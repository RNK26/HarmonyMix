"""
model_persistence.py — Save and load fitted recommender objects via joblib/pickle.

Fitting the ContentBasedRecommender on 50 k tracks takes ~10–30 s.
Persisting the fitted object to disk means subsequent app restarts load in < 1 s.

Usage
-----
    from src.utils.model_persistence import save_model, load_model

    # On first run (or after data changes):
    save_model(content_engine, 'models/content_engine.pkl')

    # On subsequent runs:
    engine = load_model('models/content_engine.pkl')
    if engine is None:
        engine = ContentBasedRecommender(df)        # fallback: re-fit
        save_model(engine, 'models/content_engine.pkl')
"""
import logging
import os

logger = logging.getLogger(__name__)

try:
    import joblib
    _BACKEND = 'joblib'
except ImportError:
    import pickle
    _BACKEND = 'pickle'


def save_model(obj, filepath: str) -> None:
    """Serialises *obj* to *filepath*.

    Writes to a temporary file first, then atomically renames it into place
    (`os.replace`) so an interrupted write never leaves a truncated file at
    the real path.

    Args:
        obj:      Any Python object (recommender instance, transformer, …).
        filepath: Destination path (created including parent directories).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = f"{filepath}.tmp"
    if _BACKEND == 'joblib':
        joblib.dump(obj, tmp_path, compress=3)
    else:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, filepath)
    logger.info("Model saved to %s (backend=%s)", filepath, _BACKEND)


def load_model(filepath: str):
    """Deserialises a model from *filepath*.

    Never raises on a missing or corrupted file -- callers already treat a
    `None` return as "not cached, (re)build it" (see the module docstring),
    so a corrupted pickle degrades to exactly the same recovery path as a
    missing one instead of crashing the app.

    Args:
        filepath: Path to the serialised model file.

    Returns:
        The deserialised object, or None if the file is missing, empty, or
        fails to deserialise (truncated/corrupted).
    """
    if not os.path.exists(filepath):
        logger.warning("Model file not found: %s", filepath)
        return None
    if os.path.getsize(filepath) == 0:
        logger.warning("Model file is empty (0 bytes), treating as missing: %s", filepath)
        return None
    try:
        if _BACKEND == 'joblib':
            return joblib.load(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # Covers EOFError (truncated write), UnpicklingError (corrupted
        # bytes), and anything else a broken pickle file can raise -- all
        # mean the same thing here: this file is unusable, rebuild from data.
        logger.exception("Model file at %s is corrupted or unreadable; treating as missing", filepath)
        return None
