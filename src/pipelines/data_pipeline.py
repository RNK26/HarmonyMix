import os
import sys
import logging

# Running this file directly (`python src/pipelines/data_pipeline.py`, which is
# what `dvc repro` does) puts src/pipelines/ on sys.path, not the project root,
# so the `src.*` absolute import below fails with ModuleNotFoundError unless the
# package was pip-installed. Insert the project root so it works either way,
# matching the same fix already used in tests/conftest.py and app/app.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocess import load_data, clean_and_prepare_dataset

logger = logging.getLogger(__name__)

def run_data_pipeline(raw_tracks_path: str, output_tracks_path: str):
    """Loads the raw tracks CSV, cleans it and writes the processed CSV.

    Args:
        raw_tracks_path: Source CSV for tracks.
        output_tracks_path: Output clean CSV path.
    """
    logger.info("Loading data from %s", raw_tracks_path)
    df = load_data(raw_tracks_path)

    logger.info("Cleaning dataset columns")
    df = clean_and_prepare_dataset(df)

    os.makedirs(os.path.dirname(output_tracks_path), exist_ok=True)
    logger.info("Saving preprocessed track data to %s", output_tracks_path)
    df.to_csv(output_tracks_path, index=False)
    logger.info("Preprocessing pipeline step completed (%d rows)", len(df))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_data_pipeline(
        raw_tracks_path='data/raw/tracks.csv',
        output_tracks_path='data/processed/tracks_cleaned.csv'
    )

