"""Builds the small data bundle the deployed app runs on.

The full pipeline needs playlists.csv (575 MB) and a fitted collaborative
engine (250 MB). Neither can go in a Git repository -- GitHub rejects files
over 100 MB -- so the hosted demo runs on a trimmed version of the same data.

Two things get written:

    data/tracks_demo.csv    the 10,000 most-listened tracks, real names
    data/collab_matrix.npz  their co-occurrence matrix, top 100 neighbours each

Subsetting tracks alone does not help. The co-occurrence submatrix for the
1,000 most popular tracks is 99.5% dense, because popular tracks appear
alongside almost everything. Keeping only each track's strongest 100
neighbours is what makes it small: 82 million non-zeros down to 1 million,
250 MB down to under 3 MB.

Run:  python scripts/build_demo_bundle.py
Needs the full local dataset and a fitted models/collab_engine.pkl.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.model_persistence import load_model  # noqa: E402

TRACKS_CSV = 'data/raw/tracks.csv'
OUT_TRACKS = 'data/tracks_demo.csv'
OUT_MATRIX = 'data/collab_matrix.npz'

N_TRACKS = 10_000
N_NEIGHBOURS = 100


def top_neighbours(matrix, keep):
    """Keeps only the `keep` highest-scoring entries in each row.

    The diagonal is a track's own listen count, which is always the largest
    value in its row, so it survives this without being special-cased. It has
    to survive: get_recommendations divides by it to turn co-occurrence counts
    into cosine similarity.
    """
    matrix = matrix.tocsr()
    rows, cols, vals = [], [], []
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        idx, data = matrix.indices[start:end], matrix.data[start:end]
        if len(data) > keep:
            top = np.argpartition(-data, keep)[:keep]
            idx, data = idx[top], data[top]
        rows.append(np.full(len(idx), i))
        cols.append(idx)
        vals.append(data)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
    )


def main():
    engine = load_model('models/collab_engine.pkl')
    if engine is None:
        raise SystemExit(
            "models/collab_engine.pkl not found. Run the app once against the "
            "full dataset to fit and cache the engine first."
        )

    tracks = pd.read_csv(TRACKS_CSV)
    full = engine.cooccurrence_matrix.tocsr()
    listens = np.asarray(full.diagonal()).flatten()
    print(f"full matrix: {full.shape[0]:,} tracks, {full.nnz:,} non-zeros, "
          f"{full.nnz / full.shape[0]**2 * 100:.2f}% dense")

    # Rank by listen count, but only keep tracks that tracks.csv can name --
    # a recommendation the app cannot label is no use in a demo.
    nameable = set(tracks['track_id'])
    order = np.argsort(-listens)
    chosen = [i for i in order if engine.tracks[i] in nameable][:N_TRACKS]
    chosen = np.array(chosen)
    print(f"selected {len(chosen):,} tracks, listen counts "
          f"{listens[chosen].min():,.0f} to {listens[chosen].max():,.0f}")

    sub = full[chosen][:, chosen]
    print(f"subset before truncation: {sub.nnz:,} non-zeros "
          f"({sub.nnz / len(chosen)**2 * 100:.1f}% dense)")

    trimmed = top_neighbours(sub, N_NEIGHBOURS).astype(np.float32)
    sp.save_npz(OUT_MATRIX, trimmed, compressed=True)

    chosen_ids = engine.tracks[chosen]
    demo_tracks = (tracks[tracks['track_id'].isin(set(chosen_ids))]
                   .drop_duplicates('track_id')
                   .set_index('track_id')
                   .loc[chosen_ids]        # same order as the matrix
                   .reset_index())
    demo_tracks.to_csv(OUT_TRACKS, index=False)

    # The app pairs these two by position, so a mismatch would silently label
    # every recommendation with the wrong song.
    assert list(demo_tracks['track_id']) == list(chosen_ids), "track order drifted"

    print(f"\nwrote {OUT_MATRIX}  {trimmed.nnz:,} non-zeros, "
          f"{os.path.getsize(OUT_MATRIX)/1e6:.2f} MB")
    print(f"wrote {OUT_TRACKS}  {len(demo_tracks):,} rows, "
          f"{os.path.getsize(OUT_TRACKS)/1e6:.2f} MB")


if __name__ == '__main__':
    main()
