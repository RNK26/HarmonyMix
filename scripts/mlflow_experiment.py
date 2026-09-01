"""Sweeps the hybrid content/collaborative weight and logs each run to MLflow.

The weight trades Diversity against Novelty. joblib saves the fitted model but
not the numbers behind a given weight, so until now that sweep only existed in
a notebook.

Run:  python scripts/mlflow_experiment.py
View: mlflow ui --backend-store-uri sqlite:///mlflow.db

MLflow 3.x dropped the old ./mlruns file store, so runs go to mlflow.db.
"""

import functools
import os
import sys

import mlflow
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.evaluate import evaluate_diversity_novelty  # noqa: E402
from src.recommenders.hybrid_recommender import HybridRecommender  # noqa: E402
from src.utils.model_persistence import load_model  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_QUERIES = 200          # query tracks sampled per run
TOP_N = 10
SEED = 42


def main() -> None:
    content_engine = load_model('models/content_engine.pkl')
    collab_engine = load_model('models/collab_engine.pkl')
    if content_engine is None or collab_engine is None:
        raise SystemExit("Fitted engines not found -- run the Streamlit app once first.")

    hybrid = HybridRecommender(content_engine, collab_engine)

    # Novelty needs a popularity proxy; the co-occurrence diagonal is the
    # listener count per track.
    freq = np.asarray(collab_engine.cooccurrence_matrix.diagonal()).flatten()
    interaction_counts = dict(zip(collab_engine.tracks, freq))

    # Sample query tracks that HAVE collaborative history, so the weight sweep
    # actually exercises the blend rather than the cold-start fallback.
    rng = np.random.default_rng(SEED)
    eligible = [t for t in collab_engine.tracks if t in content_engine.df['track_id'].values]
    query_ids = list(rng.choice(eligible, size=min(N_QUERIES, len(eligible)), replace=False))

    mlflow.set_experiment("harmonymix-hybrid-weight-sweep")
    rows = []

    for w in WEIGHTS:
        with mlflow.start_run(run_name=f"w_content={w}"):
            mlflow.log_param("w_content", w)
            mlflow.log_param("w_collaborative", round(1.0 - w, 3))
            mlflow.log_param("top_n", TOP_N)
            mlflow.log_param("n_query_tracks", len(query_ids))
            mlflow.log_param("catalogue_size", len(content_engine.df))

            # partial fixes w_content so the evaluator can call the engine with
            # just (track_id, top_n).
            fn = functools.partial(hybrid.get_recommendations, w_content=w)
            result = evaluate_diversity_novelty(
                content_engine, fn, query_ids,
                interaction_counts=interaction_counts, top_n=TOP_N,
            )

            mlflow.log_metric("diversity", result["diversity"])
            mlflow.log_metric("novelty", result["novelty"])
            mlflow.log_metric("queries_evaluated", result["queries_evaluated"])

            rows.append({"w_content": w, **result})
            print(f"w_content={w:.2f}  diversity={result['diversity']:.4f}  "
                  f"novelty={result['novelty']:.4f}  n={result['queries_evaluated']}")

    df = pd.DataFrame(rows)
    print("\n--- sweep summary ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nlogged to MLflow experiment 'harmonymix-hybrid-weight-sweep'")
    print("view with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
