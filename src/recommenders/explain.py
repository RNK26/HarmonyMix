"""
explain.py — turns the score columns HybridRecommender already computes into a
short, human-readable reason string. No new model or signal: content_score and
collaborative_score are exactly what src/recommenders/hybrid_recommender.py
already produces per row; this just formats them.
"""
import pandas as pd


def explain_recommendation(rec_row: pd.Series, w_content: float) -> str:
    """Builds a one-line explanation for a single hybrid recommendation.

    Args:
        rec_row: One row from HybridRecommender.get_recommendations() — must
            have 'content_score' and/or 'collaborative_score' columns. If it
            also has a 'w_content_used' column (the weight the hybrid engine
            actually applied -- see hybrid_recommender.py's adaptive
            cold-start override), that's used instead of `w_content` so the
            explanation reflects what really happened, not just what was
            requested.
        w_content: The weight requested by the caller (e.g. the app's
            slider value). Also what's used if the row has no
            'w_content_used' (e.g. in tests that build a plain Series).

    Returns:
        A short explanation string, e.g.
        "68% feature similarity and 12% co-listening similarity with your
        pick, leaning toward content-based matching."
    """
    content = float(rec_row.get('content_score', 0.0) or 0.0)
    collab = float(rec_row.get('collaborative_score', 0.0) or 0.0)
    effective_w = rec_row.get('w_content_used', w_content)
    effective_w = w_content if effective_w is None else float(effective_w)

    parts = []
    if content > 0:
        # "feature similarity", not "audio similarity": the content score
        # combines audio features with artist and other metadata, and artist
        # is the largest single block of that feature matrix.
        parts.append(f"{content:.0%} feature similarity")
    if collab > 0:
        parts.append(f"{collab:.0%} co-listening similarity")

    if not parts:
        return "No strong similarity signal — a low-confidence fallback pick."

    reason = " and ".join(parts) + " with your pick"

    if effective_w != w_content:
        # Adaptive override kicked in (query track had no collaborative
        # history) -- say so instead of silently ignoring the slider.
        return (f"{reason}. This song has no playlist interaction data in this "
                f"dataset, so recommendations rely on feature similarity only.")

    if content > 0 and collab > 0:
        lean = "content-based matching" if effective_w >= 0.5 else "co-listening matching"
        return f"{reason}, leaning toward {lean}."
    return f"{reason}."
