# Future Improvements

Ideas I'd like to try later. None of these are built yet.

- Faster similarity lookup with approximate nearest neighbours (Annoy or FAISS)
  instead of computing cosine similarity against the whole catalogue each time.
- Point DVC at a real remote (local folder or Google Drive) so the data can be
  pulled instead of kept locally.
- A thumbs up / thumbs down feedback loop to adjust recommendations over time.
- More evaluation metrics such as NDCG and Precision@K.
- Mood filtering using energy and valence thresholds (happy, sad, chill, etc.).
- Lyrics embeddings with sentence transformers as an extra content signal.
