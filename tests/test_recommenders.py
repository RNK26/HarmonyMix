import unittest
import pandas as pd
import numpy as np
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from src.recommenders.hybrid_recommender import HybridRecommender

class TestContentBasedRecommender(unittest.TestCase):
    def setUp(self):
        # Create minimal track dataset matching expected preprocessing columns
        self.tracks_df = pd.DataFrame({
            'track_id': ['s1', 's2', 's3'],
            'name': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist A', 'Artist B', 'Artist C'],
            'year': [2004, 2006, 2004],
            'duration_ms': [220000, 250000, 220000],
            'danceability': [0.8, 0.2, 0.75],
            'energy': [0.9, 0.1, 0.85],
            'key': [1, 2, 1],
            'loudness': [-5.0, -10.0, -5.0],
            'time_signature': [4, 4, 4],
            'speechiness': [0.05, 0.03, 0.05],
            'acousticness': [0.01, 0.8, 0.01],
            'instrumentalness': [0.0, 0.1, 0.0],
            'liveness': [0.1, 0.2, 0.1],
            'valence': [0.5, 0.3, 0.5],
            'tempo': [120.0, 90.0, 120.0],
            'tags': ['rock, indie', 'pop, acoustic', 'rock, indie'],
            'spotify_id': ['sp1', 'sp2', 'sp3']
        })
        self.recommender = ContentBasedRecommender(self.tracks_df)

    def test_recommendation_count(self):
        # Request recommendations for Song A (s1)
        recs = self.recommender.get_recommendations('s1', top_n=2)
        self.assertEqual(len(recs), 2)
        # Verify the target song is excluded
        self.assertNotIn('s1', recs['track_id'].values)

    def test_similarity_logic(self):
        # Song C (s3) is closer to Song A (s1) than Song B (s2) is.
        recs = self.recommender.get_recommendations('s1', top_n=2)
        top_rec_id = recs.iloc[0]['track_id']
        self.assertEqual(top_rec_id, 's3')

    def test_raw_row_count_matches_input(self):
        # app.py's cache-invalidation check relies on this being the *raw*
        # (pre-cleaning) row count, not len(self.df).
        self.assertEqual(self.recommender.raw_row_count, len(self.tracks_df))


class TestCollaborativeFilteringRecommender(unittest.TestCase):
    def setUp(self):
        # Create minimal playlist dataset where s1 and s2 co-occur
        self.playlist_df = pd.DataFrame({
            'playlist_id': ['p1', 'p1', 'p2', 'p2', 'p3'],
            'track_id': ['s1', 's2', 's1', 's3', 's4']
        })
        self.recommender = CollaborativeFilteringRecommender(self.playlist_df)

    def test_cooccurrence_counts(self):
        recs = self.recommender.get_recommendations('s1', top_n=2)
        # Verify that recommendations contain s2 and s3
        rec_ids = recs['track_id'].values
        self.assertIn('s2', rec_ids)
        self.assertIn('s3', rec_ids)

    def test_raw_row_count_matches_input(self):
        self.assertEqual(self.recommender.raw_row_count, len(self.playlist_df))

    def test_does_not_retain_raw_playlist_dataframe(self):
        # Retaining the full raw interaction table (up to ~9.7M rows / ~700 MB
        # on the real dataset) as instance state would mean every save/load of
        # this engine serializes that dead weight for no benefit -- nothing in
        # get_recommendations() reads it. Guards against re-introducing it.
        self.assertFalse(hasattr(self.recommender, 'playlist_df'))
        self.assertFalse(hasattr(self.recommender, 'idx_to_track'))


class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        tracks_df = pd.DataFrame({
            'track_id': ['s1', 's2', 's3', 's4'],
            'name': ['Song A', 'Song B', 'Song C', 'Song D'],
            'artist': ['Artist A', 'Artist B', 'Artist C', 'Artist D'],
            'year': [2004, 2006, 2004, 2010],
            'duration_ms': [220000, 250000, 220000, 200000],
            'danceability': [0.8, 0.2, 0.75, 0.6],
            'energy': [0.9, 0.1, 0.85, 0.5],
            'key': [1, 2, 1, 3],
            'loudness': [-5.0, -10.0, -5.0, -7.0],
            'time_signature': [4, 4, 4, 4],
            'speechiness': [0.05, 0.03, 0.05, 0.04],
            'acousticness': [0.01, 0.8, 0.01, 0.3],
            'instrumentalness': [0.0, 0.1, 0.0, 0.05],
            'liveness': [0.1, 0.2, 0.1, 0.15],
            'valence': [0.5, 0.3, 0.5, 0.4],
            'tempo': [120.0, 90.0, 120.0, 110.0],
            'tags': ['rock, indie', 'pop, acoustic', 'rock, indie', 'rock, indie'],
            'spotify_id': ['sp1', 'sp2', 'sp3', 'sp4'],
            'popularity': [80, 50, 90, 60]
        })
        # s4 deliberately has no playlist entry -- cold start for collaborative.
        playlists_df = pd.DataFrame({
            'playlist_id': ['p1', 'p1', 'p2'],
            'track_id': ['s1', 's2', 's3']
        })

        content_rec = ContentBasedRecommender(tracks_df)
        collab_rec = CollaborativeFilteringRecommender(playlists_df)
        self.hybrid_rec = HybridRecommender(content_rec, collab_rec)

    def test_hybrid_recs(self):
        recs = self.hybrid_rec.get_recommendations('s1', w_content=0.5, top_n=2)
        self.assertEqual(len(recs), 2)
        self.assertTrue('hybrid_score' in recs.columns)

    def test_weight_extremes_match_single_engines(self):
        # w_content=1.0 should rank purely by content score.
        recs = self.hybrid_rec.get_recommendations('s1', w_content=1.0, top_n=2)
        self.assertTrue((recs['hybrid_score'] == recs['content_score']).all())

    def test_cold_start_query_overrides_to_pure_content(self):
        # s4 has no collaborative history at all. Even with a collaborative-
        # leaning request (w_content=0.2), the adaptive override should use
        # pure content instead of shrinking the score by blending in a fake
        # "zero" collaborative signal.
        recs = self.hybrid_rec.get_recommendations('s4', w_content=0.2, top_n=3)
        self.assertTrue((recs['hybrid_score'] == recs['content_score']).all())
        self.assertTrue((recs['w_content_used'] == 1.0).all())

    def test_non_cold_start_query_respects_requested_weight(self):
        # s1 does have collaborative history, so the requested weight should
        # be used as-is (no override).
        recs = self.hybrid_rec.get_recommendations('s1', w_content=0.2, top_n=2)
        self.assertTrue((recs['w_content_used'] == 0.2).all())


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tracks_df = pd.DataFrame({
            'track_id': ['s1', 's2', 's3'],
            'name': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist A', 'Artist B', 'Artist C'],
            'year': [2004, 2006, 2004],
            'duration_ms': [220000, 250000, 220000],
            'danceability': [0.8, 0.2, 0.75],
            'energy': [0.9, 0.1, 0.85],
            'key': [1, 2, 1],
            'loudness': [-5.0, -10.0, -5.0],
            'time_signature': [4, 4, 4],
            'speechiness': [0.05, 0.03, 0.05],
            'acousticness': [0.01, 0.8, 0.01],
            'instrumentalness': [0.0, 0.1, 0.0],
            'liveness': [0.1, 0.2, 0.1],
            'valence': [0.5, 0.3, 0.5],
            'tempo': [120.0, 90.0, 120.0],
            'tags': ['rock, indie', 'pop, acoustic', 'rock, indie'],
            'spotify_id': ['sp1', 'sp2', 'sp3']
        })

    def test_content_unknown_track_raises_keyerror(self):
        rec = ContentBasedRecommender(self.tracks_df)
        with self.assertRaises(KeyError):
            rec.get_recommendations('does_not_exist', top_n=5)

    def test_content_missing_optional_column(self):
        # Dropping 'mode' (which isn't even present) and 'tags' must not crash.
        df = self.tracks_df.drop(columns=['tags'])
        rec = ContentBasedRecommender(df)
        recs = rec.get_recommendations('s1', top_n=2)
        self.assertEqual(len(recs), 2)

    def test_content_missing_values_are_handled(self):
        df = self.tracks_df.copy()
        df.loc[0, 'energy'] = np.nan
        df.loc[1, 'tags'] = np.nan
        rec = ContentBasedRecommender(df)
        recs = rec.get_recommendations('s2', top_n=2)
        self.assertEqual(len(recs), 2)

    def test_collab_unknown_track_returns_empty(self):
        playlist_df = pd.DataFrame({'playlist_id': ['p1', 'p1'], 'track_id': ['s1', 's2']})
        rec = CollaborativeFilteringRecommender(playlist_df)
        recs = rec.get_recommendations('unknown', top_n=5)
        self.assertTrue(recs.empty)

    def test_topn_larger_than_catalogue(self):
        rec = ContentBasedRecommender(self.tracks_df)
        recs = rec.get_recommendations('s1', top_n=999)
        # Only two other tracks exist besides the query.
        self.assertEqual(len(recs), 2)


if __name__ == '__main__':
    unittest.main()

