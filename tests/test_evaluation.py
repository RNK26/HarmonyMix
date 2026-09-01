import unittest

import pandas as pd

from src.evaluation.evaluate import evaluate_collaborative, evaluate_diversity_novelty
from src.recommenders.content_based import ContentBasedRecommender


class TestEvaluation(unittest.TestCase):
    def test_leave_one_out_finds_cooccurring_track(self):
        # Two users, each with the same two co-occurring tracks. Holding out one
        # and querying the other should recover it, giving a perfect hit rate.
        playlist_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'track_id': ['a', 'b', 'a', 'b'],
        })
        result = evaluate_collaborative(playlist_df, k=5, sample_users=2)
        self.assertEqual(result['users_evaluated'], 2)
        self.assertEqual(result['hit_rate'], 1.0)
        self.assertGreater(result['coverage'], 0.0)

    def test_no_eligible_users_returns_zero(self):
        # Every user has a single track, so leave-one-out is impossible.
        playlist_df = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'track_id': ['a', 'b'],
        })
        result = evaluate_collaborative(playlist_df, k=5)
        self.assertEqual(result['users_evaluated'], 0)
        self.assertEqual(result['hit_rate'], 0.0)


class TestDiversityNovelty(unittest.TestCase):
    def setUp(self):
        # s1/s2 are near-identical audio features; s3 is deliberately different.
        self.tracks_df = pd.DataFrame({
            'track_id': ['s1', 's2', 's3', 's4'],
            'name': ['A', 'B', 'C', 'D'],
            'artist': ['Artist A', 'Artist B', 'Artist C', 'Artist D'],
            'year': [2004, 2004, 2010, 2015],
            'duration_ms': [220000, 220000, 200000, 210000],
            'danceability': [0.8, 0.8, 0.1, 0.5],
            'energy': [0.9, 0.9, 0.1, 0.5],
            'key': [1, 1, 5, 3],
            'loudness': [-5.0, -5.0, -20.0, -10.0],
            'time_signature': [4, 4, 4, 4],
            'speechiness': [0.05, 0.05, 0.4, 0.2],
            'acousticness': [0.01, 0.01, 0.9, 0.5],
            'instrumentalness': [0.0, 0.0, 0.8, 0.3],
            'liveness': [0.1, 0.1, 0.1, 0.1],
            'valence': [0.5, 0.5, 0.1, 0.3],
            'tempo': [120.0, 120.0, 70.0, 95.0],
            'tags': ['rock', 'rock', 'ambient', 'folk'],
            'spotify_id': ['sp1', 'sp2', 'sp3', 'sp4'],
        })
        self.content_rec = ContentBasedRecommender(self.tracks_df)

    def test_low_diversity_when_recommendations_are_near_identical(self):
        # A recommender that always returns the two near-identical tracks
        # should score low diversity.
        result = evaluate_diversity_novelty(
            self.content_rec,
            get_recommendations_fn=lambda tid, top_n: pd.DataFrame({'track_id': ['s1', 's2']}),
            query_track_ids=['s3'],
            top_n=2,
        )
        self.assertEqual(result['queries_evaluated'], 1)
        self.assertLess(result['diversity'], 0.2)

    def test_high_diversity_when_recommendations_are_varied(self):
        result = evaluate_diversity_novelty(
            self.content_rec,
            get_recommendations_fn=lambda tid, top_n: pd.DataFrame({'track_id': ['s2', 's3', 's4']}),
            query_track_ids=['s1'],
            top_n=3,
        )
        self.assertGreater(result['diversity'], 0.2)

    def test_novelty_favours_less_frequently_heard_tracks(self):
        interaction_counts = {'s1': 1000, 's2': 1000, 's3': 0, 's4': 0}
        popular_only = evaluate_diversity_novelty(
            self.content_rec,
            get_recommendations_fn=lambda tid, top_n: pd.DataFrame({'track_id': ['s1', 's2']}),
            query_track_ids=['s3'],
            interaction_counts=interaction_counts,
            top_n=2,
        )
        novel_only = evaluate_diversity_novelty(
            self.content_rec,
            get_recommendations_fn=lambda tid, top_n: pd.DataFrame({'track_id': ['s3', 's4']}),
            query_track_ids=['s1'],
            interaction_counts=interaction_counts,
            top_n=2,
        )
        self.assertGreater(novel_only['novelty'], popular_only['novelty'])

    def test_no_interaction_counts_means_no_novelty_key(self):
        result = evaluate_diversity_novelty(
            self.content_rec,
            get_recommendations_fn=lambda tid, top_n: pd.DataFrame({'track_id': ['s2', 's3']}),
            query_track_ids=['s1'],
            top_n=2,
        )
        self.assertNotIn('novelty', result)


if __name__ == '__main__':
    unittest.main()
