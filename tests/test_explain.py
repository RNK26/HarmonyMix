import unittest
import pandas as pd

from src.recommenders.explain import explain_recommendation


class TestExplainRecommendation(unittest.TestCase):
    def test_both_signals_present_mentions_both_and_leaning(self):
        row = pd.Series({'content_score': 0.68, 'collaborative_score': 0.12})
        text = explain_recommendation(row, w_content=0.6)
        self.assertIn('68%', text)
        self.assertIn('12%', text)
        self.assertIn('content-based matching', text)

    def test_collaborative_leaning_when_weight_below_half(self):
        row = pd.Series({'content_score': 0.5, 'collaborative_score': 0.5})
        text = explain_recommendation(row, w_content=0.3)
        self.assertIn('co-listening matching', text)

    def test_content_only_signal(self):
        row = pd.Series({'content_score': 0.9, 'collaborative_score': 0.0})
        text = explain_recommendation(row, w_content=1.0)
        self.assertIn('90%', text)
        self.assertNotIn('co-listening', text)

    def test_no_signal_returns_fallback_message(self):
        row = pd.Series({'content_score': 0.0, 'collaborative_score': 0.0})
        text = explain_recommendation(row, w_content=0.5)
        self.assertIn('fallback', text)

    def test_adaptive_override_is_called_out(self):
        # HybridRecommender sets w_content_used=1.0 when the query track has
        # no collaborative history, regardless of the requested w_content.
        row = pd.Series({'content_score': 0.75, 'collaborative_score': 0.0, 'w_content_used': 1.0})
        text = explain_recommendation(row, w_content=0.2)
        self.assertIn('no playlist interaction data', text)

    def test_no_override_when_weight_matches(self):
        row = pd.Series({'content_score': 0.5, 'collaborative_score': 0.5, 'w_content_used': 0.6})
        text = explain_recommendation(row, w_content=0.6)
        self.assertNotIn('no playlist interaction data', text)
        self.assertIn('content-based matching', text)


if __name__ == '__main__':
    unittest.main()
