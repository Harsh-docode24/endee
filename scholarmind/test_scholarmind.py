"""
ScholarMind — Smoke Tests
Simple, fast tests that verify core functions work without needing Endee running.
"""

import json
import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


class TestNormalizeYear(unittest.TestCase):
    """Test the year normalization function used for Endee $range filters."""

    def test_min_year(self):
        from ingest import normalize_year
        self.assertEqual(normalize_year(2010), 0)

    def test_max_year(self):
        from ingest import normalize_year
        self.assertEqual(normalize_year(2025), 999)

    def test_mid_year(self):
        from ingest import normalize_year
        result = normalize_year(2017)
        self.assertTrue(0 < result < 999)
        # 2017 is ~46.7% through range → ~466
        self.assertAlmostEqual(result, 466, delta=5)

    def test_below_min_clamps(self):
        from ingest import normalize_year
        self.assertEqual(normalize_year(2000), 0)

    def test_above_max_clamps(self):
        from ingest import normalize_year
        self.assertEqual(normalize_year(2030), 999)

    def test_monotonic(self):
        """Years should be monotonically increasing when normalized."""
        from ingest import normalize_year
        years = [2013, 2015, 2018, 2020, 2023]
        normalized = [normalize_year(y) for y in years]
        self.assertEqual(normalized, sorted(normalized))


class TestResultFormatter(unittest.TestCase):
    """Test the search result formatter."""

    def test_format_basic_result(self):
        from search import ScholarSearch
        raw = [
            {
                "id": "paper_001",
                "similarity": 0.9532,
                "meta": {
                    "title": "Test Paper",
                    "authors": "Author A, Author B",
                    "abstract": "This is a test abstract.",
                    "year": 2023,
                    "category": "machine_learning",
                    "keywords": "test, ml",
                },
            }
        ]
        formatted = ScholarSearch._format_results(raw)
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["id"], "paper_001")
        self.assertEqual(formatted[0]["title"], "Test Paper")
        self.assertAlmostEqual(formatted[0]["similarity"], 0.9532, places=4)

    def test_format_empty_results(self):
        from search import ScholarSearch
        formatted = ScholarSearch._format_results([])
        self.assertEqual(formatted, [])

    def test_format_missing_meta(self):
        from search import ScholarSearch
        raw = [{"id": "paper_x", "similarity": 0.5, "meta": {}}]
        formatted = ScholarSearch._format_results(raw)
        self.assertEqual(formatted[0]["title"], "Unknown")
        self.assertEqual(formatted[0]["authors"], "Unknown")


class TestEvalMetrics(unittest.TestCase):
    """Test evaluation metric functions."""

    def test_recall_at_k_all_found(self):
        from eval import recall_at_k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 5), 1.0)

    def test_recall_at_k_none_found(self):
        from eval import recall_at_k
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 3), 0.0)

    def test_recall_at_k_partial(self):
        from eval import recall_at_k
        retrieved = ["a", "x", "y", "b", "z"]
        relevant = ["a", "b", "c"]
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 5), 2 / 3)

    def test_mrr_first_position(self):
        from eval import mrr_at_k
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant, 10), 1.0)

    def test_mrr_third_position(self):
        from eval import mrr_at_k
        retrieved = ["x", "y", "a", "b"]
        relevant = ["a"]
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant, 10), 1 / 3)

    def test_mrr_not_found(self):
        from eval import mrr_at_k
        retrieved = ["x", "y", "z"]
        relevant = ["a"]
        self.assertAlmostEqual(mrr_at_k(retrieved, relevant, 3), 0.0)


class TestConfig(unittest.TestCase):
    """Test configuration values."""

    def test_embedding_dimension(self):
        import config
        self.assertEqual(config.EMBEDDING_DIMENSION, 384)

    def test_space_type(self):
        import config
        self.assertEqual(config.SPACE_TYPE, "cosine")

    def test_sample_data_path_exists(self):
        import config
        self.assertTrue(os.path.exists(config.SAMPLE_DATA_PATH))


class TestSampleData(unittest.TestCase):
    """Test sample data integrity."""

    def test_papers_json_loads(self):
        import config
        with open(config.SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        self.assertEqual(len(papers), 45)

    def test_papers_have_required_fields(self):
        import config
        with open(config.SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        required = {"id", "title", "authors", "year", "category", "area", "keywords", "abstract"}
        for paper in papers:
            self.assertTrue(required.issubset(paper.keys()), f"Missing fields in {paper.get('id')}")

    def test_all_ids_unique(self):
        import config
        with open(config.SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        ids = [p["id"] for p in papers]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
