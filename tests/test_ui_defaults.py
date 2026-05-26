import unittest
from pathlib import Path


class UiDefaultsTests(unittest.TestCase):
    def test_default_demo_paths_point_to_full_corpus_index(self):
        from ui.defaults import demo_bm25_path, demo_sqlite_path

        root = Path("/repo")

        self.assertEqual(demo_sqlite_path(root), root / "outputs" / "full_corpus_index" / "contracts.sqlite")
        self.assertEqual(demo_bm25_path(root), root / "outputs" / "full_corpus_index" / "bm25_chunks.pkl")


if __name__ == "__main__":
    unittest.main()
