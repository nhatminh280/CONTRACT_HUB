import unittest
from pathlib import Path
import tempfile


def _make_chunk(contract_id: str, clause: str = "1"):
    from ingestion.chunker import Chunk

    return Chunk(
        id=f"{contract_id}_{clause}",
        text=f"text for {contract_id} {clause}",
        contract_id=contract_id,
        clause_number=clause,
        page_start=1,
        page_end=1,
    )


class UiUploadsTests(unittest.TestCase):
    def test_classifies_single_pdf_upload(self):
        from ui.uploads import classify_uploads

        self.assertEqual(classify_uploads(["contract.pdf"]), "pdf")

    def test_classifies_multiple_pdf_uploads(self):
        from ui.uploads import classify_uploads

        self.assertEqual(classify_uploads(["a.pdf", "b.pdf"]), "pdf")

    def test_classifies_image_uploads(self):
        from ui.uploads import classify_uploads

        self.assertEqual(classify_uploads(["page_001.jpg", "page_002.png"]), "images")

    def test_rejects_mixed_pdf_and_image_uploads(self):
        from ui.uploads import classify_uploads

        with self.assertRaises(ValueError):
            classify_uploads(["contract.pdf", "page_001.jpg"])

    def test_reset_upload_dir_removes_previous_images(self):
        from ui.uploads import reset_upload_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "contract"
            folder.mkdir()
            stale = folder / "old.png"
            stale.write_bytes(b"old")

            reset_upload_dir(folder)

            self.assertTrue(folder.exists())
            self.assertFalse(stale.exists())


class NormalizeContractIdTests(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        from ui.uploads import normalize_contract_id

        self.assertEqual(normalize_contract_id("  HD-2024-001  "), "HD-2024-001")

    def test_preserves_interior_spaces(self):
        from ui.uploads import normalize_contract_id

        self.assertEqual(normalize_contract_id("HD 2024 001"), "HD 2024 001")

    def test_rejects_empty_string(self):
        from ui.uploads import normalize_contract_id

        with self.assertRaises(ValueError):
            normalize_contract_id("")

    def test_rejects_whitespace_only(self):
        from ui.uploads import normalize_contract_id

        with self.assertRaises(ValueError):
            normalize_contract_id("   \t\n  ")


class NextContractIdTests(unittest.TestCase):
    def test_returns_base_when_taken_empty(self):
        from ui.uploads import next_contract_id

        self.assertEqual(next_contract_id("HD-2024-001", set()), "HD-2024-001")

    def test_returns_base_when_base_not_taken(self):
        from ui.uploads import next_contract_id

        self.assertEqual(next_contract_id("HD-2024-001", {"OTHER"}), "HD-2024-001")

    def test_suffixes_002_when_base_taken(self):
        from ui.uploads import next_contract_id

        self.assertEqual(next_contract_id("HD-2024-001", {"HD-2024-001"}), "HD-2024-001-002")

    def test_skips_to_003_when_002_also_taken(self):
        from ui.uploads import next_contract_id

        taken = {"HD-2024-001", "HD-2024-001-002"}
        self.assertEqual(next_contract_id("HD-2024-001", taken), "HD-2024-001-003")

    def test_hd_1_does_not_collide_with_hd_10(self):
        from ui.uploads import next_contract_id

        self.assertEqual(next_contract_id("HD-1", {"HD-10"}), "HD-1")


class RemoveContractChunksTests(unittest.TestCase):
    def test_removes_only_matching_contract_id(self):
        from ui.uploads import remove_contract_chunks

        chunks = [_make_chunk("A"), _make_chunk("B"), _make_chunk("A", "2")]

        result = remove_contract_chunks(chunks, "A")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].contract_id, "B")

    def test_preserves_order_of_survivors(self):
        from ui.uploads import remove_contract_chunks

        chunks = [_make_chunk("B", "1"), _make_chunk("A"), _make_chunk("B", "2")]

        result = remove_contract_chunks(chunks, "A")

        self.assertEqual([c.clause_number for c in result], ["1", "2"])

    def test_noop_when_contract_id_absent(self):
        from ui.uploads import remove_contract_chunks

        chunks = [_make_chunk("A"), _make_chunk("B")]

        result = remove_contract_chunks(chunks, "Z")

        self.assertEqual(len(result), 2)

    def test_does_not_mutate_input(self):
        from ui.uploads import remove_contract_chunks

        chunks = [_make_chunk("A"), _make_chunk("B")]
        before = list(chunks)

        remove_contract_chunks(chunks, "A")

        self.assertEqual(chunks, before)


class ReplaceContractChunksTests(unittest.TestCase):
    def test_replaces_old_contract_chunks_with_new(self):
        from ui.uploads import replace_contract_chunks

        existing = [_make_chunk("A", "1"), _make_chunk("B"), _make_chunk("A", "2")]
        new_chunks = [_make_chunk("A", "9")]

        result = replace_contract_chunks(existing, new_chunks, "A")

        a_chunks = [c for c in result if c.contract_id == "A"]
        self.assertEqual(len(a_chunks), 1)
        self.assertEqual(a_chunks[0].clause_number, "9")

    def test_appends_new_when_contract_not_previously_present(self):
        from ui.uploads import replace_contract_chunks

        existing = [_make_chunk("B")]
        new_chunks = [_make_chunk("A")]

        result = replace_contract_chunks(existing, new_chunks, "A")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1].contract_id, "A")

    def test_does_not_mutate_inputs(self):
        from ui.uploads import replace_contract_chunks

        existing = [_make_chunk("A"), _make_chunk("B")]
        new_chunks = [_make_chunk("A", "9")]
        existing_snapshot = list(existing)
        new_snapshot = list(new_chunks)

        replace_contract_chunks(existing, new_chunks, "A")

        self.assertEqual(existing, existing_snapshot)
        self.assertEqual(new_chunks, new_snapshot)


class SanitizeWidgetKeyTests(unittest.TestCase):
    def test_replaces_disallowed_characters(self):
        from ui.uploads import sanitize_widget_key

        self.assertEqual(sanitize_widget_key("HD/2024 001.PDF"), "HD_2024_001_PDF")

    def test_preserves_alphanumerics_dash_underscore(self):
        from ui.uploads import sanitize_widget_key

        self.assertEqual(sanitize_widget_key("HD-2024_001"), "HD-2024_001")

    def test_deterministic(self):
        from ui.uploads import sanitize_widget_key

        self.assertEqual(sanitize_widget_key("foo bar"), sanitize_widget_key("foo bar"))

    def test_raises_on_empty(self):
        from ui.uploads import sanitize_widget_key

        with self.assertRaises(ValueError):
            sanitize_widget_key("")


if __name__ == "__main__":
    unittest.main()
