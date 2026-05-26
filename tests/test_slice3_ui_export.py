import csv
import io
import unittest


class SliceThreeUiExportTests(unittest.TestCase):
    def test_rows_to_csv_includes_headers_and_escapes_values(self):
        from ui.export import rows_to_csv

        csv_text = rows_to_csv(
            [
                {"contract": "contract_001", "summary": "Payment, net 30"},
                {"contract": "contract_002", "summary": "Line one\nLine two"},
            ],
            columns=["contract", "summary"],
        )

        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(rows[0]["contract"], "contract_001")
        self.assertEqual(rows[0]["summary"], "Payment, net 30")
        self.assertEqual(rows[1]["summary"], "Line one\nLine two")

    def test_rows_to_csv_returns_empty_string_without_rows(self):
        from ui.export import rows_to_csv

        self.assertEqual(rows_to_csv([], columns=["contract"]), "")
