import unittest

from ingestion.chunker import Chunk
from scripts.slice2_multi_contract_smoke_test import evaluate_query_results, retrieval_query


class SliceTwoMultiContractTests(unittest.TestCase):
    def test_evaluate_query_results_requires_contract_page_and_terms(self):
        chunks = [
            Chunk(
                id="contract_004_doc",
                text="Master Services Agreement entered February, 2020.",
                contract_id="contract_004",
                clause_number="Document",
                page_start=1,
                page_end=1,
            ),
            Chunk(
                id="contract_005_doc",
                text="This Exhibit B is entered into as of the 26th day of March 2020.",
                contract_id="contract_005",
                clause_number="Document",
                page_start=1,
                page_end=1,
            ),
        ]
        cases = [
            {
                "query_id": "q011",
                "query": "Highlight the parts related to \"Agreement Date\".",
                "expected_contract_id": "contract_004",
                "expected_page": 1,
                "expected_contains": ["February,", "2020"],
            },
            {
                "query_id": "q003",
                "query": "Highlight the parts related to \"Agreement Date\".",
                "expected_contract_id": "contract_005",
                "expected_page": 1,
                "expected_contains": ["26th", "March", "2020"],
            },
        ]
        top_hits = {
            "q011": chunks[0],
            "q003": chunks[1],
        }

        results = evaluate_query_results(cases, top_hits)

        self.assertEqual([result["query_id"] for result in results], ["q011", "q003"])
        self.assertTrue(all(result["passed"] for result in results))
        self.assertEqual(results[0]["top_citation"], "[Document, trang 1, contract_004]")
        self.assertEqual(results[1]["top_citation"], "[Document, trang 1, contract_005]")

    def test_evaluate_query_results_accepts_expected_match_in_top_three(self):
        wrong = Chunk(
            id="contract_005_wrong",
            text="Termination and renewal terms from a different contract.",
            contract_id="contract_005",
            clause_number="NON-SOLICITATION",
            page_start=1,
            page_end=1,
        )
        expected = Chunk(
            id="contract_004_expected",
            text="Either Party may terminate this Agreement for any reason following the Initial Term.",
            contract_id="contract_004",
            clause_number="Document",
            page_start=1,
            page_end=2,
        )
        cases = [
            {
                "query_id": "q016",
                "query": "Highlight the parts related to \"Termination For Convenience\".",
                "expected_contract_id": "contract_004",
                "expected_page": 1,
                "expected_contains": ["Either", "Party", "terminate"],
            }
        ]

        results = evaluate_query_results(cases, {"q016": [wrong, expected]})

        self.assertTrue(results[0]["passed"])
        self.assertEqual(results[0]["top_citation"], "[NON-SOLICITATION, trang 1, contract_005]")
        self.assertEqual(results[0]["matched_citation"], "[Document, trang 1-2, contract_004]")

    def test_retrieval_query_keeps_category_and_details(self):
        case = {
            "query": (
                "Highlight the parts related to \"Expiration Date\". "
                "Details: On what date will the contract's initial term expire?"
            )
        }

        query = retrieval_query(case)

        self.assertIn("Expiration Date", query)
        self.assertIn("initial term expire", query)


if __name__ == "__main__":
    unittest.main()
