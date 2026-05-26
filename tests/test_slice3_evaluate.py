import unittest

from ingestion.chunker import Chunk


class SliceThreeEvaluateTests(unittest.TestCase):
    def test_evaluate_ranked_results_scores_precision_and_citations(self):
        from eval.evaluate import evaluate_ranked_results, summarize_evaluations

        expected = {
            "query_id": "q001",
            "query": 'Highlight "Payment". Details: payment terms',
            "expected_contract_id": "contract_a",
            "expected_page": 2,
            "expected_contains": ["net", "30"],
        }
        wrong = Chunk(
            id="wrong",
            text="Termination without payment terms.",
            contract_id="contract_b",
            clause_number="Section 9",
            page_start=5,
            page_end=5,
        )
        match = Chunk(
            id="match",
            text="Invoices are due net 30 days.",
            contract_id="contract_a",
            clause_number="Section 4",
            page_start=2,
            page_end=2,
        )

        evaluations = evaluate_ranked_results([expected], {"q001": [wrong, match]})
        summary = summarize_evaluations(evaluations)

        self.assertEqual(len(evaluations), 1)
        self.assertTrue(evaluations[0].precision_at_3_hit)
        self.assertTrue(evaluations[0].citation_correct)
        self.assertEqual(evaluations[0].matched_citation, "[Section 4, trang 2, contract_a]")
        self.assertEqual(summary.total_cases, 1)
        self.assertEqual(summary.precision_at_3, 1.0)
        self.assertEqual(summary.citation_accuracy, 1.0)

    def test_summarize_evaluations_counts_misses(self):
        from eval.evaluate import QueryEvaluation, summarize_evaluations

        summary = summarize_evaluations(
            [
                QueryEvaluation(
                    query_id="q001",
                    query="query one",
                    expected_contract_id="contract_a",
                    expected_page=1,
                    expected_contains=["alpha"],
                    top_citation="[Section 1, trang 1, contract_a]",
                    matched_citation="[Section 1, trang 1, contract_a]",
                    precision_at_3_hit=True,
                    citation_correct=True,
                    answer_contains_expected=True,
                ),
                QueryEvaluation(
                    query_id="q002",
                    query="query two",
                    expected_contract_id="contract_b",
                    expected_page=4,
                    expected_contains=["beta"],
                    top_citation=None,
                    matched_citation=None,
                    precision_at_3_hit=False,
                    citation_correct=False,
                    answer_contains_expected=False,
                ),
            ]
        )

        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.precision_at_3, 0.5)
        self.assertEqual(summary.citation_accuracy, 0.5)
        self.assertEqual(summary.answer_contains_accuracy, 0.5)

    def test_evaluate_cases_scopes_retrieval_to_expected_contract(self):
        from eval.evaluate import evaluate_cases

        expected = {
            "query_id": "q001",
            "query": 'Highlight "Document Name". Details: contract name',
            "expected_contract_id": "contract_a",
            "expected_page": 1,
            "expected_contains": ["Master"],
        }
        wrong_contract = Chunk(
            id="wrong",
            text="Document name Master agreement in another contract.",
            contract_id="contract_b",
            clause_number="Document",
            page_start=1,
            page_end=1,
        )
        right_contract = Chunk(
            id="right",
            text="Document name Master services agreement.",
            contract_id="contract_a",
            clause_number="Document",
            page_start=1,
            page_end=1,
        )

        evaluations = evaluate_cases([expected], [wrong_contract, right_contract])

        self.assertEqual(evaluations[0].top_citation, "[Document, trang 1, contract_a]")
        self.assertTrue(evaluations[0].precision_at_3_hit)

    def test_focused_query_expands_cuad_clause_categories(self):
        from eval.evaluate import focused_query

        document_name = focused_query(
            {
                "query": 'Highlight the parts related to "Document Name". Details: The name of the contract',
            }
        )
        revenue_share = focused_query(
            {
                "query": 'Highlight the parts related to "Revenue/Profit Sharing". Details: Is one party required to share revenue?',
            }
        )
        parties = focused_query(
            {
                "query": 'Highlight the parts related to "Parties". Details: The two or more parties who signed the contract',
            }
        )

        self.assertIn("exhibit", document_name.lower())
        self.assertIn("schedule", document_name.lower())
        self.assertIn("monthly revenue", revenue_share.lower())
        self.assertIn("below threshold", revenue_share.lower())
        self.assertIn("by and between", parties.lower())

    def test_threshold_enforcement_fails_when_metrics_drop_below_target(self):
        from eval.evaluate import EvaluationSummary, enforce_thresholds

        with self.assertRaises(SystemExit) as context:
            enforce_thresholds(
                EvaluationSummary(
                    total_cases=10,
                    precision_at_3=0.89,
                    citation_accuracy=0.95,
                    answer_contains_accuracy=0.95,
                ),
                min_precision_at_3=0.90,
                min_citation_accuracy=0.90,
            )

        self.assertEqual(context.exception.code, 1)

    def test_threshold_enforcement_allows_metrics_at_target(self):
        from eval.evaluate import EvaluationSummary, enforce_thresholds

        enforce_thresholds(
            EvaluationSummary(
                total_cases=10,
                precision_at_3=0.90,
                citation_accuracy=0.90,
                answer_contains_accuracy=0.90,
            ),
            min_precision_at_3=0.90,
            min_citation_accuracy=0.90,
        )


if __name__ == "__main__":
    unittest.main()
