import unittest

from retrieval.router import classify_intent


class SliceTwoRouterTests(unittest.TestCase):
    def test_classifies_clause_references_as_keyword(self):
        self.assertEqual(classify_intent("Tìm Điều 8.2 trong hợp đồng"), "keyword")
        self.assertEqual(classify_intent("Show Section 12.1"), "keyword")
        self.assertEqual(classify_intent("HĐ-2024-001 có điều gì?"), "keyword")

    def test_classifies_filters_and_aggregations_as_structured(self):
        self.assertEqual(classify_intent("Hợp đồng nào sắp hết hạn trong 30 ngày?"), "structured")
        self.assertEqual(classify_intent("Tổng giá trị hợp đồng với Công ty A?"), "structured")
        self.assertEqual(classify_intent("Contracts with expiry date before 2025-01-01"), "structured")

    def test_classifies_clause_meaning_questions_as_semantic(self):
        self.assertEqual(classify_intent("Điều khoản phạt vi phạm tiến độ là gì?"), "semantic")
        self.assertEqual(classify_intent("Explain the non-solicitation obligation"), "semantic")
        self.assertEqual(
            classify_intent(
                "No-Solicit Of Employees. Is there a restriction whether during the contract or after it ends?"
            ),
            "semantic",
        )


if __name__ == "__main__":
    unittest.main()
