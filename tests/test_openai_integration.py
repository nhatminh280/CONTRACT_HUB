import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk


class OpenAIIntegrationTests(unittest.TestCase):
    def test_answer_with_citations_uses_openai_responses_api(self):
        from generation.answer import answer_with_citations

        captured = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(output_text="OpenAI answer")

        class FakeOpenAI:
            def __init__(self, api_key=None):
                captured["api_key"] = api_key
                self.responses = FakeResponses()

        chunk = Chunk(
            id="c1",
            text="Payment is due net 30 days.",
            contract_id="contract_001",
            clause_number="Section 4",
            page_start=2,
            page_end=2,
        )

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            answer = answer_with_citations(
                "When is payment due?",
                [ScoredChunk(chunk=chunk, score=1.0)],
                api_key="test-key",
                model="gpt-test",
            )

        self.assertEqual(answer, "OpenAI answer")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["model"], "gpt-test")
        self.assertIn("Không có trong tài liệu", captured["instructions"])
        self.assertIn("[Section 4, trang 2, contract_001]", captured["input"])
        self.assertEqual(captured["max_output_tokens"], 1200)

    def test_extract_structured_json_uses_openai_responses_api(self):
        from ingestion.extractor import extract_structured_json

        captured = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(output_text='{"contract_id": "contract_001", "title": "Agreement"}')

        class FakeOpenAI:
            def __init__(self, api_key=None):
                captured["api_key"] = api_key
                self.responses = FakeResponses()

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            result = extract_structured_json("MASTER AGREEMENT", api_key="test-key", model="gpt-test")

        self.assertEqual(result["contract_id"], "contract_001")
        self.assertEqual(result["title"], "Agreement")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["model"], "gpt-test")
        self.assertIn("strict JSON", captured["instructions"])
        self.assertEqual(captured["input"], "MASTER AGREEMENT")
        self.assertEqual(captured["max_output_tokens"], 2000)


if __name__ == "__main__":
    unittest.main()
