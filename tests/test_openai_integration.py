import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk


class OpenAIIntegrationTests(unittest.TestCase):
    def test_answer_with_citations_uses_gemini_chat_completions(self):
        from generation.answer import answer_with_citations

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Gemini answer"))]
                )

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                self.chat = FakeChat()

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
                model="gemini-test",
            )

        self.assertEqual(answer, "Gemini answer")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(captured["model"], "gemini-test")
        self.assertEqual(captured["max_tokens"], 1200)
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertIn("Không có trong tài liệu", captured["messages"][0]["content"])
        self.assertIn("[Section 4, trang 2, contract_001]", captured["messages"][1]["content"])

    def test_extract_structured_json_uses_gemini_chat_completions(self):
        from ingestion.extractor import extract_structured_json

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"contract_id": "contract_001", "title": "Agreement"}')
                        )
                    ]
                )

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                self.chat = FakeChat()

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            result = extract_structured_json("MASTER AGREEMENT", api_key="test-key", model="gemini-test")

        self.assertEqual(result["contract_id"], "contract_001")
        self.assertEqual(result["title"], "Agreement")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(captured["model"], "gemini-test")
        self.assertEqual(captured["max_tokens"], 2000)
        self.assertIn("strict JSON", captured["messages"][0]["content"])
        self.assertEqual(captured["messages"][1]["content"], "MASTER AGREEMENT")


if __name__ == "__main__":
    unittest.main()
