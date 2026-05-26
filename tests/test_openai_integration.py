import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk


class OpenAIIntegrationTests(unittest.TestCase):
    def test_contract_record_from_llm_json_maps_gemini_payload(self):
        from ingestion.extractor import contract_record_from_llm_json

        chunk = Chunk(
            id="c1",
            text="Payment terms summary.",
            contract_id="contract_001",
            clause_number="Section 4",
            page_start=2,
            page_end=2,
        )
        payload = {
            "contract_id": "ignored_by_caller",
            "title": "Master Agreement",
            "parties": ["Kubient Inc.", {"name": "Associated Press", "role": "customer"}],
            "effective_date": "2020-02-05",
            "expiry_date": "2021-02-05",
            "contract_value": "1250000",
            "currency": "USD",
            "governing_law": "State of Delaware",
            "clauses": [
                {
                    "clause_number": "Section 4",
                    "clause_type": "payment",
                    "page": 2,
                    "summary": "Payment terms.",
                }
            ],
        }

        record = contract_record_from_llm_json("contract_001", payload, [chunk])

        self.assertEqual(record.contract_id, "contract_001")
        self.assertEqual(record.title, "Master Agreement")
        self.assertEqual(record.value, 1250000.0)
        self.assertEqual(record.currency, "USD")
        self.assertEqual(record.effective_date, "2020-02-05")
        self.assertEqual(record.expiry_date, "2021-02-05")
        self.assertEqual(record.governing_law, "State of Delaware")
        self.assertEqual(
            record.parties,
            [
                {"name": "Kubient Inc.", "role": "party"},
                {"name": "Associated Press", "role": "customer"},
            ],
        )
        self.assertEqual(record.clauses[0]["number"], "Section 4")
        self.assertEqual(record.clauses[0]["type"], "payment")

    def test_contract_record_from_llm_json_falls_back_to_chunk_clauses(self):
        from ingestion.extractor import contract_record_from_llm_json

        chunk = Chunk(
            id="c1",
            text="Long clause text for fallback summary.",
            contract_id="contract_001",
            clause_number="Section 7",
            page_start=4,
            page_end=4,
            clause_type="termination",
        )

        record = contract_record_from_llm_json("contract_001", {"title": ""}, [chunk])

        self.assertEqual(record.title, "contract_001")
        self.assertEqual(record.clauses[0]["number"], "Section 7")
        self.assertEqual(record.clauses[0]["type"], "termination")
        self.assertEqual(record.clauses[0]["page"], 4)

    def test_format_chunks_for_llm_extraction_caps_long_context(self):
        from ingestion.extractor import format_chunks_for_llm_extraction

        chunk = Chunk(
            id="c1",
            text="A" * 5000,
            contract_id="contract_001",
            clause_number="Section 1",
            page_start=1,
            page_end=2,
        )

        text = format_chunks_for_llm_extraction([chunk], max_chars=600)

        self.assertLessEqual(len(text), 600)
        self.assertIn("contract_001", text)
        self.assertIn("Section 1", text)
        self.assertIn("Text:", text)

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

        with (
            patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
            patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}, clear=False),
        ):
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
        self.assertIn("using ONLY the retrieved contract context", captured["messages"][0]["content"])
        self.assertIn("Not found in the provided context.", captured["messages"][0]["content"])
        self.assertIn("Answer:", captured["messages"][0]["content"])
        self.assertIn("Confidence:", captured["messages"][0]["content"])
        self.assertIn("[Section 4, trang 2, contract_001]", captured["messages"][1]["content"])
        self.assertIn("User question:", captured["messages"][1]["content"])
        self.assertIn("Detected intent:", captured["messages"][1]["content"])

    def test_answer_with_citations_uses_openai_when_provider_is_openai(self):
        from generation.answer import answer_with_citations

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI answer"))]
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

        with (
            patch.dict(
                sys.modules,
                {"openai": SimpleNamespace(OpenAI=FakeOpenAI)},
            ),
            patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "openai",
                    "OPENAI_API_KEY": "openai-key",
                    "OPENAI_MODEL": "gpt-test",
                },
                clear=False,
            ),
        ):
            answer = answer_with_citations(
                "When is payment due?",
                [ScoredChunk(chunk=chunk, score=1.0)],
            )

        self.assertEqual(answer, "OpenAI answer")
        self.assertEqual(captured["api_key"], "openai-key")
        self.assertIsNone(captured["base_url"])
        self.assertEqual(captured["model"], "gpt-test")

    def test_answer_with_citations_uses_anthropic_messages_when_provider_is_anthropic(self):
        from generation.answer import answer_with_citations

        captured = {}

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(content=[SimpleNamespace(text="Claude answer")])

        class FakeAnthropic:
            def __init__(self, api_key=None):
                captured["api_key"] = api_key
                self.messages = FakeMessages()

        chunk = Chunk(
            id="c1",
            text="Payment is due net 30 days.",
            contract_id="contract_001",
            clause_number="Section 4",
            page_start=2,
            page_end=2,
        )

        with (
            patch.dict(sys.modules, {"anthropic": SimpleNamespace(Anthropic=FakeAnthropic)}),
            patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "anthropic",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                    "ANTHROPIC_MODEL": "claude-test",
                },
                clear=False,
            ),
        ):
            answer = answer_with_citations(
                "When is payment due?",
                [ScoredChunk(chunk=chunk, score=1.0)],
            )

        self.assertEqual(answer, "Claude answer")
        self.assertEqual(captured["api_key"], "anthropic-key")
        self.assertEqual(captured["model"], "claude-test")
        self.assertEqual(captured["max_tokens"], 1200)
        self.assertIn("using ONLY the retrieved contract context", captured["system"])
        self.assertEqual(captured["messages"][0]["role"], "user")
        self.assertIn("When is payment due?", captured["messages"][0]["content"])
        self.assertIn("[Section 4, trang 2, contract_001]", captured["messages"][0]["content"])

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

        with (
            patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
            patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}, clear=False),
        ):
            result = extract_structured_json("MASTER AGREEMENT", api_key="test-key", model="gemini-test")

        self.assertEqual(result["contract_id"], "contract_001")
        self.assertEqual(result["title"], "Agreement")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(captured["model"], "gemini-test")
        self.assertEqual(captured["max_tokens"], 4000)
        self.assertEqual(captured["temperature"], 0)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertIn("strict JSON", captured["messages"][0]["content"])
        self.assertEqual(captured["messages"][1]["content"], "MASTER AGREEMENT")

    def test_extract_structured_json_uses_openai_when_provider_is_openai(self):
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

        with (
            patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
            patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "openai",
                    "OPENAI_API_KEY": "openai-key",
                    "OPENAI_MODEL": "gpt-test",
                },
                clear=False,
            ),
        ):
            result = extract_structured_json("MASTER AGREEMENT")

        self.assertEqual(result["contract_id"], "contract_001")
        self.assertEqual(captured["api_key"], "openai-key")
        self.assertIsNone(captured["base_url"])
        self.assertEqual(captured["model"], "gpt-test")

    def test_extract_structured_json_uses_anthropic_messages_when_provider_is_anthropic(self):
        from ingestion.extractor import extract_structured_json

        captured = {}

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(text='{"contract_id": "contract_001", "title": "Agreement"}')]
                )

        class FakeAnthropic:
            def __init__(self, api_key=None):
                captured["api_key"] = api_key
                self.messages = FakeMessages()

        with (
            patch.dict(sys.modules, {"anthropic": SimpleNamespace(Anthropic=FakeAnthropic)}),
            patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "anthropic",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                    "ANTHROPIC_MODEL": "claude-test",
                },
                clear=False,
            ),
        ):
            result = extract_structured_json("MASTER AGREEMENT")

        self.assertEqual(result["contract_id"], "contract_001")
        self.assertEqual(captured["api_key"], "anthropic-key")
        self.assertEqual(captured["model"], "claude-test")
        self.assertEqual(captured["max_tokens"], 4000)
        self.assertIn("strict JSON", captured["system"])
        self.assertEqual(captured["messages"][0]["role"], "user")
        self.assertEqual(captured["messages"][0]["content"], "MASTER AGREEMENT")


if __name__ == "__main__":
    unittest.main()
