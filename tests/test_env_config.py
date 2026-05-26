import os
from pathlib import Path
import tempfile
import unittest


class EnvConfigTests(unittest.TestCase):
    def test_load_env_file_populates_missing_values_without_overwriting(self):
        from config.env import load_env_file

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=file-value",
                        "QUOTED_VALUE=\"quoted file value\"",
                        "COMMENTED=value # comment",
                    ]
                ),
                encoding="utf-8",
            )
            old_gemini = os.environ.get("GEMINI_API_KEY")
            old_quoted = os.environ.get("QUOTED_VALUE")
            old_commented = os.environ.get("COMMENTED")
            try:
                os.environ["GEMINI_API_KEY"] = "existing-value"
                os.environ.pop("QUOTED_VALUE", None)
                os.environ.pop("COMMENTED", None)

                loaded = load_env_file(env_path)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "existing-value")
                self.assertEqual(os.environ["QUOTED_VALUE"], "quoted file value")
                self.assertEqual(os.environ["COMMENTED"], "value")
            finally:
                if old_gemini is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = old_gemini
                if old_quoted is None:
                    os.environ.pop("QUOTED_VALUE", None)
                else:
                    os.environ["QUOTED_VALUE"] = old_quoted
                if old_commented is None:
                    os.environ.pop("COMMENTED", None)
                else:
                    os.environ["COMMENTED"] = old_commented

    def test_load_env_file_returns_false_for_missing_file(self):
        from config.env import load_env_file

        self.assertFalse(load_env_file(Path("/tmp/definitely-missing-contract-hub.env")))

    def test_llm_config_prefers_openai_when_openai_key_is_available(self):
        from config.llm import llm_api_key, llm_base_url, llm_model, llm_ocr_model, llm_provider

        old_values = {
            key: os.environ.get(key)
            for key in [
                "LLM_PROVIDER",
                "OPENAI_API_KEY",
                "OPENAI_MODEL",
                "OPENAI_OCR_MODEL",
                "OPENAI_BASE_URL",
                "GEMINI_API_KEY",
            ]
        }
        try:
            os.environ["LLM_PROVIDER"] = ""
            os.environ["OPENAI_API_KEY"] = "openai-key"
            os.environ["OPENAI_MODEL"] = "gpt-test"
            os.environ["OPENAI_OCR_MODEL"] = "gpt-ocr-test"
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ["GEMINI_API_KEY"] = "gemini-key"

            self.assertEqual(llm_provider(), "openai")
            self.assertEqual(llm_api_key(), "openai-key")
            self.assertIsNone(llm_base_url())
            self.assertEqual(llm_model(), "gpt-test")
            self.assertEqual(llm_ocr_model(), "gpt-ocr-test")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_llm_config_can_force_gemini_provider(self):
        from config.llm import llm_api_key, llm_base_url, llm_model, llm_provider

        old_values = {
            key: os.environ.get(key)
            for key in [
                "LLM_PROVIDER",
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "GEMINI_MODEL",
                "GEMINI_BASE_URL",
            ]
        }
        try:
            os.environ["LLM_PROVIDER"] = "gemini"
            os.environ["OPENAI_API_KEY"] = "openai-key"
            os.environ["GEMINI_API_KEY"] = "gemini-key"
            os.environ["GEMINI_MODEL"] = "gemini-test"
            os.environ["GEMINI_BASE_URL"] = "https://gemini.example/v1/"

            self.assertEqual(llm_provider(), "gemini")
            self.assertEqual(llm_api_key(), "gemini-key")
            self.assertEqual(llm_base_url(), "https://gemini.example/v1/")
            self.assertEqual(llm_model(), "gemini-test")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_llm_config_can_force_anthropic_provider(self):
        from config.llm import llm_api_key, llm_base_url, llm_model, llm_ocr_model, llm_provider

        old_values = {
            key: os.environ.get(key)
            for key in [
                "LLM_PROVIDER",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_MODEL",
                "ANTHROPIC_OCR_MODEL",
                "OPENAI_API_KEY",
            ]
        }
        try:
            os.environ["LLM_PROVIDER"] = "anthropic"
            os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"
            os.environ["ANTHROPIC_MODEL"] = "claude-test"
            os.environ["ANTHROPIC_OCR_MODEL"] = "claude-ocr-test"
            os.environ["OPENAI_API_KEY"] = "openai-key"

            self.assertEqual(llm_provider(), "anthropic")
            self.assertEqual(llm_api_key(), "anthropic-key")
            self.assertIsNone(llm_base_url())
            self.assertEqual(llm_model(), "claude-test")
            self.assertEqual(llm_ocr_model(), "claude-ocr-test")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
