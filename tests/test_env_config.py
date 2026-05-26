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


if __name__ == "__main__":
    unittest.main()
