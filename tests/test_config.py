from __future__ import annotations

import unittest

from backend.config import Settings


class SettingsTestCase(unittest.TestCase):
    def test_bool_and_mode_parsing(self) -> None:
        settings = Settings(
            use_mock_openai="true",
            enable_llm_reranking="yes",
            llm_rerank_modes="multimodal, text ,multimodal",
            allowed_origins="http://localhost:3000, http://localhost:8000",
        )

        self.assertTrue(settings.use_mock_openai)
        self.assertTrue(settings.enable_llm_reranking)
        self.assertEqual(settings.normalized_llm_rerank_modes, {"multimodal", "text"})
        self.assertEqual(
            settings.normalized_allowed_origins,
            ["http://localhost:3000", "http://localhost:8000"],
        )


if __name__ == "__main__":
    unittest.main()
