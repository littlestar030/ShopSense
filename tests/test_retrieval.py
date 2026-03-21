from __future__ import annotations

import unittest

from backend import retrieval


class RetrievalStrategyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.original_enable = retrieval.settings.enable_llm_reranking
        self.original_modes = retrieval.settings.llm_rerank_modes
        self.original_threshold = retrieval.settings.llm_rerank_margin_threshold

    def tearDown(self) -> None:
        retrieval.settings.enable_llm_reranking = self.original_enable
        retrieval.settings.llm_rerank_modes = self.original_modes
        retrieval.settings.llm_rerank_margin_threshold = self.original_threshold

    @staticmethod
    def _candidate(score: float) -> dict:
        return {"id": f"p-{score}", "match_debug": {"hybrid_score": score}}

    def test_llm_reranking_applies_to_multimodal_mode(self) -> None:
        retrieval.settings.enable_llm_reranking = True
        retrieval.settings.llm_rerank_modes = "multimodal"
        retrieval.settings.llm_rerank_margin_threshold = 0.01

        candidates = [self._candidate(0.9), self._candidate(0.2)]

        self.assertTrue(
            retrieval._should_apply_llm_reranking(
                query="find something like this but better for running",
                mode="multimodal",
                candidates=candidates,
            )
        )

    def test_llm_reranking_skips_non_ambiguous_text_queries(self) -> None:
        retrieval.settings.enable_llm_reranking = True
        retrieval.settings.llm_rerank_modes = "multimodal,text"
        retrieval.settings.llm_rerank_margin_threshold = 0.05

        candidates = [self._candidate(0.9), self._candidate(0.6)]

        self.assertFalse(
            retrieval._should_apply_llm_reranking(
                query="black jacket for everyday layering",
                mode="text",
                candidates=candidates,
            )
        )

    def test_llm_reranking_applies_to_ambiguous_text_queries_when_enabled(self) -> None:
        retrieval.settings.enable_llm_reranking = True
        retrieval.settings.llm_rerank_modes = "text"
        retrieval.settings.llm_rerank_margin_threshold = 0.05

        candidates = [self._candidate(0.81), self._candidate(0.78)]

        self.assertTrue(
            retrieval._should_apply_llm_reranking(
                query="white t-shirt for weekend wear",
                mode="text",
                candidates=candidates,
            )
        )


if __name__ == "__main__":
    unittest.main()
