from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main


class ApiSmokeTestCase(unittest.TestCase):
    def test_healthz(self) -> None:
        with patch.object(main.settings, "enable_background_warmup", False):
            with TestClient(main.app) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_readyz(self) -> None:
        with patch.object(main.settings, "enable_background_warmup", False):
            with patch("backend.main.ensure_indexes_loaded") as ensure_indexes_loaded:
                with TestClient(main.app) as client:
                    response = client.get("/readyz")

        self.assertEqual(response.status_code, 200)
        ensure_indexes_loaded.assert_called_once_with(force_rebuild=False)

    def test_chat_returns_mocked_agent_response(self) -> None:
        fake_response = {"type": "chat", "response": "hello", "products": []}
        with patch.object(main.settings, "enable_background_warmup", False):
            with patch("backend.main.get_agent_response", return_value=fake_response) as get_agent_response:
                with TestClient(main.app) as client:
                    response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fake_response)
        get_agent_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
