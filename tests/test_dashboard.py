"""Integration tests for the local standard-library dashboard."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import app as dashboard  # noqa: E402  (path set up above)


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = dashboard.create_server("127.0.0.1", 0)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path: str) -> tuple[int, str, bytes]:
        with urlopen(self.base_url + path, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()

    def test_index_page(self) -> None:
        status, content_type, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        text = body.decode("utf-8")
        self.assertIn("Smart Gas AI", text)
        self.assertIn("/api/predict", text)

    def test_info_endpoint(self) -> None:
        status, content_type, body = self._get("/api/info")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        payload = json.loads(body)
        self.assertEqual(payload["model"]["candidate"], "gaussian_nb_adc")
        self.assertEqual(payload["model"]["features"], ["mq5_adc"])
        self.assertEqual(payload["classes"], ["Normal", "Warning", "Danger"])
        self.assertEqual(payload["adc_range"], [0, 4095])
        self.assertAlmostEqual(payload["metrics"]["test_accuracy"], 0.9394, places=4)

    def test_predict_endpoint(self) -> None:
        status, _, body = self._get("/api/predict?adc=1500")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertIn(payload["label"], ["Normal", "Warning", "Danger"])
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0, places=6)
        self.assertFalse(payload["context_used_by_model"])

    def test_invalid_adc_returns_400(self) -> None:
        with self.assertRaises(HTTPError) as context:
            self._get("/api/predict?adc=5000")
        error = context.exception
        try:
            self.assertEqual(error.code, 400)
            payload = json.loads(error.read())
            self.assertIn("error", payload)
        finally:
            error.close()

    def test_missing_adc_returns_400(self) -> None:
        with self.assertRaises(HTTPError) as context:
            self._get("/api/predict")
        error = context.exception
        try:
            self.assertEqual(error.code, 400)
        finally:
            error.close()

    def test_unknown_route_returns_404(self) -> None:
        with self.assertRaises(HTTPError) as context:
            self._get("/nope")
        error = context.exception
        try:
            self.assertEqual(error.code, 404)
        finally:
            error.close()

    def test_post_is_rejected(self) -> None:
        request = Request(
            self.base_url + "/api/predict?adc=1500",
            data=b"",
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=10)
        error = context.exception
        try:
            self.assertEqual(error.code, 405)
        finally:
            error.close()


if __name__ == "__main__":
    unittest.main()
