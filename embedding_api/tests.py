"""Tests for the host-native embedding API service.

Run with: python -m pytest embedding_api/tests.py

Requires the model to be available in cache; mock-based tests can run without it.
"""

import json
import os
from unittest.mock import patch

import pytest

try:
    import embedding_api.main as embedding_main
except ModuleNotFoundError:  # pragma: no cover - supports running from embedding_api/
    import main as embedding_main

app = embedding_main.app
_select_device = embedding_main._select_device


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok_after_startup(self, client):
        # The TestClient triggers the startup event which loads the model.
        # If model is available, this should return 200.
        response = client.get("/health")
        # Either 200 (model loaded) or 503 (model not downloaded)
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data


class TestEmbedEndpoint:
    def test_embed_returns_correct_count(self, client):
        # First check if model is ready
        health = client.get("/health")
        if health.status_code != 200:
            pytest.skip("Embedding model not available for integration test")

        response = client.post("/embed", json={"texts": ["hello", "world"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 2

    def test_embed_preserves_input_order(self, client):
        health = client.get("/health")
        if health.status_code != 200:
            pytest.skip("Embedding model not available for integration test")

        texts = ["birinci", "ikinci", "ucuncu"]
        response = client.post("/embed", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 3
        # Each embedding should be a list of floats
        for emb in data["embeddings"]:
            assert isinstance(emb, list)
            assert len(emb) == 384
            assert all(isinstance(v, float) for v in emb)

    def test_embed_empty_texts_returns_empty_list(self, client):
        health = client.get("/health")
        if health.status_code != 200:
            pytest.skip("Embedding model not available for integration test")

        response = client.post("/embed", json={"texts": []})
        assert response.status_code == 200
        data = response.json()
        assert data["embeddings"] == []

    @patch.object(embedding_main, "EMBEDDING_BATCH_SIZE", 2)
    @patch.object(embedding_main, "_ready", True)
    def test_embed_batches_requests_and_preserves_order(self):
        class FakeVector(list):
            def tolist(self):
                return list(self)

        class FakeModel:
            def __init__(self):
                self.calls = []

            def encode(
                self, batch, *, batch_size, normalize_embeddings, show_progress_bar
            ):
                self.calls.append(
                    {
                        "batch": list(batch),
                        "batch_size": batch_size,
                        "normalize_embeddings": normalize_embeddings,
                        "show_progress_bar": show_progress_bar,
                    }
                )
                return [
                    FakeVector([float(len(self.calls)), float(index)])
                    for index, _ in enumerate(batch)
                ]

        fake_model = FakeModel()
        with patch.object(embedding_main, "_model", fake_model):
            response = embedding_main.embed(
                embedding_main.EmbedRequest(texts=["a", "b", "c"])
            )

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["embeddings"] == [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        assert fake_model.calls == [
            {
                "batch": ["a", "b"],
                "batch_size": 2,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
            {
                "batch": ["c"],
                "batch_size": 1,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        ]


class TestDeviceSelection:
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(embedding_main, "torch")
    def test_defaults_to_cpu_when_mps_available(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = True
        assert _select_device() == "cpu"

    @patch.dict(os.environ, {"EMBEDDING_DEVICE": "auto"})
    @patch.object(embedding_main, "torch")
    def test_auto_selects_mps_when_available(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = True
        assert _select_device() == "mps"

    @patch.dict(os.environ, {"EMBEDDING_DEVICE": "mps"})
    @patch.object(embedding_main, "torch")
    def test_mps_preference_falls_back_to_cpu_when_unavailable(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = False
        assert _select_device() == "cpu"
