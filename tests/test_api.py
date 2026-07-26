from fastapi.testclient import TestClient
import pytest

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    with client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "device" in data


def test_info_endpoint():
    with client:
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "MyGPT-Micro"
        assert data["num_parameters"] > 0
        assert data["vocab_size"] > 0
        assert data["d_model"] == 128
        assert data["n_layer"] == 4
        assert data["n_head"] == 4


def test_generate_endpoint():
    with client:
        payload = {
            "prompt": "Building a custom",
            "max_new_tokens": 10,
            "temperature": 0.8,
            "top_k": 20,
            "greedy": False,
        }
        response = client.post("/api/v1/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Building a custom"
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)


def test_chat_endpoint():
    with client:
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_new_tokens": 10,
            "temperature": 0.7,
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert isinstance(data["message"]["content"], str)


def test_invalid_request_validation():
    with client:
        # Invalid max_new_tokens < 1
        payload = {
            "prompt": "Test prompt",
            "max_new_tokens": -5,
        }
        response = client.post("/api/v1/generate", json=payload)
        assert response.status_code == 422
