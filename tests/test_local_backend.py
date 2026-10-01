"""Offline contract tests for the loopback decision backend.

scripts/local_backend.py is exercised end-to-end over a real loopback socket
with the Ollama call mocked out — no paid APIs, no external network, and the
request-hardening gates (Host/Origin/path/token) are verified as HTTP
behavior, not just unit internals.
"""

import http.client
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import pytest

BACKEND_PATH = Path(__file__).resolve().parent.parent / "scripts" / "local_backend.py"
spec = importlib.util.spec_from_file_location("jev_local_backend", BACKEND_PATH)
local_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_backend)


def _body():
    """A minimal but well-formed SystemOne finite-choice request."""
    return {
        "model": "test",
        "state": {
            "page": {"url": "https://example.test", "title": "Search", "text": "Search"},
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "criteria": {"CLICK": "click a control", "TYPE_TEXT": "type text into a field"},
                "instructions": {"goal": "find a book", "rules": "r"},
            },
            "click_target": {
                "criteria": {"1": {"element": "Open Search"}, "2": {"element": "Go"}},
                "instructions": {"goal": "find a book"},
            },
            "type_text_target": {
                "criteria": {"1": {"element": "Search box"}},
                "instructions": {"goal": "find a book"},
            },
        },
    }


@pytest.fixture
def port(monkeypatch):
    """Serve the real handler on an ephemeral loopback port with ask() mocked."""
    monkeypatch.setattr(
        local_backend,
        "ask",
        Mock(
            side_effect=[
                ("CLICK", 0.9, {"prompt_tokens": 5, "completion_tokens": 3}),
                ("2", 0.8, {"prompt_tokens": 4, "completion_tokens": 2}),
            ]
        ),
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), local_backend.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body) if body is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    connection.request(method, path, body=payload, headers=request_headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = raw
    return response.status, data


def _post(port, body=None, headers=None, path="/v1/systemone"):
    return _request(port, "POST", path, body=body if body is not None else _body(), headers=headers)


def test_decision_round_trip_answers_both_heads(port):
    status, data = _post(port)
    assert status == 200
    operation = data["answers"]["operation"]
    assert operation["choice"] == "CLICK"
    assert operation["probabilities"]["CLICK"] == max(operation["probabilities"].values())
    assert abs(sum(operation["probabilities"].values()) - 1.0) < 1e-6
    target = data["answers"]["click_target"]
    assert target["choice"] == "2"
    assert data["usage"]["prompt_tokens"] == 9  # both ask() calls accounted


def test_non_loopback_host_is_rejected(port):
    status, data = _post(port, headers={"Host": "evil.example"})
    assert status == 403
    assert "error" in data


def test_cross_site_origin_is_rejected(port):
    status, _ = _post(port, headers={"Origin": "https://evil.example"})
    assert status == 403


def test_wrong_path_is_rejected(port):
    status, _ = _post(port, path="/v1/other")
    assert status == 404


def test_get_is_rejected(port):
    status, _ = _request(port, "GET", "/v1/systemone")
    assert status == 404


def test_token_required_when_configured(port, monkeypatch):
    monkeypatch.setenv("JEV_LOCAL_TOKEN", "shared-secret")
    assert _post(port)[0] == 403
    assert _post(port, headers={"Authorization": "Bearer wrong"})[0] == 403
    assert _post(port, headers={"Authorization": "Bearer shared-secret"})[0] == 200


def test_malformed_json_is_a_clean_error(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST", "/v1/systemone", body="{not json", headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    data = json.loads(response.read())
    connection.close()
    assert response.status == 502
    assert "error" in data


def test_missing_operation_question_is_a_clean_error(port):
    status, data = _post(port, body={"state": {}, "questions": {}})
    assert status == 502
    assert "operation" in data["error"].lower()


def test_oversized_body_is_rejected(port):
    """A declared Content-Length over the 4 MiB cap is refused at the header
    gate — the service never reads that body into memory."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST",
        "/v1/systemone",
        body="{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(4 * 1024 * 1024 + 1),
        },
    )
    response = connection.getresponse()
    data = json.loads(response.read())
    connection.close()
    assert response.status == 502
    assert "size" in data["error"].lower()


def test_answer_probabilities_peak_on_choice():
    payload = local_backend.answer("b", 0.9, {"a": "x", "b": "y", "c": "z"})
    assert payload["choice"] == "b"
    assert payload["probabilities"]["b"] == max(payload["probabilities"].values())
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-5


def test_ask_never_fabricates_a_choice(monkeypatch):
    """A model that keeps naming keys outside the criteria exhausts retries —
    the shim raises rather than returning an invented answer."""
    monkeypatch.setattr(
        local_backend,
        "ollama_chat",
        Mock(return_value=({"choice": "invented", "confidence": 0.9},
                           {"prompt_tokens": 1, "completion_tokens": 1})),
    )
    with pytest.raises(RuntimeError, match="could not answer"):
        local_backend.ask({}, {"criteria": {"a": "x"}}, context="test")
