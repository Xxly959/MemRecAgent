"""模拟外部HTTP，断言真实适配器的请求、失败和凭证保护行为。"""

import json

import pytest

from memrecagent.deepseek_client import DeepSeekClient, ProviderError


class Response:
    def __init__(self, body, status=200):
        self.status = status
        self.body = body

    def read(self, size):
        return self.body[:size]


def response_data(**changes):
    return {
        "model": "deepseek-flash",
        "system_fingerprint": "fp-synthetic",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"synthetic":true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    } | changes


def install_connection(monkeypatch, *, data=None, body=None, status=200, error=None):
    captured = []

    class Connection:
        def __init__(self, host, timeout):
            captured.append({"host": host, "timeout": timeout})

        def request(self, method, path, body, headers):
            captured[-1].update(method=method, path=path, body=json.loads(body), headers=headers)
            if error:
                raise error

        def getresponse(self):
            return Response(
                body if body is not None else json.dumps(data or response_data()).encode(), status
            )

        def close(self):
            captured[-1]["closed"] = True

    monkeypatch.setattr("memrecagent.deepseek_client.http.client.HTTPSConnection", Connection)
    return captured


def test_fixed_endpoint_json_payload_and_final_content(monkeypatch):
    sent = install_connection(monkeypatch)
    client = DeepSeekClient("secret-test-only")
    reply = client.complete([{"role": "user", "content": "输出JSON"}])
    assert sent[0]["host"] == "api.deepseek.com"
    assert sent[0]["path"] == "/chat/completions" and sent[0]["method"] == "POST"
    assert sent[0]["body"]["thinking"] == {"type": "disabled"}
    assert sent[0]["body"]["response_format"] == {"type": "json_object"}
    assert sent[0]["body"]["max_tokens"] == 4096 and sent[0]["body"]["stream"] is False
    assert reply.content == '{"synthetic":true}' and reply.usage["total_tokens"] == 15
    assert reply.fingerprint == "fp-synthetic" and reply.latency_seconds >= 0
    assert "secret-test-only" not in repr(client) and sent[0]["closed"]


@pytest.mark.parametrize("status", [301, 302, 307, 401, 403, 404, 429, 500])
def test_provider_errors_do_not_retry_or_echo_response(monkeypatch, status):
    sent = install_connection(monkeypatch, body=b"secret-test-only private response", status=status)
    with pytest.raises(ProviderError) as error:
        DeepSeekClient("secret-test-only").complete([{"role": "user", "content": "JSON"}])
    assert len(sent) == 1 and sent[0]["closed"]
    assert error.value.status == status
    assert "secret-test-only" not in str(error.value) and "private response" not in str(error.value)


@pytest.mark.parametrize(
    "data",
    [
        response_data(choices=[]),
        response_data(choices=[{"finish_reason": "length", "message": {"content": "{}"}}]),
        response_data(choices=[{"finish_reason": "stop", "message": {"content": ""}}]),
        response_data(choices=[{"finish_reason": "tool_calls", "message": {"content": "{}"}}]),
        response_data(choices=[{"finish_reason": "stop", "message": {"content": None}}]),
        response_data(
            choices=[{"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{}]}}]
        ),
    ],
)
def test_incomplete_or_non_text_outputs_fail(monkeypatch, data):
    install_connection(monkeypatch, data=data)
    with pytest.raises(ProviderError):
        DeepSeekClient("synthetic").complete([{"role": "user", "content": "JSON"}])


@pytest.mark.parametrize("body", [b"\xff", b"{", b"x" * 65537, b'{"choices":[],"choices":[]}'])
def test_invalid_encoding_json_duplicates_or_oversize_fail(monkeypatch, body):
    install_connection(monkeypatch, body=body)
    with pytest.raises(ProviderError):
        DeepSeekClient("synthetic").complete([{"role": "user", "content": "JSON"}])


def test_timeout_is_sanitized_and_missing_usage_is_unavailable(monkeypatch):
    sent = install_connection(monkeypatch, error=TimeoutError("secret-test-only"))
    with pytest.raises(ProviderError) as error:
        DeepSeekClient("secret-test-only").complete([{"role": "user", "content": "JSON"}])
    assert len(sent) == 1 and "secret-test-only" not in str(error.value)
    install_connection(monkeypatch, data=response_data(usage=None))
    assert DeepSeekClient("synthetic").complete([{"role": "user", "content": "JSON"}]).usage is None
