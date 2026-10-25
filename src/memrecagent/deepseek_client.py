"""固定DeepSeek官方端点的最小适配器；无重试、不记录凭证或推理过程。"""

import http.client
import json
import math
import time
from dataclasses import dataclass, field

from memrecagent.memory_extraction import ExtractionError, unique_object


class ProviderError(RuntimeError):
    """供应商错误只暴露类型和状态码，不含响应原文。"""

    def __init__(self, code: str, status: int | None = None):
        self.code = code
        self.status = status
        super().__init__(f"模型调用失败：{code}" + (f" HTTP {status}" if status else ""))


@dataclass(frozen=True)
class ModelReply:
    content: str = field(repr=False)
    model: str
    fingerprint: str | None
    usage: dict[str, int] | None
    latency_seconds: float


class DeepSeekClient:
    """OpenAI兼容Chat Completions协议；生产依赖不借用dev HTTP库。"""

    def __init__(self, api_key: str, *, timeout: float = 60.0):
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or any(c.isspace() for c in api_key)
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ProviderError("configuration")
        self._api_key = api_key
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        """单次请求，截断、工具调用与无内容响应都不能充当提取成功。"""
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or m["role"] not in {"system", "user", "assistant"}
                or not isinstance(m["content"], str)
                for m in messages
            )
        ):
            raise ProviderError("messages")
        payload = {
            "model": "deepseek-flash",
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "stream": False,
        }
        connection = None
        started = time.perf_counter()
        try:
            connection = http.client.HTTPSConnection("api.deepseek.com", timeout=self.timeout)
            connection.request(
                "POST",
                "/chat/completions",
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self._api_key,
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ProviderError("http", response.status)
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ProviderError("response_size")
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
            if not isinstance(data, dict):
                raise ProviderError("response_protocol")
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ProviderError("response_protocol")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ProviderError("response_incomplete")
            message = choice.get("message")
            if (
                not isinstance(message, dict)
                or message.get("tool_calls")
                or message.get("role", "assistant") != "assistant"
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ProviderError("response_content")
            model, fingerprint = data.get("model"), data.get("system_fingerprint")
            if (
                not isinstance(model, str)
                or not model.strip()
                or (fingerprint is not None and not isinstance(fingerprint, str))
            ):
                raise ProviderError("response_protocol")
            usage = data.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise ProviderError("response_usage")
                counts = {}
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens",
                ):
                    if key in usage:
                        if type(usage[key]) is not int or usage[key] < 0:
                            raise ProviderError("response_usage")
                        counts[key] = usage[key]
                usage = counts or None
            return ModelReply(
                message["content"], model, fingerprint, usage, time.perf_counter() - started
            )
        except ProviderError:
            raise
        except (OSError, http.client.HTTPException):
            raise ProviderError("transport") from None
        except (ValueError, TypeError, KeyError, RecursionError, ExtractionError):
            raise ProviderError("response_protocol") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
