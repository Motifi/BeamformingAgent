"""Small client for OpenAI-compatible Chat Completions endpoints."""

from dataclasses import dataclass
import http.client
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """A request failed or the provider did not return a usable completion."""


@dataclass(frozen=True)
class Completion:
    content: str
    model: str | None
    usage: dict[str, int]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def chat_endpoint(base_url: str, endpoint_url: str | None = None) -> str:
    """Append the standard route to a base URL or use a complete endpoint."""
    url = base_url if endpoint_url is None else endpoint_url
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise ValueError("The API URL is invalid.") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or port == 0
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or any(char.isspace() or ord(char) < 32 for char in url)
    ):
        raise ValueError("Use an HTTP(S) API URL without credentials or a fragment.")
    if endpoint_url is not None:
        return parts.geturl()
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return parts._replace(path=path).geturl()


def complete(
    *,
    base_url: str,
    endpoint_url: str | None = None,
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 30.0,
    max_tokens: int = 512,
    json_mode: bool = True,
    extra_body: dict | None = None,
) -> Completion:
    """Send a single non-streaming request and return its first text choice."""
    endpoint = chat_endpoint(base_url, endpoint_url)
    if not model.strip():
        raise ValueError("Specify a model name.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The request timeout must be positive and finite.")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("The output token limit must be a positive integer.")
    if api_key is not None and any(not 33 <= ord(char) <= 126 for char in api_key):
        raise ValueError("The API key contains invalid characters.")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("Extra request parameters must be a JSON object.")
    extra = extra_body or {}
    reserved = {"model", "messages", "max_tokens", "response_format", "stream"}
    if reserved.intersection(extra):
        raise ValueError("The model, messages, max_tokens, response_format, and stream fields are reserved.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        **extra,
    }
    if "max_completion_tokens" not in extra:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=json.dumps(payload, allow_nan=False).encode(), headers=headers, method="POST")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw_response = response.read()
    except HTTPError as error:
        status = error.code
        error.close()
        if 300 <= status < 400:
            raise ProviderError(f"The API returned HTTP {status}. Configure the final endpoint URL.") from None
        raise ProviderError(f"The API returned HTTP {status}.") from None
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise ProviderError("The API request timed out.") from None
        raise ProviderError("Could not connect to the API endpoint.") from None
    except TimeoutError:
        raise ProviderError("The API request timed out.") from None
    except (OSError, http.client.HTTPException, ValueError):
        raise ProviderError("The API connection failed.") from None

    try:
        result = json.loads(raw_response)
        choice = result["choices"][0]
        message = choice["message"]
        content = message["content"]
        reason = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("The API response did not contain a text completion.") from None
    if reason not in (None, "stop", "end_turn"):
        raise ProviderError("The model did not finish its text completion.")
    if not isinstance(content, str) or not content.strip() or message.get("refusal"):
        raise ProviderError("The API response did not contain a text completion.")
    returned_model = result.get("model")
    if not isinstance(returned_model, str) or (api_key and api_key in returned_model):
        returned_model = None
    raw_usage = result.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(name)
            if type(value) is int and value >= 0:
                usage[name] = value
    return Completion(content, returned_model, usage)
