"""Read provider connection settings for the D2D beam selector."""

import json
import math
import os
from pathlib import Path
import re

from .provider import chat_endpoint

DEFAULT_CONFIG_PATH = Path("d2d_provider.json")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

_TEXT_FIELDS = ("base_url", "api_url", "model", "api_key_env", "api_key_file")
_BOOL_FIELDS = ("no_auth", "prompt_json")
_CONFIG_FIELDS = {*_TEXT_FIELDS, *_BOOL_FIELDS, "timeout", "max_tokens", "extra_body"}


def load_provider_config(path: Path | None = None) -> dict:
    """Load provider settings, resolving credential file paths relative to the JSON file."""
    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        if path is None:
            return {}
        raise
    except UnicodeDecodeError:
        raise ValueError("The provider configuration must be UTF-8 JSON.") from None
    try:
        config = json.loads(raw)
        json.dumps(config, allow_nan=False)
    except ValueError:
        raise ValueError("The provider configuration must be valid JSON with finite numbers.") from None
    if not isinstance(config, dict):
        raise ValueError("The provider configuration must be a JSON object.")
    if config.keys() - _CONFIG_FIELDS:
        raise ValueError("The provider configuration contains unsupported fields. Use api_key_env or api_key_file for credentials.")

    for field in _TEXT_FIELDS:
        if field in config and (not isinstance(config[field], str) or not config[field].strip()):
            raise ValueError(f"The provider configuration field {field} must be a nonempty string.")
    for field in _BOOL_FIELDS:
        if field in config and type(config[field]) is not bool:
            raise ValueError(f"The provider configuration field {field} must be true or false.")
    if "timeout" in config:
        timeout = config["timeout"]
        try:
            valid_timeout = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("The provider configuration timeout must be a positive finite number.")
    if "max_tokens" in config and (type(config["max_tokens"]) is not int or config["max_tokens"] <= 0):
        raise ValueError("The provider configuration max_tokens must be a positive integer.")
    if "extra_body" in config and not isinstance(config["extra_body"], dict):
        raise ValueError("The provider configuration extra_body must be a JSON object.")

    if "base_url" in config:
        chat_endpoint(config["base_url"])
    if "api_url" in config:
        chat_endpoint(DEFAULT_BASE_URL, config["api_url"])
    if sum(("api_key_env" in config, "api_key_file" in config, config.get("no_auth", False))) > 1:
        raise ValueError("Choose one of api_key_env, api_key_file, or no_auth in the provider configuration.")
    if "api_key_env" in config and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api_key_env"]):
        raise ValueError("The provider configuration api_key_env must be an environment variable name.")
    if "api_key_file" in config:
        try:
            credential_path = Path(config["api_key_file"]).expanduser()
            if "\0" in os.fspath(credential_path):
                raise ValueError
            config["api_key_file"] = Path(os.path.abspath(source.parent / credential_path))
        except (RuntimeError, ValueError):
            raise ValueError("The provider configuration api_key_file must be a valid path.") from None
    return config
