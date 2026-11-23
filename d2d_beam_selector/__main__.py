"""Select a D2D codebook beam per transmitter and verify the sum rate."""

import argparse
import json
import math
import os
import stat
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, load_provider_config
from .provider import ProviderError, chat_endpoint, complete

_SYSTEM_PROMPT = """Select jointly one beam per transmitting device to maximize weighted sum spectral efficiency.
The input JSON contains gain[i][j][b], the linear power gain at receiver i from transmitter j using beam b.
Receiver i is paired with transmitter i. The beam ID b_j is a zero-based index for transmitter j.
Rate_i = log2(1 + power_w[i]*gain[i][i][b_i] /
    (noise_w[i] + sum over j != i of power_w[j]*gain[i][j][b_j])).
Maximize sum_i weight[i]*Rate_i. Use cross-link interference, not only each desired-link gain.
Return exactly one JSON object with key "beam_ids", containing one integer per transmitter in input order.
"""

def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return number


def _json_object(value):
    try:
        parsed = json.loads(value)
        json.dumps(parsed, allow_nan=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _beam_ids(content, link_count, beam_count):
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Provider returned invalid JSON for beam selection.") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a beam selection JSON object.")
    ids = data.get("beam_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != link_count
        or any(type(index) is not int or not 0 <= index < beam_count for index in ids)
    ):
        raise ValueError(f"Expected beam_ids to contain {link_count} integers from 0 to {beam_count - 1}.")
    return tuple(ids)


def _power_from_dbm(dbm, name):
    if not math.isfinite(dbm):
        raise ValueError(f"{name} must be finite.")
    try:
        watts = 10 ** ((dbm - 30) / 10)
    except OverflowError as exc:
        raise ValueError(f"{name} is out of range.") from exc
    if not math.isfinite(watts) or watts <= 0:
        raise ValueError(f"{name} is out of range.")
    return watts


def _origin(endpoint):
    url = urlsplit(endpoint)
    return (url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80))


def _provider_settings(args):
    config = load_provider_config(args.config)
    configured_endpoint = chat_endpoint(
        config.get("base_url", DEFAULT_BASE_URL), config.get("api_url"),
    )
    base_override = args.base_url if args.base_url is not None else os.environ.get("D2D_API_BASE_URL")
    if args.api_url is not None or base_override is not None:
        endpoint = chat_endpoint(
            base_override if base_override is not None else DEFAULT_BASE_URL,
            args.api_url,
        )
    else:
        endpoint = configured_endpoint

    same_origin = _origin(endpoint) == _origin(configured_endpoint)
    provider_config = config if same_origin else {}
    cli_auth = args.no_auth or args.api_key_env is not None or args.api_key_file is not None
    auth = {} if cli_auth else provider_config
    model = args.model if args.model is not None else os.environ.get("D2D_API_MODEL")
    return {
        "endpoint": endpoint,
        "model": model if model is not None else provider_config.get("model", DEFAULT_MODEL),
        "no_auth": args.no_auth or auth.get("no_auth", False),
        "api_key_env": args.api_key_env if cli_auth else auth.get("api_key_env"),
        "api_key_file": args.api_key_file if cli_auth else auth.get("api_key_file"),
        "timeout": args.timeout if args.timeout is not None else config.get("timeout", 30.0),
        "max_tokens": args.max_tokens if args.max_tokens is not None else config.get("max_tokens", 512),
        "prompt_json": args.prompt_json if args.prompt_json is not None else provider_config.get("prompt_json", False),
        "extra_body": args.extra_body if args.extra_body is not None else provider_config.get("extra_body"),
    }


def _key(settings):
    if settings["no_auth"]:
        return None
    env_name = settings["api_key_env"]
    if env_name:
        key = os.environ.get(env_name)
        if not key:
            raise ValueError(f"Set the API key in {env_name}.")
        return key
    if settings["api_key_file"] is not None:
        path = Path(settings["api_key_file"]).expanduser()
        if not path.is_file():
            raise ValueError("The API key file must be an existing regular file.")
        with path.open(encoding="utf-8") as stream:
            mode = os.fstat(stream.fileno()).st_mode
            if os.name == "posix" and stat.S_IMODE(mode) & 0o077:
                raise ValueError("The API key file must be private (chmod 600).")
            key = stream.read().strip()
        if not key:
            raise ValueError("The API key file is empty.")
        return key
    key = os.environ.get("D2D_API_KEY")
    if key:
        return key
    if _origin(settings["endpoint"]) == ("https", "api.deepseek.com", 443):
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key
    raise ValueError("Set D2D_API_KEY or configure api_key_env or api_key_file. Use --no-auth for a keyless server.")


def _check_config(args):
    settings = _provider_settings(args)
    endpoint = urlsplit(settings["endpoint"])._replace(query="").geturl()
    report = {
        "api_url": endpoint,
        "model": settings["model"],
        "auth": "not_required" if settings["no_auth"] else "ready",
        "json_mode": not settings["prompt_json"],
    }
    try:
        _key(settings)
    except (OSError, ValueError) as exc:
        report["auth"] = "unavailable"
        report["auth_detail"] = str(exc)
    return report


def _selection(indices, wsr, rates, codebook, powers_w):
    weights = codebook[list(indices)] * (powers_w**0.5)[:, None]
    return {
        "beam_ids": list(indices),
        "wsr_bits_s_hz": wsr,
        "link_rates_bits_s_hz": rates.tolist(),
        "beam_weights_real": weights.real.tolist(),
        "beam_weights_imag": weights.imag.tolist(),
    }


def _solve(args):
    import numpy as np

    from .core import beam_gains, demo_scene, dft_codebook, exhaustive_best, load_scene, score

    scene = demo_scene() if args.demo else load_scene(
        args.csi, args.pairs, time_index=args.time_index, subcarrier_index=args.subcarrier_index,
    )
    links, _, ports = scene.channels.shape
    powers_w = np.full(links, _power_from_dbm(args.power_dbm, "Transmit power in dBm"))
    noise_w = np.full(links, _power_from_dbm(args.noise_dbm, "Noise power in dBm"))
    weights = np.asarray(args.weights if args.weights is not None else np.ones(links), dtype=float)
    if weights.shape != (links,) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError(f"Link weights must contain {links} finite, nonnegative values, at least one positive.")
    codebook = dft_codebook(ports, args.beams)
    gains = beam_gains(scene.channels, codebook)

    best_ids, best_wsr, best_rates = exhaustive_best(gains, powers_w, noise_w, weights)
    link_indices = np.arange(links)
    independent_ids = tuple(int(index) for index in gains[link_indices, link_indices].argmax(axis=1))
    independent_wsr, independent_rates = score(gains, independent_ids, powers_w, noise_w, weights)
    result = {
        "input": {
            "source": "deterministic_demo" if args.demo else str(args.csi),
            "pairs": [{"tx_id": tx, "rx_id": rx} for tx, rx in scene.pairs],
            "channel_shape_rx_tx_port": list(scene.channels.shape),
            "time_index": args.time_index,
            "subcarrier_index": scene.subcarrier_index,
            "tx_power_dbm": args.power_dbm,
            "rx_noise_dbm": args.noise_dbm,
            "link_weights": weights.tolist(),
        },
        "codebook": {"type": "dft_over_tx_ports", "size": args.beams, "combinations": args.beams ** links},
        "exhaustive": _selection(best_ids, best_wsr, best_rates, codebook, powers_w),
        "independent_strongest_beam": _selection(
            independent_ids, independent_wsr, independent_rates, codebook, powers_w,
        ),
    }
    if args.offline:
        return result

    settings = _provider_settings(args)
    api_key = _key(settings)
    payload = json.dumps(
        {"gain": gains.tolist(), "power_w": powers_w.tolist(),
         "noise_w": noise_w.tolist(), "weight": weights.tolist()},
        separators=(",", ":"), allow_nan=False,
    )
    started = perf_counter()
    response = complete(
        base_url=DEFAULT_BASE_URL, endpoint_url=settings["endpoint"], model=settings["model"], api_key=api_key,
        system_prompt=_SYSTEM_PROMPT, user_prompt=payload, timeout=settings["timeout"],
        max_tokens=settings["max_tokens"], json_mode=not settings["prompt_json"], extra_body=settings["extra_body"],
    )
    elapsed = perf_counter() - started
    selected_ids = _beam_ids(response.content, links, args.beams)
    wsr, rates = score(gains, selected_ids, powers_w, noise_w, weights)
    result["model"] = {
        "requested_model": settings["model"],
        "returned_model": response.model,
        "latency_ms": round(1000 * elapsed, 1),
        "usage": response.usage,
        **_selection(selected_ids, wsr, rates, codebook, powers_w),
        "optimality_gap_bits_s_hz": max(0.0, best_wsr - wsr),
        "wsr_fraction_of_optimum": wsr / best_wsr if best_wsr > 0 else None,
    }
    return result


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csi", type=Path, help="SionnaRTStudio CSI ZIP or csi.npz with CFR")
    input_group.add_argument("--demo", action="store_true", help="use a deterministic four-link input")
    input_group.add_argument("--check-config", action="store_true", help="show the effective API settings without a request")
    parser.add_argument("--pairs", type=Path, help="JSON array of tx_id/rx_id pairs in link order")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--subcarrier-index", type=int, help="default: nearest to zero baseband frequency")
    parser.add_argument("--beams", type=int, default=4, help="number of unit-norm DFT beams per transmitter")
    parser.add_argument("--power-dbm", type=float, help="fixed transmit power per device (demo: 20)")
    parser.add_argument("--noise-dbm", type=float, help="receiver noise on the selected subcarrier (demo: -110)")
    parser.add_argument("--weights", type=float, nargs="+", help="one WSR weight per pair (default: equal)")
    parser.add_argument("--offline", action="store_true", help="evaluate local baselines without contacting a provider")
    parser.add_argument("--config", type=Path, help="provider JSON config (default: ./d2d_provider.json if present)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, including any /v1 prefix")
    parser.add_argument("--api-url", help="exact Chat Completions endpoint URL; overrides --base-url")
    parser.add_argument("--model", help="override the configured provider model")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--api-key-env", help="name of the environment variable containing your API key")
    auth.add_argument("--api-key-file", type=Path, help="read the API key from a private file")
    auth.add_argument("--no-auth", action="store_true", help="call a server that does not require a key")
    parser.add_argument("--timeout", type=_positive_float, help="request timeout in seconds")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--prompt-json", action=argparse.BooleanOptionalAction,
                        help="use the prompt alone to request JSON from the provider")
    parser.add_argument("--extra-body", type=_json_object, help="extra provider-specific request fields as JSON")
    parser.add_argument("--output", type=Path, help="save the same report printed to stdout")
    args = parser.parse_args()
    if args.csi and (not args.pairs or args.power_dbm is None or args.noise_dbm is None):
        parser.error("--csi requires --pairs, --power-dbm and --noise-dbm.")
    if args.demo:
        if args.pairs or args.time_index != 0 or args.subcarrier_index is not None:
            parser.error("--pairs, --time-index and --subcarrier-index apply to --csi only.")
        if args.power_dbm is None:
            args.power_dbm = 20.0
        if args.noise_dbm is None:
            args.noise_dbm = -110.0
    if args.time_index < 0 or (args.subcarrier_index is not None and args.subcarrier_index < 0):
        parser.error("CSI indices must be nonnegative.")
    if args.beams < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("--beams and --max-tokens must be positive.")
    return args


def main():
    args = _arguments()
    try:
        result = _check_config(args) if args.check_config else _solve(args)
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
    except (OSError, ValueError, ProviderError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
