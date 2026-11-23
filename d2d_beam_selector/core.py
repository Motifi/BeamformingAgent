"""Load narrowband MISO CSI and evaluate joint D2D beam selections."""

from dataclasses import dataclass
from itertools import islice, product
import json
from math import log
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np


@dataclass(frozen=True)
class Scene:
    """Channels indexed by paired receiver, paired transmitter, and Tx port."""

    channels: np.ndarray
    pairs: tuple[tuple[str, str], ...]
    subcarrier_index: int

    def __post_init__(self):
        channels = np.array(self.channels, copy=True)
        channels.setflags(write=False)
        object.__setattr__(self, "channels", channels)


def _pairing(path: Path) -> tuple[tuple[str, str], ...]:
    with path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError("The pairing file must be a nonempty JSON array.")
    pairs = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each pair must be an object with tx_id and rx_id.")
        tx, rx = record.get("tx_id"), record.get("rx_id")
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in (tx, rx)):
            raise ValueError("Each pair needs a nonempty string tx_id and rx_id.")
        pairs.append((tx, rx))
    if any(len(set(side)) != len(pairs) for side in zip(*pairs)):
        raise ValueError("A transmitter or receiver can appear only once in the pairing file.")
    return tuple(pairs)


def _indices(data: np.lib.npyio.NpzFile, axis: str, identifiers, size: int) -> list[int]:
    name = f"{axis}_ids"
    if name not in data:
        raise ValueError(f"CSI archive is missing {name}.")
    raw_ids = data[name]
    if raw_ids.shape != (size,) or raw_ids.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must contain one string ID per {axis.upper()} device.")
    stored = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_ids]
    if len(set(stored)) != size or any(not value.strip() for value in stored):
        raise ValueError(f"{name} contains a duplicate or empty device ID.")
    lookup = {identifier: index for index, identifier in enumerate(stored)}
    try:
        return [lookup[identifier] for identifier in identifiers]
    except KeyError as exc:
        raise ValueError(f"{axis.upper()} device {exc.args[0]!r} is absent from the CSI archive.") from exc


def _read_scene(data, pairs, time_index, subcarrier_index) -> Scene:
    if "h_freq" not in data:
        raise ValueError("CSI archive is missing h_freq. Export CIR + CFR in SionnaRTStudio.")
    h_freq = data["h_freq"]
    if h_freq.ndim != 6 or not np.iscomplexobj(h_freq) or 0 in h_freq.shape:
        raise ValueError("h_freq must be complex with shape [time, Rx, Rx port, Tx, Tx port, subcarrier].")
    times, rx_count, rx_ports, tx_count, _, subcarriers = h_freq.shape
    if rx_ports != 1:
        raise ValueError("The MISO selector requires exactly one Rx port.")
    if not 0 <= time_index < times:
        raise ValueError(f"time_index must be in [0, {times - 1}].")
    if subcarrier_index is None:
        subcarrier_index = subcarriers // 2
        if "frequencies" in data:
            frequencies = data["frequencies"]
            if (
                frequencies.shape != (subcarriers,)
                or frequencies.dtype.kind not in "iuf"
                or not np.isfinite(frequencies).all()
            ):
                raise ValueError("frequencies must contain one finite baseband offset per subcarrier.")
            subcarrier_index = int(np.abs(frequencies.astype(float)).argmin())
    if not 0 <= subcarrier_index < subcarriers:
        raise ValueError(f"subcarrier_index must be in [0, {subcarriers - 1}].")
    tx_ids, rx_ids = zip(*pairs)
    rx_indices = _indices(data, "rx", rx_ids, rx_count)
    tx_indices = _indices(data, "tx", tx_ids, tx_count)
    snapshot = h_freq[time_index, :, 0, :, :, subcarrier_index]
    channels = snapshot[np.ix_(rx_indices, tx_indices)]
    if not np.isfinite(channels).all():
        raise ValueError("The selected CSI contains a nonfinite coefficient.")
    return Scene(channels, pairs, subcarrier_index)


def load_scene(
    path: Path, pairs_path: Path, *, time_index: int = 0, subcarrier_index: int | None = None
) -> Scene:
    """Read a SionnaRTStudio NPZ or replay ZIP and align its two device axes."""
    path = Path(path)
    pairs = _pairing(Path(pairs_path))
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                return _read_scene(arrays, pairs, time_index, subcarrier_index)
        if suffix == ".zip":
            with ZipFile(path) as archive:
                members = [
                    member for member in archive.infolist()
                    if not member.is_dir() and PurePosixPath(member.filename).name == "csi.npz"
                ]
                if len(members) != 1:
                    raise ValueError("The replay ZIP must contain exactly one csi.npz.")
                with archive.open(members[0]) as source, np.load(source, allow_pickle=False) as arrays:
                    return _read_scene(arrays, pairs, time_index, subcarrier_index)
    except BadZipFile as exc:
        raise ValueError(f"Invalid CSI archive: {path}") from exc
    raise ValueError("Use a SionnaRTStudio .npz or replay .zip file.")


def dft_codebook(num_ports: int, num_beams: int) -> np.ndarray:
    """Return unit-norm, uniformly spaced DFT steering vectors [beam, port]."""
    if num_ports < 1 or num_beams < 1:
        raise ValueError("The number of ports and beams must both be positive.")
    angles = 2j * np.pi * np.arange(num_beams)[:, None] * np.arange(num_ports) / num_beams
    return np.exp(angles) / np.sqrt(num_ports)


def demo_scene() -> Scene:
    """Construct a deterministic four-pair numerical example without ray tracing."""
    pairs = tuple((f"tx-{index}", f"rx-{index}") for index in range(1, 5))
    rng = np.random.default_rng(437)
    coefficients = (
        rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    ) * 5e-6
    indices = np.arange(4)
    coefficients[indices, indices, indices] += 4e-5
    coefficients[(indices + 1) % 4, indices, indices] += 5e-5
    channels = coefficients @ dft_codebook(4, 4).conj()
    return Scene(channels, pairs, 0)


def beam_gains(channels: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Return |h[r,t] @ v[b]|² for every directed link and codeword."""
    if (
        channels.ndim != 3 or codebook.ndim != 2
        or channels.shape[-1] != codebook.shape[-1]
        or 0 in channels.shape or 0 in codebook.shape
    ):
        raise ValueError("Expected compatible channel [Rx, Tx, port] and codebook [beam, port] arrays.")
    return np.abs(channels @ codebook.T) ** 2


def _validate_scoring(gains, powers_w, noise_w, weights):
    if (
        gains.ndim != 3 or gains.shape[0] != gains.shape[1] or 0 in gains.shape
        or gains.dtype.kind not in "iuf" or not np.isfinite(gains).all()
        or np.any(gains < 0)
    ):
        raise ValueError("Beam gains must be finite nonnegative numbers with shape [K, K, beams].")
    for name, values, allow_zero in (
        ("powers_w", powers_w, True),
        ("noise_w", noise_w, False),
        ("weights", weights, True),
    ):
        if (
            values.shape != (gains.shape[0],) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or np.any(values < 0)
            or (not allow_zero and np.any(values == 0))
        ):
            sign = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must contain one finite {sign} number per pair.")


def _score_batch(gains, beam_ids, powers_w, noise_w, weights):
    count = gains.shape[0]
    links = np.arange(count)
    received = gains[links[None, :, None], links[None, None, :], beam_ids[:, None, :]] * powers_w
    desired = np.diagonal(received, axis1=-2, axis2=-1)
    interference = (received * ~np.eye(count, dtype=bool)).sum(axis=-1)
    rates = np.log1p(desired / (interference + noise_w)) / log(2)
    return rates @ weights, rates


def score(
    gains: np.ndarray, beam_ids, powers_w: np.ndarray, noise_w: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Score one set of beam IDs and return WSR and per-receiver spectral efficiencies."""
    _validate_scoring(gains, powers_w, noise_w, weights)
    ids = np.asarray(beam_ids)
    if (
        ids.shape != (gains.shape[0],) or ids.dtype.kind not in "iu"
        or np.any(ids < 0) or np.any(ids >= gains.shape[-1])
    ):
        raise ValueError("Provide one valid integer beam ID per pair.")
    wsrs, rates = _score_batch(gains, ids[None, :], powers_w, noise_w, weights)
    return float(wsrs[0]), rates[0]


def exhaustive_best(
    gains: np.ndarray, powers_w: np.ndarray, noise_w: np.ndarray, weights: np.ndarray
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Find the exact codebook optimum, breaking ties by lexicographic order."""
    _validate_scoring(gains, powers_w, noise_w, weights)
    count, _, beams = gains.shape
    total = beams ** count
    if total > 1_000_000:
        raise ValueError(f"Exact search has {total} combinations; use fewer pairs or beams.")
    combinations = product(range(beams), repeat=count)
    batch_size = max(1, min(4096, 524_288 // count**2))
    best_ids = ()
    best_wsr = -np.inf
    best_rates = np.empty(count)
    for _ in range(0, total, batch_size):
        choices = np.asarray(list(islice(combinations, batch_size)))
        wsrs, rates = _score_batch(gains, choices, powers_w, noise_w, weights)
        index = int(wsrs.argmax())
        if wsrs[index] > best_wsr:
            best_ids = tuple(choices[index].tolist())
            best_wsr = float(wsrs[index])
            best_rates = rates[index].copy()
    return best_ids, best_wsr, best_rates
