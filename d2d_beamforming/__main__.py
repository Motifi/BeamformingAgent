"""Fit continuous beam readouts and compare them on held-out CSI."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from d2d_beam_selector.core import load_scene

from .model import BeamHead, LinkEncoder, conventional_beams, link_features, spectral_efficiency


DEFAULT_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"


def synthetic(count: int, generator: torch.Generator, links: int = 4) -> torch.Tensor:
    channels = torch.randn(count, links, links, 4, dtype=torch.complex64, generator=generator)
    cross = 0.1 + 0.9 * torch.rand(count, links, links, 1, generator=generator)
    diagonal = torch.eye(links, dtype=torch.bool)[None, :, :, None]
    return channels * torch.where(diagonal, 1.0, cross) * math.sqrt(10)


def encode(encoder: LinkEncoder, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batches = [encoder(link_features(batch)) for batch in channels.split(16)]
    projected, encoded = zip(*batches)
    return torch.cat(projected), torch.cat(encoded)


def fit_head(features: torch.Tensor, csi: torch.Tensor, steps: int, seed: int) -> BeamHead:
    torch.manual_seed(seed)
    head = BeamHead(features.shape[-1], csi.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(len(csi), (32,), generator=generator)
        optimizer.zero_grad()
        rates = spectral_efficiency(csi[indices], head(features[indices]))
        loss = -rates.sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
    return head.eval()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("deepseek", "tiny"), default="deepseek")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--csi", type=Path, help="optional four-pair CSI ZIP or NPZ to score after the synthetic evaluation")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--power-dbm", type=float)
    parser.add_argument("--noise-dbm", type=float)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--subcarrier-index", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.train, args.test, args.steps) < 1:
        parser.error("--train, --test and --steps must be positive")
    scene_args = (args.pairs, args.power_dbm, args.noise_dbm)
    if args.csi and (None in scene_args or not all(map(math.isfinite, scene_args[1:]))):
        parser.error("--csi requires --pairs and finite --power-dbm and --noise-dbm")
    if not args.csi and any(value is not None for value in scene_args):
        parser.error("--pairs, --power-dbm and --noise-dbm require --csi")
    return args


def main() -> None:
    args = arguments()
    scene = None
    power_watts = 1.0
    if args.csi:
        scene = load_scene(
            args.csi, args.pairs, time_index=args.time_index, subcarrier_index=args.subcarrier_index
        )
        if scene.channels.shape != (4, 4, 4):
            raise SystemExit("The probe expects four pairs and four transmit ports.")
        try:
            power_watts = 10 ** ((args.power_dbm - 30) / 10)
            amplitude = 10 ** ((args.power_dbm - args.noise_dbm) / 20)
        except OverflowError as exc:
            raise SystemExit("Power or noise is out of range.") from exc
        example_csi = torch.tensor(scene.channels * amplitude, dtype=torch.complex64)[None]
        if power_watts <= 0 or amplitude <= 0 or not torch.isfinite(example_csi.abs().square()).all():
            raise SystemExit("Power or noise is out of range.")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    training = synthetic(args.train, torch.Generator().manual_seed(args.seed))
    testing = synthetic(args.test, torch.Generator().manual_seed(args.seed + 1))
    encoder = LinkEncoder(link_features(training[:1]).shape[-1], args.backbone, args.model, args.local_files_only)
    start = perf_counter()
    train_tokens, train_hidden = encode(encoder, training)
    test_tokens, test_hidden = encode(encoder, testing)
    encode_seconds = perf_counter() - start

    projection_head = fit_head(train_tokens, training, args.steps, args.seed + 2)
    transformer_head = fit_head(train_hidden, training, args.steps, args.seed + 2)
    with torch.no_grad():
        beams = {
            "projection_only": projection_head(test_tokens),
            args.backbone: transformer_head(test_hidden),
            **conventional_beams(testing),
        }
        mean_rates = {
            name: spectral_efficiency(testing, weights).sum(dim=-1).mean().item()
            for name, weights in beams.items()
        }
        if scene is None:
            example_csi = testing[:1]
            example_beams = {name: weights[:1] for name, weights in beams.items()}
        else:
            example_tokens, example_hidden = encode(encoder, example_csi)
            example_beams = {
                "projection_only": projection_head(example_tokens),
                args.backbone: transformer_head(example_hidden),
                **conventional_beams(example_csi),
            }
        weights = example_beams[args.backbone][0] * math.sqrt(power_watts)
        example = {
            "source": str(args.csi) if scene is not None else "first_held_out_synthetic_snapshot",
            "sum_rate_bits_s_hz": {
                name: spectral_efficiency(example_csi, beams).sum().item()
                for name, beams in example_beams.items()
            },
            "beam_weights_real": weights.real.tolist(),
            "beam_weights_imag": weights.imag.tolist(),
            "beam_squared_norms": weights.abs().square().sum(dim=-1).tolist(),
        }
        if scene is not None:
            example["pairs"] = scene.pairs
            example["subcarrier_index"] = scene.subcarrier_index
            example["power_watts"] = power_watts
    result = {
        "backbone": args.model if args.backbone == "deepseek" else "tiny_random_transformer",
        "seed": args.seed,
        "train_scenes": args.train,
        "test_scenes": args.test,
        "readout_steps": args.steps,
        "encode_seconds": encode_seconds,
        "test_mean_sum_rate_bits_s_hz": mean_rates,
        "example": example,
    }
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
