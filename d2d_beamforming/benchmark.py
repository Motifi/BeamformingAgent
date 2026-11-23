"""Compare trainable numerical encoders for fixed-power D2D beamforming."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from .__main__ import synthetic
from .graph import GraphBeamformer
from .model import conventional_beams, leakage_beams, spectral_efficiency, unit_beams


MODES = ("graph", "transformer", "graph_transformer")


def dataset(count: int, seed: int, links: int = 4) -> torch.Tensor:
    return synthetic(count, torch.Generator().manual_seed(seed), links)


def rates(channels: torch.Tensor, beams: torch.Tensor) -> torch.Tensor:
    return spectral_efficiency(channels, beams).sum(-1)


def predict(model: GraphBeamformer, channels: torch.Tensor) -> torch.Tensor:
    weights, regularization = model(channels)
    return leakage_beams(channels, weights, regularization)


def evaluate(model: GraphBeamformer, channels: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([rates(batch, predict(model, batch)) for batch in channels.split(128)])


def refine(channels: torch.Tensor, beams: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Benchmark a numerical refinement under the same fixed-power constraint."""
    for _ in range(steps):
        current = beams.detach().requires_grad_(True)
        current_rates = rates(channels, current)
        gradient, = torch.autograd.grad(current_rates.sum(), current)
        tangent = gradient - (gradient * current.conj()).real.sum(-1, keepdim=True) * current
        with torch.no_grad():
            best = current_rates
            for step in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
                candidate = unit_beams(current + step * tangent)
                candidate_rates = rates(channels, candidate)
                beams = torch.where((candidate_rates > best)[:, None, None], candidate, beams)
                best = torch.maximum(best, candidate_rates)
    return beams


def fit(mode: str, train: torch.Tensor, validation: torch.Tensor, steps: int, seed: int, include_reference: bool):
    torch.manual_seed(seed)
    model = GraphBeamformer(mode=mode, include_reference=include_reference)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_rate = evaluate(model, validation).mean().item()
    best_state = deepcopy(model.state_dict())
    best_step = 0
    for step in range(1, steps + 1):
        model.train()
        batch = train[torch.randint(len(train), (64,), generator=generator)]
        optimizer.zero_grad()
        loss = -rates(batch, predict(model, batch)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps:
            validation_rate = evaluate(model, validation).mean().item()
            if validation_rate > best_rate:
                best_rate = validation_rate
                best_state = deepcopy(model.state_dict())
                best_step = step
    model.load_state_dict(best_state)
    return model, best_step


def summarize(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "seed_std": stdev(values) if len(values) > 1 else 0.0,
        "per_seed": values,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=4096)
    parser.add_argument("--validation", type=int, default=512)
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--test-seed", type=int, default=27003)
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--raw-tokens", action="store_true", help="omit measured reference-waveform features")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test, args.steps, args.links) < 1:
        parser.error("Dataset sizes, training steps and link count must be positive.")
    if len(set([*args.seeds, 27001, args.test_seed])) != len(args.seeds) + 2:
        parser.error("Training, validation and test seeds must be distinct.")
    return args


def main() -> None:
    args = arguments()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    validation = dataset(args.validation, 27001, args.links)
    test = dataset(args.test, args.test_seed, args.links)

    with torch.no_grad():
        candidates = torch.logspace(-2, 2, 41).tolist()
        tuned_lambda = max(
            candidates,
            key=lambda candidate: rates(validation, leakage_beams(validation, regularization=candidate)).mean().item(),
        )
        beams = {
            **conventional_beams(test),
            "tuned_regularized": leakage_beams(test, regularization=tuned_lambda),
        }
        baseline_rates = {name: rates(test, weights) for name, weights in beams.items()}
        tuned_reference = baseline_rates["tuned_regularized"]
    baseline_rates["projected_gradient_10"] = rates(test, refine(test, beams["tuned_regularized"]))

    models = {}
    example = {}
    for mode in MODES:
        test_rates, best_steps, gains, wins = [], [], [], []
        for seed in args.seeds:
            train = dataset(args.train, seed, args.links)
            model, best_step = fit(mode, train, validation, args.steps, seed, not args.raw_tokens)
            learned = evaluate(model, test)
            test_rates.append(learned.mean().item())
            best_steps.append(best_step)
            gains.append((learned - tuned_reference).mean().item())
            wins.append((learned > tuned_reference).float().mean().item())
            if seed == args.seeds[0]:
                with torch.no_grad():
                    weights = predict(model, test[:1])[0]
                example[mode] = {
                    "beam_real": weights.real.tolist(),
                    "beam_imag": weights.imag.tolist(),
                    "power": weights.abs().square().sum(-1).tolist(),
                }
        models[mode] = {
            "mean_sum_rate_bits_s_hz": summarize(test_rates),
            "gain_vs_tuned_regularized_bits_s_hz": summarize(gains),
            "fraction_better_than_tuned_regularized": summarize(wins),
            "best_validation_steps": best_steps,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        }
    report = {
        "links": args.links,
        "train_scenes_per_seed": args.train,
        "validation_scenes": args.validation,
        "test_scenes": args.test,
        "training_seeds": args.seeds,
        "validation_seed": 27001,
        "test_seed": args.test_seed,
        "include_reference_features": not args.raw_tokens,
        "steps": args.steps,
        "tuned_regularization": tuned_lambda,
        "baseline_mean_sum_rate_bits_s_hz": {
            name: values.mean().item() for name, values in baseline_rates.items()
        },
        "models": models,
        "first_test_scene_unit_power_beams": example,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
