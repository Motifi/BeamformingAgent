"""Test frozen DeepSeek features on top of a trained D2D beamformer."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from .__main__ import DEFAULT_MODEL
from .benchmark import dataset, fit, rates
from .model import LinkEncoder, leakage_beams


@torch.no_grad()
def extract(base, encoder, channels):
    batches = []
    for batch in channels.split(32):
        nodes, edges = base.encode_graph(batch)
        projected, encoded = encoder(nodes, exchangeable_links=True)
        batches.append((base.transformer(nodes), edges, projected, encoded))
    return tuple(torch.cat(group) for group in zip(*batches))


def predict(base, channels, nodes, edges, features, adapter=None):
    if adapter is not None:
        nodes = nodes + adapter(features)
    weights, regularization = base.readout(nodes, edges)
    return rates(channels, leakage_beams(channels, weights, regularization))


def fit_adapter(base, train, validation, train_features, validation_features, steps, seed):
    width = train_features[0].shape[-1]
    adapter = nn.Linear(train_features[2].shape[-1], width, bias=False)
    nn.init.zeros_(adapter.weight)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        best = predict(base, validation, *validation_features, adapter).mean().item()
    state = deepcopy(adapter.state_dict())
    best_step = 0
    for step in range(1, steps + 1):
        subset = torch.randint(len(train), (64,), generator=generator)
        optimizer.zero_grad()
        loss = -predict(base, train[subset], *(f[subset] for f in train_features), adapter).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                score = predict(base, validation, *validation_features, adapter).mean().item()
            if score > best:
                best, best_step = score, step
                state = deepcopy(adapter.state_dict())
    adapter.load_state_dict(state)
    return adapter, best_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter-train", type=int, default=512)
    parser.add_argument("--adapter-validation", type=int, default=256)
    parser.add_argument("--adapter-steps", type=int, default=200)
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.links, args.adapter_train, args.adapter_validation, args.adapter_steps, args.test) < 1:
        parser.error("All link counts, scene counts and step counts must be positive.")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(307)
    encoder = LinkEncoder(48, "deepseek", args.model, args.local_files_only)
    seed, validation_seed, test_seed = 17, 27001, 27003
    report = {
        "backbone": args.model,
        "training_seed": seed,
        "test_seed": test_seed,
        "numerical_train_scenes": 4096,
        "numerical_validation_scenes": 512,
        "adapter_train_scenes": args.adapter_train,
        "adapter_validation_scenes": args.adapter_validation,
        "adapter_max_steps": args.adapter_steps,
        "test_scenes": args.test,
        "loads": {},
    }
    for links in args.links:
        train = dataset(4096, seed, links)
        validation = dataset(512, validation_seed, links)
        test = dataset(args.test, test_seed, links)
        base, best_step = fit("graph_transformer", train, validation, 400, seed, True)
        base.eval().requires_grad_(False)
        adapter_train = dataset(args.adapter_train, 27004, links)
        adapter_validation = dataset(args.adapter_validation, 27005, links)
        start = perf_counter()
        train_features = extract(base, encoder, adapter_train)
        validation_features = extract(base, encoder, adapter_validation)
        test_features = extract(base, encoder, test)
        encode_seconds = perf_counter() - start

        with torch.no_grad():
            baseline = predict(base, test, *test_features[:3])
        row = {
            "base_validation_step": best_step,
            "encode_seconds": encode_seconds,
            "numerical_base_rate": baseline.mean().item(),
        }
        for name, feature_index in (("projection_control", 2), ("deepseek", 3)):
            adapter, step = fit_adapter(
                base, adapter_train, adapter_validation,
                (*train_features[:2], train_features[feature_index]),
                (*validation_features[:2], validation_features[feature_index]),
                args.adapter_steps, seed,
            )
            with torch.no_grad():
                achieved = predict(base, test, *test_features[:2], test_features[feature_index], adapter)
                row[name] = {
                    "validation_step": step,
                    "mean_rate": achieved.mean().item(),
                    "mean_delta_vs_base": (achieved - baseline).mean().item(),
                    "fraction_improved_vs_base": (achieved > baseline + 1e-6).float().mean().item(),
                }
        report["loads"][str(links)] = row
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({links: row}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
