"""Learn DeepSeek beam candidates and benchmark WMMSE warm starts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from .__main__ import DEFAULT_MODEL
from .benchmark import dataset, fit, rates
from .llm_benchmark import extract
from .model import LinkEncoder, leakage_beams, unit_beams
from .wmmse import solve


class CandidateHead(nn.Module):
    """Generate several feasible complex beam updates from contextual tokens."""

    def __init__(self, width: int, context_width: int, candidates: int):
        super().__init__()
        self.candidates = candidates
        self.layers = nn.Sequential(
            nn.Linear(context_width, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.readout = nn.Sequential(
            nn.Linear(width + 64, 64), nn.GELU(), nn.Linear(64, candidates * 8)
        )
        nn.init.normal_(self.readout[-1].weight, std=0.005)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, nodes, context, anchor):
        context = self.layers(context)
        delta = self.readout(torch.cat((nodes, context), dim=-1)).tanh()
        delta = delta.reshape(*nodes.shape[:2], self.candidates, 4, 2).contiguous()
        delta = torch.view_as_complex(delta).transpose(1, 2)
        return unit_beams(anchor[:, None] + 0.4 * delta)


@torch.no_grad()
def prepare(base, encoder, channels):
    nodes, edges, projected, encoded = extract(base, encoder, channels)
    weights, regularization = base.readout(nodes, edges)
    anchor = leakage_beams(channels, weights, regularization)
    return nodes, anchor, projected, encoded


def candidate_rates(channels, beams):
    batch, candidates, links, ports = beams.shape
    csi = channels[:, None].expand(-1, candidates, -1, -1, -1).reshape(-1, links, links, ports)
    return rates(csi, beams.reshape(-1, links, ports)).reshape(batch, candidates)


@torch.no_grad()
def choose(head, channels, features):
    nodes, anchor, context = features
    proposed = head(nodes, context, anchor)
    best = candidate_rates(channels, proposed).argmax(-1)
    return proposed[torch.arange(len(channels)), best]


def train_head(train, validation, train_data, val_data, steps, seed, candidates):
    torch.manual_seed(seed)
    head = CandidateHead(train_data[0].shape[-1], train_data[2].shape[-1], candidates)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    highest = -float("inf")
    best_step = 0
    state = deepcopy(head.state_dict())
    for step in range(1, steps + 1):
        indices = torch.randint(len(train), (64,), generator=generator)
        optimizer.zero_grad()
        beams = head(*(train_data[i][indices] for i in (0, 2, 1)))
        scores = candidate_rates(train[indices], beams)
        loss = -(0.85 * scores.max(-1).values + 0.15 * scores.mean(-1)).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps:
            score = rates(validation, choose(head, validation, val_data)).mean().item()
            if score > highest:
                highest, best_step, state = score, step, deepcopy(head.state_dict())
    head.load_state_dict(state)
    return head.eval(), best_step


def score(channels, initial):
    beams = initial.to(channels.dtype)
    result = {"0": rates(channels, beams).mean().item()}
    for previous, target in ((0, 5), (5, 20), (20, 600)):
        beams, _, residual, decrease = solve(channels, beams, target - previous, 1e-9)
        result[str(target)] = rates(channels, beams).mean().item()
        if decrease > 1e-8:
            raise RuntimeError(f"WMMSE update decreased sum rate by {decrease}")
    result["final_median_tangent_gradient"] = residual.median().item()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.links, args.train, args.validation, args.test, args.steps, args.candidates) < 1:
        parser.error("Use positive link counts, dataset sizes, steps and candidate counts.")
    if not 0 < args.regularization < float("inf"):
        parser.error("Use a finite positive regularization coefficient.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed = 17
    train = dataset(args.train, 27004, args.links)
    validation = dataset(args.validation, 27005, args.links)
    test = dataset(args.test, 27003, args.links)
    base, base_step = fit(
        "graph_transformer", dataset(4096, seed, args.links),
        dataset(512, 27001, args.links), 400, seed, True
    )
    base.eval().requires_grad_(False)
    torch.manual_seed(307)
    encoder = LinkEncoder(48, "deepseek", args.model, args.local_files_only)
    training = prepare(base, encoder, train)
    validating = prepare(base, encoder, validation)
    testing = prepare(base, encoder, test)

    high_precision = test.to(torch.complex128)
    results = {
        "regularized": score(
            high_precision, leakage_beams(high_precision, regularization=args.regularization)
        ),
        "numerical_graph": score(high_precision, testing[1]),
    }
    best_heads = {}
    for name, feature_index in (("projection_control", 2), ("deepseek", 3)):
        train_data = (*training[:2], training[feature_index])
        val_data = (*validating[:2], validating[feature_index])
        test_data = (*testing[:2], testing[feature_index])
        head, step = train_head(train, validation, train_data, val_data, args.steps, seed, args.candidates)
        initial = choose(head, test, test_data)
        results[name] = score(high_precision, initial)
        results[name]["validation_step"] = step
        best_heads[name] = head.state_dict()
    if args.checkpoint:
        torch.save({
            "model": args.model,
            "links": args.links,
            "candidates": args.candidates,
            "graph": base.state_dict(),
            "projection": encoder.projection.state_dict(),
            "deepseek_head": best_heads["deepseek"],
        }, args.checkpoint)
    report = {
        "links": args.links,
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
        "test_seed": 27003,
        "training_seed": seed,
        "graph_validation_step": base_step,
        "candidate_count": args.candidates,
        "regularization": args.regularization,
        "results_by_wmmse_step": results,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
