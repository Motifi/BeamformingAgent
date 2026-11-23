"""Multi-start WMMSE for single-stream D2D with fixed per-Tx power."""

import argparse
import json
import math
from pathlib import Path

import torch

from .benchmark import dataset, rates
from .model import conventional_beams, leakage_beams, unit_beams


def sphere_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Minimize wᴴAw − 2 Re(wᴴb) subject to ||w||² = 1."""
    eigenvalues, basis = torch.linalg.eigh(matrix)
    offsets = eigenvalues - eigenvalues[..., :1]
    projected = (basis.mH @ rhs.unsqueeze(-1)).squeeze(-1)
    power = projected.abs().square()
    # The shifted multiplier is nonnegative. Its unshifted value can be negative.
    lower = torch.zeros_like(offsets[..., :1])
    upper = power.sum(-1, keepdim=True).sqrt()
    tiny = torch.finfo(power.dtype).tiny
    for _ in range(36):
        shift = (lower + upper) / 2
        trial_power = (power / (offsets + shift).square().clamp_min(tiny)).sum(-1, keepdim=True)
        lower = torch.where(trial_power > 1, shift, lower)
        upper = torch.where(trial_power > 1, upper, shift)

    # If b is orthogonal to the lowest eigenspace and the limiting solution
    # lies inside the sphere, place the unused norm in that eigenspace.
    lowest = offsets <= 1e-12 * eigenvalues.abs().amax(-1, keepdim=True).clamp_min(1)
    boundary = projected.masked_fill(lowest, 0) / offsets.masked_fill(lowest, 1)
    remaining = 1 - boundary.abs().square().sum(-1, keepdim=True)
    hard_case = (remaining >= 0) & (
        projected.masked_fill(~lowest, 0).abs().amax(-1, keepdim=True)
        <= 1e-14 * power.sum(-1, keepdim=True).sqrt()
    )
    boundary[..., :1] = remaining.clamp_min(0).sqrt()
    solution = projected / (offsets + upper).clamp_min(tiny)
    coords = torch.where(hard_case, boundary, solution)
    return unit_beams((basis @ coords.unsqueeze(-1)).squeeze(-1))


def coefficients(channels, beams, outer, direct, mask):
    received = torch.einsum("brtn,btn->brt", channels, beams)
    power = received.abs().square()
    desired = received.diagonal(dim1=-2, dim2=-1)
    noise_interference = 1 + power.masked_fill(mask, 0).sum(-1)
    total = noise_interference + desired.abs().square()
    receiver = desired / total
    inverse_mse = total / noise_interference
    penalty = inverse_mse * receiver.abs().square()
    matrix = (outer * penalty[:, :, None, None, None]).sum(1)
    rhs = (inverse_mse * receiver)[..., None] * direct
    return matrix, rhs


@torch.no_grad()
def solve(channels: torch.Tensor, initial: torch.Tensor, steps: int, tolerance: float):
    direct = channels.diagonal(dim1=1, dim2=2).transpose(1, 2).conj()
    outer = channels.conj().unsqueeze(-1) * channels.unsqueeze(-2)
    mask = torch.eye(channels.shape[1], dtype=torch.bool, device=channels.device)
    beams = initial.clone()
    iterations = torch.full((len(channels),), steps)
    indices = torch.arange(len(channels))
    max_decrease = 0.0
    for step in range(1, steps + 1):
        active = beams[indices]
        csi = channels[indices]
        matrix, rhs = coefficients(csi, active, outer[indices], direct[indices], mask)
        updated = sphere_solve(matrix, rhs)
        change = rates(csi, updated) - rates(csi, active)
        max_decrease = max(max_decrease, -change.min().item())
        beams[indices] = updated
        done = change.abs() <= tolerance
        iterations[indices[done]] = step
        indices = indices[~done]
        if indices.numel() == 0:
            break

    matrix, rhs = coefficients(channels, beams, outer, direct, mask)
    gradient = rhs - (matrix @ beams.unsqueeze(-1)).squeeze(-1)
    tangent = gradient - (gradient * beams.conj()).real.sum(-1, keepdim=True) * beams
    residual = (2 / math.log(2)) * torch.linalg.vector_norm(tangent.flatten(1), dim=-1)
    return beams, iterations, residual, max_decrease


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--test-seed", type=int, default=27003)
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=1.0, help="leakage baseline coefficient")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.test, args.steps, args.links) < 1 or args.random_starts < 0:
        parser.error("Use positive scene, step and link counts, and a nonnegative random start count.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Tolerance must be finite and positive.")
    if not math.isfinite(args.regularization) or args.regularization <= 0:
        parser.error("Regularization must be finite and positive.")
    return args


def main():
    args = arguments()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    channels = dataset(args.test, args.test_seed, args.links).to(torch.complex128)
    starts = conventional_beams(channels)
    if args.regularization != 1.0:
        starts["regularized"] = leakage_beams(channels, regularization=args.regularization)
    generator = torch.Generator().manual_seed(1729)
    for i in range(args.random_starts):
        starts[f"random_{i + 1}"] = unit_beams(
            torch.randn(channels.shape[:1] + channels.shape[2:], dtype=channels.dtype, generator=generator)
        )
    details, candidates, weights, residuals = {}, [], [], []
    for name, initial in starts.items():
        beams, counts, residual, decrease = solve(channels, initial, args.steps, args.tolerance)
        achieved = rates(channels, beams)
        candidates.append(achieved)
        weights.append(beams)
        residuals.append(residual)
        details[name] = {
            "initial_mean_rate": rates(channels, initial).mean().item(),
            "final_mean_rate": achieved.mean().item(),
            "mean_iterations": counts.double().mean().item(),
            "fraction_at_iteration_limit": (counts == args.steps).double().mean().item(),
            "max_step_rate_decrease": decrease,
            "median_tangent_gradient_norm": residual.median().item(),
        }
    best, indices = torch.stack(candidates).max(0)
    batch = torch.arange(len(channels))
    best_beams = torch.stack(weights)[indices, batch]
    best_residual = torch.stack(residuals)[indices, batch]
    report = {
        "links": args.links,
        "test_scenes": args.test,
        "test_seed": args.test_seed,
        "rate_improvement_tolerance": args.tolerance,
        "initial_regularization": args.regularization,
        "max_iterations_per_start": args.steps,
        "starts": details,
        "best_of_starts_mean_sum_rate_bits_s_hz": best.mean().item(),
        "best_of_starts_rate_quantiles_05_50_95": best.quantile(best.new_tensor([.05, .5, .95])).tolist(),
        "best_of_starts_tangent_gradient_norm_quantiles_50_95_100": (
            best_residual.quantile(best.new_tensor([.5, .95, 1])).tolist()
        ),
        "best_of_starts_max_unit_power_error": (best_beams.abs().square().sum(-1) - 1).abs().max().item(),
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
