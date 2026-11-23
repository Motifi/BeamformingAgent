"""Run a trained DeepSeek beamforming solver on one CSI snapshot."""

import argparse
import json
import math
from pathlib import Path

import torch

from d2d_beam_selector.core import load_scene

from .benchmark import dataset, rates
from .graph import GraphBeamformer
from .llm_solver import CandidateHead, choose, prepare
from .model import LinkEncoder
from .wmmse import solve


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", help="override the saved DeepSeek model path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--csi", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--power-dbm", type=float)
    parser.add_argument("--noise-dbm", type=float)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--subcarrier-index", type=int)
    parser.add_argument("--refine", type=int, default=0, help="WMMSE iterations after the DeepSeek proposal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.refine < 0:
        parser.error("--refine must be nonnegative")
    if args.csi:
        if args.pairs is None or any(
            x is None or not math.isfinite(x) for x in (args.power_dbm, args.noise_dbm)
        ):
            parser.error("--csi requires --pairs and finite --power-dbm and --noise-dbm")
    elif any(x is not None for x in (args.pairs, args.power_dbm, args.noise_dbm)):
        parser.error("--pairs, --power-dbm and --noise-dbm require --csi")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    links = checkpoint["links"]
    if args.csi:
        scene = load_scene(
            args.csi, args.pairs, time_index=args.time_index, subcarrier_index=args.subcarrier_index
        )
        if scene.channels.shape != (links, links, 4):
            parser.error(f"Checkpoint requires {links} pairs and four transmit ports per pair")
        try:
            power = 10 ** ((args.power_dbm - 30) / 10)
            amplitude = 10 ** ((args.power_dbm - args.noise_dbm) / 20)
        except OverflowError:
            parser.error("Power or noise is out of range")
        channels = torch.tensor(scene.channels * amplitude, dtype=torch.complex64)[None]
        if power <= 0 or amplitude <= 0 or not channels.abs().square().isfinite().all():
            parser.error("Power or noise is out of range")
        pairs = scene.pairs
    else:
        channels = dataset(1, 27006, links)
        power = 1.0
        pairs = None

    base = GraphBeamformer().eval()
    base.load_state_dict(checkpoint["graph"])
    model = args.model or checkpoint["model"]
    encoder = LinkEncoder(48, "deepseek", model, args.local_files_only)
    encoder.projection.load_state_dict(checkpoint["projection"])
    head = CandidateHead(48, encoder.width, checkpoint["candidates"]).eval()
    head.load_state_dict(checkpoint["deepseek_head"])
    nodes, anchor, _, encoded = prepare(base, encoder, channels)
    initial = choose(head, channels, (nodes, anchor, encoded))
    beams = initial
    if args.refine:
        beams, _, _, _ = solve(channels.to(torch.complex128), initial.to(torch.complex128), args.refine, 1e-9)
    weights = beams[0] * math.sqrt(power)
    output = {
        "backbone": model,
        "pairs": pairs,
        "links": links,
        "wmmse_steps_max": args.refine,
        "initial_rate_bits_s_hz": rates(channels, initial).item(),
        "final_rate_bits_s_hz": rates(channels.to(beams.dtype), beams).item(),
        "beam_real": weights.real.tolist(),
        "beam_imag": weights.imag.tolist(),
        "beam_power_w": weights.abs().square().sum(-1).tolist(),
    }
    report = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
