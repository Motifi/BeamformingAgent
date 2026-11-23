"""Run the SWAN-ISAC network on complex CSI tensors."""

import argparse
from pathlib import Path

import torch

from model import SwanModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help=".pt dictionary with communication_csi and sensing_csi")
    parser.add_argument("--output", type=Path, help="write the four output tensors to a .pt file")
    parser.add_argument("--backbone", choices=("tiny", "gpt2-random", "gpt2-pretrained"), default="tiny")
    parser.add_argument("--gpt2", default="openai-community/gpt2", help="GPT-2 model ID or local directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--weights", type=Path, help="complete state_dict for this SWAN model")
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(11)
    if args.input:
        tensors = torch.load(args.input, map_location="cpu", weights_only=True)
        communication_csi = tensors["communication_csi"]
        sensing_csi = tensors["sensing_csi"]
    else:
        communication_csi = torch.randn(1, 2, 40, dtype=torch.complex64)
        sensing_csi = torch.randn(1, 1, 40, dtype=torch.complex64)

    model = SwanModel(
        antennas=communication_csi.shape[-1],
        segments=args.segments,
        backbone=args.backbone,
        pretrained_model=args.gpt2,
        local_files_only=args.local_files_only,
    )
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    else:
        print("The task-specific graph encoder and output heads use random weights.")

    model.eval()
    with torch.inference_mode():
        outputs = model(communication_csi, sensing_csi)._asdict()
    for name, tensor in outputs.items():
        print(f"{name:24} {tuple(tensor.shape)}  {tensor.dtype}")
    if args.output:
        torch.save(outputs, args.output)


if __name__ == "__main__":
    main()
