"""Numerical feature encoding and continuous D2D beamforming."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def link_features(channels: torch.Tensor) -> torch.Tensor:
    """Map noise-normalized CSI [batch, Rx, Tx, port] to one token per pair."""
    if (
        channels.ndim != 4 or not channels.is_complex()
        or channels.shape[1] != channels.shape[2] or 0 in channels.shape
    ):
        raise ValueError("Expected complex CSI with shape [batch, link, link, port].")
    energy = channels.abs().square()
    scale = energy.mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
    normalized = torch.view_as_real(channels / scale)
    # Entry [r,t] is the strength of the directed edge t → r at unit beam norm.
    graph = energy.sum(dim=-1).log1p()
    return torch.cat(
        (
            normalized.flatten(2),
            normalized.transpose(1, 2).flatten(2),
            graph,
            graph.transpose(1, 2),
        ),
        dim=-1,
    ).float()


def unit_beams(beams: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(beams, dim=-1, keepdim=True)
    fallback = torch.zeros_like(beams)
    fallback[..., 0] = 1
    return torch.where(norms > 0, beams / norms.clamp_min(torch.finfo(norms.dtype).tiny), fallback)


def spectral_efficiency(channels: torch.Tensor, beams: torch.Tensor) -> torch.Tensor:
    """Compute per-link rates with noise normalized to one."""
    received = torch.einsum("brtn,btn->brt", channels, beams).abs().square()
    links = received.shape[-1]
    diagonal = torch.eye(links, device=received.device, dtype=torch.bool)
    signal = received.diagonal(dim1=-2, dim2=-1)
    interference = received.masked_fill(diagonal, 0).sum(dim=-1)
    return torch.log1p(signal / (1 + interference)) / math.log(2)


def conventional_beams(channels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Construct MRT and interference-aware regularized linear baselines."""
    direct = channels.diagonal(dim1=1, dim2=2).transpose(1, 2).conj()
    return {"mrt": unit_beams(direct), "regularized": leakage_beams(channels)}


def leakage_beams(
    channels: torch.Tensor,
    weights: torch.Tensor | None = None,
    regularization: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Solve a weighted leakage system per transmitter and enforce unit power."""
    links, ports = channels.shape[-2:]
    diagonal = torch.eye(links, device=channels.device, dtype=torch.bool)
    direct = channels.diagonal(dim1=1, dim2=2).transpose(1, 2).conj()
    outgoing = channels.transpose(1, 2).masked_fill(diagonal[..., None], 0).to(torch.complex128)
    weighted = outgoing if weights is None else weights.transpose(1, 2)[..., None] * outgoing
    covariance = outgoing.mH @ weighted
    identity = torch.eye(ports, dtype=outgoing.dtype, device=channels.device)
    regularization = torch.as_tensor(regularization, device=channels.device, dtype=covariance.real.dtype)
    system = covariance + regularization[..., None, None] * identity
    solution = torch.linalg.solve(system, direct.to(outgoing.dtype).unsqueeze(-1))
    return unit_beams(solution.squeeze(-1).to(channels.dtype))


class LinkEncoder(nn.Module):
    """Freeze the numerical projection and Transformer for the first probe."""

    def __init__(self, features: int, backbone: str, model_path: str, local_only: bool):
        super().__init__()
        self.kind = backbone
        if backbone == "deepseek":
            from transformers import AutoModel

            self.transformer, loading = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=local_only,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                output_loading_info=True,
            )
            if loading["missing_keys"]:
                raise ValueError(f"Checkpoint is missing encoder weights: {loading['missing_keys']}")
            self.width = self.transformer.config.hidden_size
            self.input_scale = float(self.transformer.get_input_embeddings().weight[:1024].detach().std())
        else:
            self.width = 64
            self.input_scale = 1.0
            layer = nn.TransformerEncoderLayer(
                d_model=self.width, nhead=4, dim_feedforward=128, dropout=0, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.projection = nn.Linear(features, self.width)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self, features: torch.Tensor, *, exchangeable_links: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = F.layer_norm(self.projection(features), (self.width,)) * self.input_scale
        if self.kind == "deepseek":
            batch, links, _ = tokens.shape
            # The additive 4D mask allows every pair to use every link token.
            mask = tokens.new_zeros((batch, 1, links, links))
            position_ids = torch.zeros((batch, links), dtype=torch.long, device=tokens.device) if exchangeable_links else None
            hidden = self.transformer(
                inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids, use_cache=False
            )
            hidden = hidden.last_hidden_state
        else:
            hidden = self.transformer(tokens)
        return F.layer_norm(tokens, (self.width,)), F.layer_norm(hidden.float(), (self.width,))


class BeamHead(nn.Module):
    def __init__(self, width: int, ports: int):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2 * ports))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        real_imag = self.readout(hidden).unflatten(-1, (-1, 2))
        return unit_beams(torch.view_as_complex(real_imag.contiguous()))
