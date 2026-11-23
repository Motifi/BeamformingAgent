"""Trainable, permutation-equivariant encoding of the D2D interference graph."""

import torch
from torch import nn

from .model import leakage_beams


class GraphBeamformer(nn.Module):
    """Predict positive leakage weights [B, Rx, Tx] and regularizers [B, Tx]."""

    def __init__(
        self, ports: int = 4, width: int = 48, mode: str = "graph_transformer",
        include_reference: bool = True,
    ):
        super().__init__()
        if mode not in {"graph_transformer", "transformer", "graph"}:
            raise ValueError(f"Unknown encoder mode: {mode}")
        self.ports = ports
        self.include_reference = include_reference
        feature_width = 2 * ports + (5 if include_reference else 1)
        self.direct_projection = nn.Sequential(nn.Linear(feature_width, width), nn.GELU())
        self.edge_projection = nn.Sequential(nn.Linear(feature_width, width), nn.GELU())

        self.graph_update = None
        if mode in {"graph_transformer", "graph"}:
            self.graph_update = nn.Sequential(
                nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width)
            )
        self.transformer = None
        if mode in {"graph_transformer", "transformer"}:
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=2 * width,
                dropout=0,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        self.regularizer = nn.Linear(width, 1)
        self.edge_readout = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        for layer in (self.regularizer, self.edge_readout[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode noise-normalized complex CSI [B, Rx, Tx, port]."""
        nodes, edges = self.encode_graph(channels)
        if self.transformer is not None:
            nodes = self.transformer(nodes)
        return self.readout(nodes, edges)

    def encode_graph(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one representation per link and per directed edge."""
        if (
            channels.ndim != 4 or not channels.is_complex()
            or channels.shape[1] != channels.shape[2] or channels.shape[1] == 0
            or channels.shape[-1] != self.ports
        ):
            raise ValueError("Expected complex CSI with shape [B, link, link, port].")

        links = channels.shape[1]
        ids = torch.arange(links, device=channels.device)
        diagonal = torch.eye(links, dtype=torch.bool, device=channels.device)[None, :, :, None]
        energy = channels.abs().square().sum(dim=-1, keepdim=True)
        direction = channels / energy.clamp_min(1e-12).sqrt()
        inputs = [torch.view_as_real(direction).flatten(3), energy.log1p()]
        if self.include_reference:
            with torch.no_grad():
                reference = leakage_beams(channels)
                power = torch.einsum("brtn,btn->brt", channels, reference).abs().square()
                signal = power.diagonal(dim1=1, dim2=2)
                disturbance = 1 + power.masked_fill(diagonal[..., 0], 0).sum(dim=2)
                price = signal / (disturbance * (disturbance + signal))
                inputs.extend((
                    power.log1p().unsqueeze(-1),
                    signal.log1p()[:, :, None, None].expand_as(energy),
                    disturbance.log()[:, :, None, None].expand_as(energy),
                    price.clamp_min(1e-12).log()[:, :, None, None].expand_as(energy),
                ))
        features = torch.cat(inputs, dim=-1).to(self.direct_projection[0].weight.dtype)
        nodes = self.direct_projection(features[:, ids, ids])
        edges = self.edge_projection(features).masked_fill(diagonal, 0)
        if self.graph_update is not None:
            degree = max(links - 1, 1)
            incoming = edges.sum(dim=2) / degree
            outgoing = edges.sum(dim=1) / degree
            nodes = nodes + self.graph_update(torch.cat((nodes, incoming, outgoing), dim=-1))
        return nodes, edges

    def readout(self, nodes: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        links = nodes.shape[1]
        diagonal = torch.eye(links, dtype=torch.bool, device=nodes.device)[None, :, :, None]
        receivers = nodes[:, :, None].expand_as(edges)
        transmitters = nodes[:, None, :].expand_as(edges)
        edge_logits = self.edge_readout(torch.cat((edges, receivers, transmitters), dim=-1))
        edge_logits = edge_logits.masked_fill(diagonal, 0).squeeze(-1)
        regularizer_logits = self.regularizer(nodes).squeeze(-1)
        # Bound the log-weights smoothly to keep the linear solve well conditioned.
        return (4 * (edge_logits / 4).tanh()).exp(), (4 * (regularizer_logits / 4).tanh()).exp()
