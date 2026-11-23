"""CSI graph encoder and two-head beamforming model."""
import math
from typing import NamedTuple

import torch
from torch import nn


class _LoRAProjection(nn.Module):
    """Low-rank update for a GPT-2 Conv1D projection."""

    def __init__(self, projection, rank=32, alpha=16, dropout=0.05):
        super().__init__()
        self.projection = projection
        input_size, output_size = projection.weight.shape
        self.down = nn.Linear(input_size, rank, bias=False)
        self.up = nn.Linear(rank, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        # Start at the original projection for both GPT-2 initializations.
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs):
        update = self.up(self.down(self.dropout(inputs)))
        return self.projection(inputs) + self.scale * update


def _gpt2_backbone(kind, checkpoint, local_files_only):
    try:
        from transformers import GPT2Config, GPT2Model
    except ImportError as exc:
        raise RuntimeError("Install transformers to run the GPT-2 backbones.") from exc

    config = GPT2Config()
    # Loading weights and initializing random weights consume different amounts
    # of randomness. Keep the surrounding trainable layers identical per seed.
    with torch.random.fork_rng(devices=[]):
        if kind == "gpt2-random":
            backbone = GPT2Model(config)
        else:
            backbone = GPT2Model.from_pretrained(
                checkpoint, local_files_only=local_files_only, use_safetensors=True
            )
            fields = (
                "n_embd", "n_layer", "n_head", "n_inner", "n_positions", "vocab_size",
                "activation_function", "resid_pdrop", "embd_pdrop", "attn_pdrop",
            )
            if any(getattr(backbone.config, field) != getattr(config, field) for field in fields):
                raise ValueError("The pretrained checkpoint must use the standard GPT-2 small configuration.")

    backbone.requires_grad_(False)
    for block in backbone.h:
        block.attn.c_attn = _LoRAProjection(block.attn.c_attn)
    return backbone


class SwanOutput(NamedTuple):
    positions_normalized: torch.Tensor
    segment_scores: torch.Tensor
    communication_weights: torch.Tensor
    sensing_weights: torch.Tensor


class SwanModel(nn.Module):
    """Map separate communication and sensing CSI tensors to dense predictions."""

    def __init__(
        self,
        antennas: int,
        segments: int,
        backbone: str = "tiny",
        pretrained_model: str = "openai-community/gpt2",
        local_files_only: bool = False,
    ):
        super().__init__()
        if segments < 2 or antennas < segments or antennas % segments:
            raise ValueError("Use at least two segments and equally many antennas per segment.")
        if backbone not in {"tiny", "gpt2-random", "gpt2-pretrained"}:
            raise ValueError(f"Unknown backbone: {backbone}")
        if backbone != "tiny" and antennas > 1024:
            raise ValueError("GPT-2 small supports at most 1024 antenna tokens.")

        self.antennas = antennas
        self.segments = segments
        self.backbone_kind = backbone
        # Nodes are communication users or sensing targets.
        graph_channels = 32
        self.node_input = nn.Linear(5, graph_channels)
        self.node_self = nn.Linear(graph_channels, graph_channels)
        self.node_neighbors = nn.Linear(graph_channels, graph_channels, bias=False)
        width = 64 if backbone == "tiny" else 768
        self.csi_projection = nn.Linear(2, graph_channels)
        self.antenna_projection = nn.Sequential(
            nn.Linear(2 * graph_channels, width), nn.LayerNorm(width)
        )
        self.graph_projection = nn.Linear(graph_channels, width)
        self.deployment_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, antennas + segments),
        )
        self.beamforming_head = nn.Sequential(
            nn.Linear(width + graph_channels + 1, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

        if backbone == "tiny":
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=2 * width,
                dropout=0.1, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.antenna_positions = nn.Parameter(torch.empty(antennas, width))
            nn.init.normal_(self.antenna_positions, std=0.02)
        else:
            self.encoder = _gpt2_backbone(backbone, pretrained_model, local_files_only)

    def _graph_embedding(self, csi, role):
        channel_norm = torch.linalg.vector_norm(csi, dim=-1)
        normalized = csi / channel_norm.clamp_min(1e-20).unsqueeze(-1)
        affinity = (normalized @ normalized.conj().transpose(-2, -1)).abs()
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        channel_phase = torch.angle(csi.sum(dim=-1))
        scale = channel_norm.mean(dim=-1, keepdim=True).clamp_min(1e-20)
        relative_norm = channel_norm / scale
        features = torch.stack(
            (
                relative_norm,
                channel_norm.clamp_min(1e-20).log10(),
                channel_phase.cos(),
                channel_phase.sin(),
                role.expand_as(channel_norm),
            ),
            dim=-1,
        )
        nodes = torch.nn.functional.gelu(self.node_input(features))
        nodes = torch.nn.functional.gelu(
            self.node_self(nodes) + self.node_neighbors(affinity @ nodes)
        )
        return nodes.mean(dim=1)

    def forward(self, communication_csi, sensing_csi) -> SwanOutput:
        """Take complex [B, U, N] and [B, S, N] tensors in antenna order."""
        for csi in (communication_csi, sensing_csi):
            if csi.ndim != 3 or csi.shape[1] < 1 or csi.shape[2] != self.antennas or not csi.is_complex():
                raise ValueError("Each CSI tensor must be complex [batch, entity, antennas].")
        if communication_csi.shape[0] != sensing_csi.shape[0]:
            raise ValueError("Communication and sensing CSI must have the same batch size.")
        users = communication_csi.shape[1]
        targets = sensing_csi.shape[1]
        stream_count = users + targets
        csi = torch.cat((communication_csi, sensing_csi), dim=1).to(dtype=torch.complex64)
        roles = torch.arange(stream_count, device=csi.device).ge(users).to(csi.real.dtype)
        graph = self._graph_embedding(csi, roles)
        scale = torch.linalg.vector_norm(csi, dim=(-2, -1), keepdim=True)
        scale = scale / math.sqrt(stream_count * self.antennas)
        components = torch.view_as_real(csi / scale.clamp_min(1e-20))
        projected = torch.nn.functional.gelu(self.csi_projection(components))
        communication_features = projected[:, :users].mean(dim=1)
        sensing_features = projected[:, users:].mean(dim=1)
        antenna_tokens = self.antenna_projection(torch.cat((communication_features, sensing_features), dim=-1))
        antenna_tokens = antenna_tokens + self.graph_projection(graph).unsqueeze(1)

        if self.backbone_kind == "tiny":
            encoded = self.encoder(antenna_tokens + self.antenna_positions)
        else:
            encoded = self.encoder(inputs_embeds=antenna_tokens, use_cache=False).last_hidden_state
        pooled = encoded.mean(dim=1)
        deployment = self.deployment_head(pooled)
        position_logits, segment_scores = deployment.split((self.antennas, self.segments), dim=-1)
        slots = torch.arange(self.antennas, device=csi.device, dtype=position_logits.dtype)
        positions = (slots + 0.5 + 0.49 * position_logits.tanh()) / self.antennas
        # Retain each stream's features after pooling. Interchanging two users
        # leaves deployment unchanged and interchanges their beam columns.
        antenna_context = encoded.unsqueeze(1).expand(-1, stream_count, -1, -1)
        stream_roles = roles.view(1, stream_count, 1, 1).expand(csi.shape[0], -1, self.antennas, -1)
        beam_inputs = torch.cat((antenna_context, projected, stream_roles), dim=-1)
        beamforming = self.beamforming_head(beam_inputs)
        beamforming = torch.complex(beamforming[..., 0], beamforming[..., 1]).transpose(1, 2)
        communication, sensing = beamforming.split((users, targets), dim=-1)
        return SwanOutput(positions, segment_scores, communication, sensing)
