"""Pitch-equivariant, instrument-aware multi-task temporal acoustic model."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .music.instruments import INSTRUMENTS


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, (3, 3), padding=(dilation, 1), dilation=(dilation, 1)),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x):
        return F.silu(x + self.net(x))


class EventModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(config.harmonics * 2, config.width, 3, padding=1),
            nn.GroupNorm(4, config.width),
            nn.SiLU(),
            *[ResidualBlock(config.width, 2 ** (i % 4)) for i in range(config.depth)],
        )
        self.heads = nn.ModuleDict(
            {
                k: nn.Conv2d(config.width, len(config.instruments), 1)
                for k in ["onset", "offset", "frame", "velocity"]
            }
        )
        for key, head in self.heads.items():
            nn.init.constant_(head.bias, 0 if key == "velocity" else -2)
        mask = torch.zeros(len(config.instruments), 128)
        for i, name in enumerate(config.instruments):
            inst = INSTRUMENTS[name]
            mask[i, inst.low : inst.high + 1] = 1
        self.register_buffer("pitch_mask", mask)

    def forward(self, x):
        encoded = self.encoder(x)
        return {k: head(encoded).permute(0, 2, 1, 3) for k, head in self.heads.items()}


def targets(piece, frames, config: Config):
    out = {
        k: torch.zeros(frames, len(config.instruments), 128)
        for k in ["onset", "offset", "frame", "velocity"]
    }
    for part in piece.parts:
        if part.instrument not in config.instruments:
            continue
        i = config.instruments.index(part.instrument)
        inst = INSTRUMENTS[part.instrument]
        for n in part.notes:
            if not inst.low <= n.pitch <= inst.high:
                continue
            start = max(0, min(frames - 1, round(n.onset / config.frame_seconds)))
            stop = max(start + 1, min(frames, round(n.offset / config.frame_seconds)))
            out["frame"][start:stop, i, n.pitch] = 1
            out["velocity"][start:stop, i, n.pitch] = n.velocity / 127
            for center, key in [(start, "onset"), (min(stop, frames - 1), "offset")]:
                for t in range(max(0, center - 1), min(frames, center + 2)):
                    out[key][t, i, n.pitch] = max(
                        out[key][t, i, n.pitch], 1.0 if t == center else 0.5
                    )
    return out


def event_loss(prediction, truth, pitch_mask, time_mask=None):
    if time_mask is None:
        time_mask = torch.ones(prediction["frame"].shape[:2], device=pitch_mask.device)
    mask = time_mask[:, :, None, None] * pitch_mask[None, None]
    denominator = mask.sum().clamp_min(1)
    losses = {}
    for key, weight in [("frame", 8.0), ("onset", 40.0), ("offset", 30.0)]:
        loss = F.binary_cross_entropy_with_logits(
            prediction[key],
            truth[key],
            pos_weight=torch.tensor(weight, device=mask.device),
            reduction="none",
        )
        losses[key] = (loss * mask).sum() / denominator
    active = truth["frame"] * mask
    losses["velocity"] = (
        (prediction["velocity"].sigmoid() - truth["velocity"]).square() * active
    ).sum() / active.sum().clamp_min(1)
    # Dice supplements sparse BCE without treating padding as music.
    probabilities = prediction["frame"].sigmoid() * mask
    intersection = (probabilities * truth["frame"]).sum()
    losses["frame_dice"] = 1 - (2 * intersection + 1) / (probabilities.sum() + active.sum() + 1)
    total = (
        losses["frame"]
        + losses["onset"]
        + losses["offset"]
        + 0.2 * losses["velocity"]
        + 0.5 * losses["frame_dice"]
    )
    return total, {k: float(v.detach()) for k, v in losses.items()}
