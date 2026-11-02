"""Baseline B2 -- a U-Net over the four input bands.

B0 and B1 are per-pixel by construction: the same rule at every pixel, no
notion of a neighbourhood. Two measurements in this repo say that is the
binding constraint rather than parameter count:

  * script 04's residual figure -- both baselines over-predict galaxy centres
    and under-predict the outskirts, a gradient neither *can* represent;
  * script 07 -- the y band is blurrier than z by ~1.1 px of second-moment
    width, stable across all eight redshift bins. Blur is spatial. No
    per-pixel rescaling of z can manufacture it.

B2 is the smallest architecture that answers both: an encoder-decoder with
skip connections, so a pixel's prediction depends on its surroundings at
several scales.

The width is set by ``base_channels`` and the parameter count is asserted
against the config rather than trusted, because "~2M parameters" is what makes
B2, B3 and the physics models a fair fight.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: z is the 4th of the four input bands (g, r, i, z) -- see src.config.BANDS.
Z_CHANNEL: int = 3


def make_norm(kind: str, channels: int, groups: int) -> nn.Module:
    """Normalisation layer, or a no-op.

    GroupNorm is the default deliberately. Batch statistics are a poor fit
    here: a batch mixes bright and near-blank galaxies whose pixel
    distributions differ by orders of magnitude, and the evaluation loaders are
    unshuffled, so BatchNorm's train/eval mismatch would surface as a metric
    change that has nothing to do with the model.
    """
    kind = kind.lower()
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind != "group":
        raise ValueError(f"norm must be 'group', 'batch' or 'none', got {kind!r}")
    # GroupNorm needs its group count to divide the channel count.
    g = groups
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


def make_activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"activation must be 'silu' or 'relu', got {kind!r}")


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by norm and activation.

    ``forward`` takes an optional (gamma, beta) pair so B3 can modulate the
    features without needing its own copy of the backbone. With ``None`` the
    block behaves exactly as B2's.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = make_norm(norm, out_channels, groups)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = make_norm(norm, out_channels, groups)
        self.act2 = make_activation(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_channels = out_channels

    def forward(
        self, x: torch.Tensor, film: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if film is not None:
            gamma, beta = film
            # (B, C) -> (B, C, 1, 1): one scale and shift per channel, applied
            # after the norm so the conditioning is not washed out by it.
            h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)
        return self.drop(self.act2(h))


class UNet(nn.Module):
    """Encoder-decoder with skip connections, 4 bands in, 1 band out.

    Args:
        in_channels / out_channels: band counts.
        base_channels: width at the top level; each level down doubles it.
        depth: encoder levels below the top.
        residual_on_z: when True the network output is added to the
            standardised z input, so the model starts near B0. Off by default.
        n_output_heads: 1 for B2 (the mean). B3 passes 2 to add log-variance.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 3,
        norm: str = "group",
        groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
        residual_on_z: bool = False,
        n_output_heads: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.residual_on_z = residual_on_z
        self.n_output_heads = n_output_heads

        kw = dict(norm=norm, groups=groups, activation=activation, dropout=dropout)
        widths = [base_channels * 2**i for i in range(depth + 1)]

        # Encoder: a block per level, each followed by a 2x2 max pool.
        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths[:-1]:
            self.encoders.append(ConvBlock(prev, w, **kw))
            prev = w
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(prev, widths[-1], **kw)

        # Decoder: transposed conv up, concatenate the skip, then a block.
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(prev, w, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(w * 2, w, **kw))
            prev = w

        # One 1x1 conv per head.
        self.heads = nn.ModuleList(
            [nn.Conv2d(prev, out_channels, kernel_size=1)
             for _ in range(n_output_heads)]
        )

    # -- the shared forward pass ------------------------------------------

    def trunk(
        self, x: torch.Tensor,
        films: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Encoder, bottleneck and decoder. ``films`` is one pair per block."""
        if x.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input bands, got {x.shape[1]}"
            )
        size = x.shape[-1]
        if size % 2**self.depth != 0:
            raise ValueError(
                f"Image size {size} must be divisible by {2**self.depth} for "
                f"depth {self.depth}, otherwise the skip connections misalign."
            )
        if films is not None and len(films) != self.n_film_blocks():
            raise ValueError(
                f"Expected {self.n_film_blocks()} FiLM pairs, got {len(films)}"
            )

        k = 0
        skips: list[torch.Tensor] = []
        h = x
        for enc in self.encoders:
            h = enc(h, films[k] if films else None)
            k += 1
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h, films[k] if films else None)
        k += 1

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h, films[k] if films else None)
            k += 1
        return h

    def n_film_blocks(self) -> int:
        """How many (gamma, beta) pairs a conditioning generator must supply."""
        return len(self.encoders) + 1 + len(self.decoders)

    def film_block_channels(self) -> list[int]:
        """Channel count of every block, in the order ``trunk`` uses them."""
        return (
            [e.out_channels for e in self.encoders]
            + [self.bottleneck.out_channels]
            + [d.out_channels for d in self.decoders]
        )

    def add_z_residual(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the standardised z band, when ``residual_on_z`` is set."""
        if not self.residual_on_z:
            return out
        return out + x[:, Z_CHANNEL : Z_CHANNEL + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 1, H, W), standardised units."""
        return self.add_z_residual(self.heads[0](self.trunk(x)), x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_b2(model_cfg: dict) -> UNet:
    """Construct B2 from its YAML block and verify the parameter count."""
    model = UNet(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        depth=int(model_cfg["depth"]),
        norm=str(model_cfg.get("norm", "group")),
        groups=int(model_cfg.get("groups", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_on_z=bool(model_cfg.get("residual_on_z", False)),
        n_output_heads=1,
    )
    check_parameter_count(model.count_parameters(), model_cfg, "B2")
    return model


def check_parameter_count(got: int, model_cfg: dict, label: str) -> None:
    """Fail loudly if the architecture drifted away from the agreed size.

    The whole point of B3 is that it is capacity-matched to the physics
    models, so a silent change in width would invalidate the comparison this
    project is built on.
    """
    expected = model_cfg.get("expected_parameters")
    if expected is None:
        return
    tol = float(model_cfg.get("parameter_tolerance", 0.02))
    expected = int(expected)
    if abs(got - expected) > tol * expected:
        raise ValueError(
            f"{label} has {got:,} parameters, but the config expects "
            f"{expected:,} (+/-{tol:.0%}). Either the architecture changed or "
            f"expected_parameters is stale -- update the config deliberately, "
            f"because the capacity match to the physics models depends on it."
        )
