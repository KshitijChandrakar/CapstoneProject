"""Baseline B3 -- U-Net + FiLM(redshift) + heteroscedastic Gaussian NLL.

This is the model the project is built around. It is the **capacity-matched
control** for the physics models P1-P3: same size, same conditioning on
redshift, same per-pixel error bars -- and no physics at all. Per sir's
slide 6, a physics model that beats a random baseline has proved nothing; it
has to beat this.

Three differences from B2, each with a measured reason:

1. **FiLM on redshift.** Script 05 refitted B0 independently per redshift bin
   and found the z->y relation genuinely changes with redshift -- the Ha and
   [OIII]+Hb windows both show an excess whose bootstrap interval clears zero.
   So redshift carries real information about how to map z to y, and a model
   that ignores it is leaving that on the table. FiLM injects it as a
   per-channel scale and shift rather than as an extra input plane, so one
   scalar cannot be mistaken for image structure.

2. **Two heads: mean and log-variance.** The model says how uncertain it is at
   every pixel.

3. **Gaussian NLL instead of L1.** This scores the error bars, not just the
   prediction. It is what makes the anomaly application possible: a galaxy the
   model gets badly wrong *and* was confident about is a candidate for
   something genuinely unusual -- a merger, an AGN, a calibration fault --
   rather than just a faint object.

Why a warmup matters: trained from scratch on NLL, the variance head learns
that declaring everything noisy minimises the loss long before the mean head
learns to predict anything. The mean then never fits. ``nll_warmup_epochs``
trains the mean on masked L1 first, which is the standard fix and is applied
in :func:`make_b3_loss`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.b2_unet import UNet, check_parameter_count


class FourierFeatures(nn.Module):
    """Random Fourier features of a scalar, plus the raw scalar.

    A bare redshift in [0, 0.4] is a very weak MLP input: the network has to
    build sharpness out of a nearly-constant number. Fourier features give it
    a basis in which "something changes at z = 0.36" is expressible at once,
    which is exactly the structure script 05 says exists.

    The frequencies are fixed at construction (``requires_grad=False``) so they
    are part of the architecture, not learned parameters -- that keeps B3's
    parameter count honest against the physics models.
    """

    def __init__(self, n_features: int = 8, scale: float = 10.0, seed: int = 0) -> None:
        super().__init__()
        self.n_features = int(n_features)
        if self.n_features > 0:
            generator = torch.Generator().manual_seed(seed)
            freqs = torch.randn(self.n_features, generator=generator) * float(scale)
            self.register_buffer("freqs", freqs)
        self.out_dim = 1 + 2 * self.n_features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """(B,) or (B, 1) -> (B, 1 + 2*n_features)."""
        t = t.reshape(-1, 1).float()
        if self.n_features == 0:
            return t
        ang = 2.0 * math.pi * t * self.freqs.reshape(1, -1)
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=1)


class FiLMGenerator(nn.Module):
    """Maps the conditioning scalar(s) to one (gamma, beta) pair per block.

    With ``identity_init`` the final layer starts at zero and gamma is formed
    as ``1 + dgamma``, so an untrained B3 computes exactly what B2 would. Any
    benefit from conditioning is then something the model learned, not an
    artefact of initialisation -- which matters, because B3's whole job is to
    be a control.
    """

    def __init__(
        self,
        block_channels: list[int],
        n_conditions: int = 1,
        hidden: int = 32,
        fourier_features: int = 8,
        fourier_scale: float = 10.0,
        identity_init: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.block_channels = list(block_channels)
        self.total_channels = sum(self.block_channels)
        self.n_conditions = int(n_conditions)

        self.features = FourierFeatures(fourier_features, fourier_scale, seed=seed)
        in_dim = self.features.out_dim * self.n_conditions

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * self.total_channels),
        )
        if identity_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(B,) or (B, n_conditions) -> one (gamma, beta) per block."""
        if cond.dim() == 1:
            cond = cond.unsqueeze(1)
        if cond.shape[1] != self.n_conditions:
            raise ValueError(
                f"Expected {self.n_conditions} conditioning value(s), "
                f"got {cond.shape[1]}"
            )
        feats = torch.cat(
            [self.features(cond[:, i]) for i in range(self.n_conditions)], dim=1
        )
        params = self.mlp(feats)
        dgamma, beta = params.chunk(2, dim=1)

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        at = 0
        for c in self.block_channels:
            # gamma = 1 + dgamma, so zero-init means an identity modulation.
            out.append((1.0 + dgamma[:, at : at + c], beta[:, at : at + c]))
            at += c
        return out


class UNetFiLMNLL(nn.Module):
    """B3: the B2 backbone, modulated by redshift, with two output heads.

    ``forward`` returns ``(mu, logvar)``, both (B, 1, H, W) and both in
    STANDARDISED units. Converting sigma to flux units is a multiplication by
    the target band's stored divisor -- see
    :func:`src.evaluation.uncertainty.sigma_to_physical`.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "group",
        groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
        residual_on_z: bool = False,
        n_conditions: int = 1,
        film_hidden: int = 32,
        fourier_features: int = 8,
        fourier_scale: float = 10.0,
        identity_init: bool = True,
        logvar_min: float = -14.0,
        logvar_max: float = 6.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            norm=norm,
            groups=groups,
            activation=activation,
            dropout=dropout,
            residual_on_z=residual_on_z,
            n_output_heads=2,
        )
        self.film = FiLMGenerator(
            block_channels=self.backbone.film_block_channels(),
            n_conditions=n_conditions,
            hidden=film_hidden,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
            identity_init=identity_init,
            seed=seed,
        )
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        # Start the variance head at log sigma^2 = 0, i.e. sigma = 1 in
        # standardised units -- the spread of the target band itself. A random
        # init here can put sigma at e^3, which stalls the mean's gradient.
        nn.init.zeros_(self.backbone.heads[1].weight)
        nn.init.zeros_(self.backbone.heads[1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 4, H, W) + (B,) redshift -> (mu, logvar)."""
        if cond is None:
            # Unconditioned call: identity modulation, so B3 degrades to B2
            # rather than silently conditioning on zeros.
            films = None
        else:
            films = self.film(cond)
        h = self.backbone.trunk(x, films)
        mu = self.backbone.add_z_residual(self.backbone.heads[0](h), x)
        logvar = self.backbone.heads[1](h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Where B3's parameters sit, so the B2 comparison stays honest."""
        backbone = sum(p.numel() for p in self.backbone.parameters()
                       if p.requires_grad)
        film = sum(p.numel() for p in self.film.parameters() if p.requires_grad)
        head_mu = sum(p.numel() for p in self.backbone.heads[0].parameters())
        head_logvar = sum(p.numel() for p in self.backbone.heads[1].parameters())
        return {
            "total": backbone + film,
            "backbone": backbone,
            "film_generator": film,
            "mean_head": head_mu,
            "logvar_head": head_logvar,
        }


def masked_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full: bool = False,
) -> torch.Tensor:
    """Gaussian negative log-likelihood over valid pixels only.

        0.5 * [ logvar + (target - mu)^2 / exp(logvar) ]   (+ 0.5*log(2*pi))

    The constant is dropped by default (``full=False``) because it shifts every
    model's score by the same amount and cancels in a comparison. It is
    included when the number is reported as a likelihood in its own right.

    NaN target pixels are excluded via ``mask``, which is the rule the 09-10
    meeting set: *"Null values of pixels should not be accounted for to calc.
    the value of target pixel."*
    """
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mu) ** 2 * inv_var)
    if full:
        nll = nll + 0.5 * math.log(2.0 * math.pi)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1_on_mean(
    mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked L1 on the mean head alone -- the warmup objective."""
    return ((mu - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def make_b3_loss(warmup_epochs: int, full_nll: bool = False):
    """A loss callable for :class:`src.training.trainer.Trainer`.

    Returns ``loss_fn(model_out, target, mask, epoch)``: masked L1 on the mean
    for the first ``warmup_epochs`` epochs, then the NLL. The switch is by
    epoch rather than by step so the training history shows a clean break and
    the jump in the loss curve is explainable rather than alarming.
    """

    def loss_fn(
        out: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
        epoch: int = 1,
    ) -> torch.Tensor:
        mu, logvar = out
        if epoch <= warmup_epochs:
            return masked_l1_on_mean(mu, target, mask)
        return masked_gaussian_nll(mu, logvar, target, mask, full=full_nll)

    loss_fn.warmup_epochs = int(warmup_epochs)
    return loss_fn


def build_b3(model_cfg: dict, film_cfg: dict, unc_cfg: dict,
             seed: int = 0) -> UNetFiLMNLL:
    """Construct B3 from its YAML blocks and verify the parameter count."""
    model = UNetFiLMNLL(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        depth=int(model_cfg["depth"]),
        norm=str(model_cfg.get("norm", "group")),
        groups=int(model_cfg.get("groups", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        residual_on_z=bool(model_cfg.get("residual_on_z", False)),
        n_conditions=len(film_cfg.get("conditioning", ["redshift"])),
        film_hidden=int(film_cfg.get("hidden", 32)),
        fourier_features=int(film_cfg.get("fourier_features", 8)),
        fourier_scale=float(film_cfg.get("fourier_scale", 10.0)),
        identity_init=bool(film_cfg.get("identity_init", True)),
        logvar_min=float(unc_cfg.get("logvar_min", -14.0)),
        logvar_max=float(unc_cfg.get("logvar_max", 6.0)),
        seed=seed,
    )
    check_parameter_count(model.count_parameters(), model_cfg, "B3")
    return model
