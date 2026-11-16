"""M1 -- a small neural network on catalogue metadata, no pixels at all.

The question M1 answers: **how much of the y band is already in the catalogue?**

Every other model in this project reads 64x64 images. M1 reads 59 numbers --
magnitudes, morphology, redshift -- and never sees a pixel. If a three-layer
MLP on catalogue columns matches what a 1.9M-parameter U-Net achieves on
photometry, then for photometric purposes the images were not carrying much
that the catalogue lacked. If it falls well short, the pixels are doing real
work. Either answer is worth having, and neither is obtainable from the image
ladder alone.

### What it predicts, and the comparison that is and is not fair

M1 predicts ``y_cmodel_mag`` directly -- a catalogue magnitude, from catalogue
inputs.

That is **not the same estimator** as the image models' sigma(mag), which
comes from aperture photometry performed on a predicted *image*. Both are "how
wrong is the y-band brightness, in magnitudes", and both are computed on the
same 17,072 test galaxies, so the numbers are informative side by side. But a
difference of a few per cent between them is estimator, not skill. The results
state this rather than ranking M1 against B2 as though the column meant one
thing.

### The leakage that would make this trivial

``y_cmodel_mag`` is the target, and every other ``y_*`` column -- half-light
radius, Sersic index, ellipticity, isophotal area -- is *measured from the y
image*. Hand those to the model and it is reading the answer off the back of
the page. ``src/data/metadata.py`` excludes the whole ``y_`` prefix and the
tests assert no feature survives it.

### Why a plain MLP

The point is the information content of the catalogue, not architecture
search. A deeper or wider network would blur the finding: if M1 does well, the
interesting claim is "the catalogue already contains this", which is weakened
if it took an elaborate model to extract. Three layers with dropout, trained
on L1 to match the magnitude metric, is enough to answer the question and
small enough that the answer is about the data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class M1MetadataNN(nn.Module):
    """MLP over standardised catalogue features, predicting a magnitude."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.1, target_offset: float = 0.0) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(f"need at least one feature, got {n_features}")

        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.SiLU(), nn.Dropout(dropout)]
            prev = width
        head = nn.Linear(prev, 1)
        # Zero-init the head so the model starts predicting exactly the
        # training median magnitude. Training then has to improve on
        # "guess the middle", which is the honest floor for a regressor.
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(float(target_offset))
        layers.append(head)

        self.net = nn.Sequential(*layers)
        self.n_features = int(n_features)
        self.target_offset = float(target_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        per_layer = {
            f"{i}_{type(m).__name__}": sum(p.numel() for p in m.parameters())
            for i, m in enumerate(self.net) if any(True for _ in m.parameters())
        }
        return {**per_layer, "total": self.count_parameters()}


def build_m1(model_cfg: dict, n_features: int, target_offset: float = 0.0):
    model = M1MetadataNN(
        n_features=n_features,
        hidden=tuple(int(h) for h in model_cfg.get("hidden", (128, 64))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        target_offset=target_offset,
    )
    expected = model_cfg.get("expected_parameters")
    if expected is not None and model.count_parameters() != int(expected):
        raise ValueError(
            f"M1 has {model.count_parameters()} trainable parameters, config "
            f"expects {expected}. The count is quoted in the results, so it is "
            "not allowed to drift silently."
        )
    return model


def catalogue_baseline(z_mag, median_colour: float):
    """The catalogue analogue of B0: y = z magnitude minus the median z-y colour.

    Zero parameters beyond one number measured on the training split. M1 has to
    beat this to have learned anything at all, and reporting it stops a
    mediocre network from looking impressive in isolation.
    """
    import numpy as np

    return np.asarray(z_mag, dtype=np.float64) - float(median_colour)
