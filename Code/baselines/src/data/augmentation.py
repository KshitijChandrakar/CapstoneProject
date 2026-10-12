"""Astronomically valid augmentation: 90-degree rotations and flips only.

These are the dihedral symmetries of the pixel grid. They move pixels around
without changing any pixel's value, so total flux, colours and photometry are
all preserved exactly. Anything that rescales or resamples pixels -- colour
jitter, random crops, shears, added noise -- would corrupt the photometry the
whole project is trying to measure, and is therefore not implemented here.

The same transform is applied to every band of a galaxy, so the bands stay
registered with one another.
"""

from __future__ import annotations

import numpy as np


class AstronomicalAugmentation:
    """Random element of the 8-fold dihedral group, shared across bands.

    Args:
        rot90: allow 90/180/270 degree rotations.
        hflip: allow horizontal flips.
        vflip: allow vertical flips.
        seed: seed for this instance's generator.
    """

    def __init__(
        self,
        rot90: bool = True,
        hflip: bool = True,
        vflip: bool = True,
        seed: int = 0,
    ) -> None:
        self.rot90 = rot90
        self.hflip = hflip
        self.vflip = vflip
        self._rng = np.random.default_rng(seed)

    def sample_params(self) -> dict[str, int | bool]:
        """Draw one transform. Call once per galaxy, apply to every band."""
        return {
            "k": int(self._rng.integers(0, 4)) if self.rot90 else 0,
            "hflip": bool(self.hflip and self._rng.random() < 0.5),
            "vflip": bool(self.vflip and self._rng.random() < 0.5),
        }

    @staticmethod
    def apply(x: np.ndarray, params: dict) -> np.ndarray:
        """Apply a sampled transform to an array of shape (C, H, W)."""
        if x.ndim != 3:
            raise ValueError(f"Expected (C, H, W), got shape {x.shape}")
        if params["k"]:
            x = np.rot90(x, k=params["k"], axes=(1, 2))
        if params["hflip"]:
            x = x[:, :, ::-1]
        if params["vflip"]:
            x = x[:, ::-1, :]
        return np.ascontiguousarray(x)

    def __call__(self, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        """Apply one shared transform to every array passed in.

        Passing the input bands, the target band and the validity mask together
        guarantees they stay aligned.
        """
        params = self.sample_params()
        return tuple(self.apply(a, params) for a in arrays)

    def __repr__(self) -> str:
        return (f"AstronomicalAugmentation(rot90={self.rot90}, "
                f"hflip={self.hflip}, vflip={self.vflip})")
