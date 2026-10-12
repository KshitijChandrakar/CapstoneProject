"""PyTorch Dataset over a redshift-filtered GalaxiesML HDF5 file.

The z<0.4 training split is ~7 GB of images, so the file handle is opened
lazily inside the worker process rather than in ``__init__``. Opening it in
``__init__`` and then forking would share one handle across workers, which
corrupts reads or deadlocks.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import BANDS, INPUT_BAND_IDX, TARGET_BAND_IDX
from src.data.augmentation import AstronomicalAugmentation
from src.data.preprocess import normalize


class GalaxyDataset(Dataset):
    """(g,r,i,z) -> y pairs, standardised, with the raw fluxes kept alongside.

    Each item is a dict:

        input       (4, H, W) float32  standardised g, r, i, z
        target      (1, H, W) float32  standardised y
        input_raw   (4, H, W) float32  unnormalised flux, for photometry
        target_raw  (1, H, W) float32  unnormalised y flux
        mask        (1, H, W) float32  1 where the target pixel is finite
        redshift    scalar float32
        object_id   int
        z_cmodel_mag, y_cmodel_mag  float   (catalogue magnitudes)
        index       int                     position in the file

    The y magnitude is carried only so B0 can fit its constant and so results
    can be cross-checked. It is never part of ``input``.
    """

    def __init__(
        self,
        h5_path: Path | str,
        stats: dict,
        is_train: bool = False,
        augment: bool = False,
        seed: int = 0,
        in_memory: bool = False,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"{self.h5_path} not found. Run scripts/01_prepare_data.py first."
            )
        self.stats = stats
        self.is_train = is_train
        self.in_memory = in_memory
        self._file: h5py.File | None = None
        self._images: np.ndarray | None = None

        self.augmenter = (
            AstronomicalAugmentation(seed=seed) if (augment and is_train) else None
        )
        if augment and not is_train:
            raise ValueError(
                "Augmentation is for the train split only; got is_train=False."
            )

        # Small 1-D columns are cheap, so they are read once and held.
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f["image"].shape[0])
            self.image_shape = tuple(int(v) for v in f["image"].shape[1:])
            self.object_id = self._read(f, "object_id", np.int64)
            self.redshift = self._read(f, "specz_redshift", np.float32)
            self.mags = {
                b: self._read(f, f"{b}_cmodel_mag", np.float32) for b in BANDS
            }
            if self.in_memory:
                self._images = np.asarray(f["image"][:], dtype=np.float32)

        if self.image_shape[0] != len(BANDS):
            raise ValueError(
                f"Expected {len(BANDS)} bands on axis 1, found {self.image_shape[0]} "
                f"in {self.h5_path.name}."
            )

    def _read(self, f: h5py.File, key: str, dtype) -> np.ndarray:
        if key not in f:
            return np.full(self.length, np.nan, dtype=np.float64).astype(dtype)
        return np.asarray(f[key][:]).astype(dtype, copy=False)

    # -- lazy handle -------------------------------------------------------

    def _handle(self) -> h5py.File:
        """Per-process file handle, opened on first use inside the worker."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _raw_image(self, i: int) -> np.ndarray:
        if self._images is not None:
            return self._images[i]
        return np.asarray(self._handle()["image"][i], dtype=np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        raw = self._raw_image(i)                       # (5, H, W) physical flux
        x_raw = raw[list(INPUT_BAND_IDX)]              # (4, H, W)
        y_raw = raw[TARGET_BAND_IDX : TARGET_BAND_IDX + 1]   # (1, H, W)

        x = normalize(x_raw, self.stats, INPUT_BAND_IDX)
        y = normalize(y_raw, self.stats, (TARGET_BAND_IDX,))

        # NaN pixels must not reach the optimiser. They are dropped from the
        # mask and zeroed; in standardised space zero is the band mean.
        mask = np.isfinite(y).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augmenter is not None:
            x, y, mask, x_raw, y_raw = self.augmenter(x, y, mask, x_raw, y_raw)

        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(np.ascontiguousarray(y)),
            "input_raw": torch.from_numpy(np.ascontiguousarray(x_raw)),
            "target_raw": torch.from_numpy(np.ascontiguousarray(y_raw)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "redshift": torch.tensor(float(self.redshift[i]), dtype=torch.float32),
            "object_id": int(self.object_id[i]),
            "z_cmodel_mag": float(self.mags["z"][i]),
            "y_cmodel_mag": float(self.mags["y"][i]),
            "index": int(i),
        }

    # -- cleanup -----------------------------------------------------------

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict:
        """Drop the open handle so the dataset can cross a process boundary."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __repr__(self) -> str:
        where = "in memory" if self.in_memory else "lazy"
        return (f"GalaxyDataset({self.h5_path.name}, n={self.length:,}, "
                f"{where}, augment={self.augmenter is not None})")


def make_dataloader(
    dataset: GalaxyDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """DataLoader with a seeded shuffle, so a run can be reproduced."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
        drop_last=False,
        # Never persist: a lingering worker keeps an HDF5 handle open and
        # deadlocks the next seed's loader on Windows.
        persistent_workers=False,
    )
