"""Load the YAML configuration and resolve paths.

Every script starts by calling :func:`load_config`, so there is exactly one
place where settings come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: <repo>/Code/baselines -- the project root for these baselines.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "config"

BANDS: tuple[str, ...] = ("g", "r", "i", "z", "y")
INPUT_BAND_IDX: tuple[int, ...] = (0, 1, 2, 3)
TARGET_BAND_IDX: int = 4

#: Mean cmodel magnitudes over the full (unfiltered) GalaxiesML training split,
#: taken from the team's own EDA notebooks. mean(m_z) - mean(m_y) = +0.171 mag,
#: i.e. y is the brighter band, which is what fixes the sign of B0's scale.
EDA_TRAIN_MEAN_MAG: dict[str, float] = {
    "g": 21.256327,
    "r": 20.246424,
    "i": 19.633747,
    "z": 19.325146,
    "y": 19.153789,
}

#: Galaxy counts in the official GalaxiesML files before any redshift cut.
KNOWN_RAW_SPLIT_SIZES: dict[str, int] = {
    "train": 204573,
    "val": 40914,
    "test": 40914,
}


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or inconsistent."""


@dataclass
class Config:
    """Parsed configuration, with paths already resolved to absolute."""

    raw: dict[str, Any]
    b0: dict[str, Any]
    b1: dict[str, Any]
    b2: dict[str, Any]
    b3: dict[str, Any]
    root: Path = PROJECT_ROOT

    # -- convenient accessors ---------------------------------------------

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def norm(self) -> dict[str, Any]:
        return self.raw["normalization"]

    @property
    def loader(self) -> dict[str, Any]:
        return self.raw["dataloader"]

    @property
    def photometry(self) -> dict[str, Any]:
        return self.raw["photometry"]

    @property
    def seeds(self) -> list[int]:
        return list(self.raw["seeds"])

    @property
    def eps(self) -> float:
        return float(self.norm["eps"])

    def model_config(self, name: str) -> dict[str, Any]:
        """One model's config block by name, with a clear error if absent."""
        table = {"b0": self.b0, "b1": self.b1, "b2": self.b2, "b3": self.b3}
        key = name.lower()
        if key not in table:
            raise ConfigError(f"Unknown model {name!r}; expected one of {list(table)}.")
        block = table[key]
        if not block:
            raise ConfigError(
                f"Config for {key.upper()} is missing. Expected "
                f"config/{'b2_unet' if key == 'b2' else 'b3_unet_film_nll'}.yaml."
            )
        return block

    def path(self, key: str) -> Path:
        """An absolute path from the `paths` block, created if missing."""
        p = self.root / self.paths[key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def raw_file(self, split: str) -> Path:
        """Where the untouched GalaxiesML HDF5 for `split` should be."""
        return self.root / self.paths["raw_dir"] / self.raw["raw_files"][split]

    def processed_file(self, split: str) -> Path:
        """Where the redshift-filtered subset for `split` is written."""
        zmin, zmax = self.dataset["z_min"], self.dataset["z_max"]
        return self.path("processed_dir") / f"{split}_z{zmin}_{zmax}.h5"

    def stats_file(self) -> Path:
        return self.path("processed_dir") / self.norm["stats_filename"]

    def aperture_center(self, image_size: int) -> tuple[float, float]:
        """Aperture centre; `null` in the YAML means the image centre."""
        c = self.dataset.get("aperture_center")
        if c is not None:
            return (float(c[0]), float(c[1]))
        v = float(image_size // 2)
        return (v, v)

    def use_in_memory(self, file_path: Path) -> bool:
        """Whether to hold a split's images in RAM.

        "auto" decides from the file size, so a small pilot loads instantly
        while the ~7 GB z<0.4 training split stays on disk.
        """
        choice = self.loader.get("in_memory", False)
        if isinstance(choice, bool):
            return choice
        if str(choice).lower() != "auto":
            raise ConfigError(
                f"dataloader.in_memory must be true, false or 'auto', got {choice!r}"
            )
        limit_gb = float(self.loader.get("max_in_memory_gb", 2.0))
        try:
            return file_path.stat().st_size <= limit_gb * 1e9
        except OSError:
            return False

    def device(self) -> str:
        choice = str(self.raw.get("hardware", {}).get("device", "auto")).lower()
        if choice != "auto":
            return choice
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def missing_raw_files(self) -> list[Path]:
        """Official files that are not on disk yet."""
        return [self.raw_file(s) for s in ("train", "val", "test")
                if not self.raw_file(s).exists()]

    def missing_processed_files(self) -> list[Path]:
        return [self.processed_file(s) for s in ("train", "val", "test")
                if not self.processed_file(s).exists()]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} did not parse into a mapping.")
    return data


def _read_yaml_optional(path: Path) -> dict[str, Any]:
    """Like :func:`_read_yaml`, but an absent file is an empty mapping."""
    return _read_yaml(path) if path.exists() else {}


def load_config(config_dir: Path | str | None = None) -> Config:
    """Read config.yaml and every model config.

    B2 and B3 are optional so that a checkout predating them, or a sandbox
    config directory that only copies the B0/B1 files, still loads. Scripts
    that need them raise a clear error instead.
    """
    d = Path(config_dir) if config_dir is not None else CONFIG_DIR
    cfg = Config(
        raw=_read_yaml(d / "config.yaml"),
        b0=_read_yaml(d / "b0_baseline.yaml"),
        b1=_read_yaml(d / "b1_baseline.yaml"),
        b2=_read_yaml_optional(d / "b2_unet.yaml"),
        b3=_read_yaml_optional(d / "b3_unet_film_nll.yaml"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Catch configuration mistakes early, with a clear message."""
    ds = cfg.dataset
    if ds["z_min"] >= ds["z_max"]:
        raise ConfigError(f"z_min ({ds['z_min']}) must be below z_max ({ds['z_max']}).")
    if list(ds["bands"]) != list(BANDS):
        raise ConfigError(
            f"dataset.bands must be {list(BANDS)} to match the HDF5 band axis, "
            f"got {ds['bands']}."
        )
    if ds["target_band"] in ds["input_bands"]:
        raise ConfigError(
            f"Target band '{ds['target_band']}' is also listed as an input band. "
            "That would leak the answer into the model."
        )
    if cfg.b1["model"]["n_parameters"] != cfg.b1["model"]["in_channels"] + 1:
        raise ConfigError("B1 must have in_channels + 1 parameters.")
    sign = cfg.b0["scaling"]["sign"]
    if sign not in (1, -1):
        raise ConfigError(f"b0.scaling.sign must be +1 or -1, got {sign}.")
    for name in ("b2", "b3"):
        block = {"b2": cfg.b2, "b3": cfg.b3}[name]
        if not block:
            continue
        m = block["model"]
        if int(m["in_channels"]) != len(INPUT_BAND_IDX):
            raise ConfigError(
                f"{name.upper()}.model.in_channels must be {len(INPUT_BAND_IDX)} "
                f"to match the input bands, got {m['in_channels']}."
            )
        size = int(cfg.dataset["image_size"])
        if size % 2 ** int(m["depth"]) != 0:
            raise ConfigError(
                f"{name.upper()}.model.depth {m['depth']} needs an image size "
                f"divisible by {2 ** int(m['depth'])}, but image_size is {size}."
            )
    if cfg.b3:
        unc = cfg.b3["uncertainty"]
        if unc["logvar_min"] >= unc["logvar_max"]:
            raise ConfigError(
                f"B3.uncertainty.logvar_min ({unc['logvar_min']}) must be below "
                f"logvar_max ({unc['logvar_max']})."
            )
        if int(unc["nll_warmup_epochs"]) >= int(cfg.b3["training"]["epochs"]):
            raise ConfigError(
                f"B3's NLL warmup ({unc['nll_warmup_epochs']} epochs) is as long "
                f"as the whole run ({cfg.b3['training']['epochs']} epochs), so "
                f"the NLL would never be trained."
            )

    handling = cfg.photometry["nonpositive_flux_handling"]
    if handling not in ("exclude", "clamp"):
        raise ConfigError(
            f"photometry.nonpositive_flux_handling must be 'exclude' or 'clamp', "
            f"got {handling!r}."
        )
