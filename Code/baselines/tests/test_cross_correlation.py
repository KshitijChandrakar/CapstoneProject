"""Validate script 07 against a KNOWN injected PSF difference.

The spatial cross-correlation is easy to get subtly wrong and hard to eyeball,
so this test blurs the y band by a known amount and checks that the script
recovers it. It caught two real bugs during development:

  * a missing fftshift, which put zero lag in the corner of the crop window and
    reported every peak at (-max_lag, -max_lag) with a meaningless width;
  * the raw auto-correlation width being dominated by galaxy size, which made a
    1.5 px PSF difference show up as 0.06 px until the galaxy profile was
    cancelled.

It also pins the estimator's accuracy, which is honestly imperfect: against an
injected 1.5 px it returns roughly 0.8-1.8 px. The assertions below encode that
tolerance rather than pretending the estimator is calibrated.

    python tests/test_cross_correlation.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import yaml

from tests.make_fake_data import make_fake_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "_xcorr_test"

#: Extra blur injected into the y band, in pixels.
INJECTED_SIGMA = 1.5
#: Range the estimator is known to return for that input (see module docstring).
ACCEPTABLE = (0.4, 2.6)


def check(label: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def build() -> Path:
    """Fake files with a known extra blur in y, plus a sandbox config."""
    raw = SANDBOX / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = [
        make_fake_split(raw / "5x64x64_training_with_morphology.hdf5",
                        n=300, seed=5, z_low=0.02, z_high=2.0)["path"],
        make_fake_split(raw / "5x64x64_testing_with_morphology.hdf5",
                        n=300, seed=6, z_low=0.02, z_high=2.0,
                        id_offset=5_000_000)["path"],
    ]

    from scipy.ndimage import gaussian_filter
    for fp in paths:
        with h5py.File(fp, "r+") as h:
            img = h["image"][:]
            for k in range(img.shape[0]):
                img[k, 4] = gaussian_filter(img[k, 4], sigma=INJECTED_SIGMA)
            h["image"][...] = img

    cfg_dir = SANDBOX / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    base["paths"] = {
        "raw_dir": "_xcorr_test/data/raw",
        "processed_dir": "_xcorr_test/data/processed",
        "splits_dir": "_xcorr_test/data/splits",
        "checkpoints_dir": "_xcorr_test/results/checkpoints",
        "results_dir": "_xcorr_test/results/metrics",
        "figures_dir": "_xcorr_test/results/figures",
    }
    base["dataset"]["expected_counts"] = {}
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False),
                                         encoding="utf-8")
    for name in ("b0_baseline.yaml", "b1_baseline.yaml"):
        shutil.copy(ROOT / "config" / name, cfg_dir / name)
    return cfg_dir


def main() -> int:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    failures: list[str] = []

    print("=" * 74)
    print(f"CROSS-CORRELATION VALIDATION  (y blurred by sigma={INJECTED_SIGMA} px)")
    print("=" * 74)

    cfg_dir = build()
    script = ROOT / "scripts" / "07_pixel_cross_correlation.py"
    spec = importlib.util.spec_from_file_location("xcorr", script)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [str(script), "--config", str(cfg_dir),
                "--edges", "0.0", "1.0", "2.0", "--max-lag", "16"]
    try:
        spec.loader.exec_module(mod)
        status = mod.main()
    finally:
        sys.argv = old_argv

    check("script ran", status == 0, f"exit code {status}", failures)

    import json
    out = json.loads(
        (SANDBOX / "results" / "metrics" / "pixel_cross_correlation.json")
        .read_text(encoding="utf-8")
    )
    print()

    for e in out["bins"]:
        tag = f"z {e['z_low']:.2f}-{e['z_high']:.2f}"

        # 1. Registration: with no shift injected, the peak must sit at (0, 0).
        bad = [k for k, v in e["pairs"].items()
               if v["peak_lag_x"] != 0 or v["peak_lag_y"] != 0]
        check(f"{tag} peak lag at (0,0)", not bad,
              "all pairs centred" if not bad else f"off-centre: {bad[:3]}",
              failures)

        # 2. Zero lag must reproduce an ordinary correlation: in (-1, 1].
        l0 = [v["lag0"] for v in e["pairs"].values()]
        check(f"{tag} lag0 in range", all(-1.0 <= v <= 1.0001 for v in l0),
              f"min {min(l0):.3f}, max {max(l0):.3f}", failures)

        # 3. A band with itself must have lag0 == 1 by construction.
        autos = [e["pairs"][f"{b}-{b}"]["lag0"] for b in out["bands"]
                 if f"{b}-{b}" in e["pairs"]]
        check(f"{tag} auto-correlation lag0 == 1",
              all(abs(v - 1.0) < 1e-6 for v in autos),
              f"{[round(v, 6) for v in autos]}", failures)

        # 4. The injected blur must be detected, with the right sign.
        ex = e["psf_excess_y_over"]
        check(f"{tag} y detected as blurrier than every input band",
              all(v > 0 for v in ex.values()),
              ", ".join(f"{k}:{v:+.2f}" for k, v in ex.items()), failures)

        lo, hi = ACCEPTABLE
        check(f"{tag} recovered blur within the known tolerance",
              all(lo <= v <= hi for v in ex.values()),
              f"injected {INJECTED_SIGMA}, recovered "
              f"{min(ex.values()):.2f}-{max(ex.values()):.2f} "
              f"(tolerance {lo}-{hi})", failures)

    shutil.rmtree(SANDBOX, ignore_errors=True)

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  [!] {f}")
        print("=" * 74)
        return 1
    print("PASSED -- the cross-correlation recovers a known PSF difference with")
    print("the correct sign, centred peaks, and magnitude inside the documented")
    print("tolerance. The estimator is a lower bound, not a calibrated width.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
