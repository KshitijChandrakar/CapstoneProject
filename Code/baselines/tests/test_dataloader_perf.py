"""Checks on the two dataloader fixes: chunk layout and worker seeding.

    python tests/test_dataloader_perf.py

Neither fix changes a result, which is exactly why they need testing: a silent
regression in either would be invisible in the loss curve. The rechunk must
preserve every pixel including NaNs, and the per-worker augmentation seeding
must actually produce different streams rather than appearing to.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from src.data.augmentation import AstronomicalAugmentation
from src.data.dataset import _make_worker_init

PASS, FAIL = 0, 0


def check(label: str, got, want=True, tol: float | None = None) -> None:
    global PASS, FAIL
    if tol is not None:
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


# --------------------------------------------------------------- chunk layout
def test_preprocess_writes_one_galaxy_chunks(tmp: Path) -> None:
    """A freshly prepared file must not reintroduce the 32-galaxy chunk."""
    print("\nchunk layout as written by preprocess.py")
    import inspect

    from src.data import preprocess

    src = inspect.getsource(preprocess)
    check("create_dataset asks for chunks=(1,) + img_shape",
          "chunks=(1,) + img_shape" in src)
    check("the old 32-galaxy chunk is gone",
          "chunks=(min(32, n_keep),)" not in src)


def test_rechunk_is_lossless(tmp: Path) -> None:
    """The rechunk script must preserve pixels, NaNs, columns and attributes."""
    print("\nrechunk preserves the data exactly")
    rng = np.random.default_rng(7)
    n, shape = 70, (5, 8, 8)

    images = rng.normal(size=(n,) + shape).astype(np.float32)
    # NaNs are real in this data and are the easiest thing for a copy to lose,
    # since NaN != NaN under ordinary comparison.
    images[3, 4, 2, 2] = np.nan
    images[n - 1, 0, 0, 0] = np.nan
    ids = np.arange(n, dtype=np.int64) + 1000
    z = rng.uniform(0.0, 0.4, size=n).astype(np.float32)

    path = tmp / "processed_test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=images, chunks=(32,) + shape)
        f.create_dataset("object_id", data=ids)
        f.create_dataset("specz_redshift", data=z)
        f.attrs["z_max"] = 0.4
        f.attrs["n_kept"] = n

    with h5py.File(path, "r") as f:
        check("fixture starts with a 32-galaxy chunk", f["image"].chunks[0], 32)

    from scripts import _load_rechunk  # noqa: F401  (shim defined below)

    ok = _load_rechunk().rechunk(path, keep_original=False)
    check("rechunk reported success", ok)

    with h5py.File(path, "r") as f:
        check("chunk is now one galaxy", f["image"].chunks[0], 1)
        check("row count unchanged", int(f["image"].shape[0]), n)
        check("pixels identical, NaNs included",
              np.array_equal(np.asarray(f["image"]), images, equal_nan=True))
        check("NaN at [3,4,2,2] survived as NaN",
              bool(np.isnan(f["image"][3, 4, 2, 2])))
        check("object_id column identical",
              np.array_equal(np.asarray(f["object_id"]), ids))
        check("redshift column identical",
              np.array_equal(np.asarray(f["specz_redshift"]), z))
        check("original attributes carried over", float(f.attrs["z_max"]), 0.4)
        check("provenance attribute added", "rechunked_from" in f.attrs)


def test_rechunk_refuses_to_destroy_on_mismatch(tmp: Path) -> None:
    """A file it cannot read must leave the original in place."""
    print("\nrechunk fails safe")
    path = tmp / "not_hdf5.h5"
    path.write_bytes(b"this is not an HDF5 file")
    info = _load_rechunk().describe(path)
    check("describe() returns None rather than raising", info, None)
    check("the unreadable file is still there", path.exists())


# ------------------------------------------------------- worker augmentation
def test_augmenter_reseed_changes_the_stream() -> None:
    print("\naugmenter reseeding")
    a = AstronomicalAugmentation(seed=0)
    first = [a.sample_params() for _ in range(8)]

    b = AstronomicalAugmentation(seed=0)
    same = [b.sample_params() for _ in range(8)]
    check("same seed reproduces the stream", first == same)

    b.reseed([0, 1])
    after = [b.sample_params() for _ in range(8)]
    check("reseed produces a different stream", first != after)

    c = AstronomicalAugmentation(seed=0)
    c.reseed([0, 1])
    check("reseed is deterministic", [c.sample_params() for _ in range(8)],
          after)


def test_worker_init_gives_each_worker_its_own_stream() -> None:
    """The bug this guards against: identical augmentation in every worker."""
    print("\nper-worker augmentation streams")

    class FakeDataset:
        def __init__(self) -> None:
            self.augmenter = AstronomicalAugmentation(seed=42)

    class FakeInfo:
        def __init__(self, dataset, num_workers: int = 4) -> None:
            self.dataset = dataset
            self.num_workers = num_workers

    init = _make_worker_init(42)
    streams = []
    for worker_id in range(4):
        ds = FakeDataset()
        info = FakeInfo(ds)
        original = torch.utils.data.get_worker_info
        torch.utils.data.get_worker_info = lambda _i=info: _i
        try:
            init(worker_id)
        finally:
            torch.utils.data.get_worker_info = original
        streams.append([ds.augmenter.sample_params() for _ in range(12)])

    unique = {repr(s) for s in streams}
    check("4 workers produce 4 distinct streams", len(unique), 4)

    # Without the fix every worker would match an un-reseeded augmenter.
    plain = AstronomicalAugmentation(seed=42)
    baseline = [plain.sample_params() for _ in range(12)]
    check("no worker merely repeats the unseeded stream",
          all(s != baseline for s in streams))

    init_again = _make_worker_init(42)
    ds = FakeDataset()
    info = FakeInfo(ds)
    original = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: info
    try:
        init_again(0)
    finally:
        torch.utils.data.get_worker_info = original
    check("worker 0 is reproducible across runs",
          [ds.augmenter.sample_params() for _ in range(12)], streams[0])


def test_seed_entropy_lists_are_not_a_safe_way_to_derive_streams() -> None:
    """The trap the spawn-based implementation exists to avoid.

    NumPy strips trailing zeros from seed entropy, so default_rng([s, 0]) is
    the same generator as default_rng(s). Deriving a worker stream as
    [seed, worker_id] therefore hands worker 0 the un-workered stream. This
    test records the behaviour so nobody "simplifies" the implementation back
    into it.
    """
    print("\nnumpy seed-entropy trailing zeros")
    plain = list(np.random.default_rng(42).integers(0, 4, 6))
    as_list = list(np.random.default_rng([42, 0]).integers(0, 4, 6))
    check("default_rng([42, 0]) really does equal default_rng(42)",
          as_list == plain)

    import inspect

    from src.data import dataset as ds_mod

    src = inspect.getsource(ds_mod._make_worker_init)
    check("worker_init uses SeedSequence.spawn", "spawn(" in src)
    check("worker_init does not pass a raw [seed, worker_id] list",
          "default_rng([seed, worker_id])" not in src)


def test_worker_init_is_a_noop_outside_a_worker() -> None:
    print("\nworker_init outside a worker process")
    init = _make_worker_init(1)
    try:
        init(0)                      # get_worker_info() is None in the parent
        check("returns quietly in the main process", True)
    except Exception as exc:        # noqa: BLE001
        check(f"returns quietly in the main process (raised {exc!r})", False)


def test_dataloader_wires_worker_init_only_when_needed() -> None:
    print("\nmake_dataloader wiring")
    import inspect

    from src.data import dataset as ds_mod

    src = inspect.getsource(ds_mod.make_dataloader)
    check("worker_init_fn passed when num_workers > 0",
          "_make_worker_init(seed) if num_workers > 0 else None" in src)
    check("persistent_workers still disabled",
          "persistent_workers=False" in src)
    handle = inspect.getsource(ds_mod.GalaxyDataset._handle)
    check("chunk cache raised above h5py's 1 MB default",
          "rdcc_nbytes" in handle)


# ----------------------------------------------------------------- the shim
def _load_rechunk():
    """Import 11_rechunk_processed.py, whose name is not an identifier."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "11_rechunk_processed.py"
    spec = importlib.util.spec_from_file_location("rechunk_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rechunk_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import shutil
    import tempfile

    # The test module imports a shim from "scripts"; make that resolvable.
    sys.modules.setdefault("scripts", type(sys)("scripts"))
    sys.modules["scripts"]._load_rechunk = _load_rechunk

    tmp = Path(tempfile.mkdtemp(prefix="dl_perf_"))
    print("=" * 70)
    print("DATALOADER FIXES -- chunk layout and worker seeding")
    print("=" * 70)
    try:
        test_preprocess_writes_one_galaxy_chunks(tmp)
        test_rechunk_is_lossless(tmp)
        test_rechunk_refuses_to_destroy_on_mismatch(tmp)
        test_augmenter_reseed_changes_the_stream()
        test_worker_init_gives_each_worker_its_own_stream()
        test_seed_entropy_lists_are_not_a_safe_way_to_derive_streams()
        test_worker_init_is_a_noop_outside_a_worker()
        test_dataloader_wires_worker_init_only_when_needed()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
