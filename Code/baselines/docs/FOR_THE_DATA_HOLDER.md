# Running the B0 / B1 baselines on the machine that holds the data

You have the three official GalaxiesML files. The rest of the team does not,
and they are ~23 GB, so moving them is painful. There are two ways to get the
team real numbers — **Option A is much easier for everyone.**

Everything below needs only a normal Python environment. If you ran the EDA
notebooks, you already have most of it.

---

## Option A — run the pipeline here, send back a few MB of results ✅

You run four commands; the team gets the finished table and figures. No large
transfer at all.

### 1. Get the code

Copy the `Code/baselines/` folder from the team repo
(`https://github.com/KshitijChandrakar/CapstoneProject`) onto this machine.

### 2. Install

```bash
cd baselines
python -m venv .venv
.venv\Scripts\activate            # Windows.  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python **3.10–3.12** (PyTorch has no 3.13/3.14 wheels yet).

### 3. Point it at the data

Put the three files in `baselines/data/raw/`, or edit `config/config.yaml`:

```yaml
raw_files:
  train: "5x64x64_training_with_morphology.hdf5"
  val:   "5x64x64_validation_with_morphology.hdf5"
  test:  "5x64x64_testing_with_morphology.hdf5"
```

If they live elsewhere, change `paths.raw_dir` to that folder.

### 4. Run

```bash
python scripts/01_prepare_data.py          # filter 0 <= z < 0.4, fit global stats
python scripts/02_run_b0.py                # B0
python scripts/03_train_b1.py              # B1, 3 seeds
python scripts/04_benchmark_comparison.py  # table + figures
```

Step 1 is the slow one — it streams ~17 GB and writes the filtered subsets
(~10 GB total), so allow disk space and some time. Steps 2–4 are quick.

Step 1 should report **84,911 train / 17,146 val / 17,072 test** after the
redshift cut. It prints a warning if the counts disagree, which would mean
something is different about the files.

### 4b. One more run the team needs (quick, no training)

A teammate asked whether "y is just a scaled copy of z" still holds at higher
redshift. B0 has zero parameters, so this is cheap to answer:

```powershell
python scripts/05_redshift_sweep.py
```

It refits B0 independently in redshift bins across the whole catalogue
(0 to 4, not just z < 0.4) and plots its accuracy against the redshifts where
spectral features are predicted to break the assumption. It makes **one**
streaming pass over the test images rather than re-filtering per bin, so it
takes a few minutes, not hours.

Outputs `redshift_sweep.json`, `redshift_sweep.md` and
`redshift_sweep.png` into `results/`.

### 4c. Band-to-band pixel correlations (also quick)

A teammate also asked how much each band's pixels affect the other bands, per
redshift bin:

```powershell
python scripts/06_band_correlations.py
```

One streaming pass, no training. Outputs `band_correlations.json`,
`band_correlations.md` and three figures into `results/`.

### 4d. Pixel-to-pixel band cross-correlation (also quick)

The last thing the team asked for — how a pixel in one band affects the pixels
*around* it in another band, which reveals whether the bands have different
PSFs:

```powershell
python scripts/07_pixel_cross_correlation.py
```

One streaming pass, no training. Outputs `pixel_cross_correlation.json`,
`pixel_cross_correlation.md` and two figures into `results/`.

### 4e. The two spatial models (B2 and B3) -- only if this machine has a GPU

These are the first models big enough to care about hardware: ~1.9M parameters
each, three seeds, over 84,911 training galaxies. On a GPU that is tens of
minutes; on CPU it is hours per seed. **If this machine has no GPU, skip 4e and
4f and say so** -- we will run them elsewhere, and everything above is still
the deliverable.

First check that torch can actually see the GPU, because our own
`requirements.txt` installs the CPU-only wheel:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints a version ending `+cpu`, or `False`, the models would fall back
to CPU and take hours. Build a second environment rather than changing the one
that produced the B0/B1 numbers:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It should print `2.14.1+cu130 True`. If it prints anything ending `+cpu`, or
`False`, **stop and tell us** -- do not go looking for the right wheel. That is
our mistake to fix, and it has already happened once: the CPU and GPU indexes
publish the same version number, so pip can quietly install the CPU build.
A quick sanity check is size: a real CUDA environment measured 3.4 GB, a CPU
one about 1 GB.

`is_available()` is a flag, not a measurement. If you want to be sure the card
is really being used rather than silently fallen back on, check that its
compute capability is in the wheel's compiled list and that a matmul actually
runs on it:

```powershell
python -c "import torch; print('cuda build:', torch.version.cuda); print('capability:', 'sm_%d%d' % torch.cuda.get_device_capability(0)); print('arch list :', torch.cuda.get_arch_list()); print('matmul OK :', bool((torch.randn(512,512,device='cuda') @ torch.randn(512,512,device='cuda')).isfinite().all()))"
```

The capability must appear in the arch list. CUDA 13 wheels compile for sm_75
and newer, so a pre-Turing card will print `True` for `is_available()` and then
fail or crawl.

Once that prints `True`, run the models from that environment:

```powershell
python scripts/08_train_b2.py
python scripts/09_train_b3.py
python scripts/10_benchmark_all.py
```

Short on time? `--seed 42` on either script runs one seed instead of three, and
is worth sending on its own.

Outputs `b2_metrics.json`, `b3_metrics.json`, `ladder_table.md` / `.tex`,
`b3_anomaly_candidates.csv` and two figures into `results/`.

### 4f. What to tell us about the B3 run

Three lines from the console, because they decide whether the numbers are
usable at all:

1. the **calibration** line -- the `(y-mu)/sigma` spread. 1.0 is honest, above
   1 means the model is over-confident;
2. the **FiLM response** -- if it prints a warning that the conditioning barely
   moves the prediction, that is a result we need to know;
3. whether either run **stopped early**, and at which epoch.

If a run fails with a CUDA out-of-memory error, lower `dataloader.batch_size`
in `config/config.yaml` (64 -> 32 -> 16) and tell us which value worked.

### 5. Send back

Zip and send **`results/`** — the metrics JSON, the table, and the figures.
A few MB. That is everything the team needs.

In PowerShell, from the `baselines` folder:

```powershell
Compress-Archive -Path results -DestinationPath results_official_z0_0.4.zip -Force
```

That one file is the deliverable.

If `03_train_b1.py` is slow, add `--seed 42` to run one seed first and send
that; the other seeds can follow.

---

## Option B — send a small extract instead

If you would rather the team ran it, you can send a cut-down file instead of
23 GB:

```bash
python scripts/make_sample.py "5x64x64_training_with_morphology.hdf5" \
    --zmax 0.4 --n 5000
```

That writes `sample_z0.4_5000.h5` at roughly 400 MB, holding only the galaxies
needed. It prints an `object_id hash` at the end — **send that line too**, so
the team can confirm the file arrived intact.

Repeat for the validation and testing files if you can; separate extracts keep
GalaxiesML's official train/val/test partition, which a single mixed file does
not.

---

## Sample provenance — already settled, kept for the record

The 2,000-galaxy pilot sample (`sample_z0.2_2000.h5`) was checked with
`scripts/check_sample_provenance.py` and came back **100% from the training
split** (0 in validation, 0 in testing). No held-out data was used during
development.

An earlier note in this file claimed the columns `x_coord_x`, `x_coord_y`,
`y_coord_x`, `y_coord_y` appear only in the testing file. **That was wrong** —
all four are present in all three splits. The earlier EDA notebooks only listed
1-D datasets, and these are 2-D, so they were filtered out of one listing and
not the other. Nothing was ever inferred correctly from that difference.

---

## If something goes wrong

- **`FileNotFoundError` on a raw file** — check `paths.raw_dir` and the three
  filenames in `config/config.yaml`.
- **Out of memory** — lower `normalization.chunk_size` (default 500) in
  `config/config.yaml`. Nothing ever loads a whole file.
- **Out of disk** — step 1 writes ~10 GB of filtered subsets. Point
  `paths.processed_dir` at a drive with room.
- **Want to check the code first** — `python tests/test_pipeline_fake.py`
  builds fake files and runs everything end to end. No real data needed.

Thank you — this unblocks the whole mid-sem presentation.
