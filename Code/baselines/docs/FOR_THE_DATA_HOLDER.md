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

## One quick question the team needs answered

The sample already shared (`sample_z0.2_2000.h5`) has 2,000 galaxies all at
z ≈ 0.2 (range 0.19686–0.20313), and it carries the columns `x_coord_x`,
`x_coord_y`, `y_coord_x`, `y_coord_y`, which appear in the **testing** file but
not in training or validation.

The team needs to know whether those 2,000 galaxies come from one split or are
mixed across all three, because GalaxiesML's test split must stay untouched
during model tuning.

This script answers it exactly:

```bash
python scripts/check_sample_provenance.py sample_z0.2_2000.h5 --raw-dir .
```

It prints how many of the 2,000 galaxies live in each official split and writes
`sample_provenance.json`. **Send that file back** — it takes under a minute and
settles the question.

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
