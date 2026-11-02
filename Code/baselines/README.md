# Baselines B0 & B1 — predicting the HSC *y* band from *g, r, i, z*

> ## ✅ Results are in — [`docs/results_official/`](docs/results_official/)
>
> Run on the three official GalaxiesML splits (**84,911 / 17,146 / 17,072**,
> `0.0 ≤ z < 0.4`). Test split, physical flux units:
>
> | Model | Params | MAE ↓ | PSNR ↑ | SSIM ↑ | σ(mag) ↓ | Outlier % ↓ |
> |---|---|---|---|---|---|---|
> | B0 (Copy-z) | 0 | 0.3505 | 37.431 | 0.8694 | 0.0431 | **0.504** |
> | B1 (Linear 1×1) | 5 | **0.3276** | **37.991** | **0.8785** | **0.0423** | 1.217 |
>
> - **B1 beats B0 on six of seven metrics** (MAE −6.5 %), but doubles the
>   catastrophic-outlier rate.
> - **z dominates.** Standardised weights order `z > i > r > g`, matching the
>   spec — but only z's lead is robust. r and i sit close together and swap
>   depending on the estimator (the physical weights and the partial-correlation
>   analysis both give `z > r > i > g`). Claim "z dominates", not the sequence.
> - **B0's formula sign in the spec is wrong.** Measured `F_y/F_z = 1.1656`;
>   the `+` sign gives 1.1790 (error 0.013), the spec's `−` gives 0.8481
>   (error 0.317).
> - **B1 converged** — 26,540 optimiser steps, 0.0127 from the solved optimum.

Phase 1 of the physics-informed band-prediction project. These baselines set
the floor that every later model (P1–P3 physics models, ViT) has to beat.

| | What it does | Learnable parameters | Sees shapes? | Error bars? |
|---|---|---|---|---|
| **B0** (Copy-z) | Copies the *z* image and multiplies it by one number | **0** | no | no |
| **B1** (Linear 1×1) | A weighted sum of the four input bands, same weights at every pixel | **5** | no | no |
| **B2** (U-Net) | Encoder–decoder over the four bands, so a pixel depends on its surroundings | **1,928,705** | yes | no |
| **B3** (U-Net+FiLM+NLL) | B2 conditioned on redshift, predicting a mean *and* a per-pixel sigma | **1,975,778** | yes | yes |

**B0 and B1 have official numbers; B2 and B3 are implemented and tested but
have not been run on the real data yet** — they need the GPU machine. See
*Running B2 and B3* below.

> *"If a complex physics model beats a random baseline, it proves nothing. If it
> beats a carefully capacity-matched control, we have discovered new science."*

---

## Quick start

```bash
# 1. environment  (Python 3.10-3.12; PyTorch has no 3.13/3.14 wheels yet)
python -m venv .venv
.venv\Scripts\activate              # Windows.  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. put the three official GalaxiesML files in data/raw/
#    (https://zenodo.org/records/13878122)

# 3. run the four steps
python scripts/01_prepare_data.py          # filter to 0 <= z < 0.4, fit global stats
python scripts/02_run_b0.py                # B0 on val + test
python scripts/03_train_b1.py              # B1, 3 seeds
python scripts/04_benchmark_comparison.py  # table + figures

# 4. the spatial models (needs a GPU to be quick -- see below)
python scripts/08_train_b2.py              # B2 U-Net, 3 seeds
python scripts/09_train_b3.py              # B3 U-Net+FiLM+NLL, 3 seeds
python scripts/10_benchmark_all.py         # the full ladder table
```

**No data yet?** Check the code works first:

```bash
python tests/test_pipeline_fake.py   # builds fake files, runs everything end to end
```

**Only have one extracted sample instead of the three official files?**

```bash
python scripts/01b_prepare_pilot.py path/to/sample.h5   # 70/15/15 re-split
python scripts/02_run_b0.py && python scripts/03_train_b1.py
python scripts/04_benchmark_comparison.py
```
That is a **pilot**, not the deliverable — see *Pilot mode* below.

---

## Repository layout

```
baselines/
├── config/
│   ├── config.yaml              # paths, redshift window, splits, seeds, photometry
│   ├── b0_baseline.yaml         # B0: the scale formula and its sign
│   ├── b1_baseline.yaml         # B1: architecture and hyperparameters
│   ├── b2_unet.yaml             # B2: width, depth, early stopping
│   └── b3_unet_film_nll.yaml    # B3: FiLM, the variance head, the NLL warmup
├── data/
│   ├── raw/                     # the three official GalaxiesML HDF5 files
│   ├── processed/               # redshift-filtered subsets + global_stats.json
│   └── splits/                  # audit reports
├── src/
│   ├── config.py                # YAML loader and validation
│   ├── data/
│   │   ├── preprocess.py        # redshift filtering, streaming global statistics
│   │   ├── dataset.py           # lazy HDF5 PyTorch Dataset
│   │   └── augmentation.py      # coordinated rotations and flips
│   ├── models/
│   │   ├── b0_copy_z.py         # B0 + the empirical sign check
│   │   ├── b1_linear.py         # B1 + the directly-solved reference optimum
│   │   ├── b2_unet.py           # the U-Net backbone, shared with B3
│   │   └── b3_unet_film_nll.py  # FiLM conditioning + the heteroscedastic NLL
│   ├── training/
│   │   ├── trainer.py           # training loop, validation, checkpointing
│   │   └── logger.py            # console + CSV logging
│   └── evaluation/
│       ├── metrics.py           # MAE, RMSE, PSNR, SSIM
│       ├── photometry.py        # aperture flux, NMAD, Δm, colour error
│       ├── uncertainty.py       # NLL, calibration, sharpness, anomaly ranking
│       └── evaluator.py         # unified evaluation across models
├── scripts/                     # 01 prepare | 01b pilot | 02 B0 | 03 B1 | 04 compare
│                               # 05 redshift sweep | 06 band corr | 07 pixel xcorr
│                               # 08 B2 | 09 B3 | 10 full ladder
├── tests/                       # fake-data end-to-end tests
└── docs/EDA_FINDINGS.md         # what the team's EDA notebooks tell us
```

## What comes out, in `results/`

| File | Contents |
|---|---|
| `metrics/results_table.md` / `.tex` | The slide-10 benchmark table, plus the discussion answers |
| `metrics/b0_metrics.json` | B0 on val and test, and the sign verification |
| `metrics/b1_metrics.json` | B1 per seed and summarised, weights, convergence |
| `checkpoints/b1_seed*.pt` | Best-validation weights per seed |
| `figures/dmag_histogram.png` | Magnitude residuals, B0 vs B1 |
| `figures/flux_scatter.png` | Predicted vs true aperture flux, log axes |
| `figures/b1_weights_*.png` | Learned weights with seed error bars |
| `figures/galaxy_grid.png` | Example galaxies: truth, predictions, residuals |
| `metrics/b2_metrics.json` | B2 per seed and summarised |
| `metrics/b3_metrics.json` | B3 per seed, plus calibration and the FiLM response |
| `metrics/ladder_table.md` / `.tex` | B0 → B1 → B2 → B3, and what each rung bought |
| `metrics/b3_anomaly_candidates.csv` | Galaxies B3 got confidently wrong, ranked |
| `figures/b3_calibration.png` | Nominal vs empirical coverage, against the diagonal |
| `figures/ladder_mae_vs_params.png` | MAE against parameter count, log scale |

---

## The rules this code follows

**Global per-band standardisation, train split only.**
`x_norm = (x - mean) / (std + 1e-10)`, saved to `global_stats.json`.
Never per-image min-max — that destroys inter-galaxy colour and absolute flux,
which is exactly the information the photometry depends on. `denormalize()`
inverts it exactly, and the round-trip is asserted before any metric is
computed.

**Negative pixels are kept.** These are sky-subtracted fluxes; negative values
are real noise, not errors.

**Augmentation is rotations and flips only**, applied identically to all five
bands, train split only. These move pixels without changing their values, so
photometry survives. No colour jitter, crops, shears or noise injection.

**No target leakage.** The *y* image and `y_cmodel_mag` never enter a model as
input. The y magnitude is used in exactly two places: fitting B0's single
constant on the train split, and evaluation.

**The test split is never used for tuning.** Splits are verified disjoint by
`object_id` and the check is an assertion, not a printout.

**NaN pixels are masked** out of the loss and every metric.

**Seeds** `[42, 123, 2026]` cover python, numpy and torch.

---

## Two things to know before reading the numbers

### 1. The sign in B0's formula is wrong in the spec

The spec sheet and slide 7 both write:

```
scale = 10^(−0.4 × Δ(z−y)_median)      ← minus
```

That sign is wrong. A magnitude is `m = −2.5·log₁₀(F)`, so `F = 10^(−0.4m)` and

```
F_y / F_z = 10^(−0.4·m_y) / 10^(−0.4·m_z) = 10^(+0.4·(m_z − m_y))
```

The dataset agrees. Across the full GalaxiesML training catalogue the team's own
EDA gives `mean(m_z) = 19.325` and `mean(m_y) = 19.154` — **y is 0.17 mag
brighter than z**, so the scale must come out *greater* than 1.

`02_run_b0.py` does not argue this; it **measures** it. It computes the median
of (aperture flux in y)/(aperture flux in z) over the training galaxies and
prints it next to both candidate scales. It also evaluates the spec's sign
alongside, so the cost of getting it wrong appears directly in the table.

`config/b0_baseline.yaml` holds `sign: +1`. Set it to `-1` to reproduce the
spec's version.

### 2. Good pixels and good photometry are not the same thing

B1 minimises pixel-by-pixel L1. Most pixels in a 64×64 cutout are empty sky, so
the fit is dominated by sky and B1 learns a small per-pixel bias. Summed over a
709-pixel aperture that bias becomes real flux — negligible for a bright galaxy,
large for a faint one. So B1 can beat B0 on MAE and still lose on magnitude
error for faint galaxies. The results table splits Δm into a faint half and a
bright half to make this visible.

### Convergence is measured, not assumed

B1 is linear with 5 parameters, so its optimum can be **solved for directly**.
Every run reports its distance from that solved optimum, which turns "did it
converge?" into a number.

This matters because the budget is `epochs × steps_per_epoch`:

| Dataset | Train galaxies | Optimiser steps at 20 epochs | Converges at lr=1e-3? |
|---|---|---|---|
| Official z<0.4 split | 84,911 | ~26,500 | **Yes**, comfortably |
| A 2,000-galaxy pilot | 1,400 | ~440 | **No** — dominated by the random init |

On the full dataset the prescribed hyperparameters are fine. On a small pilot
they are not, and the report will say so rather than quietly presenting
initialisation noise as a learned result.

---

## Running B2 and B3

B0 and B1 run on a laptop in minutes. B2 and B3 are ~1.9M-parameter
convolutional nets over 84,911 training galaxies, so they want a GPU; on CPU
expect hours per seed rather than minutes.

**The GPU needs a separate environment.** `requirements.txt` pins the CPU wheel
index, which is right for B0/B1 but sticky: an environment built from it keeps
resolving the `+cpu` wheel, and that reports `torch.cuda.is_available() ==
False` on perfectly good hardware. Use `requirements-gpu.txt` in its own venv
and verify before training:

```bash
python -m venv .venv-gpu
.venv-gpu\Scripts\activate
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.14.1+cu130 True`. A version ending `+cpu`, or `False`, means pip
picked the wrong wheel -- fix that before running 08 or 09, not after.

`requirements-gpu.txt` pins torch to an exact `+cu130` local version, and that
pin is load-bearing: PyPI publishes the same version number as a CPU-only
Windows build, so a loose `torch>=2.0` lets pip take whichever index offers the
higher version and install the CPU wheel without a word. The file's footer says
how to move the pin if it goes stale. Everything else is unchanged --
same splits, same standardisation, same photometry, same evaluation code, so
the ladder table is a like-for-like comparison and not four separate
experiments.

```bash
python scripts/08_train_b2.py                 # 3 seeds, masked L1
python scripts/09_train_b3.py                 # 3 seeds, L1 warmup then NLL
python scripts/10_benchmark_all.py            # B0 | B1 | B2 | B3 in one table

python scripts/08_train_b2.py --seed 42       # one seed first, if time is short
python scripts/09_train_b3.py --no-film       # ablation: is FiLM doing anything?
python scripts/08_train_b2.py --residual-on-z # predict a correction to z, not y
```

`10_benchmark_all.py` reports whichever models have been run and names the
ones that have not, so it is useful after B2 alone.

### What B2 is for

One question: **how much of the remaining error was spatial?** B1 bought 6.5%
over B0 with five parameters, which says local spectral mixing is close to
exhausted. Two separate measurements say the rest is shape:

* script 04's residual figure -- a red-centre / blue-halo pattern on typical
  galaxies, a gradient neither B0 nor B1 *can* represent;
* script 07 -- y is blurrier than z by ~1.1 px of second-moment width, stable
  across all eight redshift bins. Blur is spatial, so no per-pixel rescaling
  of z can produce it.

The ladder table's last column divides the improvement by the parameters it
cost. If a U-Net needs 1.9M parameters to buy what B1 bought with five, the
interesting conclusion is that **capacity is no longer the constraint** -- which
is the argument for the physics models rather than for a bigger network.

### What B3 is for

B3 is the **capacity-matched control** for P1-P3. Same size (within 2.4% of
B2), same redshift conditioning, same per-pixel error bars, no physics. Sir's
slide 6 is the reason it exists:

> *"If a complex physics model beats a random baseline, it proves nothing. If
> it beats a carefully capacity-matched control, we have discovered new
> science."*

Three pieces, each with a measured reason rather than a stylistic one:

**FiLM on redshift.** Script 05 refitted B0 independently per redshift bin and
found the z->y relation really does change with redshift: the Ha window shows
an excess of +0.20 (95% CI [+0.10, +0.29]) and [OIII]+Hb +0.28 (CI [+0.21,
+0.40]), both clear of zero. So redshift carries real information. FiLM
injects it as a per-channel scale and shift rather than as an extra input
plane, so one scalar cannot be mistaken for image structure.

**A variance head.** B3 predicts log sigma^2 per pixel alongside the mean.

**Gaussian NLL.** This scores the error bars, not just the prediction, and it
is the only metric on which B3 and the physics models are directly comparable.

### Reading B3's output: the NLL is meaningless without the calibration line

A model can post an excellent MAE and useless error bars. Three numbers in
`ladder_table.md` guard against each way of being fooled:

| Number | What a bad value means |
|---|---|
| `(y-mu)/sigma` spread | **> 1: over-confident** -- sigma too small, so every anomaly score is inflated and the shortlist fills with ordinary galaxies. **< 1: hedging.** 1.0 is honest. |
| coverage at 0.5 / 0.68 / 0.95 | The whole curve, not one point. Catches a model that is right on average and wrong at the tails. |
| Spearman(sigma, \|error\|) | **Near 0: one flat sigma everywhere.** That passes calibration while being useless for ranking anything. Above ~0.2 means the model knows *where* it is uncertain. |

`09_train_b3.py` also measures whether FiLM is doing anything at all: it
re-predicts one batch at several redshifts with the images held fixed and
reports how much the output moves. **A relative response near zero is a
finding, not a bug** -- it would mean B3's advantage over B2, if any, comes from
the NLL rather than from redshift. Either way it gets stated.

### Two failure modes that are designed around, not hoped away

**The variance head eating the model.** Trained on NLL from scratch, the
cheapest way to cut the loss is to declare everything noisy, and the mean then
never fits. `uncertainty.nll_warmup_epochs` trains the mean on masked L1 first
(5 epochs by default). The switch is visible as a jump in the history CSV, and
the Trainer **resets its best-validation record** at that epoch -- the two
losses are not on the same scale, and without the reset the saved checkpoint
would be frozen inside the warmup.

**Silent architecture drift.** "Capacity-matched" is a claim the whole physics
comparison rests on, so both configs record `expected_parameters` and
construction fails loudly if the built model disagrees by more than 2%.
Changing the width is fine -- it just has to be deliberate.

### Checking it without the data

```bash
python tests/test_b2_b3.py       # 56 checks, a few minutes on CPU
```

Builds fake files, trains both models for a few epochs, runs the ladder table,
and separately checks the properties that are impossible to eyeball later: the
NLL minimised at sigma = \|residual\| (its analytic optimum), masked pixels
contributing exactly nothing, the calibration metric returning 1.00 on data
that is calibrated by construction and ~2.0 when sigma is deliberately halved,
and the anomaly ranking putting a confidently-wrong galaxy above an equally
wrong one that admitted its uncertainty.

---

## Pilot mode (`01b_prepare_pilot.py`)

When only one extracted sample is available, this script applies the redshift
window and re-splits it 70/15/15 with a fixed seed, writing the same processed
files the official path produces. Scripts 02–04 then run unchanged.

**It is not equivalent to the official splits.** GalaxiesML's train/val/test
partition is a specific one; a re-split of a single file is not. Pilot outputs
are tagged `pilot_resplit` in `data/splits/pilot_report.json`, and any numbers
from them should be labelled as a pilot on a slide.

---

## Configuration

Everything tunable is in `config/`. The ones worth knowing:

| Setting | Default | Meaning |
|---|---|---|
| `dataset.z_min` / `z_max` | 0.0 / 0.4 | Redshift window |
| `dataset.require_homogeneous_flag` | true | Keep only secure redshifts |
| `dataset.aperture_radius` | 15 | Photometry aperture, in pixels |
| `dataset.aperture_center` | null | `null` = image centre, i.e. (32, 32) at 64×64 |
| `photometry.nonpositive_flux_handling` | `exclude` | `exclude` or the spec's `clamp` — both are always reported |
| `dataloader.in_memory` | false | Lazy HDF5 reads; the z<0.4 train split is ~7 GB |
| `b0.scaling.sign` | +1 | `-1` reproduces the spec's formula |
| `b1.training.lr` / `epochs` | 1e-3 / 20 | As specified |

### A note on non-positive aperture fluxes

The spec computes `m = −2.5·log₁₀(max(F, 1e-30))`. For a galaxy whose aperture
flux is ≤ 0 — normal for a faint source in sky-subtracted data — that clamp
produces a ~75 mag error, which is an artefact of the clamp rather than
anything about the model, and it wrecks the NMAD and outlier fraction.

The default is therefore to **exclude** those galaxies and report how many.
Both conventions are always computed and shown side by side, so the difference
is visible instead of hidden.

---

**Never commit the data.** `*.h5`, `*.hdf5` and `*.npz` are in `.gitignore`.
