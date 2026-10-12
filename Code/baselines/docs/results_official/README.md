# Official results — B0 & B1, GalaxiesML `0.0 ≤ z < 0.4`

Produced on the three official GalaxiesML splits. Galaxy counts match the spec
exactly: **84,911 train / 17,146 val / 17,072 test**.

All metrics are on the **test** split, in physical (sky-subtracted flux) units.
B1 is mean ± std over seeds `[42, 123, 2026]`.

## Headline table

| Model | Params | Test MAE ↓ | PSNR ↑ | SSIM ↑ | Flux NMAD ↓ | σ(mag) ↓ | Outlier % ↓ | σ(z−y) ↓ |
|---|---|---|---|---|---|---|---|---|
| **B0 (Copy-z)** | 0 | 0.3505 | 37.431 | 0.8694 | 0.0402 | 0.0431 | **0.504** | 0.0431 |
| **B1 (Linear 1×1)** | 5 | **0.3276** | **37.991** | **0.8785** | **0.0395** | **0.0423** | 1.217 | **0.0423** |

B1 wins on six of seven metrics. B0 wins on catastrophic outlier rate.

## The four discussion answers

**1. How much does B1 improve over B0?**
Modestly but consistently: MAE −6.5 %, PSNR +1.5 %, SSIM +1.0 %, flux NMAD
−1.8 %, σ(mag) −2.0 %. Five extra parameters buy a real but small gain, which
is the point of a baseline — it shows how much of the problem is *not* local
spectral interpolation.

The exception: B1's outlier rate (|Δm| > 0.5) is **1.22 %** against B0's
**0.50 %**. B1 is better on average and worse in the tail.

**2. What are the learned weights? Does z dominate?**

| Band | w (standardised) | w (physical) |
|---|---|---|
| g | −0.1463 | −0.7385 |
| r | +0.2374 | +0.5604 |
| i | +0.3650 | +0.5285 |
| **z** | **+0.5497** | **+0.6388** |
| bias | −0.0002 | +0.0402 |

**z dominates — but only that part of the prediction is robust.**

In standardised weight the ordering is `z > i > r > g`, matching the spec's
expectation from spectral proximity. But **r and i are close together**, and they
swap depending on the estimator: the physical weights and the partial-correlation
analysis in `06_band_correlations.py` both give `z > r > i > g`. Two of three
estimators put r above i, and across redshift bins the two swap repeatedly,
sometimes separated by as little as 0.01.

So the defensible claim is **"z dominates"** — true for every estimator and in 8
of 9 redshift bins — not the exact four-band sequence.

The standardised weight is the right measure here: it is the change in y per
one standard deviation of a band, so it is dimensionless and independent of how
widely each band happens to vary. The physical weight orders `z > r > i > g`
because `w_phys = w · σ_y/σ_b` divides by each band's spread, inflating the
narrow-spread r relative to the wider-spread i.

g carries a **negative** weight in both spaces — it acts as a colour
correction, subtracting blue light rather than contributing to the infrared
prediction.

**3. Where do the baselines fail?**

| Model | Δm NMAD, faint half | Δm NMAD, bright half |
|---|---|---|
| B0 | 0.0600 | 0.0315 |
| B1 | 0.0556 | 0.0303 |

Both are roughly twice as bad on the faint half. B1 beats B0 in both halves, so
at full scale it does not suffer the faint-galaxy bias seen in the small pilot.

`galaxy_grid.png` shows the structural failure: residuals have a **red core and
a blue halo** — both models over-predict the centre and under-predict the
outskirts. Neither can represent a **colour gradient**: B0 multiplies by one
number, B1 applies the same weights at every pixel. This is the direct evidence
that spatial context matters, and the argument for moving to B2/B3.

**4. Which sign of the B0 formula is correct?**

The spec writes `scale = 10^(−0.4 × Δ(z−y))`. **That sign is wrong.**

| Quantity | Value |
|---|---|
| median(m_z − m_y) on train | +0.1788 mag |
| **measured** median(F_y / F_z) from the images | **1.165553** |
| 10^(**+**0.4 × median) | 1.179040 — error **0.0135** ✅ |
| 10^(**−**0.4 × median) | 0.848147 — error **0.3174** ❌ |

The `+` sign is 24× closer to what the images actually show. Since
`m = −2.5·log₁₀(F)`, `F_y/F_z = 10^(+0.4·(m_z − m_y))`. The catalogue agrees:
y is 0.18 mag *brighter* than z, so the scale must exceed 1. The `−` sign would
make the predicted y image ~15 % dimmer than z.

## Did B1 converge?

Yes, comfortably, with the spec's hyperparameters unchanged:

- 84,911 train galaxies ÷ 64 × 20 epochs = **26,540 optimiser steps**
- distance from the directly-solved optimum: **0.0127**
- largest weight spread across the three seeds: **0.0002**

(B1 is linear with 5 parameters, so its optimum can be solved for exactly; the
distance above is a measurement, not a judgement. A 2,000-galaxy pilot gives
only ~440 steps and does *not* converge — but that is a small-sample artefact
and does not apply here.)

## A note on non-positive aperture fluxes

11 test galaxies have a non-positive true aperture flux — physically normal for
a faint source in sky-subtracted data. The spec's `max(F, 1e-30)` clamp turns
each into a ~75 mag error:

| Handling | Δm median | Δm NMAD | Outlier % |
|---|---|---|---|
| exclude (default) | −0.0129 | 0.0431 | **0.50** |
| clamp (spec) | −0.0129 | 0.0432 | **0.57** |

Small here, but it inflates the outlier rate by 14 % for no scientific reason.
Both are always reported.

## Provenance

`sample_provenance.json` records that the earlier 2,000-galaxy pilot sample came
**100 % from the training split** (0 in val, 0 in test) — so no held-out data
was touched during development.

## Files

| File | Contents |
|---|---|
| `results_table.md` / `.tex` | Full table and discussion, LaTeX ready for the report |
| `b0_metrics.json` | B0 on val and test, plus the sign verification |
| `b1_metrics.json` | B1 per seed and summarised, weights, convergence |
| `dmag_histogram.png` | Magnitude residuals, B0 vs B1 |
| `flux_scatter.png` | Predicted vs true aperture flux, log axes |
| `b1_weights_*.png` | Learned weights, both spaces, seed error bars |
| `galaxy_grid.png` | Example galaxies: truth, predictions, residuals |
| `sample_provenance.json` | Which official split the pilot sample came from |
