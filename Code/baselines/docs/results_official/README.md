# Official results — B0 & B1, GalaxiesML `0.0 ≤ z < 0.4`

Produced on the three official GalaxiesML splits. Galaxy counts match the spec
exactly: **84,911 train / 17,146 val / 17,072 test**.

All metrics are on the **test** split, in physical (sky-subtracted flux) units.
B1 is mean ± std over seeds `[42, 123, 2026]`.

## Headline table

| Model | Params | Test MAE ↓ | PSNR ↑ | SSIM ↑ | Flux NMAD ↓ | σ(mag) ↓ | Outlier % ↓ |
|---|---|---|---|---|---|---|---|
| **B0** (Copy-z) | 0 | 0.3505 | 37.431 | 0.8694 | 0.0402 | 0.0431 | 0.504 |
| **P2** (SED) | 4 | 0.3279 ± 0.0001 | 37.973 ± 0.020 | 0.8792 ± 0.0004 | 0.0399 ± 0.0001 | 0.0428 ± 0.0001 | 1.125 ± 0.075 |
| **B1** (Linear 1×1) | 5 | 0.3276 ± 0.0000 | 37.991 ± 0.000 | 0.8785 ± 0.0000 | 0.0395 ± 0.0000 | 0.0423 ± 0.0000 | 1.217 ± 0.007 |
| **P1** (PSF) | 7 | 0.3113 ± 0.0001 | 38.488 ± 0.006 | 0.8851 ± 0.0008 | 0.0337 ± 0.0001 | 0.0362 ± 0.0002 | 1.077 ± 0.167 |
| **B2** (U-Net) | 1,928,705 | **0.2744 ± 0.0006** | **40.411 ± 0.042** | **0.8917 ± 0.0001** | 0.0151 ± 0.0002 | 0.0164 ± 0.0002 | **0.264 ± 0.013** |
| **B3** (U-Net+FiLM+NLL) | 1,975,778 | 0.2803 ± 0.0004 | 39.867 ± 0.048 | 0.8913 ± 0.0001 | **0.0131 ± 0.0002** | **0.0143 ± 0.0002** | 0.287 ± 0.017 |

All six rows are the mean over the same three seeds [42, 123, 2026]; ± is the
seed-to-seed standard deviation. Rows are ordered by parameter count, which is
where the argument lives: the two physics models sit between B1 and B2.

## The physics models

### P1's gain comes from the measurement, not from fitting

This is the central result.

| | Trainable params | Test MAE |
|---|---|---|
| B1 (linear) | 5 | 0.32758 |
| P1, blur width **learned** | 7 | 0.311273 ± 0.000121 |
| P1, blur width **frozen** at script 07's measurement | **5** | 0.311754 |

Freezing the blur at the width script 07 measured independently — giving the
model no freedom to adjust it at all — costs **0.15%**. So at **the same five
parameters as B1**, P1 is 5.0% better on MAE and 14% better on σ(mag), entirely
because one number came from a measurement instead of being learned.

That is what "physics-informed" is supposed to mean: the prior is not a
regulariser or an initialisation, it is a quantity measured elsewhere that
carries over intact.

### P1's fitted width against the independent measurement

| Redshift bin | Train galaxies | P1 fitted σ (3 seeds) | Script 07 measured σ |
|---|---|---|---|
| 0.00–0.36 | 79,445 (93.6%) | 0.8178, 0.8206, 0.8222 | **0.8675** |
| 0.36–0.40 | 5,466 (6.4%) | 0.7640, 0.7837, 0.8005 | 1.2288 |

In the bin holding 94% of the data, **two unrelated methods agree to within
6%**. Script 07 obtained 0.8675 px by cross-correlating band images and
measuring second-moment widths; P1 obtained 0.82 px by gradient descent on
prediction error, never seeing that number except as a starting point. The
seed-to-seed spread is 0.004 px, so this is a measurement with an error bar.

**The second bin is not a contradiction and should not be quoted as agreement
either.** It holds 6.4% of the galaxies, its seed spread is 0.036 px — nine
times wider on fifteen times less data — and it is a thin sliver of script 07's
much broader 0.36–0.49 bin, so the model is fitting a lower-redshift population
than the measurement describes.

### P2 ties B1, and explains why

P2 reaches 0.3279 ± 0.0001 against B1's 0.3276 — statistically
indistinguishable. The fitted basis gains say what happened:

| Basis term | Fitted gain (3 seeds) |
|---|---|
| constant | 1.0061, 1.0071, 1.0093 |
| linear (spectral slope) | **0.5359, 0.5422, 0.5307** |
| quadratic (curvature) | **0.0485, 0.0692, 0.0372** |

The model discarded the curvature and halved the slope. Two things follow.

**Extrapolating along the measured spectral slope overshoots.** Trusting a
curvature measured over 4754–8898 Å out to 9762 Å is actively harmful; damping
it to near-zero is better. Only a model with named physical terms can report
this — the question cannot be put to B2.

**Once damped, P2 is B1.** With the quadratic gain near zero and the linear
term halved, P2 collapses to an affine function of the four band fluxes, which
is exactly what B1 is. P2 landing on B1's number is the model discovering that
the spectral information is already exhausted by a linear combination.

### The residual ablation locates the missing error

| | Params | Test MAE | σ(mag) |
|---|---|---|---|
| P2, pure SED | 4 | 0.3279 | 0.0428 |
| P2 + residual convolution | 741 | **0.2994** | **0.0349** |

737 convolution parameters buy 8.7% — more than P1's entire gain over B1. And
the fitted gains move from `[1.01, 0.54, 0.05]` to `[1.06, 0.68, 0.23]`: once
something else handles the spatial structure, **the model trusts the spectral
terms more**. The damping in pure P2 was partly compensating for missing
spatial capacity, not a statement about spectra alone.

### What the two physics models establish

The diagnostics predicted this and the models confirm it independently:

* script 06 found the spectral signal dominated by z and essentially linear
  → **P2 cannot beat B1**, and says so through its own gains;
* script 07 found y optically broader than z
  → **P1 beats B1 by 5% at equal parameter count**, using the measured width;
* B2 beats both → there is more spatial structure than one Gaussian captures.

**The remaining error is spatial, not spectral.** That was the hypothesis, and
two physics models now test it from different directions and agree.

### Are the B2 / B3 differences real? Yes — both of them

With three seeds each the two differences can finally be judged, and **neither
is seed noise**. The per-seed ranges do not overlap at all, so this needs no
distributional assumption:

| | B2 range over seeds | B3 range over seeds | gap | t |
|---|---|---|---|---|
| MAE | [0.27392, 0.27516] | [0.27979, 0.28076] | 2.2%, B2 better | ~12 |
| σ(mag) | [0.01613, 0.01665] | [0.01400, 0.01447] | 13.0%, **B3** better | ~10 |

**B3 trades pixel accuracy for photometric accuracy, and it is a real trade.**
It is reliably 2.2% worse on MAE and reliably 13% better on σ(mag), with
calibrated per-pixel uncertainty on top. Which model is "better" depends
entirely on what the band prediction is for: MAE for image fidelity, σ(mag) for
catalogue photometry.

For the physics comparison this matters twice over. B3 is the control, so its
error bars set the bar P1–P3 must clear: a physics model must beat **0.2803 ±
0.0004** on MAE or **0.0143 ± 0.0002** on σ(mag) by more than roughly 0.001 to
be making a claim rather than a noise measurement.

### What each rung bought

| Step | MAE change | Params added | MAE gain per 10k params |
|---|---|---|---|
| B0 → B1 | 0.3505 → 0.3276 (−6.5%) | 5 | 45.8 |
| B1 → B2 | 0.3276 → 0.2744 (−16.2%) | 1,928,700 | 0.000276 |
| B2 → B3 | 0.2744 → 0.2803 (+2.2%) | 47,073 | −0.00126 |

**Capacity has stopped paying.** B1 bought 6.5% with five parameters; B2 bought
16.2% with 1.9 million — a per-parameter efficiency roughly 166,000× worse.
That collapse is the argument for physics-informed models rather than a bigger
network, and it is now a measured quantity rather than an expectation.

The photometry moves the other way and more sharply: flux NMAD fell 0.0395 →
0.0151 (**−62%**) from B1 to B2, and B2's catastrophic outlier rate of
0.264 ± 0.013% is **better than B0's** 0.504% — so the spatial model repairs the
tail regression B1 introduced rather than inheriting it.

### B3's uncertainty

| Quantity | Value | Verdict |
|---|---|---|
| NLL per pixel | 0.2968 | the only metric directly comparable to P1–P3 |
| (y−μ)/σ spread | **1.037** | 1.0 is honest — **calibrated** |
| coverage at 0.50 / 0.68 / 0.95 | 53.0% / 70.7% / 94.9% | against nominal 50 / 68 / 95 |
| median σ | 0.2849 | sharpness, meaningful because calibration holds |
| Spearman(σ, \|error\|) | **+0.290** | > 0.2: it knows *where* it is uncertain |

Both conditions the anomaly search needs are met: the error bars are honest,
and σ tracks where the error actually is.

### FiLM does almost nothing — a finding, not a bug

The redshift conditioning moves B3's prediction by a relative **0.0109,
0.0124 and 0.0092 across the three seeds — about 1% every time** — when the
same images are re-predicted across redshifts. Three independent runs agree, so
this is a property of the architecture and not a quirk of one initialisation. So B3's advantages over
B2 — better flux NMAD and σ(mag), plus error bars B2 cannot produce — come from
the **variance head and the NLL objective, not from redshift**. FiLM's 47,040
parameters are close to inert.

This matters for P1–P3: it says a physics model conditioned on redshift should
not expect the conditioning alone to buy anything, and that B3 is a control
whose strength lies in its uncertainty modelling.

### Training

All six runs used 40 epochs with a cosine schedule decaying from 3.0e-4 to
4.6e-7. None stopped early and none needed to: the last five epochs moved
validation loss by **+0.13% (B2)** and **+0.01% (B3)**. Best epochs were 40/40
for every B2 seed and 38–39 for B3, which is the schedule reaching zero rather
than training being cut short — more epochs on this schedule would change the
fourth decimal place. Roughly 170 min per run on an RTX 3050 6 GB Laptop,
about 11.5 hours for the six runs in total.

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
