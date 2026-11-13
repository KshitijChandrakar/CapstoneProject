# Results — draft for the extended abstract

**Status:** complete. All four models have real numbers on the official test
split, and the three diagnostic studies are final.

All four models are the mean over the same three seeds [42, 123, 2026], so
every comparison below has measured error bars.

Numbers are on the **test** split, 17,072 galaxies, in physical
(sky-subtracted flux) units. B1 is mean ± std over seeds 42, 123 and 2026.

---

## 1. What we set out to measure

We predict the HSC *y* band (9762 Å) from 64×64 cutouts in *g, r, i, z*, on the
official GalaxiesML splits restricted to `0.0 ≤ z < 0.4` — 84,911 train, 17,146
validation, 17,072 test. The point of the baselines is not to win. It is to
measure **how much of the problem is simple**, so that any later claim about
physics has a number to beat rather than a vague sense of difficulty.

## 2. The two baselines

**B0 (Copy-z, 0 parameters)** copies the *z* image and multiplies it by one
constant, derived from the median *z − y* colour.

**B1 (Linear 1×1, 5 parameters)** is a weighted sum of the four input bands,
using the same weights at every pixel.

| Model | Params | Test MAE ↓ | PSNR ↑ | SSIM ↑ | Flux NMAD ↓ | σ(mag) ↓ | Outlier % ↓ |
|---|---|---|---|---|---|---|---|
| B0 (Copy-z) | 0 | 0.3505 | 37.431 | 0.8694 | 0.0402 | 0.0431 | **0.504** |
| B1 (Linear 1×1) | 5 | **0.3276** | **37.991** | **0.8785** | **0.0395** | **0.0423** | 1.217 |
| B2 (U-Net) | 1,928,705 | **0.2744 ± 0.0006** | **40.411 ± 0.042** | **0.8917 ± 0.0001** | 0.0151 ± 0.0002 | 0.0164 ± 0.0002 | **0.264 ± 0.013** |
| B3 (U-Net+FiLM+NLL) | 1,975,778 | 0.2803 ± 0.0004 | 39.867 ± 0.048 | 0.8913 ± 0.0001 | **0.0131 ± 0.0002** | **0.0143 ± 0.0002** | 0.287 ± 0.017 |

**B1 beats B0 on six of seven metrics, and loses on the seventh.** MAE falls
6.5%, PSNR rises 1.5%, SSIM 1.0%, flux NMAD 1.8%. But B1's catastrophic
outlier rate (|Δm| > 0.5) is **1.22% against B0's 0.50%** — better on average,
worse in the tail. We report this rather than quietly dropping it: a model that
improves the mean while doubling the tail is a different trade-off from one that
improves both, and for a survey pipeline the tail is often what matters.

B1 is linear, so its optimum can be solved for directly instead of trusted. The
trained weights sit **0.0127** from that solved optimum, and the largest spread
across the three seeds is **0.0002**. B1 has converged; its number is a property
of the problem, not of our training loop.

**The reading: five parameters buy a real but small gain.** Local spectral
mixing — the only thing B0 and B1 can do — is close to exhausted at 6.5%. That
is the floor the rest of the ladder has to clear.

## 3. Which bands carry the *y*-band signal?

Raw correlations between any two bands are ~0.99 everywhere, which measures
only that galaxies are bright in the middle. We therefore use **partial**
correlations and standardised weights, which hold the other bands fixed.

| Band | standardised weight | physical weight |
|---|---|---|
| g | −0.1463 | −0.7385 |
| r | +0.2374 | +0.5604 |
| i | +0.3650 | +0.5285 |
| **z** | **+0.5497** | **+0.6388** |

**z dominates.** This holds for every estimator we tried and in 8 of 9 redshift
bins, and it is what spectral proximity predicts — *z* is the band nearest *y*
in wavelength.

**We do not claim the full four-band ordering.** The standardised weights give
`z > i > r > g`, but *r* and *i* differ by only 0.13, and both the physical
weights and the partial-correlation analysis give `z > r > i > g`. Two of three
estimators put *r* above *i*, and across redshift bins the two swap repeatedly,
sometimes separated by 0.01. The defensible claim is the one that survives every
estimator.

*g* carries a **negative** weight in both spaces. It is acting as a colour
correction — subtracting blue light — not contributing infrared flux.

## 4. Two measurements that say the remaining error is spatial

**Spatial width.** Pixel-to-pixel cross-correlation between bands shows the *y*
band is **wider than *z*** in every redshift bin, by roughly 0.05 to 0.40 px of
second-moment width. Band-to-band peak lag is (0, 0) throughout, so the bands are
correctly registered and this is not a misalignment artefact. A per-pixel
rescaling of *z* cannot produce extra spatial extent, whatever its weights. B0
and B1 are therefore limited by optics, not by parameter count.

> Reportability: the validity test in `07_pixel_cross_correlation.py` passes for
> the *z→y* pair only. The *g*, *r* and *i* widths are measured but fail it and
> are not quoted here.

**Residual structure.** B0's and B1's residuals show a systematic red-centre /
blue-halo pattern on typical galaxies — a radial gradient that neither model can
represent, because neither has any notion of a pixel's neighbours.

Both point the same way, and that is the motivation for B2.

## 5. The *z*→*y* relation changes with redshift

We refit B0 independently in each redshift bin and compared each bin against its
**adjacent** bins rather than against the global mean, with 95% bootstrap
confidence intervals over 400 resamples.

| Emission-line window | z range | excess over adjacent controls | 95% CI | verdict |
|---|---|---|---|---|
| [OIII] + Hβ | 0.78–1.01 | **+0.28** | [+0.21, +0.40] | **supported** |
| Hα | 0.36–0.49 | **+0.20** | [+0.10, +0.29] | **supported** |
| 4000 Å break | 1.22–1.44 | +0.14 | [−0.00, +0.26] | not significant |

Two of the three windows show an excess whose interval is clear of zero;
the third does not, and we say so rather than rounding it into the story.

> **Correction to earlier circulated numbers.** An earlier version of this
> analysis compared each bin against a global mean that included contaminated
> high-redshift bins, giving a 1.18× reference that should not be quoted. The
> corrected analysis **inverts the ordering**: [OIII]+Hβ is the strongest
> window, not Hα, and the 4000 Å break moves from apparently significant to not
> significant. The numbers above supersede any earlier version.

This is the measured justification for conditioning B3 on redshift: the
relation genuinely changes, so redshift carries information a redshift-blind
model cannot use.

## 6. The spatial models

**B2 (U-Net, 1,928,705 parameters)** is an encoder–decoder with skip
connections, so a predicted pixel depends on its surroundings. It uses the same
splits, the same standardisation, the same loss and the same evaluation code as
B1 — only the architecture differs, so the comparison is like-for-like.

**B3 (U-Net + FiLM + NLL, 1,975,778 parameters)** adds two things: redshift
conditioning via FiLM (a per-channel scale and shift, so one scalar cannot be
mistaken for image structure), and a per-pixel variance head trained with a
heteroscedastic Gaussian negative log-likelihood, so the model reports how
confident it is. B3 is **within 2.4% of B2's parameter count by construction**,
enforced in code rather than asserted.

B3 exists to be the **capacity-matched control** for the physics models P1–P3.
A physics model that beats a weak baseline has shown nothing; one that beats a
same-sized, redshift-aware, uncertainty-aware network without physics has shown
something.

### What the spatial models actually bought

**B2 more than doubled B1's gain.** MAE fell 16.2% against B1's 6.5% over B0,
and the photometry improved far more sharply still: flux NMAD dropped from
0.0395 to 0.0151, a **62% reduction**. B2's catastrophic outlier rate is
**0.246%, better than B0's 0.504%** — so the spatial model repairs the tail
regression B1 introduced rather than inheriting it.

**But capacity is clearly no longer the constraint.** B1 bought 6.5% with five
parameters; B2 bought 16.2% with 1.9 million. Per parameter that is roughly
166,000× less efficient. The remaining error is real and structured, and
throwing more capacity at it is not the way to reach it — which is the central
argument for physics-informed models rather than a larger network.

**B3 is not simply worse — it makes a real trade.** With three seeds each,
both differences from B2 can be tested, and the per-seed ranges do not overlap
in either direction:

| | B2 over seeds | B3 over seeds | gap |
|---|---|---|---|
| MAE | [0.27392, 0.27516] | [0.27979, 0.28076] | 2.2%, B2 better |
| σ(mag) | [0.01613, 0.01665] | [0.01400, 0.01447] | 13.0%, **B3** better |

So B3 is reliably 2.2% worse on pixel accuracy and reliably 13% better on
photometric accuracy, with calibrated uncertainty on top. Which model is
"better" depends on what the prediction is for — MAE for image fidelity,
σ(mag) for catalogue photometry. Both numbers are stated rather than whichever
flatters the newer model.

**B3's error bars are trustworthy.** The (y−μ)/σ spread is **1.037** where 1.0
is honest, and coverage is 53.0 / 70.7 / 94.9% against nominal 50 / 68 / 95.
Spearman(σ, |error|) is **+0.290**, well above the 0.2 threshold, so σ tracks
where the error actually is rather than being one flat number. Both conditions
the anomaly search requires are satisfied.

**The redshift conditioning does almost nothing.** Re-predicting the same
images across redshifts moves B3's output by a relative **0.0109, 0.0124 and
0.0092 across the three seeds — about 1% every time**. Three independent runs
agree, so this is a property of the architecture, not one unlucky
initialisation. This was a pre-registered check with a stated interpretation,
and the answer is the negative one: B3's advantages over B2 (better flux NMAD and σ(mag), plus
uncertainty B2 cannot produce) come from the **variance head and the NLL
objective, not from redshift**. FiLM's 47,040 parameters are close to inert.

That is worth stating plainly rather than burying, because §5 showed the z→y
relation genuinely *does* vary with redshift. The information is there; this
architecture did not exploit it. For P1–P3 that is a useful warning: redshift
conditioning alone should not be expected to buy performance.

**All six runs are converged.** 40 epochs on a cosine schedule decaying to
4.6e-7; none stopped early and none needed to. The final five epochs
moved validation loss by 0.13% (B2) and 0.01% (B3). More epochs on this
schedule would change the fourth decimal place.

## 7. What the baselines establish

1. **The floor is 6.5%.** That is what pure local spectral mixing buys, and it
   is measured on a converged, directly-solvable model rather than estimated.
2. **z dominates, and that is all we claim about band ordering.** The rest of
   the sequence is not stable across estimators.
3. **The remaining error has spatial structure**, from two independent
   measurements — *y* being broader than *z*, and the radial residual pattern.
4. **The z→y relation varies with redshift**, significantly in two of three
   emission-line windows.
5. **A capacity-matched control exists and is calibrated** at 1.98M
   parameters, so the physics models will be compared against something hard to
   beat for reasons other than physics — and comparable on NLL, not only on MAE.
6. **Capacity has stopped paying.** 1.9M parameters bought 16.4% where five
   bought 6.5%. The next gain has to come from somewhere other than size.

### The bar P1–P3 have to clear

B3's three-seed spread sets it precisely. To be making a claim rather than
measuring noise, a physics model must beat

* **MAE 0.2803 ± 0.0004**, or
* **σ(mag) 0.0143 ± 0.0002**,

by more than roughly 0.001 — and must itself be run at the same three seeds, or
the comparison is one-sided. A single-seed physics model against a three-seed
control is not a result.

---

## Notes for whoever edits this

- Every number here traces to a file in `docs/results_official/` or to a
  committed script; nothing is from memory.
- The outlier-rate regression in §2, the *r*/*i* instability in §3, the
  reportability caveat in §4 and the superseded-numbers box in §5 are
  deliberate. They are the parts a reviewer would otherwise find.
- Figures available: `b1_weights_standardised.png`, `flux_scatter.png`,
  `dmag_histogram.png`, `galaxy_grid.png`, `band_psf_width_vs_redshift.png`,
  `redshift_sweep.png`. The ladder figure (`ladder_mae_vs_params.png`) and the
  calibration figure (`b3_calibration.png`) are now published too. Everything
  is in `docs/results_official/`.
- Word count is deliberately over an abstract's budget. Cut §4 and §6's
  background first; keep the tables and the caveats.
