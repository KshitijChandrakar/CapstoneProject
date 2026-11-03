# Results — draft for the extended abstract

**Status:** B0, B1 and the three diagnostic studies are final. B2 and B3 are
implemented and tested but have no real numbers yet; every place they belong is
marked `[B2/B3 PENDING]` with the sentence that will go there once the run
lands. Nothing below needs rewriting when they arrive — only filling in.

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
| B2 (U-Net) | 1,928,705 | `[B2/B3 PENDING]` | | | | | |
| B3 (U-Net+FiLM+NLL) | 1,975,778 | `[B2/B3 PENDING]` | | | | | |

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

`[B2/B3 PENDING]` — the three sentences that go here once the run lands:

1. *How much the 1.9M parameters bought over B1's five, in MAE and in MAE per
   10,000 parameters.* **Either answer is a result.** A large gain says the
   remaining error was spatial and capacity was the constraint. A small gain is
   the more interesting outcome: it says capacity is *not* the constraint, which
   is the central argument for pursuing physics rather than a bigger network.
2. *Whether B3's error bars are honest* — the spread of (y − μ)/σ, which should
   be 1.0, with coverage at the 50%, 68% and 95% levels. An excellent MAE with
   badly calibrated σ is a model that cannot be used for anomaly ranking, so
   this is reported beside the MAE, not after it.
3. *Whether the redshift conditioning does anything* — we re-predict a fixed
   batch at several redshifts and measure how far the output moves. A response
   near zero is a **finding, not a bug**: it would mean B3's advantage over B2,
   if any, comes from the uncertainty head rather than from redshift.

## 7. What the baselines establish

1. **The floor is 6.5%.** That is what pure local spectral mixing buys, and it
   is measured on a converged, directly-solvable model rather than estimated.
2. **z dominates, and that is all we claim about band ordering.** The rest of
   the sequence is not stable across estimators.
3. **The remaining error has spatial structure**, from two independent
   measurements — *y* being broader than *z*, and the radial residual pattern.
4. **The z→y relation varies with redshift**, significantly in two of three
   emission-line windows.
5. **A capacity-matched control exists** at 1.98M parameters, so the physics
   models will be compared against something that is hard to beat for reasons
   other than physics.

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
  calibration figure (`b3_calibration.png`) arrive with B2/B3.
- Word count is deliberately over an abstract's budget. Cut §4 and §6's
  background first; keep the tables and the caveats.
