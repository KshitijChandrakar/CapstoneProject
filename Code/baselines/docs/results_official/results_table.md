# Baselines B0 & B1 — predicting the HSC y band from g, r, i, z

Redshift window 0.0 ≤ z < 0.4. Evaluated on the **test** split (17,072 galaxies), in physical (sky-subtracted flux) units. B1 is mean ± std over seeds [42, 123, 2026].

Aperture photometry: circular, radius 15 px, centred at (32.0, 32.0). NMAD = 1.4826 × median(|x − median(x)|). Outlier % = galaxies with |Δm| > 0.5.

## Benchmark table (slide 10)

| Model | Params | Test MAE | PSNR (dB) | SSIM | Flux NMAD | sigma(mag) | Outlier % | sigma(z-y) |
|---|---|---|---|---|---|---|---|---|
| **B0 (Copy-z)** | 0 | 0.350498 | 37.431 | 0.8694 | 0.0402 | 0.0431 | 0.504 | 0.0431 |
| **B1 (as specified)** | 5 | 0.327580 ± 0.000000 | 37.991 ± 0.000 | 0.8785 ± 0.0000 | 0.0395 ± 0.0000 | 0.0423 ± 0.0000 | 1.217 ± 0.007 | 0.0423 ± 0.0000 |

## 1. How much does B1 improve over B0?

### B1 (as specified) vs B0

- **Test MAE**: B0 0.350498 -> 0.32758  (6.5% lower; **better**)
- **PSNR (dB)**: B0 37.4309 -> 37.9906  (1.5% higher; **better**)
- **SSIM**: B0 0.869391 -> 0.878502  (1.0% higher; **better**)
- **Flux NMAD**: B0 0.0402185 -> 0.0394761  (1.8% lower; **better**)
- **sigma(mag)**: B0 0.0431373 -> 0.0422718  (2.0% lower; **better**)
- **Outlier %**: B0 0.504074 -> 1.2172  (141.5% higher; **worse**)
- **sigma(z-y)**: B0 0.0431373 -> 0.0422718  (2.0% lower; **better**)


## 2. What are the learned weights? Does z dominate?

The spec predicts `w_z > w_i > w_r > w_g`, reasoning from spectral proximity — z is the band nearest y in wavelength. That is a claim about which band is most **informative**, and the measure for that is the **standardised** weight: the change in y per one standard deviation of that band. It is dimensionless and does not depend on how widely each band happens to vary.

The physical weight answers a different question — how much y-flux per unit of band flux, via `y_flux ≈ Σ_b w_b · x_b_flux + bias`. Since `w_phys_b = w_b · σ_y / σ_b` divides by each band's own spread, a narrow-spread band (r) is inflated relative to a wide-spread one (i), so the physical ordering is not a clean ranking of importance. Both are given below.

| Band | w (physical) | w (standardised) |
|---|---|---|
| g | -0.7385 ± 0.0003 | -0.1463 ± 0.0001 |
| r | +0.5604 ± 0.0002 | +0.2374 ± 0.0001 |
| i | +0.5285 ± 0.0003 | +0.3650 ± 0.0002 |
| z | +0.6388 ± 0.0002 | +0.5497 ± 0.0002 |
| bias | +0.04019 ± 0.000275 | -0.0002 ± 0.0000 |

- Ordering by **standardised** weight: **z > i > r > g**  ← compare with the spec's expected **z > i > r > g**
- Ordering by physical weight: **z > r > i > g**

**z dominates: True.** Matches the spec's expected ordering: **True** — but the exact sequence is not robust, see below.

> **The exact ordering is not robust.** The middle two bands differ by only 0.1276 in standardised weight, and the physical weights order them the other way round (z > r > i > g). z's lead is solid; the r-versus-i order is not. Claim "z dominates" rather than the full four-band sequence.

### Did B1 converge?

B1 is linear with 5 parameters, so its optimum can be solved for directly. The distance below is between the trained weights and that solved optimum (standardised space) — small means converged.

- optimiser steps: **26,540**
- distance from the solved optimum: **0.0127**
- largest weight spread across seeds: 0.0002
- converged: **True**

## 3. Where do the baselines fail?

Galaxies split at the median true aperture flux. Any constant offset in a prediction is a large fraction of a faint galaxy's total flux, so faint galaxies are where these models break.

| Model | Δm NMAD, faint half | Δm NMAD, bright half |
|---|---|---|
| B0 | 0.0600 | 0.0315 |
| B1 | 0.0556 ± 0.0000 | 0.0303 ± 0.0000 |

## 4. Which sign of the B0 formula is correct?

The spec sheet and slide 7 write `scale = 10^(−0.4 × Δ(z−y))`. That sign is wrong. Since `m = −2.5·log₁₀(F)`:

```
F_y / F_z = 10^(−0.4·m_y) / 10^(−0.4·m_z) = 10^(+0.4·(m_z − m_y))
```

Measured on the training images rather than argued:

| Quantity | Value |
|---|---|
| median(m_z − m_y) on train | +0.1788 mag |
| **measured** median(F_y / F_z) from the images | **1.165553** |
| 10^(**+**0.4 × median) | 1.179040 (error 0.013487) |
| 10^(**−**0.4 × median) | 0.848147 (error 0.317405) |

The data matches the **plus** sign (4,000 train galaxies). B0 therefore uses scale = 1.179040.

Note: because the same measured z flux is used on both sides of the colour, Δ(z−y) = −Δm exactly, so the two NMAD columns are identical.

## 5. A note on non-positive aperture fluxes

The spec computes magnitudes as `−2.5·log₁₀(max(F, 1e-30))`. For a galaxy whose aperture flux is ≤ 0 (a faint non-detection, which is physically normal for sky-subtracted data) that clamp produces a ~75 mag error — an artefact of the clamp, not a property of the model. Excluding those galaxies and reporting the count is the honest alternative. Both are shown:

| Handling | Δm median | Δm NMAD | Outlier % |
|---|---|---|---|
| exclude | -0.0129 | 0.0431 | 0.50 |
| clamp | -0.0129 | 0.0432 | 0.57 |

Galaxies with non-positive true flux: 11; predicted flux: 0.
