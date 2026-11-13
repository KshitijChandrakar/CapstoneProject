| Model | Params | Test MAE ↓ | PSNR (dB) ↑ | SSIM ↑ | Flux NMAD ↓ | sigma(mag) ↓ | Outlier % ↓ |
|---|---|---|---|---|---|---|---|
| B0 (Copy-z) | 0 | 0.3505 | 37.431 | 0.8694 | 0.0402 | 0.0431 | 0.504 |
| B1 (as specified) | 5 | 0.3276 ± 0.0000 | 37.991 ± 0.000 | 0.8785 ± 0.0000 | 0.0395 ± 0.0000 | 0.0423 ± 0.0000 | 1.217 ± 0.007 |
| B2 (U-Net) | 1,928,705 | **0.2744 ± 0.0006** | **40.411 ± 0.042** | **0.8917 ± 0.0001** | 0.0151 ± 0.0002 | 0.0164 ± 0.0002 | **0.264 ± 0.013** |
| B3 (U-Net+FiLM+NLL) | 1,975,778 | 0.2803 ± 0.0004 | 39.867 ± 0.048 | 0.8913 ± 0.0001 | **0.0131 ± 0.0002** | **0.0143 ± 0.0002** | 0.287 ± 0.017 |

Evaluated on the official GalaxiesML **test** split: **17,072 galaxies**, 0.0 ≤ z < 0.4, in physical flux units.  Trained models are the mean over seeds [42, 123, 2026]; ± is the seed-to-seed standard deviation. Arrows mark the good direction and the best value in each column is bold.

### What each rung bought

| Step | MAE change | Parameters added | MAE improvement per 10k params |
|---|---|---|---|
| B0 -> B1 | 0.3505 -> 0.3276 (-6.5%) | 5 | 45.8 |
| B1 -> B2 | 0.3276 -> 0.2744 (-16.2%) | 1,928,700 | 0.000276 |
| B2 -> B3 | 0.2744 -> 0.2803 (+2.2%) | 47,073 | -0.00126 |

The last column is the honest one. B1 bought 6.5% with five parameters; a U-Net buys whatever it buys with ~1.9 million. If the per-parameter column collapses, the ladder has reached the point where capacity is no longer the constraint -- which is exactly the argument for the physics models rather than a bigger network.

### B3's uncertainty, and whether it can be trusted

| Quantity | Value | Reads as |
|---|---|---|
| NLL per pixel | 0.2968 | lower is better; the only metric comparable to P1-P3 |
| (y-mu)/sigma spread | 1.037 | 1.0 is honest -- **calibrated** |
| 0.50 interval coverage | 53.0% | should be 50% |
| 0.68 interval coverage | 70.7% | should be 68% |
| 0.95 interval coverage | 94.9% | should be 95% |
| median sigma | 0.2849 | sharpness: lower is better, given calibration holds |
| Spearman(sigma, abs error) | +0.290 | 0 means one constant sigma everywhere; >0.2 means it knows where it is uncertain |

Calibration is what makes the anomaly search possible. An over-confident model inflates every score, so the shortlist fills with ordinary galaxies; a model with a flat sigma cannot rank at all.
