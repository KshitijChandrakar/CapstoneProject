# Pixel-to-pixel spatial cross-correlation between bands

Computed on the **test** split, per redshift bin, for every band pair, out to ±12 px of lag.

## What this measures, and why it matters

Script 06 asked how a pixel in one band relates to the *same* pixel in another. This asks how it relates to the pixels *around* it — i.e. how far each band's light spreads.

- `width_px` for a band against **itself** is that band's PSF proxy.
- **If the y band is wider than z, then y contains spatial structure that no per-pixel rescaling of z can reproduce.** B0 and B1 would then be limited by optics rather than by parameter count — which is a hard argument for a spatial model (B2/B3) over a wider linear one.
- `peak_lag` should be (0, 0). Anything else means the bands are misregistered, which would be a data problem.

## PSF proxy per band (spatial width, px)

| z range | n galaxies | g–g | r–r | i–i | z–z | y–y | y − z |
|---|---|---|---|---|---|---|---|
| 0.00–0.36 | 15,998 | 8.17 | 8.02 | 7.81 | 7.86 | 7.91 | **+0.05** |
| 0.36–0.49 | 4,608 | 6.18 | 7.06 | 6.99 | 7.07 | 7.17 | **+0.10** |
| 0.49–0.78 | 12,438 | 5.34 | 6.18 | 6.46 | 6.37 | 6.56 | **+0.19** |
| 0.78–1.01 | 3,001 | 4.77 | 4.71 | 4.95 | 5.41 | 5.49 | **+0.08** |
| 1.01–1.22 | 1,146 | 4.63 | 4.63 | 4.09 | 4.68 | 5.09 | **+0.40** |
| 1.22–1.44 | 715 | 4.69 | 4.39 | 3.90 | 4.43 | 4.71 | **+0.28** |
| 1.44–2.00 | 1,086 | 4.67 | 4.40 | 3.83 | 4.25 | 4.46 | **+0.21** |
| 2.00–4.00 | 1,922 | 4.70 | 4.49 | 3.93 | 4.36 | 4.73 | **+0.37** |

## Zero-lag correlation and peak offset, each band against y

| z range | g→y lag0 | r→y lag0 | i→y lag0 | z→y lag0 | g→y peak | r→y peak | i→y peak | z→y peak |
|---|---|---|---|---|---|---|---|---|
| 0.00–0.36 | 0.897 | 0.965 | 0.978 | 0.985 | (0,0) | (0,0) | (0,0) | (0,0) |
| 0.36–0.49 | 0.654 | 0.894 | 0.930 | 0.957 | (0,0) | (0,0) | (0,0) | (0,0) |
| 0.49–0.78 | 0.465 | 0.792 | 0.815 | 0.939 | (0,0) | (0,0) | (0,0) | (0,0) |
| 0.78–1.01 | 0.589 | 0.708 | 0.792 | 0.889 | (0,0) | (0,0) | (0,0) | (0,0) |
| 1.01–1.22 | 0.728 | 0.797 | 0.894 | 0.942 | (0,0) | (0,0) | (0,0) | (0,0) |
| 1.22–1.44 | 0.727 | 0.771 | 0.699 | 0.942 | (0,0) | (0,0) | (0,0) | (0,0) |
| 1.44–2.00 | 0.810 | 0.857 | 0.898 | 0.938 | (0,0) | (0,0) | (0,0) | (0,0) |
| 2.00–4.00 | 0.857 | 0.894 | 0.919 | 0.937 | (0,0) | (0,0) | (0,0) | (0,0) |
