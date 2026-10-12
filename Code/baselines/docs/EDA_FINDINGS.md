# What the team's EDA tells us (and what it doesn't)

Read from `eda.zip` — five notebooks covering the GalaxiesML training, validation
and testing splits at both cutout sizes. This note records everything in there
that affects the B0/B1 baselines, so nobody has to re-open the notebooks.

Source file the notebooks opened:
`C:/Users/ancha/Downloads/5x64x64_training_with_morphology.hdf5`

---

## 1. The shape of the data

| Split | Galaxies | `image` shape | Metadata columns |
|---|---|---|---|
| Training | **204,573** | (204573, 5, 64, 64) float32 | 84 |
| Validation | 40,914 | (40914, 5, 64, 64) float32 | 84 |
| Testing | 40,914 | (40914, 5, 64, 64) float32 | 84 (+4 extra coord columns) |

The 127×127 versions hold the *same* galaxies in bigger cutouts — validation is
40,914 in both. This matters: it means `preprocessedMetadata.csv`, which was
built from the **127×127** training CSV, describes the same 204,573 galaxies as
the 64×64 training file. (It has 204,572 rows because `PreprocessMetadata.py`
drops one row with a missing `major_minor_axis_ratio`.)

**Correction (2026-10-13).** An earlier version of this note claimed that
`x_coord_x`, `x_coord_y`, `y_coord_x` and `y_coord_y` appear only in the testing
file. **That is wrong** — all four are present in all three splits. The claim
came from comparing the training notebook's 84-column table (which lists only
1-D datasets) against the testing notebook's raw key listing (which lists
everything). These four columns are 2-D, shape (N, 2), so they were silently
dropped from one list and not the other. No conclusion should be drawn from
that difference.

## 2. Data quality — the images and catalogue are clean

| Check | Result |
|---|---|
| Missing values in the 84 metadata columns | **0** (all three splits) |
| Duplicate rows | **0** |
| Infinite values | **0** |
| NaN pixels in the images | **0** (full scan of all 40,914 validation 127×127 images) |
| Infinite pixels in the images | **0** (same scan) |
| `specz_flag_homogeneous` | **True for every galaxy** — all redshifts are secure |

So the NaN-masking in `metrics.py` and the loss is a safety net, not something
these files will actually trigger. Leave it in — the sample file and any future
re-cut could differ — but don't expect it to do anything.

No sentinel magnitudes either: the cmodel magnitudes run from ~12.9 to ~33.5,
with no 99 or −99 values. The faint tail (y up to 33.5 mag) is real but
essentially noise, which is worth remembering when the worst galaxies turn up in
the results grid.

## 3. The magnitudes — these settle the B0 sign question

Mean cmodel magnitudes over the **full training split** (204,573 galaxies):

| Band | g | r | i | z | y |
|---|---|---|---|---|---|
| mean mag | 21.256 | 20.246 | 19.634 | **19.325** | **19.154** |

A smaller magnitude means a *brighter* object. So:

```
mean(m_z) − mean(m_y) = 19.3251 − 19.1538 = +0.1714 mag
```

The validation split agrees almost exactly (+0.1719 mag), so this is a stable
property of the dataset, not a fluke of one split.

**y is the brighter band.** Brighter means more flux, so `F_y / F_z` must be
greater than 1. That is the whole argument:

| Formula | Value | Verdict |
|---|---|---|
| `scale = 10^(+0.4 × 0.1714)` | **1.171** | > 1 — matches "y is brighter" ✅ |
| `scale = 10^(−0.4 × 0.1714)` | 0.854 | < 1 — says y is *fainter*, contradicting the measured magnitudes ❌ |

Using the medians instead of the means gives +0.196 mag and a scale of 1.198 —
same conclusion.

So the **`+` sign is correct**, confirmed on 204,573 real HSC galaxies, and this
can be said without touching an image. The `−` version would make the predicted
y image about 15% *dimmer* than z, when the catalogue says it should be about
17% brighter.

> ⚠️ One honest caveat: these numbers are for galaxies at **all** redshifts. Our
> sample is cut to z < 0.2, and its median colour will not be exactly +0.17.
> The *sign* is what's being established here, and that won't flip — but quote
> the number that `run_b0.py` prints from the actual sample, not +0.17.

## 4. Redshift — how much room we have

From the training split: median redshift **0.489**, 25th percentile **0.213**,
range 0.0101 to 3.998.

Counting from the metadata catalogue:

| Cut | Galaxies available |
|---|---|
| z < 0.15 (the value in `09-10-2026.md`) | 32,812 |
| **z < 0.2 (our brief)** | **47,704** |
| z < 0.3 | 69,647 |

Our 2,000-galaxy sample is a ~4% draw from the z < 0.2 pool. There is plenty of
headroom to widen the sample later — `sample_size` in `config.py` is the only
thing to change.

## 5. Other things the EDA shows

- **Band brightness ordering** is consistent (g faintest → y brightest), which
  is what you'd expect for red, low-redshift galaxies and is a good reason to
  expect z to be the most useful input band for predicting y.
- **`g_half_light_radius` has a minimum of −11.244** — a negative radius, which
  is unphysical. It's a measurement failure flag in the catalogue. We don't use
  half-light radius, so it doesn't affect us, but anyone using morphology
  features later should filter these.
- Several morphology columns bottom out at exactly 0.0 (`sersic_index`,
  `petro_rad`, `minor_axis`), which usually also means "fit failed".
- `specz_name` shows the redshifts come mostly from SDSS DR12/DR14 and WiggleZ.

## 6. What the EDA could **not** give us

The notebooks computed per-band pixel statistics and mean images, but only ever
**plotted** them — the numbers were never printed. So we still do not have:

- the per-band pixel **mean and standard deviation** needed for normalisation
  (these must be computed from the train split anyway, per the professor's rule,
  so this is no loss — but it does mean no shortcut);
- the actual images, obviously;
- the exact median `z − y` colour **of the 2,000-galaxy z < 0.2 sample**.

All three come out automatically the moment the real HDF5 is available.

---

## Consequences for our code

1. `config.py` now records the known split sizes (204,573 / 40,914), so
   `inspect_data.py` announces which GalaxiesML file it has been handed.
2. `inspect_data.py` now prints the measured median `z − y` next to the EDA's
   full-split value, and **raises a problem if the median comes out negative** —
   which would be the signature of the image band order not being g, r, i, z, y.
   That was flagged as an "important assumption" in `30-09-2026.md` and this is
   now an automatic check rather than a hope.
3. The candidate file paths now include the real GalaxiesML filenames, so
   dropping any of the three splits in place is enough for the code to find it.
