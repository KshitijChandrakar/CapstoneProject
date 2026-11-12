# Does "y is a scaled copy of z" hold at higher redshift?

B0 refitted and re-evaluated independently in each redshift bin, on the **test** split. B0 has zero learnable parameters, so each bin gets its own median z−y colour and therefore its own scale.

> **Read the controlled column, not the raw one.** Higher-redshift galaxies are much fainter (median m_z runs from ~17 to ~22 across this range), and faint photometry is noisier whatever the filters are doing. The raw Δm NMAD therefore mostly tracks brightness. The *controlled* ratio holds apparent magnitude fixed and is the only column that speaks to the spectral question.

| z range | window | n train | n eval | median m_z | median(m_z − m_y) | B0 scale | Δm NMAD (raw) | ratio (controlled) | z band rest | y band rest |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00–0.36 | control | 79,445 | 15,998 | 17.73 | +0.1782 | 1.1784 | 0.0426 | **1.00×** | 7541 Å | 8273 Å |
| 0.36–0.49 | Ha | 23,103 | 4,608 | 18.96 | +0.2029 | 1.2055 | 0.0313 | **1.17×** | 6244 Å | 6851 Å |
| 0.49–0.78 | control | 61,831 | 12,438 | 19.62 | +0.2009 | 1.2032 | 0.0511 | **0.94×** | 5442 Å | 5971 Å |
| 0.78–1.01 | [OIII] + Hb | 15,504 | 3,001 | 21.41 | +0.1490 | 1.1470 | 0.1507 | **1.26×** | 4696 Å | 5151 Å |
| 1.01–1.22 | control | 5,523 | 1,146 | 21.81 | +0.1745 | 1.1743 | 0.1478 | **1.01×** | 4207 Å | 4616 Å |
| 1.22–1.44 | 4000A break | 3,799 | 715 | 22.06 | +0.0978 | 1.0943 | 0.2099 | **1.25×** | 3819 Å | 4190 Å |
| 1.44–2.00 | control | 5,214 | 1,086 | 20.59 | +0.0262 | 1.0244 | 0.1094 | **1.19×** | 3271 Å | 3589 Å |
| 2.00–4.00 | control | 10,154 | 1,922 | 20.62 | +0.1071 | 1.1037 | 0.1537 | **1.78×** | 2224 Å | 2440 Å |

Raw degradation, lowest bin to worst: **4.9×** (0.0426 → 0.2099). **Brightness-controlled: 1.8×** (worst bin z=2.00-4.00). The gap between those two numbers is the brightness effect, not a spectral one.

### Per-window test (controlled)

Each window bin's controlled ratio against the mean of its **adjacent** control bins, with a 95% bootstrap interval (400 resamples) on the excess. A window counts as supported only when that interval clears zero — not from the point estimate alone.

| Window | z range | controlled ratio | adjacent controls | their mean | excess | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| Ha | 0.36–0.49 | 1.17× | 0.00-0.36, 0.49-0.78 | 0.97× | +0.20 | [+0.10, +0.29] | **SUPPORTED** |
| [OIII] + Hb | 0.78–1.01 | 1.26× | 0.49-0.78, 1.01-1.22 | 0.98× | +0.28 | [+0.21, +0.40] | **SUPPORTED** |
| 4000A break | 1.22–1.44 | 1.25× | 1.01-1.22, 1.44-2.00 | 1.10× | +0.14 | [-0.00, +0.26] | **not significant** |

Controls are the adjacent bins below z=2. Above it both filters sample rest-frame UV, a different regime rather than a control, so those bins are measured but excluded from the baseline. For reference, the mean over **all** control bins including the high-z ones is 1.18× — that is the contaminated value, retained only so the difference is visible. Do not quote it.

## Predicted break points

The z and y filters are only 864 Å apart (8898 and 9762 Å observed), so the single-scale assumption should fail where a spectral feature sits in one filter but not the other:

| Feature | Rest λ | Enters z band | Enters y band |
|---|---|---|---|
| Ha | 6563 Å | z = 0.36 | z = 0.49 |
| [OIII] | 5007 Å | z = 0.78 | z = 0.95 |
| Hb | 4861 Å | z = 0.83 | z = 1.01 |
| 4000A break | 4000 Å | z = 1.22 | z = 1.44 |

See `redshift_sweep.png` for the measurement against these predictions.
