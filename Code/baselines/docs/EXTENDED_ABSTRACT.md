# Where does the missing band live? A capacity-matched study of near-infrared band prediction in HSC imaging

**Extended abstract**

---

## Abstract

Predicting an unobserved photometric band from observed ones is usually
treated as an image-to-image regression problem and solved with a large
convolutional network. We argue that the more useful question is not *how well*
such a model performs, but *what kind of information it is exploiting* — and we
show that this can be answered by building a ladder of models whose parameter
counts span six orders of magnitude and whose physical content is explicit.

Using 119,129 Subaru Hyper Suprime-Cam galaxies at `0 ≤ z < 0.4`, we predict
the *y* band (9762 Å) from *g, r, i, z* 64×64 cutouts. We first measure three
properties of the data directly, then build models that each encode exactly one
of them. The central result is that a model with **five parameters, one of
which is a width measured independently from the data rather than fitted,
outperforms a five-parameter learned linear model by 5.0%** — while a
1.9-million-parameter U-Net buys only a further 11.9%. Per parameter, the
U-Net is roughly 166,000× less efficient than the simplest baseline. We
conclude that for this task the remaining error is **spatial rather than
spectral**, that it is largely attributable to a measurable optical effect, and
that capacity has ceased to be the binding constraint.

---

## 1. Motivation

Multi-band astronomical surveys are routinely incomplete: a field may be
imaged deeply in some filters and shallowly or not at all in others. Predicting
a missing band is therefore of practical value, and modern image-to-image
networks do it well. But a network that reaches a good score tells you little
about *why* the problem was hard, and nothing about whether a different
instrument, exposure or filter set would change the answer.

We therefore invert the usual order of work. Rather than fitting a large model
and interpreting it afterwards, we measure properties of the data first, then
construct minimal models that encode one property each. A model built this way
makes a falsifiable claim: if the property it encodes is not the one that
matters, the model will not improve on a baseline that lacks it.

## 2. Data

We use the GalaxiesML release of Subaru HSC imaging with spectroscopic
redshifts, restricted to `0 ≤ z < 0.4` and using the survey's official splits:
**84,911 training, 17,146 validation and 17,072 test galaxies**. Each galaxy is
a 64×64 pixel cutout in five bands. The four bluer bands are inputs; *y* is the
target.

Two facts set the shape of the problem. First, *y* at 9762 Å lies **redward of
every input** — 864 Å (9.7%) beyond *z* at 8898 Å — so the prediction is an
extrapolation off the end of the filter set, not an interpolation between
observed bands. Second, *z* and *y* are separated by only 9% in wavelength, so
the two filters see very similar light: simply copying *z* with a single
multiplicative colour correction already performs respectably, and the
remaining error is correspondingly subtle.

Images are standardised per band using statistics computed on the training
split only. Non-finite pixels are masked out of both the loss and every metric.

## 3. Three measurements

Before modelling, we characterise the data with three independent analyses.

**(a) The band-to-band relation varies with redshift.** Refitting a copy-*z*
model independently in each redshift bin and comparing each bin against its
*adjacent* bins — not against a global mean, which is contaminated by
high-redshift outliers — gives excesses of **+0.28 (95% CI [+0.21, +0.40])** in
the [O III]+Hβ window and **+0.20 (CI [+0.10, +0.29])** for Hα. The 4000 Å
break window gives **+0.14 (CI [−0.00, +0.26])**, which we report as not
significant. Confidence intervals are from 400 bootstrap resamples.

**(b) The spectral signal is dominated by one band and is close to linear.**
Partial correlations, holding the other bands fixed, give standardised weights
of **z +0.55, i +0.37, r +0.24, g −0.15**. That *z* dominates is robust across
every estimator we tried and in 8 of 9 redshift bins. The *r*-versus-*i*
ordering is **not** robust — they differ by 0.13 and swap between estimators —
so we claim only the former. The negative weight on *g* is a colour correction,
subtracting blue light rather than contributing infrared flux.

**(c) The y band is spatially broader than z.** Pixel-to-pixel
cross-correlation between bands shows *y* wider than *z* in all eight redshift
bins, by **+0.05 to +0.40 px** of second-moment width, with band-to-band peak
lag at (0, 0) throughout, confirming the bands are correctly registered. This
is the decisive measurement: **no per-pixel rescaling of an image can change
its width.** A model restricted to reweighting pixels is therefore limited by
optics, not by parameter count, however many parameters it has.

## 4. Models

Every model shares the same splits, standardisation, loss, evaluation code and
aperture photometry, so the only thing that differs between rows is the model
itself.

| | Model | Params | Encodes |
|---|---|---|---|
| B0 | Copy-*z*, one multiplicative constant | 0 | — |
| B1 | Pixelwise linear, 1×1 convolution | 5 | colour |
| B2 | U-Net encoder–decoder | 1,928,705 | space, learned freely |
| B3 | B2 + FiLM redshift conditioning + heteroscedastic Gaussian NLL | 1,975,778 | space, redshift, uncertainty |
| **P1** | Spectral mix, then a Gaussian blur whose width comes from (c) | **7** | optics |
| **P2** | Per-pixel SED fit in log wavelength, evaluated at 9762 Å | **4** | wavelength |
| M1 | MLP on 59 catalogue columns, **no pixels** | 16,001 | catalogue content |

B3 is the capacity-matched control: within 2.4% of B2's parameter count by
construction, so a physics model that beats it cannot be dismissed as merely
having more capacity.

## 5. Results

All rows are the mean over three seeds (42, 123, 2026) on the 17,072-galaxy
test split, in physical flux units.

| Model | Params | Test MAE ↓ | σ(mag) ↓ | Outlier % ↓ |
|---|---|---|---|---|
| B0 (Copy-*z*) | 0 | 0.3505 | 0.0431 | 0.504 |
| **P2 (SED)** | 4 | 0.3279 ± 0.0001 | 0.0428 | 1.125 |
| B1 (Linear) | 5 | 0.3276 ± 0.0000 | 0.0423 | 1.217 |
| **P1 (PSF)** | 7 | **0.3113 ± 0.0001** | 0.0362 | 1.077 |
| B2 (U-Net) | 1,928,705 | **0.2744 ± 0.0006** | 0.0164 | **0.264** |
| B3 (U-Net+FiLM+NLL) | 1,975,778 | 0.2803 ± 0.0004 | **0.0143** | 0.287 |

### 5.1 A measured parameter beats a fitted one

P1's blur width is initialised at the value measured in §3(c) but is free to
move. Freezing it — removing that freedom entirely — changes almost nothing:

| | Trainable params | Test MAE |
|---|---|---|
| B1 (linear) | 5 | 0.32758 |
| P1, width learned | 7 | 0.311273 ± 0.000121 |
| P1, width **frozen at the measured value** | **5** | 0.311754 |

The difference is **0.15%**. So **at the same parameter count as B1, P1 is 5.0%
better on MAE and 14% better on σ(mag)**, entirely because one number came from
an independent measurement rather than from gradient descent. The prior is not
a regulariser or an initialisation; it is a measured quantity that transfers.

Where the learned width can be compared with the measurement, the two agree. In
the redshift bin containing 93.6% of the galaxies, P1 fits **0.8178, 0.8206,
0.8222 px** across three seeds against **0.8675 px** measured by
cross-correlation — within 6%, with a seed-to-seed spread of 0.004 px. Two
unrelated methods, one number. The sparse second bin (6.4% of galaxies, nine
times the seed spread, and a thin slice of a much wider measurement bin) is
recorded as a caveat rather than a second confirmation.

### 5.2 A negative result that explains itself

P2 reaches 0.3279 against B1's 0.3276 — indistinguishable. Its own fitted
coefficients say why: it retains the constant term (gain 1.01) but **halves the
spectral slope (0.53) and discards the curvature (0.05)**. Extrapolating along
a curvature measured over 4754–8898 Å out to 9762 Å overshoots. Once damped,
P2 *is* an affine function of the four band fluxes — which is exactly B1. The
model independently rediscovers that the spectral dimension is already
exhausted by a linear combination, and reports this through named physical
terms. The question cannot be put to B2.

### 5.3 Locating the residual error

Adding a 737-parameter convolutional correction to P2 reaches **0.2994**
(−8.7%), a larger gain than P1's entire improvement, and its spectral gains
then rise from `[1.01, 0.54, 0.05]` to `[1.06, 0.68, 0.23]`. With spatial
structure handled elsewhere, the spectral terms become trustworthy again: the
damping in §5.2 was partly compensating for missing spatial capacity.

### 5.4 Uncertainty, and a negative result on conditioning

B3's predicted uncertainties are well calibrated: the spread of (y−μ)/σ is
**1.037** where 1.0 is ideal, with coverage of **53.0 / 70.7 / 94.9%** against
nominal 50 / 68 / 95, and Spearman(σ, |error|) = **+0.290**, so σ tracks where
the error actually is rather than being a single flat number.

However, FiLM conditioning on redshift moves B3's prediction by a relative
**0.0109, 0.0124 and 0.0092 across the three seeds — about 1% each time**. B3's
advantage over B2 in photometry comes from the variance head and the NLL
objective, **not** from redshift. This is notable because §3(a) shows the
band relation genuinely *does* vary with redshift: the information is present,
and this architecture did not exploit it.

### 5.5 How much is in the catalogue alone?

M1 sees no pixels. On the same test galaxies it reaches σ(mag) **0.0391 ±
0.0005**, against 0.0545 for a zero-parameter catalogue baseline — and against
0.0431 and 0.0423 for B0 and B1, which *do* see images. A small network on
catalogue columns therefore matches or beats pixel-based linear models, while
remaining 2.4× worse than B2. The imaging carries real information, and this
quantifies how much. (M1 predicts a catalogue magnitude directly while the
image models perform aperture photometry on a predicted image; a few per cent
between the two columns is estimator, not skill.)

## 6. Discussion

**Capacity is no longer the binding constraint.** B1 buys 6.5% over B0 with
five parameters; B2 buys a further 16.2% with 1.9 million. Per parameter that
is roughly 166,000× less efficient. A collapse of this size is the signature of
a problem where additional capacity is being spent on progressively smaller
residuals rather than on a missing mechanism.

**The residual is spatial, not spectral.** Three independent lines of evidence
agree: the spectral analysis of §3(b) predicts that a linear model exhausts the
colour information, and P2 confirms it by converging to exactly that; the
optical measurement of §3(c) predicts that spatial extent is missing, and P1
confirms it by recovering 5% of error with one measured width; and the residual
ablation of §5.3 shows the largest single gain comes from adding spatial
capacity to a purely spectral model.

**On negative results.** Three of our findings are negative — P2 ties B1, FiLM
is inert, the 4000 Å break window is not significant — and each is more
informative than the corresponding positive would have been, because each is
accompanied by a mechanism. We regard the ability to state *why* a component
failed as the main practical argument for small, interpretable models alongside
large ones.

## 7. Conclusions

1. A single width measured by cross-correlation, inserted into a five-parameter
   model without being fitted, outperforms an equally sized learned model by
   5.0% — and the fitted value, where free, reproduces the measurement to 6%.
2. The spectral content of the problem is exhausted by a linear combination of
   bands; a physically parameterised SED model converges to exactly that and
   reports it through its own coefficients.
3. The remaining error is spatial, consistent with the *y* filter's broader
   point spread function.
4. A calibrated uncertainty head is achievable at this scale; naive redshift
   conditioning is not useful, despite the band relation genuinely varying with
   redshift.
5. Catalogue metadata alone reaches the performance of pixel-based linear
   models, bounding the marginal value of the imaging.

**Limitations.** All results are confined to `0 ≤ z < 0.4`, where the measured
PSF difference is small (+0.05 to +0.10 px); the larger differences at higher
redshift are untested here, and P1 should be expected to matter more there.
P1's second redshift bin is sparsely populated. A pretrained vision transformer
is the obvious next comparison and has not yet been run.

---

### Reproducibility

Every number traces to a committed script and a published artefact. Six models
share identical splits, standardisation, loss and evaluation code; all trained
rows use seeds 42, 123 and 2026 with seed-to-seed spreads reported. The
pipeline has 227 unit checks and three end-to-end tests on synthetic data, and
reproduces its own published baseline bitwise from a fresh data preparation on
a different machine.
