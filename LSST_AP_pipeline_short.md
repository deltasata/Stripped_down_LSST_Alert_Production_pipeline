###v30_0_7 stack

### Why a stripped-down pipeline

The full Rubin AP pipeline is built for real LSST exposures and everything around them -- ISR,
calibration, templates, database writes, alert packaging. Our images are custom lensed-supernova
mocks and we need millions of them at speed, so the notebooks keep only the stages that decide
what gets detected and how it is grouped: subtraction, detection, measurement, association into
DIAObjects, and the real/bogus cutouts. The output is training data, so this document certifies
from the pipeline's own code that those stages behave as production does, and where they don't.


---

### Q1 — How is an alert made, start to finish?
Correct the raw visit image (ISR, calibration) → fetch a template → solve for the matching kernel,
**convolve the template with it so the template's PSF matches the visit's, then subtract it from the
visit to form the difference image** → **decorrelate** the difference image (undo the pixel-to-pixel
noise correlation the matching convolution introduced) → **convolve the difference image with its own
PSF to form the matched-filter image, and threshold that at 5σ** to find blobs (footprints) and peaks
→ drop junk (sky sources, trailed sources) → score each DIASource with the real/bogus CNN and discard
scores below 0.5 → associate each surviving DIASource with a DIAObject (within 1 arcsec) → update the
DIAObject → write to the database and send the alert.
[pipeline file `ApPipe.yaml`]


### Q2 — How many times is the image convolved, and what is the result of each?
**Three times on the full image**, plus a fourth local one during measurement:

1. **Template ⊗ matching kernel** → the *matched template*, which by construction now carries the
   **visit image's PSF**. Subtracting it from the visit gives the **difference image**.
   [`subtractImages.py:659-665`]
2. **Difference image ⊗ decorrelation kernel** → the **decorrelated difference image**. Convolution 1
   correlated neighbouring pixels' noise; this undoes it so the noise is white again and the 5σ
   threshold means what it says. It is done as a multiply in Fourier space, which is the same thing
   as a convolution. [`imageDecorrelation.py:643-646`, on by default]
3. **Decorrelated difference image ⊗ PSF** → the **matched-filter image**. 
4. **(per source, during measurement)** a small box around the peak is convolved with the PSF's local
   kernel inside the centroid fit. [`SdssCentroid.cc:243-277`]

**What is done on which image.** Footprints, peaks, the 5σ threshold (5 × the per-pixel √variance
*of the matched-filter image*) and the peak significances are all defined on the **matched-filter
image** (3). But every *measurement* — centroid, PSF flux, moments, extendedness — is then made on
the ordinary **decorrelated difference image** (2), not on the matched-filter image. Filtering
maximises the signal-to-noise for *finding* a point source; it would bias the *measuring* of one.
[`detection.py:837,842,847`; `detectAndMeasure.py:1050`]

### Q3 — How much is a detected blob grown before it becomes a DIASource?
After thresholding, each footprint is dilated outward by 2.4 × σ pixels, rounded to a whole number,
where σ is the Gaussian σ of the difference image's PSF (which is the visit image's PSF). Touching footprints are merged into one DIASource. 
[`detection.py:671`]

### Q4 — How is a DIASource's position (and the stamp centre) decided?
Not the flux-weighted centroid of the whole footprint. It **starts at the brightest peak** and
**refines it to sub-pixel accuracy** by fitting a small polynomial to the difference image *after a
local patch of it has been convolved with the PSF*. That refined position is the DIASource position
and the centre of the postage stamp.
[`SdssCentroid.cc`, `detectAndMeasure.py`]

### Q4a — Which peak does it start from?
The brightest peak in the footprint. (Peaks are sorted by pixel value, brightest first; it takes the
first one.)
[`SdssCentroid.cc`, `InputUtilities.cc`]

### Q4b — What pixels does it use, and how big is that box?
Only a small box around the brightest peak — not the whole footprint, and nothing is thresholded
away. Box side = (PSF kernel stamp width + 4) pixels, about 15–29 px in our runs. That size is set by
the **difference image's PSF** (i.e. the visit PSF), **not** by the matching kernel. It grows only if
the source is broad enough to trigger the adaptive binning.
[`SdssCentroid.cc`]


### Q4c — Does light from other images in the same footprint move the DIASource position?
Yes, but only if another image falls inside that small box. A close companion drags the position
toward itself; a well-separated one does nothing. (Measured: two sources 6 px apart — the shift grew
from 0.1 px to 1.8 px as the companion brightened from zero to equal.)
[`runs/q4_centroid/q4d_mechanism.py`]


### Q5 — Is there an extendedness value in production, and how is it calculated exactly?
Yes. Every DIASource gets a number **`extendedness`** (0 = point-like → 1 = extended). It is only
recorded, never used to cut alerts. Exact recipe:
- Fit an elliptical Gaussian to the source and, separately, to the PSF; each fit's "size" is the
  trace of its second-moment matrix (Ixx + Iyy).
- `extendedness = 1 − exp( −½ · (source_size − PSF_size)² / PSF_size )`.
The Gaussian fitting is adaptive-moment fitting done by an outside library (GalSim's
`FindAdaptiveMom`), not by LSST code. As a blend flag it is weak: reliable only for very close pairs
(< ~1.6″); at 2–2.4″ it can wrongly say "point-like"; and it is noisy for isolated point sources in
poor seeing. The DIAObject columns `extendednessMax/Median/Min` exist in the database schema but are
**not computed** in this version.
[`plugins.py`, GalSim `PSFCorr.cpp` v2.8.4, `DiaSource.yaml`, `apdb.yaml`]

### Q6 — How is a DIAObject's position updated?
Production recomputes the DIAObject position every visit as the unweighted spherical mean of all its
DIASource positions so far, and matches the next visit's DIASources against that updated position.

### Q7 — What stamps are in an alert, and what goes to the real/bogus model?
Two separate sets of cutouts, both centred on the DIASource position (the refined brightest-peak
position from Q4):

**In the alert** — exactly three cutouts, named `cutoutDifference`, `cutoutScience` (the visit),
`cutoutTemplate`, in that order (checked in the alert schema itself). Each is photometrically
calibrated to nJy. **For an ordinary point-source transient the stamp is 30×30 pixels.** The size
rule is (I have to check with Ian): take (2 × the largest distance from the DIASource position to the edge of its footprint) and
clamp it between 30 and 102 pixels. A point source's footprint is small, so this falls below 30 and
is pinned to the 30-pixel floor → 30×30. It only grows (up to 102) when the footprint is large — i.e.
blended/merged images or poor seeing. (Verified on the loaded stack, and by measuring our own point
sources: footprint ~13–16 px → 30×30; merged 4-image footprint ~40 px → ~55–59 px.) No config
anywhere in the stack changes the 30/102 limits. Off the detector edge → a null cutout.

**To the real/bogus model** — three cutouts stacked in the order **difference, visit (science),
template** (channel 0, 1, 2). These are a **fixed 51×51 pixels**, and use **raw** (uncalibrated)
pixel values. Off the edge they are zero-filled.

Same three images, same centre; but the model's cutouts are a fixed 51×51 raw, while the alert's are
size-to-fit (30–102) and calibrated. Our notebook's real/bogus path matches the model input exactly
(51×51, raw, same order, same centre).
[alert schema `lsst.v10_0.alert.avsc`, `packageAlerts.py`, `rbTransiNetTask.py`,
`rbTransiNetInterface.py`]

