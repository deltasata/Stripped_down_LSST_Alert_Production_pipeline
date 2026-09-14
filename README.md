# Stripped-down LSST Alert Production for simulated lensed transients

## Goal

Work out how **DIASources** (the detections that become alerts) and **DIAObjects** (the objects
that tie those detections together into a transient's time series) would be produced by Rubin/LSST
if a **lensed transient** were observed — and reproduce that faithfully on our own simulations,
fast enough to run at scale.

The unit of interest is the multiply-imaged transient: several images of the same source, close
together on the sky, varying with time delays between them. What we need to know is which of those
images become separate DIASources, which merge, how they get grouped into DIAObjects across visits,
and what the resulting light curves and postage stamps look like.

## Two ways to do this

**Option 1 — run the full LSST pipeline in production mode** on LSST-like exposures after
injecting the simulated transients.

* *Pros*: guaranteed to be exactly what production does. No reimplementation risk.
* *Cons*: very large overheads and long run times. Practical mainly when injecting into **real**
  LSST exposures (e.g. DP1/DP2); building suitable exposures from scratch is difficult in its own
  right.

**Option 2 — understand the AP pipeline, keep only the tasks that matter for us, and run a
stripped-down version** on our own simulations, on cutouts that can be much smaller than a full
CCD.

* *Pros*: much faster, so it is feasible to run at scale. Full control over every input. Many
  production overheads are unnecessary here because our simulations *already know* the answer —
  the PSF and the PSF-matching kernel are constructed as part of the simulation, so they do not
  have to be estimated.
* *Cons*: the retained tasks must be applied correctly, so that the end result is the same as if
  the object had really been observed by LSST. And the stripped-down version has to be re-checked
  against each update of the production pipeline.

**This repository takes Option 2**, for speed and for processing at scale.

## What is actually run

Both notebooks call the **real LSST tasks**, not reimplementations of them:

| Stage | Task (from the LSST stack) |
|---|---|
| PSF matching + subtraction | `lsst.ip.diffim.subtractImages.AlardLuptonSubtractTask` |
| Detection + measurement on the difference | `lsst.ip.diffim.detectAndMeasure.DetectAndMeasureTask` |
| DIASource → DIAObject association | `lsst.ap.association.association.AssociationTask` |

Per visit the loop does: build the science and template `ExposureF` → `_prepareInputs` →
`runConvolveTemplate` with the simulation's own matching kernel → `DetectAndMeasureTask.run` →
`AssociationTask.run` against the current DIAObject table → update DIAObject positions → score
every DIASource with two real/bogus CNNs → keep the postage-stamp triplets.

**DIAObject positions** are handled as production does: after every visit each object's `ra`/`dec`
is recomputed as the *unweighted spherical mean* of its full DIASource history
(`lsst.geom.averageSpherePoint`, matching `ap_meanPosition`/`MeanDiaPosition`), and the next visit
is matched against those updated positions. The notebooks print the whole DIAObject table after
each visit, so the positions can be watched as they move.

Association is one-to-one: a DIAObject takes at most one DIASource per visit, and a DIASource whose
nearest object is already taken starts a new DIAObject. The loop asserts this
(`handshake_violations`, which must stay 0).

## What is deliberately skipped

Because the simulation supplies these exactly, or because they are not relevant to this study:

* PSF estimation — the PSF is an input to the simulation.
* Fitting the Alard–Lupton matching kernel — the analytic matching kernel between the template and
  visit PSFs is constructed by the simulation and injected directly.
* ISR, calibration and template generation — the frames are simulated on a common photometric and
  astrometric footing.
* Deblending (off by default in the DIA path), sky sources, streak masking, background
  estimation, aperture correction.
* APDB writes, alert packaging, and the in-pipeline reliability cut.

Detection is configured with `thresholdValue=5.0`, `thresholdType='pixel_stdev'`,
`thresholdPolarity='both'`, `doSkySources=False`, `doMaskStreaks=False`,
`doSubtractBackground=False`, `reEstimateBackground=False`, `doTempLocalBackground=False`,
and the edge flags removed from `badSourceFlags` — see the task-setup cell.

## Real/bogus models

Two packaged CNNs are shipped in `rb_models/`, each a self-contained directory holding the
architecture, the trained weights and a metadata file. Both take a 51×51×3 cutout stacked in the
order **(difference, science, template)** — the same order the production interface uses — and
return P(real).

| Column | Directory | Version | Trained on |
|---|---|---|---|
| `rb_score_v01` | `tac_cnn_comcam_2025-02-18` | 0.1 | ComCam + injections |
| `rb_score_v03` | `tac_cnn_lsstcam_2026-02-26` | 0.3 | LSSTCam + injections |

Both notebooks load them through the same `load_rb_model()` function and score every DIASource with
both. The scores are recorded only; nothing is filtered on them here.

## Repository layout

```
FakeLSne_DIA_analysis_SetupA.ipynb    free parameters (params2.py + sie_quads.py SIE solver)
FakeLSne_DIA_analysis_SetupB.ipynb    GLEE catalogue: physical magnifications, delays, geometry
Fake_LSNe_gen.py                      image/light-curve simulation and matching kernels
params2.py                            Setup A parameters
sie_quads.py                          SIE lens solver -> image positions
glee_catalog2.npz                     GLEE lens catalogue used by Setup B
rb_models/                            the two real/bogus CNN packages
```

**The two notebooks are identical except for the input cell** (cell 3) and the title. Everything
downstream — helpers, task setup, the per-visit loop, the plots — is byte-identical between them,
so any difference in the results comes from the inputs alone.

Each runs to a light-curve plot and a postage-stamp grid, one row per DIAObject.

## Environment

Everything runs inside the Rubin Science Platform container:

* **`sciplat-lab_r30_0_7_rsp2827.sif`** (≈3.8 GB), built from
  `docker://lsstsqre/sciplat-lab:r30_0_7_rsp2827`
* LSST Science Pipelines release **v30_0_7**, conda env `lsst-scipipe-12.2.0-exact`, Python 3.13.9
* Base image Debian 13 (trixie); `torch` 2.7.1 is available inside for the RB models

```bash
apptainer exec --bind /path/to/workdir sciplat-lab_r30_0_7_rsp2827.sif \
    jupyter notebook --no-browser
```

Run the notebooks on the **stack kernel** — a conda environment that cannot `import lsst.*` will
fail at the first import cell.

Results are release-dependent: the pipeline internals change between releases, so re-check against
the version you run.

## Scope of this README

This file covers *what is run and how*. The scientific discussion — what the results mean for
lensed transients, where the stripped-down version is and is not equivalent to production, and the
limits of that equivalence — is documented separately.
