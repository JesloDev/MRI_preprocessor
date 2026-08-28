# MRI / fMRI Preprocessing Platform

A modular, dependency-aware, provenance-tracked MRI/fMRI preprocessing
engine with a REST API front end. Built so that new modalities and
processing steps can be added as isolated, self-registering modules
without touching the API, dependency resolver, or execution engine.

## What's implemented in this scaffold

- Full architecture: ingestion -> detection -> QC -> pipeline registry ->
  dependency resolution -> DAG execution -> provenance -> REST API.
- Real, working steps (not stubs) for:
  - **Structural** (T1/T2/FLAIR/PD): reorientation, N4 bias correction,
    brain extraction (placeholder), registration, resampling, intensity
    normalization.
  - **Diffusion** (DWI/DTI): denoising, Gibbs-ringing correction,
    motion/eddy correction, tensor fitting (FA/MD/AD/RD).
  - **Functional** (fMRI): dummy-volume removal, motion correction,
    temporal filtering, confound regression, spatial smoothing.
- Automatic prerequisite-step insertion (select `registration`, get
  `reorientation` -> `bias_field_correction` -> `brain_extraction`
  inserted automatically).
- Machine-readable, append-only provenance log per job.
- Standardized output directory tree, raw input never modified.
- Full pytest suite using synthetic (randomly generated) NIfTI data --
  no real patient data required to verify the engine works.

See `docs/ARCHITECTURE.md` for the layer-by-layer breakdown and
`docs/CAPABILITY_MATRIX.yaml` for the engine/license tracking per step.

## What's intentionally NOT done yet (by design)

This is a working core, not a finished clinical/research product:

- No cortical reconstruction (FreeSurfer) module yet.
- No tractography module yet.
- Brain extraction is a placeholder (Otsu threshold), not SynthStrip/HD-BET.
- No frontend UI -- REST API only (see `docs/API.md`); wire up a
  React/Tauri frontend against it whenever you're ready.
- No packaging/bundling yet -- see the conda-constructor plan discussed
  separately; this repo is the thing you'd package.

## Quickstart (Linux)

```bash
./setup_linux.sh        # creates .venv, installs deps, runs tests
source .venv/bin/activate
./run_server.sh          # starts the REST API on http://127.0.0.1:5000
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"input_dir": "/path/to/your/mri/data"}'
```

See `docs/API.md` for the full endpoint reference and a complete
job-submission example.
