# Architecture

This mirrors the build order agreed on in planning: the GUI/API is a
thin front end over a real orchestration engine, not a place where
logic lives.

```
app/
├── config.py            Central config (output tree layout, ports)
├── ingestion/            Layer 1: dataset discovery
│   └── scanner.py        Recursive directory walk, subject inference
├── detection/            Layer 2: modality classification
│   └── classifier.py     Filename/path heuristic classifier
├── qc/                   Layer 3: metadata/QC engine
│   └── validators.py     NIfTI validity checks
├── pipeline/             Layer 4: pipeline/step architecture
│   ├── step_registry.py  ProcessingStep schema + registry + @step decorator
│   ├── steps_structural.py   Structural MRI step implementations
│   ├── steps_diffusion.py    Diffusion MRI step implementations
│   ├── steps_functional.py   Functional MRI step implementations
│   ├── dependency_resolver.py Auto-inserts missing prerequisite steps
│   ├── dag.py             Executes an ordered step list against one file
│   └── bootstrap.py       Imports all step modules to trigger registration
├── execution/            Layer 5: job orchestration
│   └── engine.py          Background-thread job runner + in-memory job store
├── provenance/           Layer 6 (cross-cutting): processing history
│   └── tracker.py         Append-only JSON-lines provenance log per job
├── output/               Output tree management
│   └── structure.py       Creates the standard output directory layout
└── main.py               Layer 7: REST API (Flask) -- thin, no business logic
```

## Data flow for one job

```
POST /api/jobs {input_dir, output_dir, step_ids, step_params}
        │
        ▼
execution.engine.start_job()  (spawns background thread)
        │
        ▼
ingestion.scanner.scan_directory()      -> manifest of files + inferred subjects
        │
        ▼
detection.classifier.classify_manifest() -> each file tagged with a modality
        │
        ▼
qc.validators.validate_manifest()        -> invalid files dropped, reported
        │
        ▼
pipeline.dependency_resolver.resolve_pipeline()
        -> expands user's step selection with any missing prerequisites,
           returns a topologically valid execution order
        │
        ▼
for each valid, classified file:
    pipeline.dag.run_pipeline_for_file()
        -> runs each applicable step in order, feeding output -> input,
           logging every step to provenance.tracker
        │
        ▼
execution.engine updates job status/progress (pollable via GET /api/jobs/<id>)
```

## Adding a new processing step

1. Pick the right `steps_*.py` file (or create a new one for a new modality).
2. Write a function with signature
   `fn(input_path, output_dir, params, context) -> dict`.
3. Decorate it with `@step(id=..., name=..., modalities=[...], engine=...,
   depends_on=[...], produces=[...], default_params={...})`.
4. Import the new module from `pipeline/bootstrap.py` if it's a new file.
5. Add an entry to `docs/CAPABILITY_MATRIX.yaml`.
6. Add a test in `tests/`.

Nothing else needs to change -- the REST API, dependency resolver, and
DAG executor all discover new steps automatically through the registry.

## Known simplifications (by design, for this scaffold)

- `brain_extraction` is a placeholder Otsu-threshold implementation,
  not SynthStrip/HD-BET.
- `fmri_motion_correction` and `dwi_motion_eddy_correction` are
  reference SimpleITK/DIPY implementations, not FSL MCFLIRT/eddy.
- The job store is in-memory (lost on server restart). Fine for local
  single-user use; swap for a DB-backed store for multi-user/production.
- Modality classification is filename-heuristic. Swap for pybids or
  DICOM-header-based classification for stricter datasets.
