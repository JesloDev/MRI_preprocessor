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
├── data_model/           Layer 4 (NEW): dataset/subject/acquisition model
│   ├── schema.py          Dataset, Subject, Acquisition, ClassificationResult
│   ├── relationships.py   Sidecar discovery (bval/bvec/json/confounds)
│   ├── subject_inference.py  BIDS / DirectoryDepth / Regex / Auto strategies
│   ├── builder.py         Flat manifest -> Dataset tree
│   └── subject_context.py  SubjectContext: shared per-subject state passed
│                           into run_pipeline_for_file as `context`
├── pipeline/             Layer 5: pipeline/step architecture
│   ├── step_registry.py  ProcessingStep schema + registry + @step decorator
│   ├── steps_structural.py   Structural MRI step implementations
│   ├── steps_diffusion.py    Diffusion MRI step implementations
│   ├── steps_functional.py   Functional MRI step implementations
│   ├── dependency_resolver.py Auto-inserts missing prerequisite steps
│   ├── dag.py             Executes an ordered step list against one file
│   └── bootstrap.py       Imports all step modules to trigger registration
├── execution/            Layer 6: job orchestration
│   └── engine.py          Background-thread job runner + in-memory job store
│                           Builds Dataset -> groups by Subject -> builds one
│                           SubjectContext per Subject -> calls
│                           run_pipeline_for_file per Acquisition
├── provenance/           Layer 7 (cross-cutting): processing history
│   └── tracker.py         Append-only JSON-lines provenance log per job
├── output/               Output tree management
│   └── structure.py       Creates the standard output directory layout
└── main.py               Layer 8: REST API (Flask) -- thin, no business logic
```

## Data flow for one job

```
POST /api/jobs {input_dir, output_dir, step_ids, step_params}
        │
        ▼
execution.engine.start_job()  (spawns background thread)
        │
        ▼
ingestion.scanner.scan_directory()      -> flat manifest of files
        │
        ▼
detection.classifier.classify_manifest() -> each file tagged with a modality
        │
        ▼
qc.validators.validate_manifest()        -> invalid files dropped, reported
        │
        ▼
data_model.builder.build_dataset()      -> Dataset tree:
        │                                  Dataset -> Subject[] -> Acquisition[]
        │                                  (sidecars discovered, header metadata
        │                                   read, ClassificationResult attached)
        ▼
pipeline.dependency_resolver.resolve_pipeline()
        -> expands user's step selection with any missing prerequisites,
           returns a topologically valid execution order
        │
        ▼
for each Subject in Dataset:
    SubjectContext(subject)   <- one per subject, shared across its acquisitions
        │                       (carries sibling-acquisition refs, sidecars,
        │                        masks produced so far, transforms, etc.)
        ▼
    for each Acquisition of that Subject:
        pipeline.dag.run_pipeline_for_file()
            -> runs each applicable step in order, feeding output -> input,
               logging every step to provenance.tracker
            -> step outputs (brain_mask, transform, ...) are written
               back into the SubjectContext via context.update({...}),
               so subsequent acquisitions see them.
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

## The Dataset / Subject / Acquisition data model (PR1)

The flat manifest from `ingestion/scanner.py` treats every file as an
independent unit. That works for single-file steps (reorientation, N4,
resampling) but breaks the moment a step needs to know about another
file belonging to the same subject (FLAIR-to-T1 registration, DWI
tensor fitting that needs bval/bvec, fMRI confound regression).

`app/data_model/` introduces a Dataset tree:

    Dataset
      └── Subject
            └── Acquisition (image + associated_files + metadata + classification)

`build_dataset(input_dir, subject_strategy=...)` walks the scanner
manifest, applies a subject-inference strategy (BIDS / DirectoryDepth /
Regex / Auto), discovers sibling sidecars (bval/bvec/json/confounds),
reads lightweight NIfTI header metadata (shape, ndim, voxel_size), and
runs the existing filename classifier to produce a `ClassificationResult`
on each acquisition.

`SubjectContext` is the seam between the dataset-aware engine and the
file-level step contract. It is built once per subject and passed as
the `context` argument to `run_pipeline_for_file` for every acquisition
of that subject. It:

  - Quacks like a dict (`.get()`, `.update()`, `__getitem__`,
    `__setitem__`, `__contains__`, `.keys()`, `.items()`, `.values()`)
    so existing step code that does `context.get("bval_path")` or
    `context.update({"brain_mask": ...})` keeps working unchanged.
  - Exposes structured subject/acquisition/sidecar data via
    `get_acquisitions(modality)`, `get_primary_acquisition(modality)`,
    `get_reference_image(modality)`.
  - Surfaces flat convenience keys for the primary acquisition of each
    modality (`t1_path`, `dwi_path`, `fmri_path`, `bval_path`,
    `bvec_path`, `confounds_path`, `reference_image`).
  - Routes step-produced `*_mask` and `*transform` writes into the
    structured `.masks` and `.transforms` collections, so they're
    queryable structurally as well as via `.get()`.

A subject may have multiple acquisitions of the same modality (two T1s,
two DWIs). `acquisitions` is `Dict[str, List[Acquisition]]` keyed by
modality; flat keys surface the primary (first) acquisition only.
Secondary acquisitions are accessible via the structured helpers.

## Known simplifications (by design, for this scaffold)

- `brain_extraction` is a placeholder Otsu-threshold implementation,
  not SynthStrip/HD-BET.
- `fmri_motion_correction` and `dwi_motion_eddy_correction` are
  reference SimpleITK/DIPY implementations, not FSL MCFLIRT/eddy.
- The job store is in-memory (lost on server restart). Fine for local
  single-user use; swap for a DB-backed store for multi-user/production.
- Modality classification is filename-heuristic. Swap for pybids or
  DICOM-header-based classification for stricter datasets.
