# REST API Reference

Base URL (default): `http://127.0.0.1:5000`

## GET /api/health
Returns server status and number of registered processing steps.

## POST /api/scan
Body: `{"input_dir": "/path/to/raw/data"}`
Scans the directory, classifies modalities, runs QC, returns a dataset
summary (subjects, file counts, modality counts, failed files).

## GET /api/capabilities
Returns every registered `ProcessingStep`, keyed by step id. This is
what a UI would use to render the "Available preprocessing" checklist.

## GET /api/capabilities/<modality>
Returns only the steps applicable to one modality (e.g. `T1`, `DWI`, `fMRI`).

## POST /api/pipeline/resolve
Body: `{"step_ids": ["registration"]}`
Returns the fully expanded, dependency-resolved, topologically ordered
pipeline -- this is what powers the "This operation requires N
prerequisite steps. Add automatically?" UI prompt.

## POST /api/jobs
Body:
```json
{
  "input_dir": "/path/to/raw/data",
  "output_dir": "/path/to/processed/data",
  "step_ids": ["bias_field_correction", "brain_extraction", "resampling"],
  "step_params": {
    "resampling": {"voxel_size": [1.0, 1.0, 1.0]}
  },
  "modalities": ["T1"]
}
```
Starts a background job, returns `{"job_id": "...", "status": "started"}`.

## GET /api/jobs
Lists all jobs known to this server process (in-memory).

## GET /api/jobs/<job_id>
Returns job status (`pending`/`running`/`completed`/`failed`), progress,
and (on completion) the full result including per-file outcomes and QC report.

## GET /api/jobs/<job_id>/provenance
Returns the full, ordered provenance log for that job -- every step
applied to every file, with parameters and metrics.
