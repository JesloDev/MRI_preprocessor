"""
Central configuration for the MRI Preprocessing Platform.

Nothing modality-specific lives here. This module only defines the
standard output directory layout and a couple of runtime constants that
every other layer depends on.
"""
import os

# Standard output tree created under the user's chosen output directory.
OUTPUT_SUBDIRS = [
    "metadata",
    "qc",
    "qc/images",
    "logs",
    "processed",
    "derivatives",
    "provenance",
    "ML",
]

# Recognized NIfTI extensions during ingestion scanning.
NIFTI_EXTENSIONS = (".nii", ".nii.gz")

# Where the job registry / in-memory state lives for this process.
# (Swap for Redis/DB if you need multi-process job execution later.)
JOB_STORE_PATH = os.environ.get(
    "MRIPREP_JOB_STORE",
    os.path.join(os.path.expanduser("~"), ".mripreprocessor", "jobs.json"),
)

HOST = os.environ.get("MRIPREP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MRIPREP_PORT", "5000"))
