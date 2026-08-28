"""
Output tree management: creates the standard output directory structure
under the user's chosen output directory. The user never has to create
or manage these folders manually.
"""
import os

from app.config import OUTPUT_SUBDIRS


def initialize_output_tree(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    for sub in OUTPUT_SUBDIRS:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    return output_dir


def subject_processed_dir(output_dir: str, subject_id: str, modality: str) -> str:
    path = os.path.join(output_dir, "processed", subject_id, modality)
    os.makedirs(path, exist_ok=True)
    return path


def derivatives_dir(output_dir: str, derivative_name: str) -> str:
    path = os.path.join(output_dir, "derivatives", derivative_name)
    os.makedirs(path, exist_ok=True)
    return path
