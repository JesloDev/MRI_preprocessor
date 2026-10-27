"""
Engine manifest schema + loader (PR13).

An EngineManifest is a structured declaration of ONE external engine --
its name, provider, capability, version, source (how to install it),
runtime support (CPU/CUDA/container), input requirements, parameters,
and outputs. It's the single source of truth for "what is this engine
and how do I get it".

Manifests live as YAML files in `app/engines/manifests/<id>.yaml`.
The loader reads all manifests at startup and exposes them via the
EngineManager.

Why YAML (not code):
  - Operators can audit/modify manifests without touching Python.
  - Future GUI can render installation dialogs directly from the
    manifest schema (provider, license, source, runtime options) --
    no hardcoded engine-specific UI logic.
  - Reproducibility: a manifest pin (version + checksums) makes a
    given engine installation bit-for-bit reproducible across
    machines, which is critical for ML preprocessing provenance.

The manifest is intentionally a flat model (no nested SourceSpec /
RuntimeSpec submodels) so it serializes cleanly to JSON for the API
and is easy for non-Python clients to consume.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class EngineManifest(BaseModel):
    """Structured declaration of one external engine.

    A manifest answers:
      - What is this engine?  (id, name, provider, description)
      - What does it do?      (capability, requirements, outputs)
      - How do I install it?  (source_type, source_channel, source_package)
      - What runtime?         (cpu, cuda, container)
      - License?              (license_type, license_notes, license_file_required)
      - Which version?        (version, checksums)
    """
    # Identity
    id: str                          # "synthstrip", "eddy", "mcflirt"
    name: str                        # display name: "SynthStrip"
    provider: str                    # "FreeSurfer", "FSL", "DKFZ"
    capability: str                  # "brain_extraction", "dwi_correction", ...
    version: str                     # "7.4.1"
    description: str = ""

    # The binary name that this engine's adapter looks for via
    # binary_discovery.find_binary(). This MUST match the
    # DiscoveryConfig.binary_name in the adapter file. The EngineManager
    # uses this to look up the binary in the local store; the StubInstaller
    # uses it to create a fake binary with the right name.
    binary_name: str = ""

    # License
    license_type: str = "non-commercial"   # "open-source" | "non-commercial" | "commercial"
    license_notes: Optional[str] = None
    license_file_required: bool = False   # True for FreeSurfer (license.txt)

    # Source (how to install)
    source_type: str = "conda"             # "conda" | "url" | "local" | "container" | "bundle"
    source_channel: Optional[str] = None  # conda channel URL
    source_package: Optional[str] = None  # conda package name
    source_url: Optional[str] = None       # direct download URL (source_type=url)
    source_container: Optional[str] = None # docker image (source_type=container)

    # PR16: Platform support. When empty, the engine is considered
    # platform-independent (pure Python). When non-empty, the engine
    # only runs on the listed platforms. This is what the PlatformResolver
    # uses to select the correct bundle: if the current platform isn't
    # in this list, the engine can't be installed natively.
    #
    # Platform identifiers follow the conda convention:
    #   linux-64, linux-aarch64, osx-64, osx-arm64, win-64
    platforms: List[str] = Field(default_factory=list)

    # PR16: Bundle base directory. When source_type="bundle", this is
    # the directory containing per-platform subdirectories. The
    # PlatformResolver appends the platform identifier to get the
    # concrete bundle path:
    #   <bundle_base>/<platform>/bin/<binary_name>
    # Example: /app/bundles/synthstrip/linux-64/bin/mri_synthstrip
    bundle_base: Optional[str] = None

    # Runtime
    runtime_cpu: bool = True
    runtime_cuda: bool = False
    runtime_container: Optional[str] = None  # docker image if containerized

    # Inputs / outputs (for UI hints + validation)
    requirements: List[str] = []   # ["dwi", "bval", "bvec", "brain_mask"]
    outputs: List[str] = []         # ["corrected_dwi", "rotated_bvecs"]

    # Reproducibility: pin the package with checksums per platform
    # (linux-64, osx-64, osx-arm64, win-64). Empty by default; populated
    # for engines we want to make bit-for-bit reproducible.
    checksums: Dict[str, str] = {}

    class Config:
        extra = "ignore"

    def to_info_dict(self, installed: bool = False,
                      install_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns a JSON-serializable info dict for the CLI / API.

        This is what `engines info <id>` and `GET /api/engines/<id>/info`
        return. It surfaces everything the user needs to decide whether
        to install + use this engine.
        """
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "capability": self.capability,
            "version": self.version,
            "description": self.description,
            "license": {
                "type": self.license_type,
                "notes": self.license_notes,
                "file_required": self.license_file_required,
            },
            "source": {
                "type": self.source_type,
                "channel": self.source_channel,
                "package": self.source_package,
                "url": self.source_url,
                "container": self.source_container,
                "bundle_base": self.bundle_base,
            },
            "runtime": {
                "cpu": self.runtime_cpu,
                "cuda": self.runtime_cuda,
                "container": self.runtime_container,
            },
            "requirements": list(self.requirements),
            "outputs": list(self.outputs),
            "platforms": list(self.platforms),
            "checksums": dict(self.checksums),
            "installed": installed,
            "install_path": install_path,
        }


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #

MANIFESTS_DIR = os.path.join(os.path.dirname(__file__), "manifests")


def load_manifest(manifest_path: str) -> EngineManifest:
    """Loads one YAML manifest file into an EngineManifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EngineManifest(**data)


def load_all_manifests(manifests_dir: str = MANIFESTS_DIR) -> Dict[str, EngineManifest]:
    """Loads every YAML file in `manifests_dir` into a dict keyed by id.

    Skips files that fail validation (corrupt YAML or schema mismatch)
    rather than crashing -- the platform should still start up if one
    manifest is broken.
    """
    if not os.path.isdir(manifests_dir):
        return {}

    result: Dict[str, EngineManifest] = {}
    for filename in sorted(os.listdir(manifests_dir)):
        if not filename.endswith(".yaml") and not filename.endswith(".yml"):
            continue
        path = os.path.join(manifests_dir, filename)
        try:
            manifest = load_manifest(path)
            result[manifest.id] = manifest
        except Exception as exc:  # noqa: BLE001 -- skip corrupt files
            # In production we'd log this; for now, silently skip so
            # the platform still starts.
            print(f"Warning: failed to load manifest {path}: {exc}")
    return result
