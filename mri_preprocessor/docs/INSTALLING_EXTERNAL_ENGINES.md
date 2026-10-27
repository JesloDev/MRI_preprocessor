# Engine Installation (PR13+)

The platform includes an **Engine Manager** that installs and manages
external engines locally. Users install capabilities ("SynthStrip"),
not software suites ("FreeSurfer"). The platform handles discovery,
installation, and isolation.

## Quick start

```bash
# List all known engines
python -m app.cli engines list

# Install an engine (uses conda/micromamba)
python -m app.cli engines install synthstrip

# Install with the stub installer (for testing -- no conda needed)
python -m app.cli engines install synthstrip --stub

# Check install status + manifest details
python -m app.cli engines info synthstrip

# Uninstall
python -m app.cli engines uninstall synthstrip

# List only installed engines
python -m app.cli engines list --installed
```

## Architecture

```
                    MRI PREPROCESSOR
                           │
                           ▼
                    ENGINE MANAGER
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
         Bundled       Managed local    Remote
         engines          engines       engines
            │              │              │
       SimpleITK       SynthStrip       Optional
       DIPY            FSL EDDY         cloud/service
       NiBabel         MCFLIRT
       NumPy           HD-BET
                       mri_deface
                       TOPUP
                             │
                             ▼
                       ENGINE EXECUTION
                             │
                             ▼
                      OUTPUT VALIDATION
                             │
                             ▼
                        PROVENANCE
```

### Engine store layout

```
engines/                            # store root (project-level)
├── synthstrip/
│   ├── 7.4.1/                      # version-pinned install
│   │   ├── bin/
│   │   │   └── mri_synthstrip
│   │   └── lib/
│   ├── manifest.yaml              # local manifest copy
│   └── installed                   # marker file
├── eddy/
│   └── 6.0.7/
│       └── bin/
│           └── eddy
└── ...
```

Store location (priority):
1. `$MRI_PREPROCESSOR_ENGINES_DIR` env var (override for testing / custom deployments)
2. `<project_root>/engines/` (default, alongside `app/`)

### Binary discovery chain

When an engine adapter needs to find its binary, the discovery chain
checks in this order:

1. **EngineManager local store** (PR13) — if the engine was installed
   via `engines install`, the binary is at
   `<store>/<id>/<version>/bin/<binary>`. This check runs FIRST so a
   managed install always wins over PATH.
2. **Explicit parameter** — `params["binary_path"]` (per-job override)
3. **Engine-specific env var** — `$MRI_PREPROCESSOR_<ENGINE_NAME>`
4. **PATH** — `shutil.which(binary_name)`
5. **Tool-suite fallback** — `$<TOOL_SUITE_HOME>/bin/<binary_name>`

This means: if a user installs SynthStrip via
`python -m app.cli engines install synthstrip`, the discovery chain
finds it in the local store — no `$FREESURFER_HOME`, no PATH
manipulation, no system-wide FreeSurfer installation needed.

## Engine manifests

Each engine has a YAML manifest at `app/engines/manifests/<id>.yaml`
declaring its metadata:

```yaml
id: synthstrip
name: SynthStrip
provider: FreeSurfer
capability: brain_extraction
version: "7.4.1"
binary_name: "mri_synthstrip"     # MUST match the adapter's DiscoveryConfig

license_type: non-commercial
license_file_required: true       # FreeSurfer requires license.txt

source_type: conda                 # how to install
source_channel: https://fsl.fmrib.ox.ac.uk/fsl/conda/public
source_package: fsl-synthstrip

runtime_cpu: true
runtime_cuda: true

requirements:                     # input requirements (for UI hints)
  - structural_mri

outputs:                           # produced artifacts
  - brain.nii.gz
  - brain_mask.nii.gz

checksums: {}                     # per-platform (populated for reproducibility)
```

The manifest is the source of truth — the CLI, API, and future GUI all
discover engine metadata from it. No hardcoded engine-specific UI logic.

## Installers

The EngineManager supports pluggable installer strategies:

| Installer | When to use | How it works |
|---|---|---|
| `CondaInstaller` | Production (default) | `conda create -p <store>/<id>/<version> -c <channel> <package>=<version>` |
| `StubInstaller` | Testing / development | Creates a fake binary that exits 0 |
| *(future)* `DockerInstaller` | Containerized engines | `docker pull <image>` + run via `docker run` |
| *(future)* `URLInstaller` | Direct tarball downloads | `wget` + extract |
| *(future)* `BundleInstaller` | Offline distribution | Register pre-bundled engines in the store (no download) |

The `BundleInstaller` is particularly important for the offline
distribution goal: when the platform is packaged as a standalone
application, engines can be pre-bundled in the installer and registered
in the store at install time — no internet access needed.

## FSL-specific notes

FSL publishes its components as **individual Conda packages** (not just
the full FSL distribution). This means `conda create -p <path> -c
https://fsl.fmrib.ox.ac.uk/fsl/conda/public fsl-eddy=6.0.7` installs
ONLY eddy + its dependencies, not the entire FSL suite.

Each FSL engine in the platform has its own isolated conda env inside
the store, so FSL eddy's dependencies can't break FSL mcflirt's
dependencies (or DIPY's, or the platform's own Python env).

## FreeSurfer-specific notes

FreeSurfer engines (SynthStrip, mri_deface) require a license file at
`$FREESURFER_HOME/license.txt`. The manifest declares
`license_file_required: true`. The Engine Manager surfaces this in
`engines info` so the user knows they need a license before the engine
can run.

*(Future work: the Engine Manager should prompt for the license file
during `engines install` and store it in the engine's install dir.)*

## License compliance

All external engines have non-commercial licenses (FSL, FreeSurfer) or
custom non-commercial licenses (HD-BET). The manifest's `license_type`
and `license_notes` fields surface this. The `maturity: external_engine`
field on each step also communicates the license caveat.

Before deploying this platform in any commercial context, re-review
the FSL and FreeSurfer license terms. The pure-Python engines (otsu,
simpleitk, dipy, placeholder) are the only engines safe for commercial
use without separate licensing.

## Roadmap

### Phase 1 (PR13 — DONE)
- EngineManifest schema + YAML manifests
- EngineStore filesystem layout
- EngineManager central API
- CondaInstaller + StubInstaller
- Binary discovery integration (store checked first)
- CLI: `engines install / uninstall / info / list`

### Phase 2 (next)
- Real Conda installation (test with actual micromamba + FSL channel)
- FreeSurfer license file management (prompt + validate during install)
- Version pinning with checksums (reproducible installs)
- `presets install <name>` that resolves required engines and installs
  missing ones

### Phase 3
- Runtime selection (native / container / managed / remote)
- `BundleInstaller` for offline distribution (pre-bundled engines)
- OS/architecture-specific engine resolution (linux-x86_64, macos-arm64, etc.)

### Phase 4
- Automatic engine acquisition (pipeline → resolver → required engines
  → manager → install → execute)
- Container-based engines (Docker / Podman)
- Remote execution (explicit user opt-in only — data never leaves the
  machine without consent)
