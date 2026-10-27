# Packaging Guide (PR19)

This document covers how to build standalone distributions of the MRI
Preprocessor that run without requiring the user to install Python,
conda, or any dependencies.

## Quick start (development install)

```bash
# Clone or extract the project
cd mri_preprocessor

# Create a venv + install deps + run tests
./setup_linux.sh        # Linux/macOS
setup_windows.bat        # Windows

# Use the CLI
./bin/mri-preprocessor scan /data
./bin/mri-preprocessor run /in /out --preset ml-ready

# Or use the REST API
./run_server.sh          # http://127.0.0.1:5000
```

## Standalone distribution (PyInstaller)

Build a self-contained directory that includes the Python runtime +
all dependencies + the application code. Users extract it and run
`./mri-preprocessor` without installing anything.

### Prerequisites

```bash
pip install pyinstaller
```

### Build

```bash
cd mri_preprocessor
pyinstaller packaging/mri_preprocessor.spec
```

### Output

```
dist/mri-preprocessor/
├── mri-preprocessor           (main executable)
├── _internal/                  (Python runtime + all libs)
│   ├── app/                    (application code)
│   ├── nibabel/
│   ├── SimpleITK/
│   ├── dipy/
│   ├── nilearn/
│   └── ...
└── ...
```

### Distribute

```bash
# Create a tarball for the current platform
cd dist
tar -czf mri-preprocessor-linux-x86_64.tar.gz mri-preprocessor/

# Or zip
zip -r mri-preprocessor-linux-x86_64.zip mri-preprocessor/
```

Users extract and run:

```bash
tar -xzf mri-preprocessor-linux-x86_64.tar.gz
cd mri-preprocessor
./mri-preprocessor scan /data
```

### Platform-specific builds

PyInstaller builds are platform-specific. Build on each target:

| Platform | Build on | Output |
|---|---|---|
| Linux x86_64 | Linux x86_64 | `mri-preprocessor-linux-x86_64.tar.gz` |
| Linux ARM64 | Linux ARM64 | `mri-preprocessor-linux-aarch64.tar.gz` |
| macOS Intel | macOS Intel | `mri-preprocessor-macos-x86_64.tar.gz` |
| macOS ARM64 | macOS ARM64 | `mri-preprocessor-macos-arm64.tar.gz` |
| Windows x86_64 | Windows x86_64 | `mri-preprocessor-win-x86_64.zip` |

For cross-platform CI builds, use GitHub Actions with a matrix of
OS images.

## Bundled engine distribution

The standalone binary includes the Python runtime and the platform's
built-in engines (Otsu, SimpleITK, DIPY). External engines (SynthStrip,
MCFLIRT, FSL eddy, etc.) are NOT bundled in the PyInstaller build --
they're platform-specific compiled binaries that need separate handling.

### Option 1: Pre-bundled engines (recommended for offline distribution)

Include pre-bundled engine directories alongside the PyInstaller output:

```
mri-preprocessor-distribution/
├── mri-preprocessor/              (PyInstaller build)
│   └── ...
├── bundles/
│   ├── synthstrip/
│   │   ├── linux-64/
│   │   │   └── bin/mri_synthstrip
│   │   ├── osx-arm64/
│   │   │   └── bin/mri_synthstrip
│   │   └── ...
│   ├── eddy/
│   │   └── ...
│   └── ...
└── install.sh                      (first-run setup)
```

At first launch, `install.sh` registers the bundles:

```bash
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
MP="$DIR/mri-preprocessor/mri-preprocessor"

# Register all bundled engines via symlink (zero disk overhead)
for engine_dir in "$DIR"/bundles/*/; do
    engine_id=$(basename "$engine_dir")
    "$MP" engines install "$engine_id" \
        --bundle "$engine_dir" --symlink 2>/dev/null
done

echo "All bundled engines registered."
echo "Run: $MP --help"
```

### Option 2: Conda installation (requires internet)

Users install engines on demand:

```bash
mri-preprocessor engines install synthstrip    # via conda
mri-preprocessor presets install ml-ready      # auto-installs all missing
```

This requires micromamba/conda on the user's PATH and internet access
to the FSL conda channel.

## Launcher script behavior

The `bin/mri-preprocessor` launcher auto-detects the runtime environment:

1. **Bundled Python** (`./runtime/bin/python3`): used in a standalone
   distribution that ships its own Python. No system Python required.
2. **Local venv** (`.venv/bin/python3`): used in development after
   running `./setup_linux.sh`.
3. **Active venv** (`$VIRTUAL_ENV`): used when the user has already
   activated a venv.
4. **System Python**: last resort. The user must have the deps
   installed via `pip install .`

## REST API as a service

For running the REST API as a long-lived service:

### Linux (systemd)

```ini
# /etc/systemd/system/mri-preprocessor.service
[Unit]
Description=MRI Preprocessor REST API
After=network.target

[Service]
Type=simple
User=mripreprocessor
WorkingDirectory=/opt/mri-preprocessor
ExecStart=/opt/mri-preprocessor/bin/mri-preprocessor-server
Environment=MRIPREP_HOST=0.0.0.0
Environment=MRIPREP_PORT=5000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### macOS (launchd)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ...>
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mri-preprocessor.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/MRI-Preprocessor/bin/mri-preprocessor-server</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MRIPREP_HOST</key>
        <string>127.0.0.1</string>
        <key>MRIPREP_PORT</key>
        <string>5000</string>
    </dict>
</dict>
</plist>
```

## Directory layout for a full offline distribution

```
MRI-Preprocessor/                  (distribution root)
├── mri-preprocessor/              (PyInstaller build)
│   ├── mri-preprocessor           (main executable)
│   ├── mri-preprocessor-server    (REST API server)
│   └── _internal/                 (Python runtime + deps)
├── runtime/                      (optional: standalone Python)
│   └── bin/python3
├── bundles/                       (pre-bundled engine binaries)
│   ├── synthstrip/
│   │   ├── linux-64/bin/mri_synthstrip
│   │   ├── osx-arm64/bin/mri_synthstrip
│   │   └── ...
│   ├── eddy/
│   └── ...
├── licenses/                      (FreeSurfer license files)
│   └── freesurfer-license.txt
├── install.sh                     (first-run: registers bundles)
├── install.bat                    (Windows first-run)
└── README.md
```

Users:
1. Extract the archive
2. Run `./install.sh` (registers all bundled engines)
3. Run `./mri-preprocessor/mri-preprocessor run /data /output --preset ml-ready`

No Python, conda, FSL, or FreeSurfer installation required.
