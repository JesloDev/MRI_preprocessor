# Preparing Engine Bundles on Your Machine

These commands download the actual FSL/FreeSurfer binaries and prepare
them as platform-specific engine bundles that the MRI Preprocessor can
register via BundleInstaller.

## Prerequisites

```bash
# 1. Install micromamba (lightweight conda, no admin needed)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/bin bin/micromamba
export PATH="$HOME/bin:$PATH"
micromamba --version

# 2. For FreeSurfer engines (SynthStrip, mri_deface):
#    Register at https://surfer.nmr.mgh.harvard.edu/registration.html
#    Download license.txt and place it somewhere safe.
#    You'll need it for: mri-preprocessor engines license synthstrip <path>
```

## Create the bundles directory

```bash
cd /path/to/mri_preprocessor    # the extracted project root
mkdir -p bundles

# Set this so the bundles are discoverable by the platform
export MRI_PREPROCESSOR_ENGINES_DIR="$(pwd)/engines"
```

## Download FSL engines (eddy, mcflirt, topup, flirt)

FSL publishes individual conda packages. Each command creates an
isolated conda env containing ONLY that tool + its dependencies.

```bash
# --- FSL EDDY (DWI motion/eddy-current correction) ---
mkdir -p bundles/fsl_eddy/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/fsl_eddy/linux-64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    fsl-eddy

# --- FSL MCFLIRT (fMRI motion correction) ---
mkdir -p bundles/mcflirt/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/mcflirt/linux-64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    fsl-mcflirt

# --- FSL TOPUP (susceptibility distortion correction) ---
# (optional — used by eddy when fieldmaps are available)
mkdir -p bundles/topup/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/topup/linux-64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    fsl-topup

# --- FSL FLIRT (registration) ---
# (optional — used by pydeface)
mkdir -p bundles/flirt/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/flirt/linux-64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    fsl-flirt
```

## Download FreeSurfer engines (SynthStrip, mri_deface)

FreeSurfer distributes as a full tarball (~7GB). You can extract just
the tools you need, but they depend on the FreeSurfer directory
structure.

```bash
# 1. Download FreeSurfer (you need a license for the download)
#    Go to: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
#    Download the Linux tarball (e.g. freesurfer-linux-ubuntu20_x86_64-7.4.1.tar.gz)

# 2. Extract
sudo mkdir -p /opt/freesurfer
sudo tar -xzf freesurfer-linux-ubuntu20_x86_64-7.4.1.tar.gz -C /opt/freesurfer

# 3. Place your license file
sudo cp license.txt /opt/freesurfer/license.txt

# 4. Create SynthStrip bundle
#    SynthStrip is at $FREESURFER_HOME/bin/mri_synthstrip
#    It needs the FreeSurfer model weights + shared libraries
mkdir -p bundles/synthstrip/linux-64/bin
cp /opt/freesurfer/bin/mri_synthstrip bundles/synthstrip/linux-64/bin/

# 5. Create mri_deface bundle
#    mri_deface needs the face + skull atlas templates
mkdir -p bundles/mri_deface/linux-64/bin
cp /opt/freesurfer/bin/mri_deface bundles/mri_deface/linux-64/bin/

# 6. Copy the atlas templates (mri_deface requires these)
mkdir -p bundles/mri_deface/linux-64/subjects/face/face_gca
cp /opt/freesurfer/subjects/face/face_gca/face.gca bundles/mri_deface/linux-64/subjects/face/face_gca/
cp /opt/freesurfer/subjects/face/face_gca/aseg.gca bundles/mri_deface/linux-64/subjects/face/face_gca/
```

## Download HD-BET

HD-BET is a Python package that uses PyTorch. It needs model weights
downloaded separately.

```bash
# 1. Create a conda env with PyTorch + HD-BET
mkdir -p bundles/hdbet/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/hdbet/linux-64" \
    -c conda-forge \
    python=3.10 pytorch cudatoolkit=11.8
micromamba run -p "$(pwd)/bundles/hdbet/linux-64" pip install hd-bet

# 2. The model weights are downloaded automatically on first run,
#    OR you can pre-download them:
#    (HD-BET downloads weights to ~/.hd-bet on first use)
micromamba run -p "$(pwd)/bundles/hdbet/linux-64" \
    python -c "from hdbet.utils import download_pretrained_weights; download_pretrained_weights()"

# 3. Copy the weights into the bundle so they're self-contained
cp -r ~/.hd-bet bundles/hdbet/linux-64/hd-bet-models
```

## Download pydeface

pydeface is a Python package that wraps FSL's flirt.

```bash
# 1. Create a conda env with pydeface + FSL flirt
mkdir -p bundles/pydeface/linux-64
micromamba create -y \
    -p "$(pwd)/bundles/pydeface/linux-64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    -c conda-forge \
    fsl-flirt python=3.10 pip
micromamba run -p "$(pwd)/bundles/pydeface/linux-64" pip install pydeface

# 2. Verify
micromamba run -p "$(pwd)/bundles/pydeface/linux-64" pydeface --help
```

## Verify the bundle structure

After downloading, your bundles directory should look like:

```
bundles/
├── fsl_eddy/
│   └── linux-64/
│       ├── bin/
│       │   └── eddy
│       └── lib/               (shared libraries)
├── mcflirt/
│   └── linux-64/
│       ├── bin/
│       │   └── mcflirt
│       └── lib/
├── synthstrip/
│   └── linux-64/
│       └── bin/
│           └── mri_synthstrip
├── mri_deface/
│   └── linux-64/
│       ├── bin/
│       │   └── mri_deface
│       └── subjects/
│           └── face/
│               └── face_gca/
│                   ├── face.gca
│                   └── aseg.gca
├── hdbet/
│   └── linux-64/
│       ├── bin/
│       │   └── hd_bet
│       └── hd-bet-models/
└── pydeface/
    └── linux-64/
        ├── bin/
        │   └── pydeface
        └── lib/
```

## Register the bundles in the engine store

```bash
# Set the engine store location (writable)
export MRI_PREPROCESSOR_ENGINES_DIR="$HOME/.mri-preprocessor/engines"

# Register each engine (symlink = zero disk overhead)
mri-preprocessor engines install fsl_eddy --bundle bundles/fsl_eddy --symlink
mri-preprocessor engines install mcflirt --bundle bundles/mcflirt --symlink
mri-preprocessor engines install synthstrip --bundle bundles/synthstrip --symlink
mri-preprocessor engines install mri_deface --bundle bundles/mri_deface --symlink
mri-preprocessor engines install hdbet --bundle bundles/hdbet --symlink
mri-preprocessor engines install pydeface --bundle bundles/pydeface --symlink

# Provide FreeSurfer license (for SynthStrip + mri_deface)
mri-preprocessor engines license synthstrip /path/to/license.txt
mri-preprocessor engines license mri_deface /path/to/license.txt
```

## Verify health

```bash
mri-preprocessor engines health synthstrip
mri-preprocessor engines health mri_deface
mri-preprocessor engines health mcflirt
mri-preprocessor engines health fsl_eddy
mri-preprocessor engines health hdbet
mri-preprocessor engines health pydeface
```

All should report `✓ Ready`.

## Or: install all engines required by a preset at once

```bash
# Install everything needed for the ml-ready preset
mri-preprocessor presets install ml-ready --bundle-dir bundles --symlink

# Or for the full-neuroimaging preset
mri-preprocessor presets install full-neuroimaging --bundle-dir bundles --symlink
```

## For macOS (Apple Silicon)

Repeat the micromamba commands with `-c https://fsl.fmrib.ox.ac.uk/fsl/conda/public`
and a different platform prefix:

```bash
mkdir -p bundles/synthstrip/osx-arm64
micromamba create -y \
    -p "$(pwd)/bundles/synthstrip/osx-arm64" \
    -c https://fsl.fmrib.ox.ac.uk/fsl/conda/public \
    --platform osx-arm64 \
    fsl-synthstrip
```

For FreeSurfer on macOS:
```bash
# Download the macOS tarball from the FreeSurfer website
sudo tar -xzf freesurfer-darwin-arm64-7.4.1.tar.gz -C /opt/freesurfer
```

## For Windows

FSL doesn't ship native Windows binaries. On Windows, use WSL (Windows
Subsystem for Linux) and follow the Linux instructions, or use the
Docker container approach.

## Troubleshooting

### FSL conda channel returns 404

If `https://fsl.fmrib.ox.ac.uk/fsl/conda/public` returns 404:

1. Check the current channel URL at:
   https://fsl.fmrib.ox.ac.uk/fsl/docs/install/conda.html

2. FSL may have changed their channel URL. As of FSL 6.0.7+, the
   channel should be at the URL above. If it's moved, update the
   manifest YAMLs in `app/engines/manifests/*.yaml` (the
   `source_channel` field).

3. Alternative: install the full FSL distribution from
   https://fsl.fmrib.ox.ac.uk/fsl/docs/install/
   and use the system-installed FSL binaries (the binary discovery
   chain checks `$FSLDIR/bin/` as a fallback).

### FreeSurfer requires a license

FreeSurfer tools (SynthStrip, mri_deface) will refuse to run without
a license file at `$FREESURFER_HOME/license.txt`. Register at:
https://surfer.nmr.mgh.harvard.edu/registration.html

The license is free for non-commercial use.

### HD-BET model weights download fails

HD-BET's model weights are hosted on Google Drive. If the download
fails (rate limiting, etc.), you can manually download from:
https://github.com/MIC-DKFZ/HD-BET/releases

Place the model files in `bundles/hdbet/linux-64/hd-bet-models/`.
