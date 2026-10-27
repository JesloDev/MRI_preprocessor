# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for MRI Preprocessor (PR19)
#
# Builds a standalone directory bundle containing the Python runtime,
# all dependencies, and the application code. The resulting directory
# can be zipped and distributed -- users extract it and run
# `./mri-preprocessor/mri-preprocessor` without installing anything.
#
# Usage:
#   pip install pyinstaller
#   pyinstaller packaging/mri_preprocessor.spec
#
# Output:
#   dist/mri-preprocessor/         (standalone directory)
#   dist/mri-preprocessor/mri-preprocessor  (main executable)
#
# The bundle is ~300-500MB (SimpleITK + DIPY + numpy are large).
# Engine bundles (FreeSurfer, FSL binaries) are NOT included -- they're
# installed separately via `mri-preprocessor engines install --bundle`.

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect non-Python files from packages that ship data
# (e.g. nibabel's test data, SimpleITK's shared libraries)
datas = []
binaries = []

# Include engine manifest YAMLs
datas += [('app/engines/manifests/*.yaml', 'app/engines/manifests')]

# Include nilearn's data files (templates, atlases)
datas += collect_data_files('nilearn')

# Include dipy's data files
datas += collect_data_files('dipy')

# Include sklearn's data files
datas += collect_data_files('sklearn')

a = Analysis(
    ['../__main__.py'],
    pathex=[os.path.dirname(os.path.abspath('../'))],
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules('nilearn') + collect_submodules('dipy'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook',
              'jupyter', 'pytest', 'sphinx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mri-preprocessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mri-preprocessor',
)
