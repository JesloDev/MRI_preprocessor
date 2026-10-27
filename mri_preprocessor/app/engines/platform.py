"""
Platform detection for engine bundles (PR16).

When the platform distributes pre-bundled engines (via BundleInstaller),
the correct bundle must be selected for the user's OS + architecture.
A Linux x86_64 binary won't run on macOS ARM64; a Windows .exe won't
run on Linux. This module detects the current platform and provides
the canonical platform identifier used throughout the engine system.

Platform identifiers follow the conda naming convention:
    linux-64       (Linux x86_64)
    linux-aarch64  (Linux ARM64)
    osx-64         (macOS Intel)
    osx-arm64      (macOS Apple Silicon)
    win-64         (Windows x86_64)

This matches FSL's conda channel naming, so the same identifiers work
for both conda-installed and bundle-installed engines.

The module also supports a `--platform` override (for testing and for
cross-platform preparation: a user on Linux can pre-install the macOS
bundle for later transfer to a Mac).
"""
from __future__ import annotations

import os
import platform
import shutil
from typing import Optional


# Canonical platform identifiers (conda convention)
LINUX_64 = "linux-64"
LINUX_AARCH64 = "linux-aarch64"
OSX_64 = "osx-64"
OSX_ARM64 = "osx-arm64"
WIN_64 = "win-64"

ALL_PLATFORMS = [LINUX_64, LINUX_AARCH64, OSX_64, OSX_ARM64, WIN_64]


def detect_platform() -> str:
    """Detects the current OS + architecture and returns the canonical
    platform identifier.

    Detection logic:
        1. $MRI_PREPROCESSOR_PLATFORM env var (explicit override — for
           testing and cross-platform preparation)
        2. Python's platform module (os.uname + architecture)

    Returns one of: linux-64, linux-aarch64, osx-64, osx-arm64, win-64

    If the platform can't be confidently identified, returns linux-64
    as the safest default (most common in neuroimaging research).
    """
    # 1. Explicit override
    env_platform = os.environ.get("MRI_PREPROCESSOR_PLATFORM")
    if env_platform and env_platform in ALL_PLATFORMS:
        return env_platform

    # 2. Auto-detect
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return LINUX_64
        if machine in ("aarch64", "arm64"):
            return LINUX_AARCH64
        # Unknown Linux arch -- default to x86_64
        return LINUX_64

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return OSX_ARM64
        if machine in ("x86_64", "amd64"):
            return OSX_64
        # Unknown macOS arch -- default to Intel
        return OSX_64

    if system in ("windows", "win32", "cygwin"):
        if machine in ("amd64", "x86_64", "arm64", "aarch64"):
            return WIN_64
        return WIN_64

    # Unknown system -- safest default
    return LINUX_64


def is_native_platform(platform_id: str) -> bool:
    """Returns True if the given platform identifier matches the current
    machine's platform."""
    return platform_id == detect_platform()


def get_platform_display_name(platform_id: str) -> str:
    """Returns a human-readable name for a platform identifier."""
    names = {
        LINUX_64: "Linux x86_64",
        LINUX_AARCH64: "Linux ARM64",
        OSX_64: "macOS Intel",
        OSX_ARM64: "macOS Apple Silicon",
        WIN_64: "Windows x86_64",
    }
    return names.get(platform_id, platform_id)


def get_current_platform_display() -> str:
    """Convenience: returns the human-readable name of the current platform."""
    return get_platform_display_name(detect_platform())
