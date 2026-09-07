"""
ProcessingEngine Protocol + a small registry helper.

The Protocol mirrors the existing step-function contract:
    fn(input_path, output_dir, params, context) -> dict

Each engine is a class (not a bare function) so it can carry metadata
about itself: name, maturity, validation_notes, license_notes. This
metadata is what the UI surfaces in the engine-selection dropdown
("Otsu [Experimental] -- placeholder; SynthStrip [External] -- requires
FreeSurfer license; HD-BET [External] -- requires HD-BET installation").

The metadata is also what makes future audit / dashboard views easy:
"this run used engine=otsu for brain_extraction; here are its known
limitations." Because the engine choice is recorded per-step via
provenance (params includes engine=...), the full processing chain is
reproducible and auditable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class ProcessingEngine(Protocol):
    """The contract every engine implementation must satisfy.

    Identical signature to the step function contract, so an engine IS
    just a step-function-shaped callable, plus optional metadata.

    Attributes (declared on the concrete class, not enforced here --
    duck typing is fine):
        name             : str  -- short identifier ("otsu", "synthstrip")
        maturity         : str  -- one of the MaturityLevel values
        validation_notes : Optional[str]
        license_notes    : Optional[str]  -- for external engines
        requires_external_tool : bool -- True if FSL/FreeSurfer/etc.
                                          must be installed separately

    PR7: `is_available()` is a new method that returns True if the engine
    can actually run on this machine. Pure-Python engines (otsu,
    simpleitk, dipy) always return True. External engines (synthstrip,
    hdbet, mcflirt, fsl_eddy) return True only when their binary is
    discoverable -- see each engine's is_available() implementation for
    its specific discovery chain.

    The `availability_reason()` companion returns a human-readable
    explanation when `is_available()` returns False (e.g. "mri_synthstrip
    not found; checked: $MRI_PREPROCESSOR_SYNTHSTRIP, PATH, $FREESURFER_HOME/bin").
    The CLI and API surface this so the user knows what to install /
    configure before they can use the engine.
    """

    name: str
    maturity: str
    validation_notes: Optional[str]
    license_notes: Optional[str]
    requires_external_tool: bool

    def is_available(self) -> bool:
        """Returns True if this engine can run on the current machine.

        Pure-Python engines always return True. External engines probe
        for their binary / installation and return False (with a useful
        `availability_reason()`) if the tool is not found.
        """
        ...

    def availability_reason(self) -> str:
        """Returns a human-readable explanation of the availability
        state. Empty string when is_available() is True; otherwise a
        message describing what's missing and how to fix it.
        """
        ...

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Returns the same dict shape as a step function:
            {
                "output_path": str,
                "outputs":     {name: path, ...},
                "metrics":     {name: value, ...},
            }
        """
        ...


class EngineRegistry:
    """A small lookup table mapping engine-name -> engine instance.

    Each engine family (brain_extraction, fmri_motion, dwi_correction)
    has its own EngineRegistry singleton, populated at import time by
    the engine modules themselves.

    Failure mode: if a user requests an engine name that isn't
    registered, `get()` raises a KeyError with a useful message listing
    what IS registered. This is the most common runtime error an
    engine layer produces -- making it informative is more valuable than
    trying to be clever about fallbacks.
    """

    def __init__(self, family_name: str):
        self.family_name = family_name
        self._engines: Dict[str, ProcessingEngine] = {}

    def register(self, engine: ProcessingEngine) -> None:
        if engine.name in self._engines:
            raise ValueError(
                f"Engine '{engine.name}' already registered in family "
                f"'{self.family_name}'"
            )
        self._engines[engine.name] = engine

    def get(self, name: str) -> ProcessingEngine:
        if name not in self._engines:
            available = ", ".join(sorted(self._engines.keys())) or "<none>"
            raise KeyError(
                f"Unknown engine '{name}' in family '{self.family_name}'. "
                f"Available engines: {available}"
            )
        return self._engines[name]

    def get_default(self) -> str:
        """Returns the name of the default engine.

        The first engine registered is the default. Engine modules
        are imported in dependency order (built-in pure-Python first,
        external-tool wrappers second) so the default is always the
        one that works out of the box.
        """
        if not self._engines:
            raise RuntimeError(
                f"No engines registered in family '{self.family_name}'"
            )
        return next(iter(self._engines.values())).name

    def available_engines(self) -> List[Dict[str, Any]]:
        """Returns a list of engine metadata dicts, suitable for the API.

        Each entry has: name, maturity, validation_notes, license_notes,
        requires_external_tool, available (bool), availability_reason (str).

        The `available` and `availability_reason` fields let the UI
        render "✓ Available" or "✗ Not available: mri_synthstrip not found"
        in the engine-selection dropdown, BEFORE the user picks an engine
        and gets a runtime error.
        """
        result = []
        for engine in self._engines.values():
            try:
                available = engine.is_available()
                reason = engine.availability_reason() if not available else ""
            except Exception:  # noqa: BLE001 -- availability check shouldn't crash the API
                available = False
                reason = "Availability check raised an exception."
            result.append({
                "name": engine.name,
                "maturity": getattr(engine, "maturity", "reference"),
                "validation_notes": getattr(engine, "validation_notes", None),
                "license_notes": getattr(engine, "license_notes", None),
                "requires_external_tool": getattr(engine, "requires_external_tool", False),
                "available": available,
                "availability_reason": reason,
            })
        return result
