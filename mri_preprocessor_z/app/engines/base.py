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
    """

    name: str
    maturity: str
    validation_notes: Optional[str]
    license_notes: Optional[str]
    requires_external_tool: bool

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
        requires_external_tool. The UI uses this to populate its
        engine-selection dropdown.
        """
        result = []
        for engine in self._engines.values():
            result.append({
                "name": engine.name,
                "maturity": getattr(engine, "maturity", "reference"),
                "validation_notes": getattr(engine, "validation_notes", None),
                "license_notes": getattr(engine, "license_notes", None),
                "requires_external_tool": getattr(engine, "requires_external_tool", False),
            })
        return result
