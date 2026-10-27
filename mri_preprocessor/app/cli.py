"""
Command-line interface for the MRI Preprocessor.

The CLI is intentionally thin: every command calls into existing
functions from app.data_model, app.detection, app.pipeline, and
app.execution. No business logic is duplicated here.

Commands:
  scan          Scan a dataset directory and print the summary.
  capabilities  List available preprocessing steps.
  engines       List engine adapters for a given step.
  resolve       Preview the resolved pipeline (deps + ordering).
  run           Execute a pipeline against a dataset.

Run synchronously by default (calls run_job directly). The CLI is the
first real product interface for the platform -- it validates dataset
scanning, pipeline selection, dependency resolution, SubjectContext,
outputs, provenance, and engine selection before GUI complexity is
introduced.

Usage:
    python -m app.cli scan /path/to/dataset
    python -m app.cli capabilities --maturity production_ready
    python -m app.cli engines brain_extraction
    python -m app.cli resolve brain_extraction registration
    python -m app.cli presets list
    python -m app.cli presets show ml-ready
    python -m app.cli presets create my-preset --steps reorientation brain_extraction
    python -m app.cli run /data/input /data/output \
        --preset ml-ready
    python -m app.cli run /data/input /data/output \
        --steps reorientation brain_extraction \
        --modalities T1 \
        --param brain_extraction.engine=otsu
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Importing bootstrap triggers step registration side effects.
from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import registry


app = typer.Typer(
    name="mri-preprocessor",
    help="MRI / fMRI preprocessing platform CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# --------------------------------------------------------------------- #
# Helper: --param key=value parsing
# --------------------------------------------------------------------- #

def parse_params(params: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parses --param step.key=value strings into a nested dict.

    Examples:
        ["brain_extraction.engine=otsu"]
            -> {"brain_extraction": {"engine": "otsu"}}

        ["brain_extraction.engine=synthstrip",
         "resampling.voxel_size=[1.0,1.0,1.0]"]
            -> {"brain_extraction": {"engine": "synthstrip"},
                "resampling": {"voxel_size": [1.0, 1.0, 1.0]}}

    Value parsing (in priority order):
        1. JSON (handles lists, numbers, bools, null) -- e.g.
           "[1.0,1.0,1.0]" -> [1.0, 1.0, 1.0], "5" -> 5, "true" -> True
        2. Comma-separated fallback (for non-JSON-but-listy values) -- e.g.
           "1,176,256,80,1" -> ["1","176","256","80","1"], then coerced
           to int/float when all entries are numeric. This lets users
           write `--param resampling.shape=1,176,256,80,1` without the
           JSON brackets.
        3. Raw string -- the fallback if neither JSON nor comma-separated
           yields a usable value.

    This is the only CLI-specific logic. The resulting dict is passed
    straight into run_job -- the CLI is a thin wrapper.
    """
    if not params:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for raw in params:
        if "=" not in raw:
            raise typer.BadParameter(
                f"--param expects step.key=value, got {raw!r}"
            )
        key_path, value = raw.split("=", 1)
        if "." not in key_path:
            raise typer.BadParameter(
                f"--param expects step.key=value (note the dot), got {raw!r}"
            )
        step_id, param_key = key_path.split(".", 1)

        # Try JSON first (handles lists, numbers, bools, null)
        parsed_value = None
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Comma-separated fallback: split on commas, coerce numerics.
            # This is for users who write `shape=1,176,256,80,1` instead
            # of `shape=[1,176,256,80,1]`. We split, then convert each
            # element to int / float / str as appropriate. If there are
            # no commas, just keep the raw string.
            if "," in value:
                parts = [p.strip() for p in value.split(",") if p.strip()]
                coerced = []
                for p in parts:
                    # Try int first (so "5" stays 5, not 5.0)
                    try:
                        coerced.append(int(p))
                    except ValueError:
                        try:
                            coerced.append(float(p))
                        except ValueError:
                            coerced.append(p)
                parsed_value = coerced
            else:
                parsed_value = value

        result.setdefault(step_id, {})[param_key] = parsed_value
    return result


# --------------------------------------------------------------------- #
# Helper: subject inference strategy from CLI flags
# --------------------------------------------------------------------- #

def _resolve_subject_strategy(strategy: str, depth: Optional[int]):
    """Maps the --subject-strategy flag to a SubjectInferenceStrategy."""
    from app.data_model import (
        AutoSubjectStrategy,
        BIDSSubjectStrategy,
        DirectoryDepthStrategy,
        RegexStrategy,
    )
    if strategy == "auto":
        return AutoSubjectStrategy()
    if strategy == "bids":
        return BIDSSubjectStrategy()
    if strategy == "depth":
        if depth is None:
            raise typer.BadParameter(
                "--subject-strategy depth requires --subject-depth N"
            )
        return DirectoryDepthStrategy(depth=depth)
    if strategy == "regex":
        raise typer.BadParameter(
            "--subject-strategy regex is not yet exposed via the CLI; "
            "use the API directly."
        )
    raise typer.BadParameter(f"Unknown --subject-strategy: {strategy}")


# --------------------------------------------------------------------- #
# Command: scan
# --------------------------------------------------------------------- #

@app.command()
def scan(
    input_dir: str = typer.Argument(..., help="Directory to scan."),
    subject_strategy: str = typer.Option(
        "auto",
        "--subject-strategy",
        help="Subject-id inference: auto | bids | depth | regex",
    ),
    subject_depth: Optional[int] = typer.Option(
        None,
        "--subject-depth",
        help="Depth for --subject-strategy depth (e.g. 2 for Root/ScanFolder/Subject).",
    ),
    no_headers: bool = typer.Option(
        False,
        "--no-headers",
        help="Skip NIfTI header reads (faster but no metadata in the dataset).",
    ),
):
    """Scan a dataset directory and print the summary."""
    from app.data_model import build_dataset
    from app.detection.classifier import classify_manifest, modality_counts
    from app.ingestion.scanner import scan_directory, summarize_manifest
    from app.qc.validators import validate_manifest

    strategy = _resolve_subject_strategy(subject_strategy, subject_depth)

    console.print(Panel.fit("MRI Preprocessor", style="bold blue"))
    console.print(f"\nScanning [cyan]{input_dir}[/]...")

    # Flat manifest (kept for QC + modality-count display)
    manifest = scan_directory(input_dir)
    manifest = classify_manifest(manifest)
    summary = summarize_manifest(manifest)
    summary["modality_counts"] = modality_counts(manifest)

    # QC
    qc_report = validate_manifest(manifest)

    # Dataset tree (subjects -> acquisitions)
    dataset = build_dataset(
        input_dir=input_dir,
        subject_strategy=strategy,
        read_headers=not no_headers,
        manifest=manifest,
    )

    # Output: subject count + acquisition count
    console.print()
    console.print(
        f"[bold]Subjects:[/bold] {summary['num_subjects']}"
        f"    [bold]Acquisitions:[/bold] {summary['num_files']}"
    )
    console.print()

    # Output: modality counts table
    if summary["modality_counts"]:
        table = Table(title="Detected modalities", show_lines=False)
        table.add_column("Modality", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        # Sort by count descending, then by name
        sorted_counts = sorted(
            summary["modality_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for modality, count in sorted_counts:
            table.add_row(modality, str(count))
        console.print(table)
        console.print()

    unclassified = summary["modality_counts"].get("unclassified", 0)
    if unclassified:
        console.print(
            f"[yellow]Unclassified:[/yellow] {unclassified} "
            f"(will be flagged for manual review)"
        )

    # Output: QC summary
    console.print()
    console.print("[bold]QC[/bold]")
    console.print(
        f"  [green]✓[/green] Valid files: {qc_report['valid_count']}"
    )
    if qc_report["failed_count"]:
        console.print(
            f"  [red]✗[/red] Failed files: {qc_report['failed_count']}"
        )
        for failed in qc_report["failed_files"][:5]:
            console.print(
                f"      - {failed['filepath']}: {failed['error']}",
                style="red",
            )
        if len(qc_report["failed_files"]) > 5:
            console.print(
                f"      ... and {len(qc_report['failed_files']) - 5} more",
                style="red",
            )


# --------------------------------------------------------------------- #
# Command: capabilities
# --------------------------------------------------------------------- #

# Map modality groups to a display label
MODALITY_GROUPS = {
    "Structural MRI":  ["T1", "T2", "FLAIR", "PD", "SWI", "GRE"],
    "Diffusion MRI":   ["DWI", "DTI", "ADC"],
    "Functional MRI":  ["fMRI"],
    "Perfusion / Angio": ["ASL", "MRA", "MRV", "TOF"],
}


# Maturity badge per step (the load-bearing output)
MATURITY_BADGE = {
    "production_ready": ("✓", "green"),
    "reference":          ("•", "yellow"),
    "experimental":       ("⚠", "yellow"),
    "external_engine":    ("⚡", "cyan"),
    "planned":            ("○", "dim"),
}


def _step_in_any_modality(step, modalities: List[str]) -> bool:
    return any(m in step.modalities for m in modalities)


@app.command()
def capabilities(
    modality: Optional[str] = typer.Option(
        None, "--modality", "-m",
        help="Filter to a specific modality (e.g. T1, DWI, fMRI).",
    ),
    maturity: Optional[str] = typer.Option(
        None, "--maturity",
        help="Filter by maturity level (planned/experimental/reference/production_ready/external_engine).",
    ),
):
    """List available preprocessing steps."""
    all_steps = registry.all_steps()

    if modality:
        all_steps = [s for s in all_steps if modality in s.modalities]
    if maturity:
        all_steps = [s for s in all_steps if s.maturity == maturity]

    if not all_steps:
        console.print("[red]No steps match the given filters.[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit("Available preprocessing steps", style="bold blue"))
    console.print()

    if modality:
        # Single modality filter: flat list
        _print_step_list(all_steps)
    else:
        # Group by modality family
        used_step_ids = set()
        for group_label, group_modalities in MODALITY_GROUPS.items():
            group_steps = [
                s for s in all_steps
                if _step_in_any_modality(s, group_modalities)
                and s.id not in used_step_ids
            ]
            if not group_steps:
                continue
            console.print(f"[bold]{group_label}[/bold]")
            console.print("[dim]" + "-" * len(group_label) + "[/dim]")
            _print_step_list(group_steps, indent=2)
            # Mark steps that belong to multiple groups as already-shown
            for s in group_steps:
                used_step_ids.add(s.id)
            console.print()


def _print_step_list(steps, indent: int = 0):
    pad = " " * indent
    for step in steps:
        badge, color = MATURITY_BADGE.get(step.maturity, ("?", "white"))
        console.print(
            f"{pad}[{color}]{badge}[/{color}] [bold]{step.id}[/bold] "
            f"-- {step.name}"
        )
        if step.validation_notes:
            # Wrap the notes at a sensible width and indent them under
            # the step line
            import textwrap
            wrapped = textwrap.fill(
                step.validation_notes, width=80,
                initial_indent=" " * (indent + 4),
                subsequent_indent=" " * (indent + 4),
            )
            console.print(f"[dim italic]{wrapped}[/dim italic]")


# --------------------------------------------------------------------- #
# Command group: engines
# --------------------------------------------------------------------- #

# `engines` is a command group with subcommands:
#   - `engines install <id>`    : install an engine via the EngineManager
#   - `engines uninstall <id>` : remove an installed engine
#   - `engines info <id>`      : show manifest + install status
#   - `engines list`            : list all known engines (or only installed)
#   - `engines adapters <step>`: list engine adapters for a given step
#                                 (the old `engines <step>` behavior)
#
# This is the integration point between the user and the EngineManager
# (PR13). The user sees capabilities ("install SynthStrip"), not
# software suites ("install FreeSurfer").

engines_app = typer.Typer(
    name="engines",
    help="Manage engine installations: install, uninstall, info, list.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(engines_app, name="engines")


@engines_app.command("install")
def engines_install(
    engine_id: str = typer.Argument(..., help="Engine id to install (e.g. synthstrip, eddy)."),
    force: bool = typer.Option(
        False, "--force",
        help="Reinstall even if already installed.",
    ),
    stub: bool = typer.Option(
        False, "--stub",
        help="Use the stub installer (development only -- creates a fake "
             "binary that exits 0). NOT for production.",
    ),
    bundle: Optional[str] = typer.Option(
        None, "--bundle",
        help="Register a pre-bundled engine directory (PR15 offline mode). "
             "The directory must contain bin/<binary_name> (direct) OR "
             "<platform>/bin/<binary_name> (PR16 platform-specific). "
             "No internet, no conda required. Use --symlink to symlink "
             "instead of copy.",
    ),
    symlink: bool = typer.Option(
        False, "--symlink",
        help="With --bundle: symlink the bundle into the store instead of "
             "copying. Zero disk overhead; ideal for read-only application "
             "directories where the bundle already lives.",
    ),
    platform: Optional[str] = typer.Option(
        None, "--platform",
        help="Override platform detection (PR16). One of: linux-64, "
             "linux-aarch64, osx-64, osx-arm64, win-64. Useful for "
             "cross-platform preparation (e.g. pre-install the macOS "
             "arm64 bundle from a Linux machine).",
    ),
):
    """Install an engine via the EngineManager.

    Three installation modes:
      1. Default (no flags): uses CondaInstaller (requires micromamba/conda
         + internet to download from the FSL conda channel).
      2. --stub: uses StubInstaller (creates a fake binary; for testing).
      3. --bundle <path>: uses BundleInstaller (registers a pre-bundled
         engine directory; no internet needed -- the offline distribution
         path).
    """
    from app.engines.installer import StubInstaller
    from app.engines.manager import EngineManager, get_engine_manager

    # Validate: --stub and --bundle are mutually exclusive
    if stub and bundle:
        console.print(
            "[red]Error:[/red] --stub and --bundle are mutually exclusive."
        )
        raise typer.Exit(1)

    try:
        manager = get_engine_manager()
        info = manager.info(engine_id)

        # Show what we're about to install
        console.print(Panel.fit(
            f"Installing [bold]{info['name']}[/bold] "
            f"v{info['version']} ({info['provider']})",
            style="bold blue",
        ))
        console.print(f"  [dim]Capability:[/dim] {info['capability']}")

        # Show which installer mode is active
        if bundle:
            from app.engines.platform import detect_platform, get_platform_display_name
            current_platform = platform or detect_platform()
            console.print(f"  [dim]Mode:[/dim]      bundle "
                           f"({'symlink' if symlink else 'copy'})")
            console.print(f"  [dim]Bundle:[/dim]    {bundle}")
            console.print(f"  [dim]Platform:[/dim]  "
                           f"{get_platform_display_name(current_platform)} "
                           f"({current_platform})")
            console.print(f"  [dim]Offline:[/dim]   ✓ (no internet needed)")
        elif stub:
            console.print(f"  [dim]Mode:[/dim]      stub (fake binary)")
        else:
            console.print(f"  [dim]Mode:[/dim]      conda (download required)")
            console.print(f"  [dim]Source:[/dim]    {info['source']['type']} "
                           f"({info['source'].get('package', '')})")
        console.print(f"  [dim]Runtime:[/dim]  "
                       f"{'CPU' if info['runtime']['cpu'] else ''}"
                       f"{' + CUDA' if info['runtime']['cuda'] else ''}")
        if info['license']['file_required']:
            console.print(
                f"  [yellow]License:[/yellow]  {info['license']['type']} "
                f"(requires license file)"
            )
        else:
            console.print(f"  [dim]License:[/dim]  {info['license']['type']}")
        console.print()

        if bundle:
            # Bundle mode: use install_bundle() directly
            install_path = manager.install_bundle(
                engine_id, bundle, symlink=symlink, force=force,
                platform_id=platform,
            )
        elif stub:
            # Stub mode: construct a manager with StubInstaller
            manager = EngineManager(
                store=get_engine_manager().store,
                installer=StubInstaller(),
                manifests=get_engine_manager().manifests,
            )
            install_path = manager.install(engine_id, force=force)
        else:
            # Default: conda mode
            if not manager.installer.is_available():
                console.print(
                    f"[red]Installer not available:[/red] "
                    f"{manager.installer.availability_reason()}"
                )
                raise typer.Exit(1)
            install_path = manager.install(engine_id, force=force)

        console.print(
            f"[green]✓[/green] Installed [bold]{engine_id}[/bold] "
            f"-> {install_path}"
        )
    except KeyError:
        console.print(f"[red]Unknown engine:[/red] {engine_id!r}")
        console.print("Hint: run `python -m app.cli engines list` to see "
                       "available engines.")
        raise typer.Exit(1)
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]Install failed:[/red] {exc}")
        raise typer.Exit(1)


@engines_app.command("uninstall")
def engines_uninstall(
    engine_id: str = typer.Argument(..., help="Engine id to remove."),
):
    """Remove an installed engine from the local store."""
    from app.engines.manager import get_engine_manager
    try:
        if not get_engine_manager().is_installed(engine_id):
            console.print(
                f"[yellow]Engine '{engine_id}' is not installed.[/yellow]"
            )
            raise typer.Exit(1)
        deleted = get_engine_manager().uninstall(engine_id)
        if deleted:
            console.print(f"[green]✓[/green] Removed [bold]{engine_id}[/bold]")
        else:
            console.print(f"[red]Failed to remove {engine_id!r}[/red]")
            raise typer.Exit(1)
    except KeyError:
        console.print(f"[red]Unknown engine:[/red] {engine_id!r}")
        raise typer.Exit(1)


@engines_app.command("health")
def engines_health(
    engine_id: str = typer.Argument(..., help="Engine id to health-check."),
    run_binary: bool = typer.Option(
        True, "--run-binary/--no-run-binary",
        help="Attempt to run the binary with --version. Set to --no-run-binary "
             "in restricted environments.",
    ),
):
    """Check if an installed engine is actually usable (PR17).

    Distinguishes 'installed' (marker file exists) from 'ready' (binary
    exists, is executable, license valid, runs --version).
    """
    from app.engines.manager import get_engine_manager
    try:
        result = get_engine_manager().health_check(
            engine_id, run_binary=run_binary,
        )
    except KeyError:
        console.print(f"[red]Unknown engine:[/red] {engine_id!r}")
        raise typer.Exit(1)

    from app.engines.health import status_display

    console.print(Panel.fit(
        f"Health check: [bold]{engine_id}[/bold]",
        style="bold blue",
    ))
    console.print()
    console.print(f"  Status:      {status_display(result['status'])}")
    console.print(f"  Binary:      "
                   f"{'✓ exists' if result['binary_exists'] else '✗ missing'}")
    if result["binary_path"]:
        console.print(f"  Path:        {result['binary_path']}")
    console.print(f"  Executable:  "
                   f"{'✓' if result['binary_executable'] else '✗'}")
    if result["binary_runs"] is not None:
        console.print(f"  Runs:        "
                       f"{'✓ --version ok' if result['binary_runs'] else '✗ --version failed'}")
    if result["license_required"]:
        console.print(f"  License:     "
                       f"{'✓ valid' if result['license_valid'] else '✗ required but missing'}")
        if result["license_path"]:
            console.print(f"  License path: {result['license_path']}")

    if result["issues"]:
        console.print()
        console.print("[bold]Issues:[/bold]")
        for issue in result["issues"]:
            console.print(f"  [red]•[/red] {issue}")
    else:
        console.print()
        if result["status"] == "ready":
            console.print("[green]✓ Engine is ready to use.[/green]")
        else:
            console.print("[yellow]⚠ Engine is not ready.[/yellow]")


@engines_app.command("license")
def engines_license(
    engine_id: str = typer.Argument(..., help="Engine id."),
    license_path: str = typer.Argument(..., help="Path to the license file."),
):
    """Provide a license file for an installed engine (PR17).

    Copies the license file into the engine's install directory so it
    travels with the engine. After this, the health check will report
    READY (or BROKEN if the binary is still missing) instead of
    LICENSE_REQUIRED.
    """
    from app.engines.manager import get_engine_manager
    try:
        dest = get_engine_manager().store_license(engine_id, license_path)
        console.print(
            f"[green]✓[/green] License stored for [bold]{engine_id}[/bold]"
            f" -> {dest}"
        )
    except KeyError:
        console.print(f"[red]Unknown engine:[/red] {engine_id!r}")
        raise typer.Exit(1)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@engines_app.command("info")
def engines_info(
    engine_id: str = typer.Argument(..., help="Engine id."),
):
    """Show manifest + install status for one engine."""
    from app.engines.manager import get_engine_manager
    try:
        info = get_engine_manager().info(engine_id)
    except KeyError:
        console.print(f"[red]Unknown engine:[/red] {engine_id!r}")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"{info['name']} ({info['provider']})",
        style="bold blue",
    ))
    console.print()
    console.print(f"[bold]ID:[/bold]          {info['id']}")
    console.print(f"[bold]Version:[/bold]     {info['version']}")
    console.print(f"[bold]Capability:[/bold]  {info['capability']}")
    console.print(f"[bold]Description:[/bold] {info['description']}")
    console.print()
    console.print("[bold]License[/bold]")
    console.print(f"  Type:         {info['license']['type']}")
    console.print(f"  File required: {info['license']['file_required']}")
    if info['license']['notes']:
        console.print(f"  Notes:        {info['license']['notes'][:200]}...")
    console.print()
    console.print("[bold]Source[/bold]")
    console.print(f"  Type:     {info['source']['type']}")
    if info['source'].get('channel'):
        console.print(f"  Channel:  {info['source']['channel']}")
    if info['source'].get('package'):
        console.print(f"  Package:  {info['source']['package']}")
    console.print()
    console.print("[bold]Runtime[/bold]")
    console.print(f"  CPU:      {'✓' if info['runtime']['cpu'] else '✗'}")
    console.print(f"  CUDA:     {'✓' if info['runtime']['cuda'] else '✗'}")
    console.print()
    if info['requirements']:
        console.print("[bold]Requirements[/bold]")
        for r in info['requirements']:
            console.print(f"  - {r}")
        console.print()
    if info['outputs']:
        console.print("[bold]Outputs[/bold]")
        for o in info['outputs']:
            console.print(f"  - {o}")
        console.print()
    console.print("[bold]Status[/bold]")
    if info['installed']:
        console.print(f"  [green]✓ Installed[/green] at {info['install_path']}")
    else:
        console.print(f"  [yellow]✗ Not installed[/yellow]")
        console.print(f"  Installer: {info['installer']['name']}")
        if info['installer']['available']:
            console.print(
                f"  [dim]Install with:[/dim] "
                f"`python -m app.cli engines install {info['id']}`"
            )
        else:
            console.print(
                f"  [red]Installer not available:[/red] "
                f"{info['installer']['reason']}"
            )

    # PR17: health status + license state
    if info.get('health'):
        h = info['health']
        from app.engines.health import status_display
        console.print()
        console.print("[bold]Health[/bold]")
        console.print(f"  Status:  {status_display(h['status'])}")
        if h.get('license_required'):
            console.print(f"  License: "
                           f"{'✓ valid' if h.get('license_valid') else '✗ required but missing'}")
            if h.get('license_path'):
                console.print(f"  License path: {h['license_path']}")
            elif not h.get('license_valid'):
                console.print(
                    f"  [dim]Provide with:[/dim] "
                    f"`python -m app.cli engines license {info['id']} <path>`"
                )
        if h.get('issues'):
            console.print(f"  [red]Issues:[/red]")
            for issue in h['issues']:
                console.print(f"    • {issue}")


@engines_app.command("list")
def engines_list(
    installed: bool = typer.Option(
        False, "--installed",
        help="Only show installed engines.",
    ),
):
    """List all known engines (or only installed ones)."""
    from app.engines.manager import get_engine_manager

    if installed:
        engines_to_show = get_engine_manager().list_installed_engines()
        title = "Installed engines"
    else:
        engines_to_show = get_engine_manager().list_known_engines()
        title = "Known engines"

    if not engines_to_show:
        console.print(f"[yellow]No {('installed' if installed else 'known')} "
                       f"engines.[/yellow]")
        return

    console.print(Panel.fit(title, style="bold blue"))
    console.print()

    table = Table(show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Version")
    table.add_column("Capability")
    table.add_column("Status")

    for m in engines_to_show:
        is_installed = get_engine_manager().is_installed(m.id)
        status = "[green]✓ Installed[/green]" if is_installed else \
                 "[yellow]✗ Not installed[/yellow]"
        table.add_row(m.id, m.name, m.provider, m.version, m.capability, status)

    console.print(table)


@engines_app.command("adapters")
def engines_adapters(
    step_id: str = typer.Argument(..., help="Step id (e.g. brain_extraction)."),
):
    """List available engine adapters for a given step (the old `engines <step>` behavior)."""
    from app.engines import (
        BRAIN_EXTRACTION_ENGINES,
        FMRI_MOTION_ENGINES,
        DWI_CORRECTION_ENGINES,
        DEFACING_ENGINES,
    )
    engine_registries = {
        "brain_extraction":            BRAIN_EXTRACTION_ENGINES,
        "fmri_motion_correction":     FMRI_MOTION_ENGINES,
        "dwi_motion_eddy_correction": DWI_CORRECTION_ENGINES,
        "defacing":                    DEFACING_ENGINES,
    }

    try:
        registry.get(step_id)
    except KeyError:
        console.print(f"[red]Unknown step: {step_id}[/red]")
        raise typer.Exit(1)

    if step_id not in engine_registries:
        console.print(
            f"[yellow]Step '{step_id}' does not support engine selection.[/yellow]\n"
            f"It has a single implementation."
        )
        return

    engine_registry = engine_registries[step_id]
    available = engine_registry.available_engines()

    console.print(
        Panel.fit(
            f"Engine adapters for [bold]{step_id}[/bold]",
            style="bold blue",
        )
    )
    console.print(f"Default adapter: [cyan]{engine_registry.get_default()}[/cyan]\n")

    table = Table(show_lines=False)
    table.add_column("Engine", style="cyan")
    table.add_column("Status")
    table.add_column("Maturity")
    table.add_column("Notes")

    for engine in available:
        status = "[green]✓ Available[/green]" if engine["available"] \
            else f"[red]✗ Not available[/red]\n[dim]{engine['availability_reason']}[/dim]"
        badge, color = MATURITY_BADGE.get(engine["maturity"], ("?", "white"))
        maturity_str = f"[{color}]{badge} {engine['maturity']}[/{color}]"
        notes = engine["validation_notes"] or ""
        if len(notes) > 80:
            notes = notes[:77] + "..."
        table.add_row(engine["name"], status, maturity_str, notes)

    console.print(table)


# --------------------------------------------------------------------- #
# Command: resolve
# --------------------------------------------------------------------- #

@app.command()
def resolve(
    steps: List[str] = typer.Argument(..., help="Step ids to resolve."),
    param: List[str] = typer.Option(
        [], "--param", "-p",
        help="Step params as step.key=value (e.g. brain_extraction.engine=otsu).",
    ),
    skip_recommended: List[str] = typer.Option(
        [], "--skip-recommended",
        help="Recommended dependency step ids to suppress.",
    ),
):
    """Preview the resolved pipeline: requested steps, auto-inserted
    dependencies, and final execution order."""
    from app.pipeline.dependency_resolver import DependencyError, resolve_pipeline

    step_params = parse_params(param)

    try:
        resolution = resolve_pipeline(
            steps,
            step_params=step_params,
            skip_recommended=skip_recommended,
        )
    except DependencyError as exc:
        console.print(f"[red]Dependency resolution error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(Panel.fit("Pipeline Resolution", style="bold blue"))

    console.print("\n[bold]Requested:[/bold]")
    for s in resolution["requested"]:
        console.print(f"  {s}")

    breakdown = resolution["tier_breakdown"]
    if breakdown["hard"]:
        console.print("\n[bold]Required dependencies (hard):[/bold]")
        for s in breakdown["hard"]:
            console.print(f"  [green]{s}[/green]")
    if breakdown["recommended"]:
        console.print("\n[bold]Recommended dependencies:[/bold]")
        for s in breakdown["recommended"]:
            console.print(f"  [yellow]{s}[/yellow]")
    if breakdown["conditional"]:
        console.print("\n[bold]Conditional dependencies:[/bold]")
        for s in breakdown["conditional"]:
            console.print(f"  [cyan]{s}[/cyan]")
    if resolution["skipped_recommended"]:
        console.print("\n[bold]Skipped recommended:[/bold]")
        for s in resolution["skipped_recommended"]:
            console.print(f"  [dim strikethrough]{s}[/dim strikethrough]")

    console.print("\n[bold]Final execution order:[/bold]\n")
    for i, step_id in enumerate(resolution["ordered_pipeline"], 1):
        console.print(f"  [bold]{i}.[/bold] {step_id}")


@engines_app.command("resolve")
def engines_resolve(
    preset: Optional[str] = typer.Option(
        None, "--preset",
        help="Resolve engines required by this preset name.",
    ),
    steps: Optional[str] = typer.Option(
        None, "--steps",
        help="Comma-separated step ids to resolve (alternative to --preset).",
    ),
):
    """Preview which engines are required for a preset or pipeline (PR18).

    Shows which engines are ready, which are missing, and which need
    a license file -- without installing anything.
    """
    from app.engines.manager import get_engine_manager
    from app.engines.resolver import (
        resolve_engines_for_preset, resolve_engines_for_steps,
        format_resolution_report,
    )

    if not preset and not steps:
        console.print("[red]Error:[/red] either --preset or --steps is required.")
        raise typer.Exit(1)

    manager = get_engine_manager()

    if preset:
        from app.presets import get_preset as get_preset_obj
        preset_obj = get_preset_obj(preset)
        if preset_obj is None:
            console.print(f"[red]Error:[/red] unknown preset: {preset!r}")
            raise typer.Exit(1)
        report = resolve_engines_for_preset(preset_obj, manager)
    else:
        step_list = [s.strip() for s in steps.split(",") if s.strip()]
        report = resolve_engines_for_steps(step_list, {}, manager)

    console.print(Panel.fit(
        "Engine resolution",
        style="bold blue",
    ))
    console.print()
    console.print(format_resolution_report(report))


# --------------------------------------------------------------------- #
# Command group: presets
# --------------------------------------------------------------------- #

presets_app = typer.Typer(
    name="presets",
    help="List, inspect, create, and delete named pipeline presets.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(presets_app, name="presets")


@presets_app.command("list")
def presets_list():
    """List all available presets (built-ins first, then customs)."""
    from app.presets import list_presets

    all_presets = list_presets()
    if not all_presets:
        console.print("[yellow]No presets available.[/yellow]")
        return

    console.print(Panel.fit("Available presets", style="bold blue"))
    console.print()

    table = Table(show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Steps", justify="right")
    table.add_column("Description")

    for p in all_presets:
        type_str = "[dim]built-in[/dim]" if p.is_builtin else "[cyan]custom[/cyan]"
        # Truncate description for table view
        desc = p.description or ""
        if len(desc) > 70:
            desc = desc[:67] + "..."
        table.add_row(p.name, type_str, str(len(p.step_ids)), desc)

    console.print(table)


@presets_app.command("show")
def presets_show(
    name: str = typer.Argument(..., help="Preset name."),
):
    """Show preset details + the resolved pipeline (deps + order +
    tier breakdown + maturity of components)."""
    from app.presets import get_preset
    from app.pipeline.dependency_resolver import (
        DependencyError, resolve_pipeline,
    )
    from app.pipeline.step_registry import registry as step_registry

    preset = get_preset(name)
    if preset is None:
        console.print(f"[red]Error:[/red] unknown preset: {name!r}")
        console.print("Hint: run `python -m app.cli presets list` to see "
                       "available presets.")
        raise typer.Exit(1)

    # Header
    console.print(Panel.fit(
        f"{preset.name.upper()}"
        + (" [dim](built-in)[/dim]" if preset.is_builtin else " [cyan](custom)[/cyan]"),
        style="bold blue",
    ))
    console.print()

    # Purpose
    console.print("[bold]Purpose[/bold]")
    console.print(f"  {preset.description}")
    console.print()

    # Requested steps
    console.print("[bold]Requested steps[/bold]")
    for s in preset.step_ids:
        console.print(f"  {s}")
    console.print()

    # Step params (if any)
    if preset.step_params:
        console.print("[bold]Step parameters[/bold]")
        for step_id, params in preset.step_params.items():
            for k, v in params.items():
                console.print(f"  [cyan]{step_id}.{k}[/cyan] = {v!r}")
        console.print()

    # Skip recommended (if any)
    if preset.skip_recommended:
        console.print("[bold]Skip recommended[/bold]")
        for s in preset.skip_recommended:
            console.print(f"  [dim strikethrough]{s}[/dim strikethrough]")
        console.print()

    # Modality filter (if any)
    if preset.target_modalities:
        console.print("[bold]Modality filter[/bold]")
        console.print(f"  {', '.join(preset.target_modalities)}")
        console.print()

    # Resolve the pipeline (this is the load-bearing call -- same backend
    # as `resolve` and `run`)
    try:
        resolution = resolve_pipeline(
            preset.step_ids,
            step_params=preset.step_params,
            skip_recommended=preset.skip_recommended,
        )
    except DependencyError as exc:
        console.print(f"[red]Dependency resolution error:[/red] {exc}")
        raise typer.Exit(1)

    # Tier breakdown
    breakdown = resolution["tier_breakdown"]
    console.print("[bold]Dependencies[/bold]")
    if breakdown["hard"]:
        console.print(f"  Hard:        [green]{len(breakdown['hard'])}[/green]  ({', '.join(breakdown['hard'])})")
    if breakdown["recommended"]:
        console.print(f"  Recommended: [yellow]{len(breakdown['recommended'])}[/yellow]  ({', '.join(breakdown['recommended'])})")
    if breakdown["conditional"]:
        console.print(f"  Conditional: [cyan]{len(breakdown['conditional'])}[/cyan]  ({', '.join(breakdown['conditional'])})")
    if not any(breakdown.values()):
        console.print("  [dim]None[/dim]")
    console.print()

    # Final execution order with maturity badges
    console.print("[bold]Final execution order[/bold]")
    for i, step_id in enumerate(resolution["ordered_pipeline"], 1):
        try:
            step = step_registry.get(step_id)
            badge, color = MATURITY_BADGE.get(step.maturity, ("?", "white"))
            maturity_str = f"[{color}]{badge}[/{color}]"
        except KeyError:
            maturity_str = "[red]?[/red]"
        console.print(f"  [bold]{i}.[/bold] {maturity_str} {step_id}")
    console.print()

    # Maturity summary
    production_count = sum(
        1 for sid in resolution["ordered_pipeline"]
        if (step_registry.get(sid).maturity == "production_ready"
            if _step_exists(step_registry, sid) else False)
    )
    experimental_count = sum(
        1 for sid in resolution["ordered_pipeline"]
        if (step_registry.get(sid).maturity == "experimental"
            if _step_exists(step_registry, sid) else False)
    )
    reference_count = sum(
        1 for sid in resolution["ordered_pipeline"]
        if (step_registry.get(sid).maturity == "reference"
            if _step_exists(step_registry, sid) else False)
    )
    console.print("[bold]Component maturity[/bold]")
    console.print(
        f"  [green]✓ Production-ready:[/green] {production_count}    "
        f"[yellow]• Reference:[/yellow] {reference_count}    "
        f"[yellow]⚠ Experimental:[/yellow] {experimental_count}"
    )


def _step_exists(reg, step_id):
    try:
        reg.get(step_id)
        return True
    except KeyError:
        return False


def _split_commas(items):
    """Splits comma-separated entries in a list of strings. Used by
    `--steps`, `--modalities`, etc. so users can pass either
    `--steps a --steps b` or `--steps a,b`."""
    if not items:
        return []
    out = []
    for it in items:
        out.extend(s.strip() for s in it.split(",") if s.strip())
    return out


@presets_app.command("create")
def presets_create(
    name: str = typer.Argument(..., help="Preset name (must be unique; "
                                          "cannot shadow a built-in)."),
    steps: List[str] = typer.Option(
        ..., "--steps", "-s",
        help="Step ids to include in the preset. "
             "Pass --steps multiple times: `--steps reorientation --steps "
             "brain_extraction`. Or comma-separated: `--steps reorientation,brain_extraction`.",
    ),
    description: str = typer.Option(
        "", "--description", "-d",
        help="Human-readable purpose (shown in `presets show`).",
    ),
    param: List[str] = typer.Option(
        [], "--param", "-p",
        help="Step params as step.key=value (e.g. brain_extraction.engine=otsu).",
    ),
    skip_recommended: List[str] = typer.Option(
        [], "--skip-recommended",
        help="Recommended dependency step ids to suppress when applied. "
             "Pass --skip-recommended multiple times, or comma-separated.",
    ),
    modality: Optional[List[str]] = typer.Option(
        None, "--modality", "-m",
        help="Modality filter (e.g. --modality T1 --modality FLAIR). "
             "Pass --modality multiple times, or comma-separated.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="If a custom preset with this name already exists, replace it.",
    ),
):
    """Save a custom preset to disk. Reusable via `run --preset <name>`
    or via the API.

    Step ids can be passed two ways:
        --steps reorientation --steps brain_extraction  (repeated)
        --steps reorientation,brain_extraction         (comma-separated)
    Both forms can be mixed in the same command.
    """
    from app.presets import register_preset
    from app.presets.schema import Preset

    step_ids = _split_commas(steps)
    step_params = parse_params(param)
    skip_recs = _split_commas(skip_recommended)
    mods = _split_commas(modality) if modality else None

    if not step_ids:
        console.print("[red]Error:[/red] --steps must list at least one step id.")
        raise typer.Exit(1)

    preset = Preset(
        name=name,
        description=description,
        step_ids=step_ids,
        step_params=step_params,
        skip_recommended=skip_recs,
        target_modalities=mods,
        is_builtin=False,  # forced to False on save by the registry
    )

    try:
        saved = register_preset(preset, overwrite=overwrite)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Saved preset: [bold]{saved.name}[/bold]")
    console.print(
        f"  [dim]Steps:[/dim] {saved.step_ids.__len__()} steps, "
        f"{len(saved.step_params)} param overrides"
    )
    if saved.target_modalities:
        console.print(f"  [dim]Modality filter:[/dim] {', '.join(saved.target_modalities)}")


@presets_app.command("delete")
def presets_delete(
    name: str = typer.Argument(..., help="Preset name to delete."),
):
    """Delete a custom preset. Built-ins cannot be deleted."""
    from app.presets import delete_preset

    try:
        deleted = delete_preset(name)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if not deleted:
        console.print(
            f"[red]Error:[/red] preset not found: {name!r}. "
            "Run `presets list` to see what's available."
        )
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Deleted preset: [bold]{name}[/bold]")


@presets_app.command("install")
def presets_install(
    name: str = typer.Argument(..., help="Preset name."),
    bundle_dir: Optional[str] = typer.Option(
        None, "--bundle-dir",
        help="Directory containing pre-bundled engine subdirs. If given, "
             "uses BundleInstaller (offline mode) for each missing engine. "
             "The dir should contain <engine_id>/<platform>/bin/<binary> "
             "subdirs. If not given, uses CondaInstaller (requires internet).",
    ),
    platform: Optional[str] = typer.Option(
        None, "--platform",
        help="Override platform detection for bundle resolution.",
    ),
    symlink: bool = typer.Option(
        False, "--symlink",
        help="With --bundle-dir: symlink instead of copy.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be installed without actually installing.",
    ),
):
    """Install all engines required by a preset (PR18).

    Examines the preset's steps + step_params, resolves which engines
    are needed, and installs any that are missing. Also checks for
    license requirements.

    If --bundle-dir is given, uses BundleInstaller (offline mode) for
    each missing engine. Otherwise uses CondaInstaller (requires
    internet + conda/micromamba).
    """
    from app.engines.manager import get_engine_manager
    from app.engines.resolver import resolve_engines_for_preset, format_resolution_report
    from app.presets import get_preset

    preset = get_preset(name)
    if preset is None:
        console.print(f"[red]Error:[/red] unknown preset: {name!r}")
        raise typer.Exit(1)

    manager = get_engine_manager()
    report = resolve_engines_for_preset(preset, manager)

    console.print(Panel.fit(
        f"Engine resolution for [bold]{name}[/bold]",
        style="bold blue",
    ))
    console.print()
    console.print(format_resolution_report(report))

    if report["all_ready"]:
        console.print()
        console.print("[green]All required engines are ready.[/green]")
        return

    # Collect missing + license-required engines
    to_install = report["missing"]
    if not to_install:
        console.print()
        console.print(
            "[yellow]No missing engines to install.[/yellow] "
            "Some engines may need a license file -- use "
            "`python -m app.cli engines license <id> <path>`."
        )
        return

    console.print()
    if dry_run:
        console.print(f"[dim]Dry run -- would install: {', '.join(to_install)}[/dim]")
        return

    # Install each missing engine
    for engine_id in to_install:
        console.print()
        console.print(f"[bold]Installing {engine_id}...[/bold]")
        try:
            if bundle_dir:
                install_path = manager.install_bundle(
                    engine_id,
                    os.path.join(bundle_dir, engine_id),
                    symlink=symlink,
                    platform_id=platform,
                )
            else:
                # Conda installer
                if not manager.installer.is_available():
                    console.print(
                        f"[red]Installer not available:[/red] "
                        f"{manager.installer.availability_reason()}"
                    )
                    console.print(
                        "[dim]Use --bundle-dir for offline installation.[/dim]"
                    )
                    raise typer.Exit(1)
                install_path = manager.install(engine_id)
            console.print(
                f"[green]✓[/green] Installed {engine_id} -> {install_path}"
            )
        except Exception as exc:
            console.print(f"[red]✗ Failed to install {engine_id}:[/red] {exc}")

    # Check for license requirements after install
    console.print()
    console.print("[bold]Post-install health check:[/bold]")
    for engine_id in to_install:
        status = manager.get_status(engine_id)
        from app.engines.health import status_display
        console.print(f"  {engine_id}: {status_display(status)}")
        if status == "license_required":
            console.print(
                f"    [dim]Provide license with:[/dim] "
                f"`python -m app.cli engines license {engine_id} <path>`"
            )


# --------------------------------------------------------------------- #
# Command: run
# --------------------------------------------------------------------- #

@app.command()
def run(
    input_dir: str = typer.Argument(..., help="Input dataset directory."),
    output_dir: str = typer.Argument(..., help="Output directory."),
    steps: Optional[List[str]] = typer.Option(
        None, "--steps", "-s",
        help="Step ids to execute. Pass --steps multiple times "
             "(--steps reorientation --steps brain_extraction) OR "
             "comma-separated (--steps reorientation,brain_extraction). "
             "Required unless --preset is given.",
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset",
        help="Apply a named preset (e.g. --preset ml-ready). "
             "Overrides --steps; --param and --skip-recommended still apply "
             "on top of the preset's defaults.",
    ),
    modalities: Optional[List[str]] = typer.Option(
        None, "--modalities", "-m",
        help="Filter to specific modalities (e.g. --modalities T1 FLAIR). "
             "If --preset is given and the preset has its own modality "
             "filter, this overrides it.",
    ),
    param: List[str] = typer.Option(
        [], "--param", "-p",
        help="Step params as step.key=value (e.g. brain_extraction.engine=synthstrip). "
             "If --preset is given, these override the preset's step_params.",
    ),
    skip_recommended: List[str] = typer.Option(
        [], "--skip-recommended",
        help="Recommended dependency step ids to suppress. "
             "If --preset is given, these add to the preset's skips.",
    ),
    subject_strategy: str = typer.Option(
        "auto", "--subject-strategy",
        help="Subject-id inference: auto | bids | depth",
    ),
    subject_depth: Optional[int] = typer.Option(
        None, "--subject-depth",
        help="Depth for --subject-strategy depth.",
    ),
    job_id: Optional[str] = typer.Option(
        None, "--job-id",
        help="Explicit job id (default: random UUID).",
    ),
):
    """Execute a pipeline against a dataset. Runs synchronously with live
    progress display.

    Two ways to specify the pipeline:
        1. Explicit:  --steps reorientation brain_extraction
        2. Preset:    --preset ml-ready
    If --preset is given, --param and --skip-recommended still apply on
    top of the preset's defaults (caller overrides take precedence).
    """
    from app.execution.engine import run_job

    # Validate: exactly one of --steps / --preset must be provided
    if not steps and not preset:
        console.print(
            "[red]Error:[/red] either --steps or --preset must be provided."
        )
        raise typer.Exit(1)
    if steps and preset:
        console.print(
            "[red]Error:[/red] --steps and --preset are mutually exclusive. "
            "Pick one."
        )
        raise typer.Exit(1)

    # If a preset is given, expand it into the same (step_ids, step_params,
    # skip_recommended, target_modalities) tuple that --steps would produce.
    # This is the load-bearing translation: presets aren't a separate
    # execution path, they're a configuration shorthand.
    preset_obj = None
    if preset:
        from app.presets import get_preset
        preset_obj = get_preset(preset)
        if preset_obj is None:
            console.print(f"[red]Error:[/red] unknown preset: {preset!r}")
            console.print(
                "Hint: run `python -m app.cli presets list` to see "
                "available presets."
            )
            raise typer.Exit(1)
        step_ids = list(preset_obj.step_ids)
        # Caller --param overrides preset step_params
        step_params = {**preset_obj.step_params, **parse_params(param)}
        # Caller --skip-recommended adds to preset skips
        effective_skip = list(preset_obj.skip_recommended) + list(skip_recommended)
        # Caller --modalities overrides preset target_modalities
        effective_modalities = modalities or preset_obj.target_modalities
    else:
        step_ids = _split_commas(steps)
        step_params = parse_params(param)
        effective_skip = list(skip_recommended)
        effective_modalities = modalities

    actual_job_id = job_id or str(uuid.uuid4())

    # Validate the resolved pipeline before doing anything
    from app.pipeline.dependency_resolver import DependencyError, resolve_pipeline
    try:
        resolution = resolve_pipeline(
            step_ids,
            step_params=step_params,
            skip_recommended=effective_skip,
        )
    except DependencyError as exc:
        console.print(f"[red]Dependency resolution error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"Running pipeline (job_id={actual_job_id[:8]}...)",
        style="bold blue",
    ))
    console.print(f"\nInput:  [cyan]{input_dir}[/cyan]")
    console.print(f"Output: [cyan]{output_dir}[/cyan]")
    if preset_obj:
        console.print(
            f"Preset: [bold]{preset_obj.name}[/bold] "
            f"{'[dim](built-in)[/dim]' if preset_obj.is_builtin else '[cyan](custom)[/cyan]'}"
        )
    console.print(f"Steps:  {', '.join(resolution['ordered_pipeline'])}")
    if effective_modalities:
        console.print(f"Modalities filter: {', '.join(effective_modalities)}")
    if effective_skip:
        console.print(
            f"[dim]Skipping recommended: {', '.join(effective_skip)}[/dim]"
        )
    console.print()

    # Live progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=20,
    ) as progress:
        task = progress.add_task(
            "Processing acquisitions", total=None,
        )

        def on_progress(snapshot):
            p = snapshot.get("progress") or {}
            total = p.get("total_files") or 0
            done = p.get("processed_files") or 0
            progress.update(task, total=total, completed=done)

        try:
            result = run_job(
                job_id=actual_job_id,
                input_dir=input_dir,
                output_dir=output_dir,
                selected_step_ids=step_ids,
                step_params=step_params,
                target_modalities=effective_modalities,
                skip_recommended=effective_skip or None,
                progress_callback=on_progress,
            )
        except Exception as exc:
            console.print(f"\n[red]Job failed:[/red] {exc}")
            raise typer.Exit(1)

        # Force a final refresh so the bar shows the completed state
        # before the summary panel below it.
        progress.refresh()

    # Final summary
    console.print()
    console.print("[bold green]✓ Job completed[/bold green]")

    per_file = result["per_file_results"]
    successes = [r for r in per_file if r["status"] == "success"]
    failures = [r for r in per_file if r["status"] == "failed"]

    console.print(
        f"  Processed: [green]{len(successes)}[/green] succeeded, "
        f"[red]{len(failures)}[/red] failed"
    )

    if failures:
        console.print("\n[bold red]Failed acquisitions:[/bold red]")
        for f in failures[:10]:
            console.print(
                f"  [red]✗[/red] {f['subject']}/{f['modality']}/{f['acquisition_id']}: "
                f"{f['error']}"
            )
        if len(failures) > 10:
            console.print(f"  ... and {len(failures) - 10} more")

    # Provenance pointer
    console.print()
    console.print(
        f"Provenance log: [cyan]{output_dir}/provenance/{actual_job_id}.jsonl[/cyan]"
    )


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    app()
