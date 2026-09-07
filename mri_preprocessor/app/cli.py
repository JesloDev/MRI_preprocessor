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

        # Try to parse value as JSON (so lists, numbers, bools work);
        # fall back to raw string.
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
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
# Command: engines
# --------------------------------------------------------------------- #

@app.command()
def engines(
    step_id: str = typer.Argument(..., help="Step id (e.g. brain_extraction)."),
):
    """List available engine adapters for a given step."""
    from app.engines import (
        BRAIN_EXTRACTION_ENGINES,
        FMRI_MOTION_ENGINES,
        DWI_CORRECTION_ENGINES,
    )
    engine_registries = {
        "brain_extraction":            BRAIN_EXTRACTION_ENGINES,
        "fmri_motion_correction":     FMRI_MOTION_ENGINES,
        "dwi_motion_eddy_correction": DWI_CORRECTION_ENGINES,
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
            f"Engines for [bold]{step_id}[/bold]",
            style="bold blue",
        )
    )
    console.print(f"Default engine: [cyan]{engine_registry.get_default()}[/cyan]\n")

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


# --------------------------------------------------------------------- #
# Command: run
# --------------------------------------------------------------------- #

@app.command()
def run(
    input_dir: str = typer.Argument(..., help="Input dataset directory."),
    output_dir: str = typer.Argument(..., help="Output directory."),
    steps: List[str] = typer.Option(
        ..., "--steps", "-s",
        help="Step ids to execute (e.g. --steps reorientation brain_extraction).",
    ),
    modalities: Optional[List[str]] = typer.Option(
        None, "--modalities", "-m",
        help="Filter to specific modalities (e.g. --modalities T1 FLAIR).",
    ),
    param: List[str] = typer.Option(
        [], "--param", "-p",
        help="Step params as step.key=value (e.g. brain_extraction.engine=synthstrip).",
    ),
    skip_recommended: List[str] = typer.Option(
        [], "--skip-recommended",
        help="Recommended dependency step ids to suppress.",
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
    progress display."""
    from app.execution.engine import run_job

    step_params = parse_params(param)
    actual_job_id = job_id or str(uuid.uuid4())

    # Validate the resolved pipeline before doing anything
    from app.pipeline.dependency_resolver import DependencyError, resolve_pipeline
    try:
        resolution = resolve_pipeline(
            steps,
            step_params=step_params,
            skip_recommended=skip_recommended,
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
    console.print(f"Steps:  {', '.join(resolution['ordered_pipeline'])}")
    if modalities:
        console.print(f"Modalities filter: {', '.join(modalities)}")
    if skip_recommended:
        console.print(
            f"[dim]Skipping recommended: {', '.join(skip_recommended)}[/dim]"
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
                selected_step_ids=steps,
                step_params=step_params,
                target_modalities=modalities,
                skip_recommended=skip_recommended or None,
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
