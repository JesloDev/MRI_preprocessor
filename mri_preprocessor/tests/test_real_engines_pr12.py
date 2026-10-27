"""
Tests for the 4 real subprocess wrappers (PR12).

pydeface, mri_deface, mcflirt, fsl_eddy were stubs in PR5/PR7. PR12
replaced them with real subprocess wrappers (same pattern as SynthStrip
in PR9). These tests exercise the full flow without needing the actual
external tools installed -- we use fake shell scripts that mimic each
tool's CLI surface.

Each test family covers:
  - Binary discovery via env var / PATH / tool-suite env (already tested
    in test_engine_availability.py -- PR12 doesn't change this).
  - Successful subprocess execution against the fake binary.
  - Subprocess failure (exit code != 0) raises RuntimeError with stderr.
  - Missing output file raises RuntimeError (output validation works).
  - Binary not found -> FileNotFoundError with helpful message.
  - Engine choice recorded in provenance via metrics["engine"].
  - Engine integrates with the dispatcher (calls via registry.get_fn).
"""
import os
import stat

import nibabel as nib
import numpy as np
import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import registry


# --------------------------------------------------------------------- #
# Helpers: fake-binary factories (one per engine, since their CLIs differ)
# --------------------------------------------------------------------- #

def _make_executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _make_fake_pydeface(binary_path, *, fail_with=None, skip_output=False):
    """Fake pydeface script.

    Real pydeface CLI: `pydeface <input> --outfile <output> [--force]`.
    Our fake version copies the input to the outfile.
    """
    if fail_with is not None:
        _make_executable(binary_path, f"""#!/bin/bash
echo "fake_pydeface failing on purpose" >&2
exit {fail_with}
""")
        return

    if skip_output:
        _make_executable(binary_path, """#!/bin/bash
# Don't actually produce anything
exit 0
""")
        return

    _make_executable(binary_path, """#!/bin/bash
# pydeface input.nii.gz --outfile output.nii.gz [--force]
# Capture the input (first positional) BEFORE the option loop,
# because the loop shifts through everything.
INPUT="$1"
shift
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outfile) OUTPUT="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$OUTPUT" ] && cp "$INPUT" "$OUTPUT"
exit 0
""")


def _make_fake_mri_deface(binary_path, fake_fs_home, *,
                            fail_with=None, skip_output=False):
    """Fake mri_deface script.

    Real mri_deface CLI: `mri_deface <invol> <face.gca> <aseg.gca> <outvol>`.
    The face/skull templates live under $FREESURFER_HOME/subjects/face/.
    Our fake version creates the template files (so _resolve_templates
    succeeds) and copies input to output.
    """
    # Create the template files at the expected paths
    face_dir = fake_fs_home / "subjects" / "face" / "face_gca"
    face_dir.mkdir(parents=True, exist_ok=True)
    (face_dir / "face.gca").write_text("fake_face_template")
    (face_dir / "aseg.gca").write_text("fake_skull_template")

    if fail_with is not None:
        _make_executable(binary_path, f"""#!/bin/bash
echo "fake_mri_deface failing on purpose" >&2
exit {fail_with}
""")
        return

    if skip_output:
        _make_executable(binary_path, """#!/bin/bash
# Don't produce output
exit 0
""")
        return

    _make_executable(binary_path, """#!/bin/bash
# argv: input face.gca aseg.gca output
INPUT="$1"
OUTPUT="$4"
[ -n "$OUTPUT" ] && cp "$INPUT" "$OUTPUT"
exit 0
""")


def _make_fake_mcflirt(binary_path, *, fail_with=None, skip_output=False):
    """Fake mcflirt script.

    Real mcflirt CLI: `mcflirt -in <input> -out <output_base> -plots`.
    Writes <output_base>.nii.gz and <output_base>.par.
    """
    if fail_with is not None:
        _make_executable(binary_path, f"""#!/bin/bash
echo "fake_mcflirt failing on purpose" >&2
exit {fail_with}
""")
        return

    if skip_output:
        _make_executable(binary_path, """#!/bin/bash
# Don't produce anything
exit 0
""")
        return

    _make_executable(binary_path, """#!/bin/bash
IN=""
OUT_BASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -in)     IN="$2"; shift 2 ;;
    -out)    OUT_BASE="$2"; shift 2 ;;
    -plots)  shift ;;
    -mats)   shift ;;
    *)       shift ;;
  esac
done
# Copy input to <OUT_BASE>.nii.gz
[ -n "$OUT_BASE" ] && cp "$IN" "${OUT_BASE}.nii.gz"
# Write a fake .par file (motion params: 6 cols per volume)
[ -n "$OUT_BASE" ] && python3 -c "
import nibabel as nib
img = nib.load('$IN')
n_vols = img.shape[-1] if len(img.shape) > 3 else 1
with open('${OUT_BASE}.par', 'w') as f:
    for _ in range(n_vols):
        f.write('0 0 0 0.1 0.2 0.3\\n')
"
exit 0
""")


def _make_fake_fsl_eddy(binary_path, *, fail_with=None, skip_output=False):
    """Fake FSL eddy script.

    Real eddy CLI: `eddy --imain=<...> --mask=<...> --acqp=<...>
    --index=<...> --bvals=<...> --bvecs=<...> --out=<...>`.
    Writes <out_base>.nii.gz and (optionally) <out_base>.eddy_rotated_bvecs.
    """
    if fail_with is not None:
        _make_executable(binary_path, f"""#!/bin/bash
echo "fake_eddy failing on purpose" >&2
exit {fail_with}
""")
        return

    if skip_output:
        _make_executable(binary_path, """#!/bin/bash
exit 0
""")
        return

    _make_executable(binary_path, """#!/bin/bash
IMAIN=""
OUT_BASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imain=*) IMAIN="${1#--imain=}"; shift ;;
    --out=*)   OUT_BASE="${1#--out=}"; shift ;;
    *)         shift ;;
  esac
done
# Copy imain to <OUT_BASE>.nii.gz
[ -n "$OUT_BASE" ] && cp "$IMAIN" "${OUT_BASE}.nii.gz"
# Write a fake rotated bvecs file
[ -n "$OUT_BASE" ] && python3 -c "
with open('${OUT_BASE}.eddy_rotated_bvecs', 'w') as f:
    f.write('0 1 0\\n0 0 0\\n0 0 0\\n')
"
exit 0
""")


# --------------------------------------------------------------------- #
# pydeface tests
# --------------------------------------------------------------------- #

class TestPyDefaceEngine:
    @pytest.fixture(autouse=True)
    def _setup_env(self, tmp_path, monkeypatch):
        self.binary_path = tmp_path / "fake_pydeface"
        self.tmp_path = tmp_path
        _make_fake_pydeface(self.binary_path)
        monkeypatch.setenv("MRI_PREPROCESSOR_PYDEFACE", str(self.binary_path))

    def _t1(self):
        t1 = self.tmp_path / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(t1))
        return str(t1)

    def test_successful_execution_produces_defaced_output(self):
        from app.engines.defacing.pydeface import PyDefaceEngine
        engine = PyDefaceEngine()

        output_dir = str(self.tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._t1(), output_dir, {}, {})

        assert os.path.isfile(result["output_path"])
        assert result["output_path"].endswith("defaced.nii.gz")
        assert result["metrics"]["engine"] == "pydeface"
        assert result["metrics"]["returncode"] == 0

    def test_subprocess_failure_raises_runtime_error(self):
        from app.engines.defacing.pydeface import PyDefaceEngine
        _make_fake_pydeface(self.binary_path, fail_with=42)

        engine = PyDefaceEngine()
        output_dir = str(self.tmp_path / "fail")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._t1(), output_dir, {}, {})
        assert "42" in str(exc_info.value)  # exit code
        assert "fake_pydeface failing" in str(exc_info.value)  # stderr

    def test_missing_output_raises_runtime_error(self):
        from app.engines.defacing.pydeface import PyDefaceEngine
        _make_fake_pydeface(self.binary_path, skip_output=True)

        engine = PyDefaceEngine()
        output_dir = str(self.tmp_path / "no_out")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._t1(), output_dir, {}, {})
        assert "defaced output is missing" in str(exc_info.value)

    def test_binary_missing_raises_filenotfounderror(self, monkeypatch):
        monkeypatch.delenv("MRI_PREPROCESSOR_PYDEFACE", raising=False)
        from app.engines.defacing.pydeface import PyDefaceEngine
        engine = PyDefaceEngine()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run(self._t1(), str(self.tmp_path / "no_bin"), {}, {})
        assert "pydeface" in str(exc_info.value).lower()
        assert "placeholder" in str(exc_info.value).lower()  # suggests fallback

    def test_extra_args_passed_to_subprocess(self):
        from app.engines.defacing.pydeface import PyDefaceEngine
        # Make a binary that logs its argv
        log_path = self.tmp_path / "argv.log"
        _make_executable(self.binary_path, f"""#!/bin/bash
echo "$@" > {log_path}
# Capture input (first positional) before the option loop
INPUT="$1"
shift
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outfile) OUTPUT="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$OUTPUT" ] && cp "$INPUT" "$OUTPUT"
exit 0
""")
        engine = PyDefaceEngine()
        output_dir = str(self.tmp_path / "extra_args")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._t1(), output_dir, {
            "extra_args": ["--verbose", "--cost", "mutualinfo"],
        }, {})
        logged = log_path.read_text()
        assert "--verbose" in logged
        assert "mutualinfo" in logged


# --------------------------------------------------------------------- #
# mri_deface tests
# --------------------------------------------------------------------- #

class TestMRIDefaceEngine:
    @pytest.fixture(autouse=True)
    def _setup_env(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.binary_path = tmp_path / "fake_mri_deface"
        self.fake_fs_home = tmp_path / "freesurfer"
        self.fake_fs_home.mkdir(parents=True, exist_ok=True)
        _make_fake_mri_deface(self.binary_path, self.fake_fs_home)
        monkeypatch.setenv("MRI_PREPROCESSOR_MRI_DEFACE", str(self.binary_path))
        monkeypatch.setenv("FREESURFER_HOME", str(self.fake_fs_home))

    def _t1(self):
        t1 = self.tmp_path / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(t1))
        return str(t1)

    def test_successful_execution_produces_defaced_output(self):
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        engine = MRIDefaceEngine()
        output_dir = str(self.tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._t1(), output_dir, {}, {})

        assert os.path.isfile(result["output_path"])
        assert result["output_path"].endswith("defaced.nii.gz")
        assert result["metrics"]["engine"] == "mri_deface"
        assert "face_template" in result["metrics"]
        assert "skull_template" in result["metrics"]

    def test_templates_resolved_from_freesurfer_home(self):
        """When face_template/skull_template aren't passed, they default
        to $FREESURFER_HOME/subjects/face/face_gca/{face,aseg}.gca."""
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        engine = MRIDefaceEngine()
        face, skull = engine._resolve_templates({})
        assert face.endswith("subjects/face/face_gca/face.gca")
        assert skull.endswith("subjects/face/face_gca/aseg.gca")

    def test_explicit_template_params_override(self):
        """params['face_template'] and params['skull_template'] override
        the default FreeSurfer paths."""
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        custom_face = self.tmp_path / "custom_face.gca"
        custom_skull = self.tmp_path / "custom_skull.gca"
        custom_face.write_text("custom_face")
        custom_skull.write_text("custom_skull")
        engine = MRIDefaceEngine()
        face, skull = engine._resolve_templates({
            "face_template": str(custom_face),
            "skull_template": str(custom_skull),
        })
        assert face == str(custom_face)
        assert skull == str(custom_skull)

    def test_missing_template_raises_filenotfounderror(self, monkeypatch):
        """When FREESURFER_HOME isn't set and templates aren't passed,
        the engine raises FileNotFoundError."""
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        engine = MRIDefaceEngine()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine._resolve_templates({})
        assert "template" in str(exc_info.value).lower()

    def test_nonexistent_template_file_raises(self):
        """If a passed template path doesn't exist, the engine raises."""
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        engine = MRIDefaceEngine()
        with pytest.raises(FileNotFoundError):
            engine._resolve_templates({
                "face_template": "/nonexistent/face.gca",
                "skull_template": "/nonexistent/skull.gca",
            })

    def test_subprocess_failure_raises_runtime_error(self):
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        _make_fake_mri_deface(self.binary_path, self.fake_fs_home, fail_with=7)
        engine = MRIDefaceEngine()
        output_dir = str(self.tmp_path / "fail")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._t1(), output_dir, {}, {})
        assert "7" in str(exc_info.value)

    def test_binary_missing_raises_filenotfounderror(self, monkeypatch):
        monkeypatch.delenv("MRI_PREPROCESSOR_MRI_DEFACE", raising=False)
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        from app.engines.defacing.mri_deface import MRIDefaceEngine
        engine = MRIDefaceEngine()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run(self._t1(), str(self.tmp_path / "no_bin"), {}, {})
        assert "mri_deface" in str(exc_info.value).lower()
        assert "placeholder" in str(exc_info.value).lower()


# --------------------------------------------------------------------- #
# MCFLIRT tests
# --------------------------------------------------------------------- #

class TestMCFLIRTEngine:
    @pytest.fixture(autouse=True)
    def _setup_env(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.binary_path = tmp_path / "fake_mcflirt"
        _make_fake_mcflirt(self.binary_path)
        monkeypatch.setenv("MRI_PREPROCESSOR_MCFLIRT", str(self.binary_path))
        # Make sure no FSLDIR leaks from the test environment
        monkeypatch.delenv("FSLDIR", raising=False)

    def _bold(self):
        bold = self.tmp_path / "bold.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16, 10).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(bold))
        return str(bold)

    def test_successful_execution_produces_motion_corrected_output(self):
        from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
        engine = MCFLIRTMotionCorrection()
        output_dir = str(self.tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._bold(), output_dir, {}, {})

        assert os.path.isfile(result["output_path"])
        assert result["output_path"].endswith("motion_corrected.nii.gz")
        assert "motion_params" in result["outputs"]
        assert os.path.isfile(result["outputs"]["motion_params"])
        assert result["metrics"]["engine"] == "mcflirt"
        # Mean abs translation should be computed from the .par file
        assert result["metrics"]["mean_abs_translation_mm"] is not None

    def test_subprocess_failure_raises_runtime_error(self):
        from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
        _make_fake_mcflirt(self.binary_path, fail_with=1)
        engine = MCFLIRTMotionCorrection()
        output_dir = str(self.tmp_path / "fail")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._bold(), output_dir, {}, {})
        assert "1" in str(exc_info.value)

    def test_missing_output_raises_runtime_error(self):
        from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
        _make_fake_mcflirt(self.binary_path, skip_output=True)
        engine = MCFLIRTMotionCorrection()
        output_dir = str(self.tmp_path / "no_out")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._bold(), output_dir, {}, {})
        assert "motion-corrected output is missing" in str(exc_info.value)

    def test_binary_missing_raises_filenotfounderror(self, monkeypatch):
        monkeypatch.delenv("MRI_PREPROCESSOR_MCFLIRT", raising=False)
        from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
        engine = MCFLIRTMotionCorrection()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run(self._bold(), str(self.tmp_path / "no_bin"), {}, {})
        assert "mcflirt" in str(exc_info.value).lower()
        assert "simpleitk" in str(exc_info.value).lower()  # suggests fallback

    def test_motion_params_file_is_loadable(self):
        from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
        engine = MCFLIRTMotionCorrection()
        output_dir = str(self.tmp_path / "loadable")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._bold(), output_dir, {}, {})
        # The .par file should be numpy-loadable
        params = np.loadtxt(result["outputs"]["motion_params"])
        # 10 volumes * 6 params per volume = (10, 6) array
        assert params.shape == (10, 6)


# --------------------------------------------------------------------- #
# FSL eddy tests
# --------------------------------------------------------------------- #

class TestFSLEddyEngine:
    @pytest.fixture(autouse=True)
    def _setup_env(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.binary_path = tmp_path / "fake_eddy"
        _make_fake_fsl_eddy(self.binary_path)
        monkeypatch.setenv("MRI_PREPROCESSOR_FSL_EDDY", str(self.binary_path))
        monkeypatch.delenv("FSLDIR", raising=False)

    def _dwi(self):
        dwi = self.tmp_path / "dwi.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16, 4).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(dwi))
        return str(dwi)

    def _mask(self):
        mask = self.tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((16, 16, 16), dtype='uint8'),
                                  affine=np.eye(4)), str(mask))
        return str(mask)

    def _acqp(self):
        # FSL acqp.txt: one row per acquisition with format
        # echo_time pe_direction total_readout_time
        path = self.tmp_path / "acqp.txt"
        path.write_text("1 0 0 0.05\n")
        return str(path)

    def _index(self):
        # FSL index.txt: one entry per DWI volume, each row says which
        # acqp row applies to that volume
        path = self.tmp_path / "index.txt"
        path.write_text("1 1 1 1\n")
        return str(path)

    def _bvals(self):
        path = self.tmp_path / "dwi.bval"
        path.write_text("0 1000 1000 1000\n")
        return str(path)

    def _bvecs(self):
        path = self.tmp_path / "dwi.bvec"
        path.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")
        return str(path)

    def _all_params(self):
        return {
            "mask_path": self._mask(),
            "acqp_path": self._acqp(),
            "index_path": self._index(),
            "bvals_path": self._bvals(),
            "bvecs_path": self._bvecs(),
        }

    def test_successful_execution_produces_corrected_output(self):
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        output_dir = str(self.tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)
        result = engine.run(self._dwi(), output_dir, self._all_params(), {})

        assert os.path.isfile(result["output_path"])
        assert result["output_path"].endswith("motion_eddy_corrected.nii.gz")
        assert result["metrics"]["engine"] == "fsl_eddy"
        # rotated_bvecs should be produced by the fake binary
        assert "rotated_bvecs" in result["outputs"]

    def test_required_inputs_resolved_from_context(self):
        """When params['bvals_path'] / bvecs_path aren't passed, the
        engine reads them from the SubjectContext."""
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        context = {
            "bval_path": self._bvals(),
            "bvec_path": self._bvecs(),
            "brain_mask": self._mask(),
        }
        params = {
            "acqp_path": self._acqp(),
            "index_path": self._index(),
        }
        resolved = engine._resolve_required_inputs(self._dwi(), params, context)
        assert resolved["bvals"].endswith("dwi.bval")
        assert resolved["bvecs"].endswith("dwi.bvec")
        assert resolved["mask"].endswith("mask.nii.gz")

    def test_acqp_path_required(self):
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine._resolve_required_inputs(self._dwi(), {}, {})
        assert "acqp" in str(exc_info.value).lower()

    def test_index_path_required(self):
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine._resolve_required_inputs(self._dwi(),
                                              {"acqp_path": self._acqp()}, {})
        assert "index" in str(exc_info.value).lower()

    def test_mask_resolved_from_context_brain_mask(self):
        """context['brain_mask'] (set by an earlier brain_extraction step)
        is auto-resolved as the mask path."""
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        context = {"brain_mask": self._mask()}
        params = {
            "acqp_path": self._acqp(),
            "index_path": self._index(),
            "bvals_path": self._bvals(),
            "bvecs_path": self._bvecs(),
        }
        resolved = engine._resolve_required_inputs(self._dwi(), params, context)
        assert resolved["mask"] == context["brain_mask"]

    def test_missing_input_file_raises(self):
        """If a resolved path doesn't exist on disk, raises FileNotFoundError."""
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        with pytest.raises(FileNotFoundError):
            engine._resolve_required_inputs(
                self._dwi(),
                {
                    "acqp_path": "/nonexistent/acqp.txt",
                    "index_path": "/nonexistent/index.txt",
                    "bvals_path": "/nonexistent/bvals",
                    "bvecs_path": "/nonexistent/bvecs",
                    "mask_path": "/nonexistent/mask.nii.gz",
                },
                {},
            )

    def test_subprocess_failure_raises_runtime_error(self):
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        _make_fake_fsl_eddy(self.binary_path, fail_with=99)
        engine = FSLEddyCorrection()
        output_dir = str(self.tmp_path / "fail")
        os.makedirs(output_dir, exist_ok=True)
        with pytest.raises(RuntimeError) as exc_info:
            engine.run(self._dwi(), output_dir, self._all_params(), {})
        assert "99" in str(exc_info.value)

    def test_binary_missing_raises_filenotfounderror(self, monkeypatch):
        monkeypatch.delenv("MRI_PREPROCESSOR_FSL_EDDY", raising=False)
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run(self._dwi(), str(self.tmp_path / "no_bin"),
                        self._all_params(), {})
        assert "eddy" in str(exc_info.value).lower()
        assert "dipy" in str(exc_info.value).lower()  # suggests fallback

    def test_topup_path_optional(self, monkeypatch):
        """If params['topup_path'] is given, the engine uses it (--topup appears
        in the command and metrics['used_topup'] is True)."""
        from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
        engine = FSLEddyCorrection()

        # Create a sentinel file for topup_path (the engine skips the
        # isfile check for topup specifically, so we can pass any path).
        topup_file = self.tmp_path / "topup_results.txt"
        topup_file.write_text("fake topup")

        # Make a binary that logs its argv and produces output
        log_path = self.tmp_path / "argv.log"
        _make_executable(self.binary_path, """#!/bin/bash
echo "$@" > """ + str(log_path) + """
# Parse --imain and --out, copy imain to <out>.nii.gz
IMAIN=""
OUT_BASE=""
for arg in "$@"; do
  case "$arg" in
    --imain=*) IMAIN="${arg#--imain=}";;
    --out=*)   OUT_BASE="${arg#--out=}";;
  esac
done
[ -n "$OUT_BASE" ] && cp "$IMAIN" "${OUT_BASE}.nii.gz"
exit 0
""")

        output_dir = str(self.tmp_path / "topup_out")
        os.makedirs(output_dir, exist_ok=True)
        params = self._all_params()
        params["topup_path"] = str(topup_file)
        result = engine.run(self._dwi(), output_dir, params, {})

        logged = log_path.read_text()
        assert "--topup=" in logged
        assert result["metrics"]["used_topup"] is True


# --------------------------------------------------------------------- #
# Integration: dispatcher + provenance for the new real engines
# --------------------------------------------------------------------- #

class TestDispatcherAndProvenance:
    """The 4 real wrappers all flow through the dispatcher and record
    engine choice + binary path in provenance metrics."""

    def test_defacing_pydeface_via_dispatcher_records_engine(self, tmp_path, monkeypatch):
        # Setup fake pydeface binary
        binary = tmp_path / "fake_pydeface"
        _make_fake_pydeface(binary)
        monkeypatch.setenv("MRI_PREPROCESSOR_PYDEFACE", str(binary))

        # Build a T1
        t1 = tmp_path / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(t1))

        # Call via the dispatcher (registry.get_fn)
        defacing_fn = registry.get_fn("defacing")
        output_dir = str(tmp_path / "dispatcher_out")
        os.makedirs(output_dir, exist_ok=True)

        result = defacing_fn(
            input_path=str(t1),
            output_dir=output_dir,
            params={"engine": "pydeface"},
            context={},
        )

        assert result["metrics"]["engine"] == "pydeface"
        assert "binary_path" in result["metrics"]
        assert os.path.isfile(result["output_path"])

    def test_fmri_motion_correction_mcflirt_via_dispatcher(self, tmp_path, monkeypatch):
        binary = tmp_path / "fake_mcflirt"
        _make_fake_mcflirt(binary)
        monkeypatch.setenv("MRI_PREPROCESSOR_MCFLIRT", str(binary))
        monkeypatch.delenv("FSLDIR", raising=False)

        bold = tmp_path / "bold.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16, 10).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(bold))

        fn = registry.get_fn("fmri_motion_correction")
        output_dir = str(tmp_path / "dispatcher_mcflirt")
        os.makedirs(output_dir, exist_ok=True)
        result = fn(
            input_path=str(bold),
            output_dir=output_dir,
            params={"engine": "mcflirt"},
            context={},
        )
        assert result["metrics"]["engine"] == "mcflirt"
        assert "motion_params" in result["outputs"]
        assert os.path.isfile(result["outputs"]["motion_params"])

    def test_dwi_correction_fsl_eddy_via_dispatcher(self, tmp_path, monkeypatch):
        binary = tmp_path / "fake_eddy"
        _make_fake_fsl_eddy(binary)
        monkeypatch.setenv("MRI_PREPROCESSOR_FSL_EDDY", str(binary))
        monkeypatch.delenv("FSLDIR", raising=False)

        # Build DWI + sidecars + mask
        dwi = tmp_path / "dwi.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16, 4).astype(np.float32) * 100,
                                  affine=np.eye(4)), str(dwi))
        bval = tmp_path / "dwi.bval"
        bval.write_text("0 1000 1000 1000\n")
        bvec = tmp_path / "dwi.bvec"
        bvec.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")
        mask = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((16, 16, 16), dtype='uint8'),
                                  affine=np.eye(4)), str(mask))
        acqp = tmp_path / "acqp.txt"
        acqp.write_text("1 0 0 0.05\n")
        index = tmp_path / "index.txt"
        index.write_text("1 1 1 1\n")

        fn = registry.get_fn("dwi_motion_eddy_correction")
        output_dir = str(tmp_path / "dispatcher_eddy")
        os.makedirs(output_dir, exist_ok=True)
        result = fn(
            input_path=str(dwi),
            output_dir=output_dir,
            params={
                "engine": "fsl_eddy",
                "acqp_path": str(acqp),
                "index_path": str(index),
                "mask_path": str(mask),
                "bvals_path": str(bval),
                "bvecs_path": str(bvec),
            },
            context={},
        )
        assert result["metrics"]["engine"] == "fsl_eddy"
        assert os.path.isfile(result["output_path"])
