"""
Smoke tests for the macOS .app bundle produced by `scripts/package_macos_app.sh`.

Phase 7 added the bundling + mini-dylibbundler + ad-hoc resign +
codesign + DMG packaging pipeline. These tests verify the bundle is
shippable:

  (a) Every aliceVision_* binary inside the bundle has zero
      `/opt/homebrew/...` references in its load commands. If any
      reference leaked through, the .app fails on machines without
      Homebrew installed.

  (b) Every aliceVision_* binary inside the bundle launches without
      `/opt/homebrew/lib` on the DYLD path. This is the strongest
      "shippable" signal short of running on a clean machine.

  (c) The launcher script in `Contents/MacOS/meshroom` has the right
      shebang + env-var setup.

  (d) The bundled venv's `python3` runs + can import meshroom.

The tests are gated behind `RUN_APP_BUNDLE=1` (and the bundle's
existence — they don't auto-build) because they take ~30 s to run
even on a pre-built bundle. Default `pytest` runs skip them.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_APP = REPO_ROOT / "build" / "release" / "Meshroom.app"
APP = Path(os.environ.get("MESHROOM_APP", str(DEFAULT_APP)))

_gate = pytest.mark.skipif(
    os.environ.get("RUN_APP_BUNDLE") != "1" or platform.system() != "Darwin",
    reason=(
        "Run with RUN_APP_BUNDLE=1 on macOS. Optionally point MESHROOM_APP "
        "at a non-default .app location (e.g. RUN_APP_BUNDLE=1 "
        "MESHROOM_APP=/tmp/app_test/Meshroom.app pytest ...)."
    ),
)


@pytest.fixture(scope="module")
def app_path() -> Path:
    if not APP.is_dir():
        pytest.skip(
            f"No .app at {APP} — build first with "
            f"`bash scripts/package_macos_app.sh`."
        )
    return APP


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #

@_gate
def test_app_has_expected_layout(app_path):
    contents = app_path / "Contents"
    assert (contents / "Info.plist").is_file()
    assert (contents / "MacOS" / "meshroom").is_file()
    assert (contents / "Resources" / "alicevision").is_dir()
    assert (contents / "Resources" / "meshroom-mac").is_dir()
    assert (contents / "Resources" / "meshroom-venv").is_dir()
    assert (contents / "Resources" / "plugins").is_dir()
    assert (contents / "Resources" / "python_shim").is_dir()
    # The dylib bundle dir from Phase 7's mini-dylibbundler step.
    assert (contents / "Resources" / "lib").is_dir(), (
        "Contents/Resources/lib missing — package_macos_app.sh's "
        "bundler step didn't run"
    )


@_gate
def test_launcher_script_is_executable(app_path):
    launcher = app_path / "Contents" / "MacOS" / "meshroom"
    assert os.access(launcher, os.X_OK), f"launcher not executable: {launcher}"
    # The launcher exports AV_AI_MODELS_DIR + MESHROOM_NODES_PATH; verify
    # those references survived templating.
    body = launcher.read_text()
    assert "AV_AI_MODELS_DIR" in body
    assert "MESHROOM_NODES_PATH" in body
    assert "ALICEVISION_ROOT" in body


# --------------------------------------------------------------------------- #
# (a) No /opt/homebrew leaks in load commands
# --------------------------------------------------------------------------- #

@_gate
def test_binaries_have_no_homebrew_references(app_path):
    bin_dir = app_path / "Contents" / "Resources" / "alicevision"
    binaries = [p for p in bin_dir.iterdir()
                if p.name.startswith("aliceVision_") and os.access(p, os.X_OK)]
    assert binaries, f"no aliceVision_* binaries in {bin_dir}"

    leaks: list[tuple[str, str]] = []
    for binary in binaries:
        out = subprocess.run(
            ["otool", "-L", str(binary)],
            capture_output=True, text=True, check=True,
        )
        for line in out.stdout.splitlines()[1:]:
            tok = line.strip().split()
            if not tok:
                continue
            path = tok[0]
            if path.startswith("/opt/homebrew/"):
                leaks.append((binary.name, path))

    assert not leaks, (
        f"{len(leaks)} /opt/homebrew leaks across binaries — the "
        f"mini-dylibbundler step missed rewriting these references. "
        f"First few: {leaks[:5]}"
    )


@_gate
def test_bundled_dylibs_have_no_homebrew_references(app_path):
    lib_dir = app_path / "Contents" / "Resources" / "lib"
    libs = list(lib_dir.glob("*.dylib"))
    assert libs, f"no bundled dylibs in {lib_dir}"

    leaks: list[tuple[str, str]] = []
    for lib in libs:
        out = subprocess.run(
            ["otool", "-L", str(lib)],
            capture_output=True, text=True, check=True,
        )
        for line in out.stdout.splitlines()[1:]:
            tok = line.strip().split()
            if not tok:
                continue
            path = tok[0]
            if path.startswith("/opt/homebrew/"):
                leaks.append((lib.name, path))

    assert not leaks, (
        f"{len(leaks)} /opt/homebrew leaks across bundled dylibs — "
        f"recursive bundling missed transitive deps. First: {leaks[:5]}"
    )


# --------------------------------------------------------------------------- #
# (b) Binaries launch with empty DYLD path
# --------------------------------------------------------------------------- #

@_gate
@pytest.mark.parametrize("name", [
    "aliceVision_cameraInit",
    "aliceVision_featureExtraction",
    "aliceVision_meshing",
    "aliceVision_texturing",
])
def test_binary_launches_without_homebrew(app_path, name):
    """If the bundle is self-contained, every binary must run with
    DYLD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH BOTH cleared.

    We use --help so the binary exits fast after loading all its
    dylibs + getting through boost::program_options parsing — that's
    sufficient to prove no missing-library failures.

    Success criterion: NOT SIGKILL (rc=137 → Gatekeeper killed the
    process because codesignature was invalidated by install_name_tool
    and the ad-hoc resign step didn't run). Any other exit — including
    a non-zero from "missing required arg" — proves the binary loaded
    + ran + parsed args without dyld errors.
    """
    binary = app_path / "Contents" / "Resources" / "alicevision" / name
    if not binary.is_file():
        pytest.skip(f"{name} not in bundle")

    # Clean env: nothing from the test runner's shell leaks through.
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "DYLD_LIBRARY_PATH": "",
        "DYLD_FALLBACK_LIBRARY_PATH": "",
    }

    out = subprocess.run(
        [str(binary), "--help"],
        env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode != 137, (
        f"{name} was SIGKILL'd — codesignature invalid. The ad-hoc "
        f"resign step in package_macos_app.sh didn't run, OR a later "
        f"step modified the binary without re-signing.\n"
        f"stderr tail: {out.stderr[-1500:]}"
    )
    # dyld errors land on stderr with specific prefixes; failing to load
    # a dylib produces a very recognizable banner. Detect + report it
    # specifically so users know the bundler missed a dep (the exact
    # bug the Phase 7 mini-dylibbundler was supposed to prevent).
    dyld_errors = [
        line for line in out.stderr.splitlines()
        if "dyld[" in line and "Library not loaded" in line
    ]
    assert not dyld_errors, (
        f"{name} reports missing dylibs at launch — the bundler missed "
        f"these references. First errors:\n" + "\n".join(dyld_errors[:5])
    )


# --------------------------------------------------------------------------- #
# (d) Bundled venv works
# --------------------------------------------------------------------------- #

@_gate
def test_bundled_venv_python_works(app_path):
    py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python3"
    if not py.is_file():
        py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python"
    assert py.exists(), f"no python in bundled venv at {py}"

    out = subprocess.run(
        [str(py), "-c", "import sys; print(sys.version_info[:2])"],
        capture_output=True, text=True, timeout=10,
    )
    assert out.returncode == 0, f"bundled venv python broken: {out.stderr}"
    assert "(3," in out.stdout, f"unexpected python version output: {out.stdout!r}"


@_gate
def test_bundled_venv_can_import_pyside6(app_path):
    py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python3"
    if not py.is_file():
        py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python"

    out = subprocess.run(
        [str(py), "-c", "import PySide6; print(PySide6.__version__)"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 0, (
        f"PySide6 import failed in bundled venv: {out.stderr}"
    )
    assert "6." in out.stdout, f"unexpected PySide6 version: {out.stdout!r}"


@_gate
def test_bundled_venv_can_import_coremltools(app_path):
    """coremltools is the AI-segmentation backbone; without it the
    SegmentationBiRefNet node is broken."""
    py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python3"
    if not py.is_file():
        py = app_path / "Contents" / "Resources" / "meshroom-venv" / "bin" / "python"

    out = subprocess.run(
        [str(py), "-c", "import coremltools; print(coremltools.__version__)"],
        capture_output=True, text=True, timeout=20,
    )
    assert out.returncode == 0, (
        f"coremltools import failed in bundled venv: {out.stderr}"
    )
