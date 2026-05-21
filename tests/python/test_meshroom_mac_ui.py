"""Track A — meshroom-mac Qt UI smoke + regression tests.

What this guards against (per memory/strategy_s54_unification.md §1):
  • A1 — ThreadPool semaphore leak: on SIGINT (Ctrl-C), the Qt
    aboutToQuit chain must run and terminate the three multiprocessing
    ThreadPool instances explicitly. Failure ⇒ resource_tracker warning.
  • A2 — "QQmlComponent: Component is not ready" + "Type Viewer3D
    unavailable" cascade. Caused by missing Qt3DQuickScene3D.framework
    in PySide6 < 6.11; fixed by upgrading to PySide6 6.11.1.
  • A3 — start.sh must source the same env as scripts/run_meshroom.sh,
    or every aliceVision node registers as UnknownNodeType.
  • A4 — QSettings org/app name must be ('AliceVision', 'Meshroom') so
    recent projects persist to ~/Library/Preferences/org.alicevision.*.

These are end-to-end smoke tests: they launch the real start.sh in a
subprocess, send a real SIGINT after a short delay, and inspect the
captured stdout/stderr. They are slow (~15 s each) and need a graphics
context — gated behind `RUN_MESHROOM_UI=1` so they don't break CI on
headless boxes by default.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
START_SH = REPO_ROOT / "meshroom-mac" / "start.sh"
SIGINT_LAUNCH = REPO_ROOT / "scripts" / "sigint_launch.py"


requires_ui = pytest.mark.skipif(
    os.environ.get("RUN_MESHROOM_UI") != "1",
    reason="Set RUN_MESHROOM_UI=1 to run interactive Meshroom UI smoke tests",
)


@pytest.fixture(scope="module")
def ui_output() -> str:
    """Launch start.sh, send SIGINT after 12 s, return combined stdout+stderr."""
    assert START_SH.exists(), f"missing {START_SH}"
    assert SIGINT_LAUNCH.exists(), f"missing {SIGINT_LAUNCH}"
    env = os.environ.copy()
    env["MESHROOM_OUTPUT_QML_WARNINGS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(SIGINT_LAUNCH), "12", str(START_SH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return proc.stdout + proc.stderr


@requires_ui
def test_a1_no_semaphore_leak(ui_output: str) -> None:
    """A1: clean SIGINT must produce no multiprocessing.resource_tracker warning."""
    assert "resource_tracker" not in ui_output, (
        "ThreadPool cleanup regressed; resource_tracker warning present:\n"
        + ui_output
    )
    assert "leaked semaphore" not in ui_output


@requires_ui
def test_a2_no_component_not_ready(ui_output: str) -> None:
    """A2: QML must not emit Component-is-not-ready or Viewer3D-unavailable."""
    assert "QQmlComponent: Component is not ready" not in ui_output, ui_output
    assert "Type WorkspaceView unavailable" not in ui_output, ui_output
    assert "Type Viewer3D unavailable" not in ui_output, ui_output
    assert "Qt3DQuickScene3D" not in ui_output, ui_output  # framework load error


@requires_ui
def test_a3_core_photogrammetry_nodes_register(ui_output: str) -> None:
    """A3: the 11 photogrammetry-pipeline node types we ship binaries for
    MUST register without UnknownNodeType warnings. Other upstream nodes
    (segmentation/Sam3, scene preview, detection prompt) may be unknown
    since we deliberately don't ship those binaries (see segmentation_pipeline.md).
    """
    core_nodes = (
        "CameraInit",
        "FeatureExtraction",
        "ImageMatching",
        "FeatureMatching",
        "StructureFromMotion",
        "PrepareDenseScene",
        "DepthMap",
        "DepthMapFilter",
        "Meshing",
        "MeshFiltering",
        "Texturing",
    )
    failures = []
    for ntype in core_nodes:
        # A failure looks like:
        # "WARNING:root:Compatibility issue detected for node 'CameraInit_1': UnknownNodeType"
        if re.search(rf"Compatibility issue.*'{ntype}_\d+': UnknownNodeType", ui_output):
            failures.append(ntype)
    assert not failures, (
        f"core photogrammetry node(s) failed to load: {failures}\n"
        f"(env is wrong, MESHROOM_NODES_PATH not set correctly)"
    )


@requires_ui
def test_a_no_graph_load_crash(ui_output: str) -> None:
    """No AttributeError on graph/template load (nodeDesc=None guard)."""
    assert "'NoneType' object has no attribute 'hasPreprocess'" not in ui_output, ui_output
    assert "'NoneType' object has no attribute 'hasPostprocess'" not in ui_output, ui_output


@requires_ui
def test_a3_extractmetadata_loads(ui_output: str) -> None:
    """A3 follow-up: ExtractMetadata node must load (distutils removal handled)."""
    assert "ModuleNotFoundError: No module named 'distutils'" not in ui_output


# ---- Pure-unit (non-graphical) checks below — always run, no RUN_MESHROOM_UI gate.

def test_a4_qsettings_identity_set_in_source() -> None:
    """A4: source check that setOrganizationName/setApplicationName are called."""
    app_py = (REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "app.py").read_text()
    assert "setOrganizationName('AliceVision')" in app_py, "org name missing"
    assert "setApplicationName('Meshroom')" in app_py, "app name missing"


def test_start_sh_env_unified() -> None:
    """A3 source check: start.sh must export the same env keys as run_meshroom.sh."""
    start = START_SH.read_text()
    required = (
        "ALICEVISION_ROOT",
        "MESHROOM_NODES_PATH",
        "MESHROOM_PIPELINE_TEMPLATES_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "PYTHONPATH",
    )
    for key in required:
        assert key in start, f"start.sh missing required env export: {key}"


def test_sigint_handler_routed_to_qt() -> None:
    """A1 source check: SIGINT must call QApplication.quit, not SIG_DFL.

    We inspect the actual assignment, not the substring, so explanatory
    comments mentioning 'SIG_DFL' don't trip the check.
    """
    main = (REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "__main__.py").read_text()
    assert not re.search(r"signal\.signal\(\s*signal\.SIGINT\s*,\s*signal\.SIG_DFL", main), (
        "SIG_DFL bypasses aboutToQuit; will leak pools on Ctrl-C"
    )
    assert "uiInstance.quit()" in main, "SIGINT handler must trigger Qt quit"


@requires_ui
def test_no_segfault_in_render_loop(ui_output: str) -> None:
    """Apple's OpenGL drivers on macOS 26.5 crash inside glDrawElements_*
    during QRhi::endFrame. Forcing Metal RHI via QQuickWindow.setGraphicsApi
    keeps the render loop alive. Regression test that the fix stays in place.
    """
    assert "Fatal Python error" not in ui_output, ui_output
    assert "EXC_BAD_ACCESS" not in ui_output, ui_output
    assert "glDrawElements" not in ui_output, ui_output  # OpenGL path should not be exercised


def test_metal_rhi_forced_in_source() -> None:
    """A2/render-loop source check: app.py must call setGraphicsApi(Metal)
    before QApplication is constructed. QSG_RHI_BACKEND env var is silently
    ignored on Qt 6.11.1 / macOS 26.5; the API call is the only way to
    avoid the broken OpenGL backend.
    """
    app_py = (REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "app.py").read_text()
    assert "QQuickWindow.setGraphicsApi" in app_py, (
        "QQuickWindow.setGraphicsApi(Metal) must be called before QApplication "
        "to avoid OpenGL-backend segfault on macOS 26.5"
    )
    assert "GraphicsApi.Metal" in app_py


def test_viewer3d_loader_doubly_gated() -> None:
    """The Viewer3D Loader's `active` property must be AND'd with the
    `_viewer3DAvailable` context property. Otherwise, QML `Settings`
    persistence remembers a user's previous `showViewer3D=true` and
    overrides our checked: false default — re-triggering the crash.
    """
    workspace_qml = (
        REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "qml" / "WorkspaceView.qml"
    ).read_text()
    app_py = (REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "app.py").read_text()
    assert "_viewer3DAvailable" in app_py, (
        "app.py must expose _viewer3DAvailable context property to QML"
    )
    assert "_viewer3DAvailable && settingsUILayout.showViewer3D" in workspace_qml, (
        "panel3dViewerLoader.active must AND _viewer3DAvailable with the menu toggle"
    )


def test_viewer3d_default_off_in_source() -> None:
    """Viewer3D forces Qt3D's Scene3D embedding which can't share a Metal
    context with QtQuick. Default the menu toggle to off so users get a
    working UI; the native Metal viewer arrives in Track B B4.
    """
    application_qml = (
        REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "qml" / "Application.qml"
    ).read_text()
    # The viewer3DVisibilityCB MenuItem block must contain `checked: false`.
    # Search a 200-char window around the id declaration.
    import re as _re
    block = _re.search(r"id: viewer3DVisibilityCB.{0,1200}", application_qml, _re.DOTALL)
    assert block is not None, "viewer3DVisibilityCB MenuItem not found"
    assert "checked: false" in block.group(0), (
        "viewer3DVisibilityCB must default to checked: false until a native Metal "
        "3D viewer replaces Qt3D's Scene3D (which crashes on macOS 26.5)"
    )


def test_threadpoolexecutor_in_source() -> None:
    """A1 source check: pools must use concurrent.futures.ThreadPoolExecutor.

    The original multiprocessing.pool.ThreadPool leaked POSIX semaphores
    (its SimpleQueue+Lock pair stayed registered with resource_tracker
    even after terminate()/join() because Qt's QObject parent hierarchy
    delayed the pool's garbage collection past process exit). The fix is
    to use concurrent.futures.ThreadPoolExecutor, which doesn't create
    multiprocessing semaphores at all.
    """
    files = (
        REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "graph.py",
        REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "scene.py",
        REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "components" / "thumbnail.py",
    )
    for f in files:
        text = f.read_text()
        assert "from multiprocessing.pool import ThreadPool" not in text, (
            f"{f.name}: multiprocessing.pool.ThreadPool leaks semaphores under Qt; "
            f"use concurrent.futures.ThreadPoolExecutor instead"
        )
        assert "ThreadPoolExecutor" in text, f"{f.name} must use ThreadPoolExecutor"
    scene_py = (REPO_ROOT / "meshroom-mac" / "meshroom" / "ui" / "scene.py").read_text()
    assert "def stopChildThreads" in scene_py, "Scene must override stopChildThreads"
