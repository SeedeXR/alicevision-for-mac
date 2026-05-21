"""
node_run — minimal in-process Meshroom node executor.

Called by the native SwiftUI Meshroom (via `scripts/run_python_node.sh`)
to invoke pure-Python nodes that don't have a CLI binary counterpart,
e.g. SegmentationBiRefNet.

Contract:
    python -m meshroom.bin.node_run --nodeType <T> [--<attr> <value> ...]

The runner imports `meshroom.nodes.aliceVision.<T>`, instantiates the
node class, populates attribute values from argv, synthesises a chunk
object, and dispatches `processChunk(chunk)`.

This is intentionally NOT a re-implementation of `meshroom_compute` —
that one operates on `.mg` graph files and walks the whole pipeline.
This runs ONE node, in-process, with values provided on the command
line — the shape the Swift GraphExecutor produces.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any


# ----- duck-typed chunk + node attribute shims ---------------------- #


class _AttrShim:
    """Stands in for `desc.File` / `desc.ChoiceParam` etc. instances.

    Only carries `.value`; nodes read `.value` exclusively.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _NodeShim:
    """Stands in for the `Node` instance Meshroom would normally pass.

    Each attribute name from the node's `inputs` / `outputs` list is
    accessible as `shim.<name>` and returns an `_AttrShim` with `.value`.
    """

    def __init__(self, attr_values: dict[str, Any]) -> None:
        for name, value in attr_values.items():
            object.__setattr__(self, name, _AttrShim(value))


class _LogManagerShim:
    """Subset of `meshroom.core.node.LogManager` used by node bodies.

    SegmentationBiRefNet only calls `start(level)` and `end()`.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def start(self, level: str) -> None:
        try:
            self._logger.setLevel(logging.getLevelName(str(level).upper()))
        except (TypeError, ValueError):
            self._logger.setLevel(logging.INFO)

    def end(self) -> None:
        for h in self._logger.handlers[:]:
            try:
                h.flush()
            except Exception:
                pass


class _ChunkShim:
    """Stands in for `meshroom.core.node.NodeChunk`."""

    def __init__(self, node: _NodeShim, logger: logging.Logger) -> None:
        self.node = node
        self.logger = logger
        self.logManager = _LogManagerShim(logger)


# ----- argv parsing ------------------------------------------------- #


def _split_argv(argv: list[str]) -> tuple[str, dict[str, str]]:
    """Pick out --nodeType then flatten remaining --key value pairs.

    Bool-ish values (`true`/`false`/`1`/`0`) and empty strings are
    preserved as strings; the node bodies do their own coercion via
    `int(...)` / `bool(...)` as needed (see SegmentationBiRefNet).
    """

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--nodeType", required=True)
    known, rest = ap.parse_known_args(argv)

    pairs: dict[str, str] = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            pairs[key] = rest[i + 1]
            i += 2
        else:
            pairs[key] = "true"
            i += 1
    return known.nodeType, pairs


def _coerce(raw: str) -> Any:
    """Coerce stringy argv values to the types nodes expect."""
    if raw is None:
        return None
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    return raw


# ----- main --------------------------------------------------------- #


def _bootstrap_path() -> Path:
    """Ensure `meshroom-mac/`, `meshroom-mac/nodes/`, and the python
    shim are importable. The aliceVision plugin tree lives at
    `meshroom-mac/nodes/aliceVision/` (discovered via MESHROOM_NODES_PATH
    in normal Meshroom runs); we put the parent on sys.path so plain
    `import aliceVision.<NodeType>` works.
    """
    here = Path(__file__).resolve()
    meshroom_mac = here.parent.parent.parent
    repo_root = meshroom_mac.parent
    candidates = (
        meshroom_mac,
        meshroom_mac / "nodes",
        repo_root / "src" / "python_shim",
    )
    for entry in candidates:
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo_root


def _import_node(node_type: str):
    """Import the node class.

    aliceVision/ nodes live under `meshroom-mac/nodes/aliceVision/`,
    which is added to sys.path by `_bootstrap_path()`. We try the
    aliceVision plugin namespace first; fall back to Meshroom's bundled
    `meshroom.nodes.aliceVision` namespace for cross-compat.
    """
    last_err: Exception | None = None
    for modname in (
        f"aliceVision.{node_type}",
        f"meshroom.nodes.aliceVision.{node_type}",
    ):
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, node_type, None)
            if cls is not None:
                return cls
            last_err = AttributeError(
                f"Module {modname} has no class named {node_type}"
            )
        except ImportError as exc:
            last_err = exc
    raise RuntimeError(
        f"Cannot import node {node_type}: {last_err}"
    )


def _make_logger(node_type: str) -> logging.Logger:
    log = logging.getLogger(node_type)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


def main(argv: list[str] | None = None) -> int:
    repo_root = _bootstrap_path()
    os.environ.setdefault("U2NET_HOME", str(repo_root / "ai-models"))

    node_type, raw_pairs = _split_argv(argv if argv is not None else sys.argv[1:])
    logger = _make_logger(node_type)

    node_cls = _import_node(node_type)
    attr_values = {k: _coerce(v) for k, v in raw_pairs.items()}

    # Ensure 'output' resolves to a concrete directory; the Swift
    # GraphExecutor always supplies --output <dir>, so this is just a
    # safety net for ad-hoc CLI invocations.
    if "output" not in attr_values:
        attr_values["output"] = str(repo_root / "tmp" / node_type)

    chunk = _ChunkShim(_NodeShim(attr_values), logger)
    node = node_cls()
    node.processChunk(chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
