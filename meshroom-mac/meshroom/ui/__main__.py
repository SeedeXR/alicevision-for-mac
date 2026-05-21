import faulthandler
import os
import signal
import sys

# Dump a C-level stack trace on SIGSEGV/SIGABRT/SIGBUS so we can see what
# the native Qt code was doing at the time of the crash. Costs nothing
# at runtime and is invaluable when QML widgets crash inside C++.
faulthandler.enable(file=sys.stderr, all_threads=True)

# DEBUG: probe semaphore registrations (enabled via MESHROOM_TRACE_SEMAPHORES=1).
# Kept in tree to make future leak regressions easy to bisect.
if os.environ.get("MESHROOM_TRACE_SEMAPHORES") == "1":
    import multiprocessing.resource_tracker as _rt
    import traceback as _tb
    _orig_register = _rt.register
    def _patched_reg(name, rtype):
        if rtype == "semaphore":
            sys.stderr.write(f"\n[SEM-REG {name}]\n")
            sys.stderr.write("".join(_tb.format_stack(limit=20)))
        return _orig_register(name, rtype)
    _rt.register = _patched_reg
    sys.stderr.write(f"[probe] resource_tracker.register patched pid={os.getpid()}\n")

import meshroom
from meshroom.common import Backend

meshroom.setupEnvironment(backend=Backend.PYSIDE)

import meshroom.ui
import meshroom.ui.app


meshroom.ui.uiInstance = meshroom.ui.app.MeshroomApp(sys.argv)
meshroom.ui.uiInstance.aboutToQuit.connect(meshroom.ui.uiInstance.terminateManual)

# Route SIGINT through Qt's quit machinery so aboutToQuit fires and the
# ThreadPool cleanup in terminateManual / stopChildThreads runs. The
# previous signal.SIG_DFL caused the process to die at the C level on
# Ctrl-C, bypassing all Qt teardown.
signal.signal(signal.SIGINT, lambda *_: meshroom.ui.uiInstance.quit())

meshroom.ui.uiInstance.exec()
