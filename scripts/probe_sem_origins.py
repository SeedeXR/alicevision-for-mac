"""Monkey-patch multiprocessing.resource_tracker.register so we log a stack
trace every time a semaphore is registered. Reveals what's actually creating
the 6 leaked semaphores in meshroom-mac.

Usage:  PYTHONSTARTUP=scripts/probe_sem_origins.py meshroom-mac/start.sh
"""
import multiprocessing.resource_tracker as rt
import traceback
import sys

_orig_register = rt.register

def _patched(name, rtype, *args, **kw):
    if rtype == "semaphore":
        sys.stderr.write(f"[SEM REG] {name} rtype={rtype}\n")
        sys.stderr.write("".join(traceback.format_stack(limit=8)))
        sys.stderr.write("[/SEM REG]\n")
    return _orig_register(name, rtype, *args, **kw)

rt.register = _patched
sys.stderr.write(f"[probe] resource_tracker.register patched (pid={__import__('os').getpid()})\n")
