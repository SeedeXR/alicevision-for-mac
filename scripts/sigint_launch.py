#!/usr/bin/env python3
"""Launch a subprocess, send SIGINT after <delay> seconds, capture output.

Used by the Track A semaphore-leak smoke test: SIGALRM (perl -e 'alarm')
bypasses Qt's aboutToQuit, so we must send SIGINT explicitly to exercise
the new signal handler in meshroom-mac/meshroom/ui/__main__.py.

Usage:  sigint_launch.py <delay_seconds> <cmd> [args...]
Prints the combined stdout/stderr of the child, then exits with the child's code.
"""
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    delay = float(sys.argv[1])
    argv = sys.argv[2:]
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # New process group so SIGINT reaches the whole tree.
        start_new_session=True,
    )
    # Sleep on the parent side; let the child run.
    time.sleep(delay)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        pass  # child already exited
    try:
        out, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    sys.stdout.write(out.decode(errors="replace"))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
