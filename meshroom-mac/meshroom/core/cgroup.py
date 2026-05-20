#!/usr/bin/env python

import os
import subprocess
import sys


def _darwinSysctlInt(name):
    """
    Query a sysctl key on Darwin and return its integer value, or -1 if
    the call fails. Used to surface the true hardware memory / CPU count
    on macOS where cgroup files do not exist.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", name], stderr=subprocess.DEVNULL, timeout=2
        ).decode("utf-8", errors="replace").strip()
        return int(out) if out.lstrip("-").isdigit() else -1
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


# Try to retrieve limits of memory for the current process' cgroup
def getCgroupMemorySize():

    if sys.platform == "darwin":
        # No cgroups on macOS; report the physical memory budget instead.
        return _darwinSysctlInt("hw.memsize")

    # First of all, get pid of process
    pid = os.getpid()

    # Get cgroup associated with pid
    filename = f"/proc/{pid}/cgroup"

    cgroup = None
    try:
        with open(filename) as f:

            # cgroup file is a ':' separated table
            # lookup a line where the second field is "memory"
            lines = f.readlines()
            for line in lines:
                tokens = line.rstrip("\r\n").split(":")
                if len(tokens) < 3:
                    continue
                if tokens[1] == "memory":
                    cgroup = tokens[2]
    except OSError:
        pass

    if cgroup is None:
        return -1

    size = -1
    filename = f"/sys/fs/cgroup/memory/{cgroup}/memory.limit_in_bytes"
    try:
        with open(filename) as f:
            value = f.read().rstrip("\r\n")
            if value.isnumeric():
                size = int(value)
    except OSError:
        pass

    return size


def parseNumericList(numericListString):

    nList = []
    for item in numericListString.split(','):
        if '-' in item:
            start, end = item.split('-')
            start = int(start)
            end = int(end)
            nList.extend(range(start, end + 1))
        else:
            value = int(item)
            nList.append(value)

    return nList


# Try to retrieve limits of cores for the current process' cgroup
def getCgroupCpuCount():

    if sys.platform == "darwin":
        # No cgroups on macOS; report the logical CPU count via sysctl.
        # hw.ncpu = hw.logicalcpu on Apple Silicon (perf + efficiency cores).
        return _darwinSysctlInt("hw.ncpu")

    # First of all, get pid of process
    pid = os.getpid()

    # Get cgroup associated with pid
    filename = f"/proc/{pid}/cgroup"

    cgroup = None
    try:
        with open(filename) as f:

            # cgroup file is a ':' separated table
            # lookup a line where the second field is "memory"
            lines = f.readlines()
            for line in lines:
                tokens = line.rstrip("\r\n").split(":")
                if len(tokens) < 3:
                    continue
                if tokens[1] == "cpuset":
                    cgroup = tokens[2]
    except OSError:
        pass

    if cgroup is None:
        return -1

    size = -1
    filename = f"/sys/fs/cgroup/cpuset/{cgroup}/cpuset.cpus"
    try:
        with open(filename) as f:
            value = f.read().rstrip("\r\n")
            nlist = parseNumericList(value)
            size = len(nlist)

    except OSError:
        pass

    return size
