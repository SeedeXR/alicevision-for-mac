#!/usr/bin/env python3
"""
Quick textured-OBJ vertex/face counter + sibling texture EXR resolution probe.

Usage:
  scripts/obj_stats.py path/to/texturedMesh.obj [...more.obj]
"""
import sys, os, subprocess

def stats(path):
    v = f = vt = vn = 0
    with open(path, "r", errors="replace") as fp:
        for line in fp:
            if   line.startswith("v "):  v  += 1
            elif line.startswith("f "):  f  += 1
            elif line.startswith("vt "): vt += 1
            elif line.startswith("vn "): vn += 1
    return v, f, vt, vn

def main():
    for p in sys.argv[1:]:
        v, f, vt, vn = stats(p)
        size = os.path.getsize(p)
        print(f"{p}")
        print(f"  vertices : {v:,}")
        print(f"  faces    : {f:,}")
        print(f"  uvs      : {vt:,}")
        print(f"  normals  : {vn:,}")
        print(f"  size     : {size/1024:.1f} KiB")
        tdir = os.path.dirname(p)
        for fn in sorted(os.listdir(tdir)):
            if fn.startswith("texture_") and fn.endswith(".exr"):
                tp = os.path.join(tdir, fn)
                try:
                    out = subprocess.check_output(["oiiotool", "--info", tp],
                                                  stderr=subprocess.STDOUT,
                                                  text=True)
                    print(f"  texture  : {fn}  ->  {out.strip().splitlines()[0]}")
                except Exception as e:
                    print(f"  texture  : {fn}  (oiiotool failed: {e})")

if __name__ == "__main__":
    main()
