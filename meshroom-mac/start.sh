#!/bin/bash
# Portable realpath: macOS BSD readlink lacks -f. python3 is required
# by the next line anyway, so we reuse it here. Matches `readlink -f`
# semantics on Linux (resolves symlinks in every path component).
export MESHROOM_ROOT="$(dirname "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${BASH_SOURCE[0]}")")"
export PYTHONPATH=$MESHROOM_ROOT:$PYTHONPATH

# using existing alicevision release
#export LD_LIBRARY_PATH=/foo/Meshroom-2023.2.0/aliceVision/lib/
#export PATH=$PATH:/foo/Meshroom-2023.2.0/aliceVision/bin/

# using alicevision built source
#export PATH=$PATH:/foo/build/Linux-x86_64/

python3 "$MESHROOM_ROOT/meshroom/ui"
