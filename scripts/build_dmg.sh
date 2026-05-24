#!/usr/bin/env bash
#
# scripts/build_dmg.sh — ONE-SHOT pipeline: source → DMG.
#
# Orchestrates the entire build-to-DMG flow that previously needed four
# separate invocations:
#
#   1. cmake configure         (skipped if build/ already configured)
#   2. cmake build             (60 aliceVision_* binaries + 3 av_* + 1 av_roma sublibs)
#   3. scripts/package_macos_app.sh  → build/release/Meshroom.app
#   4. scripts/codesign_macos_app.sh → re-sign (ad-hoc by default)
#   5. scripts/make_dmg.sh           → build/release/Meshroom-<v>-arm64.dmg
#
# Every step is logged to build/release/logs/<step>.log AND streamed live.
# A final SUMMARY.md is written next to the DMG with timings, file sizes,
# and a copy of the resolved CLI flags so the build is reproducible.
#
# Usage:
#   scripts/build_dmg.sh                                    # ad-hoc signed DMG
#   scripts/build_dmg.sh --identity "Developer ID Application: NAME (TEAM)"
#   scripts/build_dmg.sh --skip-cmake-configure             # cmake .. already done
#   scripts/build_dmg.sh --skip-cmake-build                 # binaries already built
#   scripts/build_dmg.sh --skip-package                     # .app already in build/release/
#   scripts/build_dmg.sh --skip-codesign                    # signed elsewhere
#   scripts/build_dmg.sh --clean                            # rm -rf build/release/Meshroom.app + DMG before starting
#   scripts/build_dmg.sh --jobs N                           # ninja -jN (default: ncpu)
#   scripts/build_dmg.sh --compression {udzo,udzo-max,ulfo,ulmo}
#                                                          # default: ulmo (34% smaller than udzo on our fixture)
#
# Exit code: 0 on success; non-zero if any step fails. Each step's log
# is named build/release/logs/<NN>_<step>.log and the last 30 lines of a
# failed step's log are echoed to stderr before exiting.
#
set -euo pipefail

# ---------- paths ----------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
RELEASE_DIR="$BUILD_DIR/release"
LOG_DIR="$RELEASE_DIR/logs"
APP="$RELEASE_DIR/Meshroom.app"

mkdir -p "$LOG_DIR"

# ---------- args ----------
IDENTITY="-"     # `-` = ad-hoc
SKIP_CMAKE_CONFIGURE=0
SKIP_CMAKE_BUILD=0
SKIP_PACKAGE=0
SKIP_CODESIGN=0
DO_CLEAN=0
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
COMPRESSION="ulmo"   # passed to make_dmg.sh; ulmo = best for distribution

while [ $# -gt 0 ]; do
    case "$1" in
        --identity)          IDENTITY="$2"; shift 2 ;;
        --skip-cmake-configure) SKIP_CMAKE_CONFIGURE=1; shift ;;
        --skip-cmake-build)  SKIP_CMAKE_BUILD=1; shift ;;
        --skip-package)      SKIP_PACKAGE=1; shift ;;
        --skip-codesign)     SKIP_CODESIGN=1; shift ;;
        --clean)             DO_CLEAN=1; shift ;;
        --jobs)              JOBS="$2"; shift 2 ;;
        --compression)       COMPRESSION="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set/p' "$0" | sed -n '/^#/p' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Validate compression up-front so we don't get to step 5 with a typo.
case "$COMPRESSION" in
    udzo|udzo-max|ulfo|ulmo) ;;
    *)
        echo "unknown --compression value: $COMPRESSION (use udzo, udzo-max, ulfo, ulmo)" >&2
        exit 2 ;;
esac

# ---------- logging helpers ----------
# Colors only when stdout is a tty.
if [ -t 1 ]; then
    BOLD="$(tput bold 2>/dev/null || echo)"
    DIM="$(tput dim 2>/dev/null || echo)"
    RED="$(tput setaf 1 2>/dev/null || echo)"
    GREEN="$(tput setaf 2 2>/dev/null || echo)"
    YELLOW="$(tput setaf 3 2>/dev/null || echo)"
    BLUE="$(tput setaf 4 2>/dev/null || echo)"
    RESET="$(tput sgr0 2>/dev/null || echo)"
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

now_epoch() { date +%s; }
fmt_dur()  { printf "%dm%02ds" $(( $1 / 60 )) $(( $1 % 60 )); }

log_info()  { printf "${BLUE}[%s] %s${RESET}\n" "$(date +%H:%M:%S)" "$*"; }
log_ok()    { printf "${GREEN}[%s] ✓ %s${RESET}\n" "$(date +%H:%M:%S)" "$*"; }
log_warn()  { printf "${YELLOW}[%s] ⚠ %s${RESET}\n" "$(date +%H:%M:%S)" "$*" >&2; }
log_err()   { printf "${RED}[%s] ✗ %s${RESET}\n" "$(date +%H:%M:%S)" "$*" >&2; }
log_header(){ printf "\n${BOLD}── %s ──${RESET}\n" "$*"; }

# run_step <number> <name> <log-suffix> -- <command...>
# Pipes both stdout and stderr to the log AND to the console.
# Records start/end epoch in STEP_<N>_T_START / STEP_<N>_T_DURATION.
run_step() {
    local num="$1"; shift
    local name="$1"; shift
    local logsuffix="$1"; shift
    [ "$1" = "--" ] && shift

    local logfile
    logfile="$LOG_DIR/$(printf '%02d' "$num")_${logsuffix}.log"

    log_header "Step $num: $name"
    log_info "log: $logfile"

    local t0
    t0=$(now_epoch)
    local rc=0

    # Stream + log via tee. set +e while the pipe runs so we can capture
    # the real rc (PIPESTATUS) without killing the script.
    set +e
    "$@" 2>&1 | tee "$logfile"
    rc=${PIPESTATUS[0]}
    set -e

    local dur=$(( $(now_epoch) - t0 ))
    eval "STEP_${num}_DURATION=$dur"
    eval "STEP_${num}_NAME=\"\$name\""
    eval "STEP_${num}_LOG=\"\$logfile\""

    if [ "$rc" -ne 0 ]; then
        log_err "Step $num ('$name') FAILED (exit $rc, $(fmt_dur "$dur"))."
        log_err "Last 30 lines of the log:"
        tail -n 30 "$logfile" >&2 || true
        exit "$rc"
    fi
    log_ok "Step $num done in $(fmt_dur "$dur")."
}

# ---------- preflight ----------
log_header "build_dmg.sh — alicevision-for-mac DMG orchestrator"
log_info "ROOT:        $ROOT"
log_info "BUILD_DIR:   $BUILD_DIR"
log_info "RELEASE_DIR: $RELEASE_DIR"
log_info "LOG_DIR:     $LOG_DIR"
log_info "IDENTITY:    $IDENTITY"
log_info "JOBS:        $JOBS"
log_info "COMPRESSION: $COMPRESSION"
log_info "ARCH:        $(uname -m)   (must be arm64 on Apple Silicon)"
log_info "macOS:       $(sw_vers -productVersion 2>/dev/null || echo unknown)"
log_info "CMake:       $(cmake --version 2>/dev/null | head -1 || echo MISSING)"
log_info "Ninja:       $(ninja --version 2>/dev/null || echo MISSING)"

# Hard preflight failures: missing toolchain.
command -v cmake >/dev/null || { log_err "cmake not on PATH — brew install cmake"; exit 2; }
command -v ninja >/dev/null || { log_err "ninja not on PATH — brew install ninja"; exit 2; }

# Soft preflight warnings — they should be fixed but the build can proceed.
if [ "$(uname -m)" != "arm64" ]; then
    log_warn "Not running on arm64 ($(uname -m)). The pipeline is Apple-Silicon-only; expect failures."
fi
[ -L "$ROOT/upstream" ] || log_warn "upstream/ symlink missing — cmake will fail unless you set it up."

# ---------- optional clean ----------
T_TOTAL_START=$(now_epoch)
if [ "$DO_CLEAN" -eq 1 ]; then
    log_header "Step 0: Clean"
    log_info "Removing $APP and any existing DMGs under $RELEASE_DIR/"
    rm -rf "$APP" "$RELEASE_DIR"/Meshroom-*.dmg 2>/dev/null || true
    log_ok "Cleaned."
fi

# ---------- step 1: cmake configure ----------
if [ "$SKIP_CMAKE_CONFIGURE" -eq 0 ]; then
    run_step 1 "cmake configure" "cmake_configure" -- \
        cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DAV_BUILD_UPSTREAM=ON \
            -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
            -DAV_BUILD_PYALICEVISION=ON
else
    log_header "Step 1: cmake configure (SKIPPED)"
    log_warn "--skip-cmake-configure passed; assuming $BUILD_DIR/CMakeCache.txt is current."
    [ -f "$BUILD_DIR/CMakeCache.txt" ] || { log_err "no CMakeCache.txt at $BUILD_DIR — drop --skip-cmake-configure"; exit 2; }
fi

# ---------- step 2: cmake build ----------
if [ "$SKIP_CMAKE_BUILD" -eq 0 ]; then
    # Build everything; ninja's dep graph handles incremental builds well.
    run_step 2 "cmake build (all targets)" "cmake_build" -- \
        cmake --build "$BUILD_DIR" -j "$JOBS"

    # Sanity: count built binaries.
    bins=$(ls "$BUILD_DIR"/aliceVision_* 2>/dev/null | wc -l | tr -d ' ')
    log_info "Built $bins aliceVision_* binaries (expected ~60)."
    if [ "$bins" -lt 50 ]; then
        log_err "Only $bins binaries built — something is wrong. Inspect $LOG_DIR/02_cmake_build.log"
        exit 1
    fi
else
    log_header "Step 2: cmake build (SKIPPED)"
fi

# ---------- step 3: package_macos_app.sh ----------
if [ "$SKIP_PACKAGE" -eq 0 ]; then
    run_step 3 "Package Meshroom.app (dylibbundler + ad-hoc resign)" "package" -- \
        bash "$ROOT/scripts/package_macos_app.sh"

    [ -d "$APP" ] || { log_err "package_macos_app.sh finished but $APP missing"; exit 1; }
    # Bundle sanity-check: no leftover /opt/homebrew references.
    leaks=$(find "$APP/Contents/MacOS" "$APP/Contents/Resources/lib" -type f \( -name 'aliceVision_*' -o -name '*.dylib' \) 2>/dev/null \
            | xargs -I {} sh -c 'otool -L "{}" 2>/dev/null' \
            | grep -c '/opt/homebrew/' || true)
    if [ "$leaks" -gt 0 ]; then
        log_warn "Found $leaks /opt/homebrew/ references in bundled Mach-Os — bundler missed some dylibs."
        log_warn "Bundle may still run on dev machines (Homebrew available) but not on clean Macs."
    else
        log_ok "Bundle is self-contained — zero /opt/homebrew/ references."
    fi
else
    log_header "Step 3: Package (SKIPPED)"
    [ -d "$APP" ] || { log_err "--skip-package passed but $APP doesn't exist"; exit 2; }
fi

# ---------- step 4: codesign ----------
if [ "$SKIP_CODESIGN" -eq 0 ]; then
    run_step 4 "Codesign Meshroom.app (identity: $IDENTITY)" "codesign" -- \
        bash "$ROOT/scripts/codesign_macos_app.sh" --app "$APP" --identity "$IDENTITY"

    # Verify signature.
    if codesign --verify --deep --strict "$APP" >/dev/null 2>&1; then
        log_ok "codesign --verify --deep --strict: OK"
    else
        log_warn "codesign --verify failed — DMG will still build but may not pass Gatekeeper."
    fi
else
    log_header "Step 4: Codesign (SKIPPED)"
fi

# ---------- step 5: make_dmg ----------
run_step 5 "Make DMG (compression: $COMPRESSION)" "make_dmg" -- \
    bash "$ROOT/scripts/make_dmg.sh" --app "$APP" --compression "$COMPRESSION"

# Find the produced DMG (make_dmg writes the path on its own).
DMG="$(ls -t "$RELEASE_DIR"/Meshroom-*-arm64.dmg 2>/dev/null | head -1 || true)"
[ -z "$DMG" ] && { log_err "make_dmg.sh succeeded but no Meshroom-*-arm64.dmg in $RELEASE_DIR"; exit 1; }

# ---------- summary ----------
T_TOTAL=$(( $(now_epoch) - T_TOTAL_START ))
APP_SIZE=$(du -sh "$APP" 2>/dev/null | awk '{print $1}')
DMG_SIZE=$(du -sh "$DMG" 2>/dev/null | awk '{print $1}')
BIN_COUNT=$(ls "$BUILD_DIR"/aliceVision_* 2>/dev/null | wc -l | tr -d ' ')
DYLIB_COUNT=$(ls "$APP/Contents/Resources/lib"/*.dylib 2>/dev/null | wc -l | tr -d ' ')

SUMMARY="$RELEASE_DIR/SUMMARY.md"
{
    echo "# alicevision-for-mac DMG build summary"
    echo
    echo "**Built**: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Total wall-clock**: $(fmt_dur "$T_TOTAL")"
    echo
    echo "## Inputs"
    echo
    echo "- ARCH: \`$(uname -m)\` on macOS \`$(sw_vers -productVersion 2>/dev/null || echo unknown)\`"
    echo "- CMake: \`$(cmake --version 2>/dev/null | head -1)\`"
    echo "- Ninja: \`$(ninja --version 2>/dev/null)\`"
    echo "- Codesign identity: \`$IDENTITY\` (\`-\` = ad-hoc)"
    echo "- Compression: \`$COMPRESSION\` (see scripts/make_dmg.sh -h for tradeoffs)"
    echo "- Jobs: \`$JOBS\`"
    echo
    echo "## Outputs"
    echo
    echo "| Artifact | Path | Size |"
    echo "|---|---|---|"
    echo "| Meshroom.app | \`$APP\` | $APP_SIZE |"
    echo "| DMG | \`$DMG\` | $DMG_SIZE |"
    echo
    echo "## Counts"
    echo
    echo "- aliceVision_* binaries: **$BIN_COUNT**"
    echo "- Bundled dylibs: **$DYLIB_COUNT**"
    echo "- /opt/homebrew/ references in bundle: $leaks"
    echo
    echo "## Step durations"
    echo
    echo "| # | Step | Duration |"
    echo "|---|---|---|"
    for i in 1 2 3 4 5; do
        dur_var="STEP_${i}_DURATION"
        name_var="STEP_${i}_NAME"
        if [ -n "${!dur_var:-}" ]; then
            echo "| $i | ${!name_var} | $(fmt_dur "${!dur_var}") |"
        else
            echo "| $i | (skipped) | — |"
        fi
    done
    echo
    echo "## Logs"
    echo
    for f in "$LOG_DIR"/*.log; do
        echo "- \`${f#$ROOT/}\`"
    done
    echo
    echo "## Next steps"
    echo
    if [ "$IDENTITY" = "-" ]; then
        echo "- DMG is **ad-hoc signed** — Gatekeeper will warn on first launch."
        echo "  Re-run with \`--identity \"Developer ID Application: NAME (TEAMID)\"\` for production."
    else
        echo "- DMG signed with **\`$IDENTITY\`**."
        echo "- Upload to Apple for notarization:"
        echo
        echo "  \`\`\`bash"
        echo "  xcrun notarytool submit \"$DMG\" \\"
        echo "      --apple-id you@example.com --team-id TEAMID \\"
        echo "      --password APP-SPECIFIC-PWD --wait"
        echo "  xcrun stapler staple \"$DMG\""
        echo "  \`\`\`"
    fi
} > "$SUMMARY"

# ---------- console final ----------
log_header "Done."
log_ok "Meshroom.app: $APP   ($APP_SIZE)"
log_ok "DMG:          $DMG   ($DMG_SIZE)"
log_ok "Summary:      $SUMMARY"
log_ok "Total time:   $(fmt_dur "$T_TOTAL")"

cat <<EOF

${BOLD}Quick install test:${RESET}
    open "$DMG"

${BOLD}Reproduce this build:${RESET}
    cat "$SUMMARY"

EOF
