#!/usr/bin/env bash
#
# scripts/codesign_macos_app.sh — sign Meshroom.app for distribution.
#
# Two modes:
#
#   1. AD-HOC (default, free)
#      Suitable for in-house testing + dev distribution. macOS Gatekeeper
#      will block the .app on first launch ("from an unidentified
#      developer"); users right-click → Open to bypass.
#
#   2. Apple Developer ID (--identity "Developer ID Application: NAME (TEAMID)")
#      Suitable for end-user distribution. Requires an active Apple
#      Developer Program membership ($99/yr) and an installed
#      `Developer ID Application` certificate (Xcode → Settings →
#      Accounts → Manage Certificates).
#
# After signing with Developer ID, run `notarytool submit` to upload to
# Apple for malware-scan + stapling (see comments at end of file).
#
# Usage:
#   scripts/codesign_macos_app.sh                                       # ad-hoc
#   scripts/codesign_macos_app.sh --identity "Developer ID Application: …"
#   scripts/codesign_macos_app.sh [--app /path/to/Meshroom.app] ...
#
set -euo pipefail

# --------- argument parse ---------
APP=""
IDENTITY="-"     # `-` = ad-hoc

while [ $# -gt 0 ]; do
    case "$1" in
        --identity)
            IDENTITY="$2"; shift 2 ;;
        --app)
            APP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set/p' "$0" | sed -n '/^# /p' | sed 's/^# //'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -z "$APP" ] && APP="$ROOT/build/release/Meshroom.app"
[ -d "$APP" ] || { echo "no .app at $APP — run scripts/package_macos_app.sh first" >&2; exit 2; }

# --------- entitlements (required for hardened runtime + Metal/JIT) ---------
ENTITLEMENTS_PLIST="$(mktemp -t mr_entitlements).plist"
trap 'rm -f "$ENTITLEMENTS_PLIST"' EXIT

cat > "$ENTITLEMENTS_PLIST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- PySide6's Python interpreter ships with embedded bytecode that
         needs JIT-style allow-unsigned-executable-memory permissions
         under hardened runtime. -->
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <!-- The pipeline writes intermediate caches under ~/Library/Caches
         and ~/Documents/Meshroom; standard user-folder access is fine.
         For sandboxed distribution we'd need finer entitlements. -->
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
PLIST

# --------- sign order matters: inner Mach-Os FIRST, .app LAST ---------
# Apple's codesign tool requires nested-content signing in dependency
# order: the innermost dylibs first, then the binaries that link them,
# then frameworks, then the .app wrapper. Otherwise the outer signature
# captures invalid hashes from the inner content and verification fails.

echo "[1/4] Signing dylibs in Resources/lib"
if [ -d "$APP/Contents/Resources/lib" ]; then
    find "$APP/Contents/Resources/lib" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | \
    while IFS= read -r -d '' lib; do
        codesign --force --options runtime --timestamp \
                 --entitlements "$ENTITLEMENTS_PLIST" \
                 --sign "$IDENTITY" "$lib"
    done
fi

echo "[2/4] Signing aliceVision_* binaries"
find "$APP/Contents/Resources/alicevision" -type f -name "aliceVision_*" -print0 | \
while IFS= read -r -d '' bin; do
    codesign --force --options runtime --timestamp \
             --entitlements "$ENTITLEMENTS_PLIST" \
             --sign "$IDENTITY" "$bin"
done

echo "[3/4] Signing Python venv frameworks"
# PySide6 ships .framework bundles + many .so files; they each need their
# own signature when shipping under hardened runtime.
if [ -d "$APP/Contents/Resources/meshroom-venv" ]; then
    find "$APP/Contents/Resources/meshroom-venv" -type f \
        \( -name "*.so" -o -name "*.dylib" \) -print0 | \
    while IFS= read -r -d '' f; do
        codesign --force --options runtime --timestamp \
                 --entitlements "$ENTITLEMENTS_PLIST" \
                 --sign "$IDENTITY" "$f" 2>/dev/null || true
    done
    # The framework Mach-Os live one level deeper.
    find "$APP/Contents/Resources/meshroom-venv" -type d -name "*.framework" -print0 | \
    while IFS= read -r -d '' fw; do
        codesign --force --options runtime --timestamp \
                 --entitlements "$ENTITLEMENTS_PLIST" \
                 --sign "$IDENTITY" "$fw" 2>/dev/null || true
    done
fi

echo "[4/4] Signing .app wrapper"
codesign --force --options runtime --timestamp \
         --entitlements "$ENTITLEMENTS_PLIST" \
         --sign "$IDENTITY" "$APP"

echo
echo "Verifying signature…"
codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -5 || true
spctl --assess --type execute --verbose=2 "$APP" 2>&1 | tail -3 || true

echo
if [ "$IDENTITY" = "-" ]; then
    cat <<EOF
Signed AD-HOC. The .app will run on this machine, but macOS will warn
end-users on first launch ("unidentified developer"). For shippable
distribution, re-run with a Developer ID:

    $0 --identity "Developer ID Application: NAME (TEAMID)" --app "$APP"

After Developer-ID signing, notarize:

    xcrun notarytool submit "<dmg or zip>" \\
        --apple-id you@example.com --team-id TEAMID --password APP-SPECIFIC-PWD \\
        --wait
    xcrun stapler staple "<dmg or zip>"
EOF
else
    cat <<EOF
Signed with $IDENTITY.

Next step: notarize. Create a DMG with scripts/make_dmg.sh, then:

    xcrun notarytool submit "$APP.dmg" \\
        --apple-id you@example.com --team-id TEAMID --password APP-SPECIFIC-PWD \\
        --wait
    xcrun stapler staple "$APP.dmg"
EOF
fi
