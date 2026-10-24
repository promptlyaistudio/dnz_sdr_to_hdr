#!/bin/bash
# Build the DNZ SDR to HDR OFX plugin and assemble the Flame bundle.
# Output: build/DNZSDRtoHDR.ofx.bundle  (ad-hoc signed, ready to copy into /Library/OFX/Plugins)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OFX="${OFX_SDK_DIR:-$HOME/projets/openfx}"
[ -f "$OFX/include/ofxCore.h" ] || OFX="$HERE/../openfx"
[ -f "$OFX/include/ofxCore.h" ] || { echo "OpenFX SDK not found (set OFX_SDK_DIR)"; exit 1; }

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DOFX_SDK_DIR="$OFX" >/dev/null
cmake --build "$HERE/build" --config Release -j

# stale bundles from earlier iterations must not survive: a leftover .ofx in the same
# folder would be dlopen'd by the host test / Flame alongside the current one
rm -rf "$HERE/build/DNZSDRtoHDR.ofx.bundle" "$HERE/build/dnz-sdr-to-hdr.ofx.bundle" \
       "$HERE/build/"*.ofx.bundle
B="$HERE/build/DNZSDRtoHDR.ofx.bundle"
mkdir -p "$B/Contents/MacOS" "$B/Contents/Resources"
cp "$HERE/cmake/Info.plist" "$B/Contents/Info.plist"
cp "$HERE/build/DNZSDRtoHDR.ofx" "$B/Contents/MacOS/DNZSDRtoHDR.ofx"
xattr -cr "$B" 2>/dev/null || true
# arm64 macOS refuses unsigned code; an ad-hoc signature is enough for a local plugin
codesign --force --sign - "$B/Contents/MacOS/DNZSDRtoHDR.ofx" >/dev/null 2>&1 || true
codesign --force --sign - "$B" >/dev/null 2>&1 || true
echo "built: $B"
