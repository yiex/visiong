#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

BUILD_DIR=${1:-build}
OUT_DIR=${2:-dist-binary}
TARGET_RPATH='$ORIGIN:/oem/usr/lib:/usr/lib:/lib'

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required tool not found: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Error: required file not found: $1" >&2
        exit 1
    fi
}

require_tool patchelf
require_tool readelf
require_tool sha256sum
require_tool file

require_file "$BUILD_DIR/libvisiong.so"
require_file "$BUILD_DIR/_visiong.so"
require_file "$BUILD_DIR/visiong.py"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp -f "$BUILD_DIR/libvisiong.so" "$OUT_DIR/"
cp -f "$BUILD_DIR/_visiong.so" "$OUT_DIR/"
cp -f "$BUILD_DIR/visiong.py" "$OUT_DIR/"
cp -f README.md "$OUT_DIR/README.md"
cp -f THIRD_PARTY_NOTICES.md "$OUT_DIR/THIRD_PARTY_NOTICES.md"
cp -f docs/BINARY_REDISTRIBUTION_POLICY.md "$OUT_DIR/BINARY_REDISTRIBUTION_POLICY.md"

cat > "$OUT_DIR/BINARY_ARTIFACT_WARNING.txt" <<'EOF'
VisionG binary artifact notice

This artifact contains VisionG binaries only.
External Rockchip and Luckfox runtime libraries are not bundled here.
Deploy it only to targets that already provide the required runtime libraries.
Review redistribution terms for every external dependency before mirroring this artifact.
EOF

for lib in libvisiong.so _visiong.so; do
    if patchelf --print-rpath "$OUT_DIR/$lib" >/dev/null 2>&1; then
        patchelf --print-rpath "$OUT_DIR/$lib" > "$OUT_DIR/${lib}.original-rpath.txt"
    else
        : > "$OUT_DIR/${lib}.original-rpath.txt"
    fi
    patchelf --set-rpath "$TARGET_RPATH" "$OUT_DIR/$lib"
    patchelf --print-rpath "$OUT_DIR/$lib" > "$OUT_DIR/${lib}.patched-rpath.txt"
    readelf -d "$OUT_DIR/$lib" | grep NEEDED > "$OUT_DIR/${lib}.needed.txt" || true
    file "$OUT_DIR/$lib" > "$OUT_DIR/${lib}.file.txt"
done

sha256sum "$OUT_DIR/libvisiong.so" "$OUT_DIR/_visiong.so" "$OUT_DIR/visiong.py" > "$OUT_DIR/SHA256SUMS.txt"
