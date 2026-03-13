#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUT_PATH_INPUT=${1:-"${REPO_ROOT}/visiong_source_release.zip"}
PUBLIC_TOP_LEVEL=(
    .github
    .gitattributes
    .gitignore
    3rdparty
    CMakeLists.txt
    CONTRIBUTING.md
    LICENSE
    README.md
    THIRD_PARTY_NOTICES.md
    build.sh
    cmake
    docs
    include
    licenses
    release
    scripts
    src
)

if [[ "${OUT_PATH_INPUT}" = /* ]]; then
    OUT_PATH="${OUT_PATH_INPUT}"
else
    OUT_PATH="${REPO_ROOT}/${OUT_PATH_INPUT}"
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: zip is required to create a source release." >&2
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGE_DIR="${TMP_DIR}/visiong"
mkdir -p "${STAGE_DIR}"

for entry in "${PUBLIC_TOP_LEVEL[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${entry}" ]]; then
        echo "Error: required public entry is missing: ${entry}" >&2
        exit 1
    fi
    cp -a "${REPO_ROOT}/${entry}" "${STAGE_DIR}/"
done

find "${STAGE_DIR}" -name '*.zip' -type f -delete
find "${STAGE_DIR}" -name '*.log' -type f -delete
find "${STAGE_DIR}/3rdparty/media-server" -type f \
    \( -name '*.vcxproj' -o -name '*.vcxproj.filters' -o -name 'Android.mk' -o -name 'Makefile' -o -name 'version.ver' \) \
    -delete || true
find "${STAGE_DIR}/3rdparty/media-server" -type d \
    \( -name '*.xcodeproj' -o -name '*.xcworkspace' -o -name test -o -name tests -o -name demo \) \
    -prune -exec rm -rf {} + || true

python3 "${STAGE_DIR}/scripts/audit_build_inputs.py"
rm -rf "${STAGE_DIR}/3rdparty/target_python"
rm -rf "${STAGE_DIR}/3rdparty/opencv"

mkdir -p "$(dirname "${OUT_PATH}")"
(
    cd "${TMP_DIR}"
    rm -f "${OUT_PATH}"
    zip -r "${OUT_PATH}" visiong >/dev/null
)
python3 "${STAGE_DIR}/scripts/audit_source_release.py" "${OUT_PATH}"

echo "Source release created: ${OUT_PATH}"
