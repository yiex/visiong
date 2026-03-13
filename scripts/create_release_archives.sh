#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR=${1:-build}
OUT_DIR=${2:-${ROOT_DIR}}
CPP_ZIP_NAME=${3:-visiong_cpp.zip}
PY_ZIP_NAME=${4:-visiong_python.zip}

if [[ "${OUT_DIR}" != /* ]]; then
    OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi
OUT_DIR=$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required tool not found: $1" >&2
        exit 1
    fi
}

cache_get() {
    local key=$1
    local line
    line=$(grep -E "^${key}:" "${BUILD_DIR}/CMakeCache.txt" | tail -n 1 || true)
    if [[ -z "${line}" ]]; then
        return 1
    fi
    printf '%s\n' "${line#*=}"
}

copy_common_docs() {
    local dst=$1
    local item
    for item in LICENSE THIRD_PARTY_NOTICES.md README.md licenses; do
        if [[ -e "${ROOT_DIR}/${item}" ]]; then
            cp -a "${ROOT_DIR}/${item}" "${dst}/"
        fi
    done
}

copy_tree_contents() {
    local src=$1
    local dst=$2
    if [[ -d "${src}" ]]; then
        mkdir -p "${dst}"
        cp -a "${src}/." "${dst}/"
    fi
}

write_component_manifest() {
    local dst=$1
    local ive_flag=$2
    local npu_flag=$3
    local gui_flag=$4
    cat > "${dst}/VISIONG_COMPONENTS.txt" <<MANIFEST
VisionG release components

IVE=${ive_flag}
NPU=${npu_flag}
GUI=${gui_flag}
MANIFEST
}

require_tool zip
require_tool grep

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "Error: ${BUILD_DIR}/CMakeCache.txt not found. Run CMake build first." >&2
    exit 1
fi

if [[ ! -f "${BUILD_DIR}/libvisiong.so" && ! -f "${BUILD_DIR}/libvisiong.a" ]]; then
    echo "Error: libvisiong artifacts are missing in ${BUILD_DIR}." >&2
    exit 1
fi

if [[ ! -f "${BUILD_DIR}/_visiong.so" || ! -f "${BUILD_DIR}/visiong.py" ]]; then
    echo "Error: Python artifacts (_visiong.so and visiong.py) are missing in ${BUILD_DIR}." >&2
    exit 1
fi

TMP_DIR=$(mktemp -d)
CPP_STAGE="${TMP_DIR}/cpp"
PY_STAGE="${TMP_DIR}/python"
mkdir -p "${CPP_STAGE}/include" "${CPP_STAGE}/lib" "${CPP_STAGE}/cmake" "${PY_STAGE}"

VISIONG_ENABLE_IVE=$(cache_get VISIONG_ENABLE_IVE || echo ON)
VISIONG_ENABLE_NPU=$(cache_get VISIONG_ENABLE_NPU || echo ON)
VISIONG_ENABLE_GUI=$(cache_get VISIONG_ENABLE_GUI || echo ON)

copy_common_docs "${CPP_STAGE}"
copy_common_docs "${PY_STAGE}"

cp -a "${ROOT_DIR}/include/." "${CPP_STAGE}/include/"
mkdir -p "${CPP_STAGE}/include/visiong/common"
if [[ -f "${BUILD_DIR}/generated/visiong/common/build_config.h" ]]; then
    cp -f "${BUILD_DIR}/generated/visiong/common/build_config.h" "${CPP_STAGE}/include/visiong/common/build_config.h"
fi

if [[ "${VISIONG_ENABLE_IVE}" == "ON" ]]; then
    :
else
    rm -f "${CPP_STAGE}/include/visiong/modules/IVE.h"
fi
if [[ "${VISIONG_ENABLE_NPU}" == "ON" ]]; then
    :
else
    rm -rf "${CPP_STAGE}/include/visiong/npu"
fi
if [[ "${VISIONG_ENABLE_GUI}" != "ON" ]]; then
    rm -f "${CPP_STAGE}/include/visiong/modules/GUIManager.h"
fi

[[ -f "${BUILD_DIR}/libvisiong.so" ]] && cp -f "${BUILD_DIR}/libvisiong.so" "${CPP_STAGE}/lib/"
[[ -f "${BUILD_DIR}/libvisiong.a" ]] && cp -f "${BUILD_DIR}/libvisiong.a" "${CPP_STAGE}/lib/"

for cmake_file in VisionGConfig.cmake VisionGConfigVersion.cmake VisionGTargets.cmake; do
    if [[ -f "${BUILD_DIR}/${cmake_file}" ]]; then
        cp -f "${BUILD_DIR}/${cmake_file}" "${CPP_STAGE}/cmake/${cmake_file}"
    fi
done
if [[ -f "${ROOT_DIR}/cmake/rv1106_toolchain.cmake" ]]; then
    cp -f "${ROOT_DIR}/cmake/rv1106_toolchain.cmake" "${CPP_STAGE}/cmake/"
fi

write_component_manifest "${CPP_STAGE}" "${VISIONG_ENABLE_IVE}" "${VISIONG_ENABLE_NPU}" "${VISIONG_ENABLE_GUI}"

cp -f "${BUILD_DIR}/_visiong.so" "${PY_STAGE}/_visiong.so"
cp -f "${BUILD_DIR}/visiong.py" "${PY_STAGE}/visiong.py"
write_component_manifest "${PY_STAGE}" "${VISIONG_ENABLE_IVE}" "${VISIONG_ENABLE_NPU}" "${VISIONG_ENABLE_GUI}"

rm -f "${OUT_DIR}/${CPP_ZIP_NAME}" "${OUT_DIR}/${PY_ZIP_NAME}"
(
    cd "${CPP_STAGE}"
    zip -qr "${OUT_DIR}/${CPP_ZIP_NAME}" .
)
(
    cd "${PY_STAGE}"
    zip -qr "${OUT_DIR}/${PY_ZIP_NAME}" .
)

rm -rf "${TMP_DIR}"

echo "Created: ${OUT_DIR}/${CPP_ZIP_NAME}"
echo "Created: ${OUT_DIR}/${PY_ZIP_NAME}"
