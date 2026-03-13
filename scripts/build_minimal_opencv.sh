#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TOOLCHAIN_FILE="${REPO_ROOT}/cmake/rv1106_toolchain.cmake"
SOURCE_DIR=""
INSTALL_PREFIX=""
BUILD_DIR=""
TOOLCHAIN_ROOT="${VISIONG_TOOLCHAIN_ROOT:-}"
JOBS="$(nproc)"
CLEAN=1

usage() {
    cat <<USAGE
Usage: ./scripts/build_minimal_opencv.sh [options]

Options:
  --source-dir <path>      Official OpenCV source root
  --install-prefix <path>  Install prefix for the staged OpenCV build
  --build-dir <path>       Out-of-tree build directory
  --toolchain-root <path>  Root of arm-rockchip830-linux-uclibcgnueabihf
  --toolchain-file <path>  Explicit CMake toolchain file
  --jobs <n>               Parallel build jobs (default: nproc)
  --no-clean               Reuse the existing build directory
  -h, --help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir) shift; SOURCE_DIR="$1" ;;
        --install-prefix) shift; INSTALL_PREFIX="$1" ;;
        --build-dir) shift; BUILD_DIR="$1" ;;
        --toolchain-root) shift; TOOLCHAIN_ROOT="$1" ;;
        --toolchain-file) shift; TOOLCHAIN_FILE="$1" ;;
        --jobs) shift; JOBS="$1" ;;
        --no-clean) CLEAN=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

[[ -n "${SOURCE_DIR}" ]] || { echo "Error: --source-dir is required." >&2; exit 1; }
[[ -d "${SOURCE_DIR}" ]] || { echo "Error: OpenCV source directory not found: ${SOURCE_DIR}" >&2; exit 1; }
[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || { echo "Error: ${SOURCE_DIR} is not an OpenCV source root." >&2; exit 1; }
[[ -n "${INSTALL_PREFIX}" ]] || { echo "Error: --install-prefix is required." >&2; exit 1; }
[[ -n "${BUILD_DIR}" ]] || { echo "Error: --build-dir is required." >&2; exit 1; }
[[ -n "${TOOLCHAIN_ROOT}" ]] || { echo "Error: --toolchain-root is required." >&2; exit 1; }
[[ -d "${TOOLCHAIN_ROOT}" ]] || { echo "Error: toolchain root not found: ${TOOLCHAIN_ROOT}" >&2; exit 1; }
[[ -f "${TOOLCHAIN_FILE}" ]] || { echo "Error: toolchain file not found: ${TOOLCHAIN_FILE}" >&2; exit 1; }

if [[ ${CLEAN} -eq 1 ]]; then
    rm -rf "${BUILD_DIR}" "${INSTALL_PREFIX}"
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_PREFIX}"
export VISIONG_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
AR_BIN="${TOOLCHAIN_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-ar"
if [[ ! -x "${AR_BIN}" ]]; then
    AR_BIN="$(command -v ar)"
fi

# Keep the staged OpenCV footprint close to visiong's real usage:
# core/imgproc/imgcodecs only, no GUI/video stacks, and no external codec pulls.
# 将 OpenCV 保持为 visiong 实际需要的最小子集：
# 只保留 core/imgproc/imgcodecs，不带 GUI/视频栈，也不额外拉取外部编解码依赖。
cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_FAT_JAVA_LIB=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_JPEG=ON \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_OPENJPEG=OFF \
    -DBUILD_PACKAGE=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_PNG=ON \
    -DBUILD_TBB=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_TIFF=OFF \
    -DBUILD_WEBP=OFF \
    -DBUILD_ZLIB=ON \
    -DENABLE_NEON=ON \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
    -DOPENCV_GENERATE_PKGCONFIG=OFF \
    -DWITH_1394=OFF \
    -DWITH_ADE=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF \
    -DWITH_IPP=OFF \
    -DWITH_ITT=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_OPENJPEG=OFF \
    -DWITH_PNG=ON \
    -DWITH_QUIRC=OFF \
    -DWITH_TBB=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF

cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --install "${BUILD_DIR}"

# OpenCV's exported static CMake package may unconditionally declare libprotobuf
# even when our minimal build does not produce it. Provide an empty archive so
# downstream find_package(OpenCV) keeps working with the trimmed module set.
# OpenCV 的静态导出包在最小裁剪构建下仍可能无条件声明 libprotobuf。
# 这里补一个空归档，保证下游 find_package(OpenCV) 在精简模块场景下可正常工作。
THIRDPARTY_DIR="${INSTALL_PREFIX}/lib/opencv4/3rdparty"
mkdir -p "${THIRDPARTY_DIR}"
if [[ ! -f "${THIRDPARTY_DIR}/liblibprotobuf.a" ]]; then
    "${AR_BIN}" crs "${THIRDPARTY_DIR}/liblibprotobuf.a"
fi
