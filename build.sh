#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
LIB_OUTPUT_DIR="${BUILD_DIR}/libs"
PY_LIB_DIR="${LIB_OUTPUT_DIR}/python"
CPP_LIB_DIR="${LIB_OUTPUT_DIR}/cpp"
CPP_ZIP_PATH="${SCRIPT_DIR}/visiong_cpp.zip"
PY_ZIP_PATH="${SCRIPT_DIR}/visiong_python.zip"
TOOLCHAIN_FILE="${SCRIPT_DIR}/cmake/rv1106_toolchain.cmake"
DEFAULT_RELEASE_CONFIG="${SCRIPT_DIR}/release/release_components.cmake"

CLEAN_BUILD=1
PACKAGE_RELEASE=1
NATIVE_BUILD=0
SDK_ROOT="${VISIONG_SDK_ROOT:-}"
DEPS_ROOT="${VISIONG_DEPS_ROOT:-}"
TOOLCHAIN_ROOT="${VISIONG_TOOLCHAIN_ROOT:-}"
CUSTOM_TOOLCHAIN_FILE="${VISIONG_TOOLCHAIN_FILE:-}"
RELEASE_CONFIG="${VISIONG_RELEASE_CONFIG:-}"
EXTRA_CMAKE_ARGS=()

usage() {
    cat <<USAGE
Usage: ./build.sh [options]

Options:
  --sdk-root <path>       Root of the Luckfox/Rockchip SDK.
  --deps-root <path>      Root of a staged dependency bundle prepared by scripts/stage_release_deps.sh.
  --toolchain-root <path> Root of the arm-rockchip830-linux-uclibcgnueabihf toolchain.
  --toolchain-file <path> Explicit CMake toolchain file.
  --release-config <path> Component release config (.cmake). Defaults to release/release_components.cmake when present.
  --native                Configure without a CMake toolchain file.
  --no-clean              Reuse the existing build directory.
  --no-package            Skip release zip generation.
  --cmake-arg <arg>       Forward one extra argument to CMake (repeatable).
  -h, --help              Show this help message.
USAGE
}

resolve_toolchain_root() {
    if [[ -n "${TOOLCHAIN_ROOT}" ]]; then
        return 0
    fi

    if [[ -d "${SCRIPT_DIR}/toolchain/arm-rockchip830-linux-uclibcgnueabihf" ]]; then
        TOOLCHAIN_ROOT="${SCRIPT_DIR}/toolchain/arm-rockchip830-linux-uclibcgnueabihf"
        return 0
    fi

    if [[ -n "${DEPS_ROOT}" ]]; then
        local found
        found=$(find "${DEPS_ROOT}" -type f -name arm-rockchip830-linux-uclibcgnueabihf-gcc -print -quit 2>/dev/null || true)
        if [[ -n "${found}" ]]; then
            TOOLCHAIN_ROOT=$(cd "$(dirname "${found}")/.." && pwd)
            return 0
        fi
    fi

    if [[ -n "${SDK_ROOT}" ]]; then
        local found
        found=$(find "${SDK_ROOT}" -type f -name arm-rockchip830-linux-uclibcgnueabihf-gcc -print -quit 2>/dev/null || true)
        if [[ -n "${found}" ]]; then
            TOOLCHAIN_ROOT=$(cd "$(dirname "${found}")/.." && pwd)
            return 0
        fi
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk-root)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --sdk-root requires a value" >&2; exit 1; }
            SDK_ROOT="$1"
            ;;
        --deps-root)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --deps-root requires a value" >&2; exit 1; }
            DEPS_ROOT="$1"
            ;;
        --toolchain-root)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --toolchain-root requires a value" >&2; exit 1; }
            TOOLCHAIN_ROOT="$1"
            ;;
        --toolchain-file)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --toolchain-file requires a value" >&2; exit 1; }
            CUSTOM_TOOLCHAIN_FILE="$1"
            ;;
        --release-config)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --release-config requires a value" >&2; exit 1; }
            RELEASE_CONFIG="$1"
            ;;
        --native)
            NATIVE_BUILD=1
            ;;
        --no-clean)
            CLEAN_BUILD=0
            ;;
        --no-package)
            PACKAGE_RELEASE=0
            ;;
        --cmake-arg)
            shift
            [[ $# -gt 0 ]] || { echo "Error: --cmake-arg requires a value" >&2; exit 1; }
            EXTRA_CMAKE_ARGS+=("$1")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ -z "${RELEASE_CONFIG}" && -f "${DEFAULT_RELEASE_CONFIG}" ]]; then
    RELEASE_CONFIG="${DEFAULT_RELEASE_CONFIG}"
fi

if [[ -n "${SDK_ROOT}" && ! -d "${SDK_ROOT}" ]]; then
    echo "Error: SDK root not found: ${SDK_ROOT}" >&2
    exit 1
fi
if [[ -n "${DEPS_ROOT}" && ! -d "${DEPS_ROOT}" ]]; then
    echo "Error: deps root not found: ${DEPS_ROOT}" >&2
    exit 1
fi
if [[ -n "${SDK_ROOT}" && -n "${DEPS_ROOT}" ]]; then
    echo "Error: use either --sdk-root or --deps-root, not both." >&2
    exit 1
fi
if [[ -n "${RELEASE_CONFIG}" && ! -f "${RELEASE_CONFIG}" ]]; then
    echo "Error: release config not found: ${RELEASE_CONFIG}" >&2
    exit 1
fi

CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
)

if [[ -n "${RELEASE_CONFIG}" ]]; then
    # Load release toggles before the main configure step. / Load release toggles before main 配置 step.
    # 在主配置阶段前加载发布组件开关。
    CMAKE_ARGS=(-C "${RELEASE_CONFIG}" "${CMAKE_ARGS[@]}")
fi

if [[ -n "${SDK_ROOT}" ]]; then
    CMAKE_ARGS+=("-DVISIONG_SDK_ROOT=${SDK_ROOT}")
fi
if [[ -n "${DEPS_ROOT}" ]]; then
    CMAKE_ARGS+=("-DVISIONG_DEPS_ROOT=${DEPS_ROOT}")
fi

if [[ ${NATIVE_BUILD} -eq 0 ]]; then
    if [[ -n "${CUSTOM_TOOLCHAIN_FILE}" ]]; then
        CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${CUSTOM_TOOLCHAIN_FILE}")
    else
        resolve_toolchain_root
        if [[ -z "${TOOLCHAIN_ROOT}" ]]; then
            cat >&2 <<ERR
Error: unable to locate the RV1106 cross toolchain.

Try one of these:
  ./build.sh --toolchain-root /path/to/arm-rockchip830-linux-uclibcgnueabihf
  ./build.sh --sdk-root /path/to/luckfox-or-rk-sdk
  ./build.sh --deps-root /path/to/visiong-deps-rv1106
  ./build.sh --native   # only when building directly on the target device
ERR
            exit 1
        fi
        CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}" "-DVISIONG_TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}")
    fi
fi

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
    rm -rf "${BUILD_DIR}"
fi

if [[ -n "${TOOLCHAIN_ROOT}" ]]; then
    export VISIONG_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}"
fi

echo "Configuring VisionG..."
[[ -n "${RELEASE_CONFIG}" ]] && echo "  - release config: ${RELEASE_CONFIG}"
cmake "${CMAKE_ARGS[@]}" "${EXTRA_CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

mkdir -p "${PY_LIB_DIR}" "${CPP_LIB_DIR}"
rm -f "${BUILD_DIR}/visiong.so" "${PY_LIB_DIR}/visiong.so"

if [[ -f "${BUILD_DIR}/_visiong.so" ]]; then
    cp -f "${BUILD_DIR}/_visiong.so" "${PY_LIB_DIR}/"
    cp -f "${SCRIPT_DIR}/src/python/visiong.py" "${BUILD_DIR}/visiong.py"
    cp -f "${SCRIPT_DIR}/src/python/visiong.py" "${PY_LIB_DIR}/"
fi
if [[ -f "${BUILD_DIR}/libvisiong.so" ]]; then
    cp -f "${BUILD_DIR}/libvisiong.so" "${CPP_LIB_DIR}/"
fi
if [[ -f "${BUILD_DIR}/libvisiong.a" ]]; then
    cp -f "${BUILD_DIR}/libvisiong.a" "${CPP_LIB_DIR}/"
fi

echo "Build success:"
[[ -f "${BUILD_DIR}/_visiong.so" ]] && echo "  - ${BUILD_DIR}/_visiong.so"
[[ -f "${BUILD_DIR}/visiong.py" ]] && echo "  - ${BUILD_DIR}/visiong.py"
[[ -f "${BUILD_DIR}/libvisiong.so" ]] && echo "  - ${BUILD_DIR}/libvisiong.so"
[[ -f "${BUILD_DIR}/libvisiong.a" ]] && echo "  - ${BUILD_DIR}/libvisiong.a"
echo "Library output folders:"
echo "  - ${PY_LIB_DIR}"
echo "  - ${CPP_LIB_DIR}"

if [[ ${PACKAGE_RELEASE} -eq 0 ]]; then
    exit 0
fi

"${SCRIPT_DIR}/scripts/create_release_archives.sh" \
    "${BUILD_DIR}" \
    "${SCRIPT_DIR}" \
    "$(basename "${CPP_ZIP_PATH}")" \
    "$(basename "${PY_ZIP_PATH}")"

echo "Release packages:"
echo "  - ${CPP_ZIP_PATH}"
echo "  - ${PY_ZIP_PATH}"
