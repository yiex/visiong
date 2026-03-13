#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUT_ROOT="${REPO_ROOT}/_stage"
LUCKFOX_SDK_ROOT=""
RKNN_TOOLKIT2_ROOT=""
LIBRGA_ROOT=""
OPENCV_SOURCE_TAR=""
OPENCV_SOURCE_DIR=""
CLEAN=1

usage() {
    cat <<USAGE
Usage: ./scripts/stage_release_deps.sh [options]

Options:
  --out <path>                 Output staging directory (default: ./_stage)
  --luckfox-sdk <path>         Path to a cloned Luckfox SDK tree
  --rknn-toolkit2 <path>       Path to a cloned airockchip/rknn-toolkit2 tree
  --librga <path>              Path to a cloned airockchip/librga tree
  --opencv-source-tar <path>   Path to an official OpenCV source archive (.tar.* or .zip)
  --opencv-source-dir <path>   Path to an official OpenCV source tree
  --no-clean                   Reuse the existing output directory
  -h, --help                   Show this help message
USAGE
}

copy_first_match() {
    local dest="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -f "${candidate}" ]]; then
            mkdir -p "$(dirname "${dest}")"
            cp -f "${candidate}" "${dest}"
            return 0
        fi
    done
    return 1
}

copy_tree_if_exists() {
    local src="$1"
    local dest="$2"
    if [[ -d "${src}" ]]; then
        mkdir -p "$(dirname "${dest}")"
        rm -rf "${dest}"
        cp -a "${src}" "${dest}"
        return 0
    fi
    return 1
}

copy_named_file_from_roots() {
    local filename="$1"
    local dest="$2"
    shift 2

    local root
    for root in "$@"; do
        [[ -d "${root}" ]] || continue
        local found
        found=$(find "${root}" -type f -name "${filename}" -print -quit 2>/dev/null || true)
        if [[ -n "${found}" ]]; then
            mkdir -p "$(dirname "${dest}")"
            cp -f "${found}" "${dest}"
            return 0
        fi
    done
    return 1
}

copy_tree_from_marker_file() {
    local marker_name="$1"
    local up_levels="$2"
    local dest="$3"
    shift 3

    local root
    for root in "$@"; do
        [[ -d "${root}" ]] || continue
        local found
        found=$(find "${root}" -type f -name "${marker_name}" -print -quit 2>/dev/null || true)
        if [[ -n "${found}" ]]; then
            local src_tree
            src_tree="${found}"
            local i
            for ((i = 0; i < up_levels; ++i)); do
                src_tree=$(dirname "${src_tree}")
            done
            copy_tree_if_exists "${src_tree}" "${dest}" && return 0
        fi
    done
    return 1
}

copy_vendor_lib_from_known_paths() {
    local libname="$1"
    local dest="$2"

    copy_first_match "${dest}" \
        "${LUCKFOX_SDK_ROOT}/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/project/app/component/fastboot_server/rootfs/usr/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/sysdrv/output/out/rootfs_uclibc_rv1106/usr/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/sysdrv/output/image/usr/lib/${libname}" || true

    if [[ -f "${dest}" ]]; then
        return 0
    fi

    # Fallback to public Luckfox source-tree locations for RV1106 prebuilt libs.
    copy_first_match "${dest}" \
        "${LUCKFOX_SDK_ROOT}/media/rga/release_rga_rv1106_arm-rockchip830-linux-uclibcgnueabihf/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/mpp/release_mpp_rv1106_arm-rockchip830-linux-uclibcgnueabihf/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/isp/release_camera_engine_rkaiq_rv1106_arm-rockchip830-linux-uclibcgnueabihf/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/isp/release_camera_engine_rkaiq_rv1106_arm-rockchip830-linux-uclibcgnueabihf/rkisp_demo/demo/libs/arm32/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/iva/iva/librockiva/rockiva-rv1106-Linux/lib/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/rockit/rockit/lib/lib32/${libname}" \
        "${LUCKFOX_SDK_ROOT}/media/common_algorithm/common_algorithm/misc/lib/arm-rockchip830-linux-uclibcgnueabihf/${libname}" || true
}

resolve_vendor_flat_root() {
    local root
    for root in "$@"; do
        [[ -d "${root}" ]] || continue

        local sample_comm
        sample_comm=$(find "${root}" -type f -name "sample_comm.h" -print -quit 2>/dev/null || true)
        if [[ -n "${sample_comm}" ]]; then
            dirname "${sample_comm}"
            return 0
        fi

        local rk_type
        rk_type=$(find "${root}" -type f -name "rk_type.h" -print -quit 2>/dev/null || true)
        if [[ -n "${rk_type}" ]]; then
            local parent
            parent=$(dirname "${rk_type}")
            if [[ "$(basename "${parent}")" == "rockchip" ]]; then
                dirname "${parent}"
            else
                echo "${parent}"
            fi
            return 0
        fi
    done
    return 1
}

resolve_toolchain_root() {
    if [[ -d "${OUT_ROOT}/toolchain/arm-rockchip830-linux-uclibcgnueabihf" ]]; then
        echo "${OUT_ROOT}/toolchain/arm-rockchip830-linux-uclibcgnueabihf"
        return 0
    fi
    if [[ -d "${LUCKFOX_SDK_ROOT}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf" ]]; then
        echo "${LUCKFOX_SDK_ROOT}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf"
        return 0
    fi

    local found
    found=$(find "${LUCKFOX_SDK_ROOT}" -type f -name "arm-rockchip830-linux-uclibcgnueabihf-gcc" -print -quit 2>/dev/null || true)
    if [[ -n "${found}" ]]; then
        cd "$(dirname "${found}")/.." && pwd
        return 0
    fi
    return 1
}

stage_opencv_from_source() {
    if [[ -f "${OUT_ROOT}/opencv/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
        return 0
    fi

    local source_dir=""
    if [[ -n "${OPENCV_SOURCE_TAR}" ]]; then
        rm -rf "${OUT_ROOT}/opencv.source" "${OUT_ROOT}/opencv.source.unpack"
        mkdir -p "${OUT_ROOT}/opencv.source.unpack"
        case "${OPENCV_SOURCE_TAR}" in
            *.zip)
                unzip -q "${OPENCV_SOURCE_TAR}" -d "${OUT_ROOT}/opencv.source.unpack"
                ;;
            *)
                tar -xf "${OPENCV_SOURCE_TAR}" -C "${OUT_ROOT}/opencv.source.unpack"
                ;;
        esac
        shopt -s nullglob
        local entries=("${OUT_ROOT}/opencv.source.unpack"/*)
        shopt -u nullglob
        if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
            mv "${entries[0]}" "${OUT_ROOT}/opencv.source"
            rm -rf "${OUT_ROOT}/opencv.source.unpack"
        else
            mv "${OUT_ROOT}/opencv.source.unpack" "${OUT_ROOT}/opencv.source"
        fi
        source_dir="${OUT_ROOT}/opencv.source"
    elif [[ -n "${OPENCV_SOURCE_DIR}" ]]; then
        copy_tree_if_exists "${OPENCV_SOURCE_DIR}" "${OUT_ROOT}/opencv.source" || true
        source_dir="${OUT_ROOT}/opencv.source"
    fi

    if [[ -z "${source_dir}" ]]; then
        return 0
    fi

    local toolchain_root
    toolchain_root=$(resolve_toolchain_root || true)
    if [[ -z "${toolchain_root}" ]]; then
        echo "Error: failed to locate the RV1106 toolchain required for the staged OpenCV build." >&2
        exit 1
    fi

    local -a build_args=(
        --source-dir "${source_dir}"
        --install-prefix "${OUT_ROOT}/opencv"
        --build-dir "${OUT_ROOT}/opencv.build"
        --toolchain-root "${toolchain_root}"
    )
    if [[ ${CLEAN} -eq 0 ]]; then
        build_args+=(--no-clean)
    fi
    "${REPO_ROOT}/scripts/build_minimal_opencv.sh" "${build_args[@]}"
}

stage_vendor_headers() {
    local out_include_dir="${OUT_ROOT}/vendor/rockchip/include"
    rm -rf "${out_include_dir}"
    mkdir -p "${out_include_dir}"

    local -a roots=("${LUCKFOX_SDK_ROOT}")

    copy_tree_from_marker_file "rk_aiq_user_api2_sysctl.h" 2 "${out_include_dir}/rkaiq" "${roots[@]}" || true
    copy_tree_from_marker_file "rk_mpi.h" 1 "${out_include_dir}/rockchip" "${roots[@]}" || true

    local flat_root
    flat_root=$(resolve_vendor_flat_root "${roots[@]}" || true)

    if [[ -n "${flat_root}" && -d "${flat_root}" ]]; then
        find "${flat_root}" -maxdepth 1 -type f -name "rk_*.h" -exec cp -f {} "${out_include_dir}/" \;
        if [[ -f "${flat_root}/rtsp_demo.h" ]]; then
            cp -f "${flat_root}/rtsp_demo.h" "${out_include_dir}/rtsp_demo.h"
        fi
    else
        copy_named_file_from_roots "rtsp_demo.h" "${out_include_dir}/rtsp_demo.h" "${roots[@]}" || true
        for header in rk_type.h rk_debug.h rk_errno.h rk_mpi_mb.h rk_mpi_sys.h rk_mpi_vi.h rk_mpi_venc.h rk_mpi_vdec.h rk_mpi_vpss.h rk_mpi_vo.h rk_mpi_rga.h; do
            copy_named_file_from_roots "${header}" "${out_include_dir}/${header}" "${roots[@]}" || true
        done
    fi
    local comm_marker
    local comm_root
    local root
    for root in "${roots[@]}"; do
        [[ -d "${root}" ]] || continue
        comm_marker=$(find "${root}" -type f -name "rk_comm_mb.h" -print -quit 2>/dev/null || true)
        if [[ -n "${comm_marker}" ]]; then
            comm_root=$(dirname "${comm_marker}")
            cp -a "${comm_root}/." "${out_include_dir}/"
            break
        fi
    done
    copy_named_file_from_roots "rtsp_demo.h" "${out_include_dir}/rtsp_demo.h" "${roots[@]}" || true


}

stage_rga_headers() {
    local out_include_dir="${OUT_ROOT}/vendor/librga/include"
    rm -rf "${out_include_dir}"
    mkdir -p "${out_include_dir}"

    local -a roots=()
    if [[ -n "${LIBRGA_ROOT}" ]]; then
        roots+=("${LIBRGA_ROOT}")
    fi
    roots+=("${LUCKFOX_SDK_ROOT}")

    copy_tree_from_marker_file "im2d.h" 1 "${out_include_dir}" "${roots[@]}" || true
}

stage_ive_headers() {
    local out_include_dir="${OUT_ROOT}/vendor/ive/include"
    rm -rf "${out_include_dir}"
    mkdir -p "${out_include_dir}"

    local -a roots=("${LUCKFOX_SDK_ROOT}")
    copy_tree_from_marker_file "rk_mpi_ive.h" 1 "${out_include_dir}" "${roots[@]}" || true
}

stage_rknn_headers() {
    local out_include_dir="${OUT_ROOT}/vendor/rknpu2/include"
    rm -rf "${out_include_dir}"
    mkdir -p "${out_include_dir}"

    local -a roots=()
    if [[ -n "${RKNN_TOOLKIT2_ROOT}" ]]; then
        roots+=("${RKNN_TOOLKIT2_ROOT}")
    fi
    roots+=("${LUCKFOX_SDK_ROOT}")

    local root
    for root in "${roots[@]}"; do
        [[ -d "${root}" ]] || continue
        local found
        found=$(find "${root}" -type f -name "rknn_api.h" -print -quit 2>/dev/null || true)
        if [[ -n "${found}" ]]; then
            local src_dir
            src_dir=$(dirname "${found}")
            cp -a "${src_dir}/." "${out_include_dir}/"
            break
        fi
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) shift; OUT_ROOT="$1" ;;
        --luckfox-sdk) shift; LUCKFOX_SDK_ROOT="$1" ;;
        --rknn-toolkit2) shift; RKNN_TOOLKIT2_ROOT="$1" ;;
        --librga) shift; LIBRGA_ROOT="$1" ;;
        --opencv-source-tar) shift; OPENCV_SOURCE_TAR="$1" ;;
        --opencv-source-dir) shift; OPENCV_SOURCE_DIR="$1" ;;
        --no-clean) CLEAN=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

if [[ "${OUT_ROOT}" != /* ]]; then
    OUT_ROOT="${REPO_ROOT}/${OUT_ROOT}"
fi
OUT_ROOT=$(mkdir -p "${OUT_ROOT}" && cd "${OUT_ROOT}" && pwd)

[[ -n "${LUCKFOX_SDK_ROOT}" ]] || { echo "Error: --luckfox-sdk is required." >&2; exit 1; }
[[ -d "${LUCKFOX_SDK_ROOT}" ]] || { echo "Error: Luckfox SDK not found: ${LUCKFOX_SDK_ROOT}" >&2; exit 1; }

if [[ ${CLEAN} -eq 1 ]]; then
    rm -rf "${OUT_ROOT}"
fi

mkdir -p "${OUT_ROOT}/lib" "${OUT_ROOT}/3rdparty/ive/ive" "${OUT_ROOT}/3rdparty"

copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf" \
    "${OUT_ROOT}/toolchain/arm-rockchip830-linux-uclibcgnueabihf" || true

copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/sysdrv/source/mcu/rt-thread/bsp/rockchip/common/drivers/python/target_python" \
    "${OUT_ROOT}/3rdparty/target_python" || true
copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/3rdparty/target_python" \
    "${OUT_ROOT}/3rdparty/target_python" || true
copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/target_python" \
    "${OUT_ROOT}/3rdparty/target_python" || true
copy_tree_if_exists \
    "${REPO_ROOT}/3rdparty/target_python" \
    "${OUT_ROOT}/3rdparty/target_python" || true

copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/media/ive/ive/lib" \
    "${OUT_ROOT}/3rdparty/ive/ive/lib" || true
copy_tree_if_exists \
    "${LUCKFOX_SDK_ROOT}/3rdparty/ive/ive/lib" \
    "${OUT_ROOT}/3rdparty/ive/ive/lib" || true

for libname in \
    librga.so librga.a \
    librockiva.so librockiva.a \
    librockchip_mpp.so.1 librockchip_mpp.so.0 librockchip_mpp.a \
    librkaiq.so librkaiq.a \
    librknnmrt.so librknnmrt.a \
    librockit.so librockit.a \
    librtsp.a; do
    copy_vendor_lib_from_known_paths "${libname}" "${OUT_ROOT}/lib/${libname}"
done

if [[ -n "${RKNN_TOOLKIT2_ROOT}" && -d "${RKNN_TOOLKIT2_ROOT}" ]]; then
    copy_first_match "${OUT_ROOT}/lib/librknnmrt.so" \
        "${RKNN_TOOLKIT2_ROOT}/rknpu2/runtime/RV1106/Linux/librknn_api/aarch32/librknnmrt.so" \
        "${RKNN_TOOLKIT2_ROOT}/rknpu2/runtime/RV1106/Linux/librknn_api/armhf-uclibc/librknnmrt.so" \
        "${RKNN_TOOLKIT2_ROOT}/rknpu2/runtime/RV1106/Linux/librknn_api/arm-linux-uclibcgnueabihf/librknnmrt.so" || true
fi

if [[ -n "${LIBRGA_ROOT}" && -d "${LIBRGA_ROOT}" ]]; then
    copy_first_match "${OUT_ROOT}/lib/librga.so" \
        "${LIBRGA_ROOT}/libs/Linux/gcc-aarch32-uclibc/librga.so" \
        "${LIBRGA_ROOT}/libs/Linux/gcc-armhf/librga.so" \
        "${LIBRGA_ROOT}/Linux/gcc-aarch32-uclibc/librga.so" || true
fi

copy_tree_if_exists \
    "${REPO_ROOT}/3rdparty/opencv" \
    "${OUT_ROOT}/opencv" || true

stage_opencv_from_source

stage_vendor_headers
stage_rga_headers
stage_ive_headers
stage_rknn_headers

if [[ ! -f "${OUT_ROOT}/vendor/rknpu2/include/rknn_api.h" ]]; then
    echo "Warning: rknn_api.h was not staged under ${OUT_ROOT}/vendor/rknpu2/include" >&2
fi
if [[ ! -f "${OUT_ROOT}/vendor/librga/include/im2d.h" ]]; then
    echo "Warning: librga headers were not staged under ${OUT_ROOT}/vendor/librga/include" >&2
fi
if [[ ! -f "${OUT_ROOT}/vendor/ive/include/rk_mpi_ive.h" ]]; then
    echo "Warning: IVE headers were not staged under ${OUT_ROOT}/vendor/ive/include" >&2
fi
if [[ ! -f "${OUT_ROOT}/3rdparty/ive/ive/lib/librve.so" && ! -f "${OUT_ROOT}/3rdparty/ive/ive/lib/librve.a" ]]; then
    echo "Warning: librve was not staged under ${OUT_ROOT}/3rdparty/ive/ive/lib" >&2
fi
if [[ ! -f "${OUT_ROOT}/3rdparty/ive/ive/lib/libivs.so" && ! -f "${OUT_ROOT}/3rdparty/ive/ive/lib/libivs.a" ]]; then
    echo "Warning: libivs was not staged under ${OUT_ROOT}/3rdparty/ive/ive/lib" >&2
fi

cat <<DONE
Staged dependencies under: ${OUT_ROOT}

Suggested next step:
  ./build.sh --deps-root ${OUT_ROOT}
DONE
