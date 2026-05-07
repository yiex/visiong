#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_DIR="${ROOT_DIR}/drivers/visiong_hw"

find_kernel_build_dir() {
    local root
    for root in "${VISIONG_KERNEL_BUILD_ROOT:-}" "${VISIONG_DEPS_ROOT:-}" "${VISIONG_SDK_ROOT:-}" "${SDK_ROOT:-}"; do
        [[ -n "${root}" && -d "${root}" ]] || continue
        for candidate in \
            "${root}/sysdrv/source/objs_kernel" \
            "${root}/sysdrv/source/kernel" \
            "${root}/kernel" \
            "${root}/linux"; do
            if [[ -d "${candidate}" && -f "${candidate}/Makefile" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done
    done
    if [[ -d "/lib/modules/$(uname -r)/build" && -f "/lib/modules/$(uname -r)/build/Makefile" ]]; then
        printf '%s\n' "/lib/modules/$(uname -r)/build"
        return 0
    fi
    return 1
}

find_cross_compile_prefix() {
    local root
    local gcc_path
    for root in "${VISIONG_TOOLCHAIN_ROOT:-}" "${TOOLCHAIN_ROOT:-}" "${VISIONG_DEPS_ROOT:-}" "${VISIONG_SDK_ROOT:-}" "${SDK_ROOT:-}"; do
        [[ -n "${root}" && -d "${root}" ]] || continue
        gcc_path=$(find "${root}" -type f \( \
            -name arm-rockchip830-linux-uclibcgnueabihf-gcc -o \
            -name arm-linux-gcc \) -print -quit 2>/dev/null || true)
        if [[ -n "${gcc_path}" ]]; then
            printf '%s\n' "${gcc_path%-gcc}-"
            return 0
        fi
    done
    return 1
}

prepare_kernel_build_dir() {
    if [[ -f "${KDIR}/include/generated/autoconf.h" && -f "${KDIR}/scripts/module.lds" ]]; then
        return 0
    fi

    local defconfig=""
    for candidate in rv1106_defconfig luckfox_rv1106_linux_defconfig rockchip_linux_defconfig rockchip_defconfig; do
        if [[ -f "${KDIR}/arch/${ARCH}/configs/${candidate}" ]]; then
            defconfig="${candidate}"
            break
        fi
    done

    if [[ -z "${defconfig}" ]]; then
        echo "error: kernel tree is not prepared and no supported defconfig was found under ${KDIR}/arch/${ARCH}/configs" >&2
        exit 1
    fi

    echo "Preparing kernel build tree with ${defconfig}..."
    make -C "${KDIR}" ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE:-}" "${defconfig}"
    make -C "${KDIR}" ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE:-}" modules_prepare
}

if [[ -z "${KDIR:-}" ]]; then
    KDIR=$(find_kernel_build_dir || true)
fi

if [[ -z "${KDIR:-}" || ! -f "${KDIR}/Makefile" ]]; then
    echo "error: kernel build tree not found; set KDIR=/path/to/kernel/build" >&2
    exit 1
fi

if [[ -z "${ARCH:-}" ]]; then
    ARCH=arm
fi

if [[ -z "${CROSS_COMPILE:-}" ]]; then
    CROSS_COMPILE=$(find_cross_compile_prefix || true)
fi

echo "KDIR=${KDIR}"
echo "ARCH=${ARCH}"
echo "CROSS_COMPILE=${CROSS_COMPILE:-<host>}"

prepare_kernel_build_dir

if [[ ! -f "${KDIR}/scripts/module.lds" ]]; then
    KERNEL_SRC="${KERNEL_SRC:-}"
    if [[ -z "${KERNEL_SRC}" ]]; then
        for root in "${VISIONG_KERNEL_SOURCE_ROOT:-}" "${VISIONG_DEPS_ROOT:-}" "${VISIONG_SDK_ROOT:-}" "${SDK_ROOT:-}"; do
            [[ -n "${root}" && -d "${root}" ]] || continue
            candidate="${root}/sysdrv/source/kernel"
            if [[ -f "${candidate}/scripts/module.lds.S" ]]; then
                KERNEL_SRC="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${KERNEL_SRC}" ]]; then
        for candidate in \
            "$(readlink -f "${KDIR}/source" 2>/dev/null || true)"; do
            if [[ -n "${candidate}" && -f "${candidate}/scripts/module.lds.S" ]]; then
                KERNEL_SRC="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${KERNEL_SRC}" && -f "${KERNEL_SRC}/scripts/module.lds.S" ]]; then
        mkdir -p "${KDIR}/scripts"
        gcc -E -P -D__KERNEL__ -D__ASSEMBLY__ \
            -I"${KDIR}/arch/${ARCH}/include/generated" \
            -I"${KERNEL_SRC}/arch/${ARCH}/include" \
            -I"${KDIR}/include" \
            -I"${KERNEL_SRC}/include" \
            -include "${KDIR}/include/generated/autoconf.h" \
            "${KERNEL_SRC}/scripts/module.lds.S" \
            -o "${KDIR}/scripts/module.lds"
    fi
fi

make -C "${KDIR}" M="${MODULE_DIR}" ARCH="${ARCH}" CROSS_COMPILE="${CROSS_COMPILE:-}" modules

echo "built: ${MODULE_DIR}/visiong_hw.ko"
