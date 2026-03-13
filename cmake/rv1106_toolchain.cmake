# SPDX-License-Identifier: LGPL-3.0-or-later
# cmake/rv1106_toolchain.cmake / 详见英文原注释。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Ensure the configured toolchain root is visible inside CMake try_compile / 确保 configured toolchain root 为 可见 inside CMake try_compile
# sub-configures (e.g. compiler ABI checks). / sub-configures (e.g. compiler ABI 检查).
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES VISIONG_TOOLCHAIN_ROOT)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(DEFINED VISIONG_TOOLCHAIN_ROOT AND NOT VISIONG_TOOLCHAIN_ROOT STREQUAL "")
    set(TC_PATH "${VISIONG_TOOLCHAIN_ROOT}")
elseif(DEFINED ENV{VISIONG_TOOLCHAIN_ROOT} AND NOT "$ENV{VISIONG_TOOLCHAIN_ROOT}" STREQUAL "")
    set(TC_PATH "$ENV{VISIONG_TOOLCHAIN_ROOT}")
else()
    set(TC_PATH "${PROJECT_ROOT}/toolchain/arm-rockchip830-linux-uclibcgnueabihf")
endif()

set(_VISIONG_GCC "${TC_PATH}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(_VISIONG_GXX "${TC_PATH}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

if(NOT EXISTS "${_VISIONG_GCC}" OR NOT EXISTS "${_VISIONG_GXX}")
    message(FATAL_ERROR
        "RV1106 toolchain not found under ${TC_PATH}.
"
        "Set -DVISIONG_TOOLCHAIN_ROOT=/path/to/arm-rockchip830-linux-uclibcgnueabihf
"
        "or export VISIONG_TOOLCHAIN_ROOT before configuring CMake."
    )
endif()

set(CMAKE_C_COMPILER "${_VISIONG_GCC}")
set(CMAKE_CXX_COMPILER "${_VISIONG_GXX}")

set(_VISIONG_SYSROOT "${TC_PATH}/arm-rockchip830-linux-uclibcgnueabihf/sysroot")
if(EXISTS "${_VISIONG_SYSROOT}")
    set(CMAKE_SYSROOT "${_VISIONG_SYSROOT}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

