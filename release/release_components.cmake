# SPDX-License-Identifier: LGPL-3.0-or-later
# VisionG release component profile / VisionG 发布组件配置
# Edit the switches below, then run ./build.sh. / 直接修改以下开关后执行 ./build.sh 即可。

set(VISIONG_ENABLE_IVE ON CACHE BOOL "Build the IVE component / 构建 IVE 组件" FORCE)
set(VISIONG_ENABLE_NPU ON CACHE BOOL "Build the NPU component / 构建 NPU 组件" FORCE)
set(VISIONG_ENABLE_GUI ON CACHE BOOL "Build the GUI component / 构建 GUI 组件" FORCE)
