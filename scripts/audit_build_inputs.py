#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that the repository only keeps files intentionally used by the build or release flow."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PRESENT = [
    ROOT / "CMakeLists.txt",
    ROOT / "build.sh",
    ROOT / "README.md",
    ROOT / "3rdparty/opencv/lib/cmake/opencv4/OpenCVConfig.cmake",
    ROOT / "cmake/rv1106_toolchain.cmake",
    ROOT / "3rdparty/media-server/LICENSE",
    ROOT / "3rdparty/nuklear/LICENSE",
    ROOT / "3rdparty/pybind11/LICENSE",
    ROOT / "3rdparty/quirc/LICENSE",
    ROOT / "3rdparty/stb/LICENSE",
    ROOT / "3rdparty/target_python/include/python3.11/Python.h",
    ROOT / "3rdparty/target_python/lib/libpython3.11.so",
    ROOT / "src/core/Camera.cpp",
    ROOT / "src/modules/DisplayRTSP.cpp",
    ROOT / "src/npu/NPU.cpp",
    ROOT / "src/npu/internal/tracking/core.cpp",
    ROOT / "src/npu/internal/npu_common.h",
    ROOT / "src/python/visiong_bindings.cpp",
    ROOT / "src/common/internal/string_utils.h",
    ROOT / "src/core/internal/logger.h",
    ROOT / "src/modules/internal/venc_utils.h",
    ROOT / "src/modules/internal/ive_memory.h",
    ROOT / "src/modules/internal/gui_nuklear_config.h",
    ROOT / "src/modules/internal/gui_nuklear_style.h",
    ROOT / "src/modules/internal/gui_command_renderer.h",
    ROOT / "src/modules/internal/gui_command_renderer.cpp",
    ROOT / "src/modules/internal/gui_manager_impl.h",
    ROOT / "src/modules/internal/gui_manager_input.cpp",
    ROOT / "src/modules/internal/gui_manager_widgets.cpp",
    ROOT / "src/modules/GUIManagerApi.cpp",
    ROOT / "src/modules/GUIManagerBridge.cpp",
    ROOT / "src/modules/GUIManagerCompat.cpp",
    ROOT / "src/common/internal/dma_alloc.h",
    ROOT / "src/c_compat/include/http-reason.h",
    ROOT / "3rdparty/quirc/CMakeLists.txt",
    ROOT / "3rdparty/media-server/libflv/source/flv-muxer.c",
    ROOT / "scripts/audit_source_release.py",
    ROOT / "scripts/audit_dependency_policy.py",
    ROOT / "scripts/audit_spdx_headers.py",
    ROOT / "scripts/audit_binary_release_policy.py",
    ROOT / "scripts/dependency_policy.json",
    ROOT / "scripts/binary_release_policy.json",
    ROOT / "scripts/build_minimal_opencv.sh",
    ROOT / "scripts/prepare_binary_artifact.sh",
    ROOT / "scripts/create_release_archives.sh",
    ROOT / "release/release_components.cmake",
    ROOT / "docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md",
    ROOT / "docs/BINARY_REDISTRIBUTION_POLICY.md",
    ROOT / "docs/BINARY_ARTIFACT_WORKFLOW.md",
    ROOT / ".github/workflows/binary-artifact.yml",
]

EXPECTED_ABSENT = [
    ROOT / "_deps",
    ROOT / "_stage",
    ROOT / "build",
    ROOT / "dist",
    ROOT / "tests",
    ROOT / "lib",
    ROOT / "opencv",
    ROOT / "toolchain",
    ROOT / "include/python3.11",
    ROOT / "include/rkaiq",
    ROOT / "include/rockchip",
    ROOT / "include/rknn",
    ROOT / "include/pybind11",
    ROOT / "include/opencv4",
    ROOT / "include/rknn_box_priors.h",
    ROOT / "include/visiong/c_compat",
    ROOT / "include/visiong/assets",
    ROOT / "include/visiong/python",
    ROOT / "include/visiong/common/string_utils.h",
    ROOT / "include/visiong/core/logger.h",
    ROOT / "include/visiong/core/log_filter.h",
    ROOT / "include/visiong/modules/font_support.h",
    ROOT / "include/visiong/modules/gui_render_utils.h",
    ROOT / "include/visiong/modules/http_socket_utils.h",
    ROOT / "include/visiong/modules/venc_utils.h",
    ROOT / "include/visiong/npu/npu_common.h",
    ROOT / "vendor",
    ROOT / "3rdparty/rknpu2",
    ROOT / "3rdparty/allocator",
    ROOT / "3rdparty/ive",
    ROOT / "3rdparty/librga",
    ROOT / "3rdparty/rockchip_samples",
    ROOT / "src/npu/trackers/nanotrack",
    ROOT / "src/npu/nanotrack",
    ROOT / "3rdparty/CMakeLists.txt",
    ROOT / "3rdparty/allocator/drm",
    ROOT / "3rdparty/quirc/demo",
    ROOT / "3rdparty/quirc/tests",
    ROOT / "3rdparty/media-server/.github",
    ROOT / "3rdparty/media-server/libdash",
    ROOT / "3rdparty/media-server/libhls",
    ROOT / "3rdparty/media-server/libmkv",
    ROOT / "3rdparty/media-server/libmpeg",
    ROOT / "3rdparty/media-server/librtmp",
    ROOT / "3rdparty/media-server/libsip",
    ROOT / "3rdparty/media-server/librtp/test",
    ROOT / "3rdparty/media-server/librtp/rtpext",
    ROOT / "3rdparty/media-server/librtsp/test",
    ROOT / "3rdparty/media-server/librtsp/source/client",
    ROOT / "3rdparty/media-server/librtsp/source/sdp",
    ROOT / "3rdparty/media-server/librtsp/source/utils",
]

ALLOWED_TOP_LEVEL = {
    ".git",
    ".github",
    ".gitattributes",
    ".gitignore",
    "3rdparty",
    "CMakeLists.txt",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "examples",
    "THIRD_PARTY_NOTICES.md",
    "build.sh",
    "cmake",
    "docs",
    "include",
    "licenses",
    "release",
    "scripts",
    "src",
}

problems: list[str] = []
for path in EXPECTED_PRESENT:
    if not path.exists():
        problems.append(f"missing required path: {path.relative_to(ROOT)}")
for path in EXPECTED_ABSENT:
    if path.exists():
        problems.append(f"path should have been removed: {path.relative_to(ROOT)}")

legacy_cc = sorted((ROOT / "src").rglob("*.cc"))
for path in legacy_cc:
    problems.append(f"legacy .cc source remains under src/: {path.relative_to(ROOT)}")

for pattern in ("rk_*.h", "sample_comm*.h", "rtsp_demo.h", "rknn_api.h"):
    for path in sorted((ROOT / "include").rglob(pattern)):
        problems.append(f"third-party SDK header leaked into include/: {path.relative_to(ROOT)}")

for path in sorted((ROOT / "include").glob("*.h")):
    problems.append(f"top-level public header should live under include/visiong/: {path.relative_to(ROOT)}")

for path in ROOT.iterdir():
    if path.name not in ALLOWED_TOP_LEVEL:
        problems.append(f"unexpected top-level path present: {path.relative_to(ROOT)}")

if problems:
    for problem in problems:
        print(f"[FAIL] {problem}")
    sys.exit(1)

print("[OK] repository layout matches the audited public-release build set")
