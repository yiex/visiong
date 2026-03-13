<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Dependency Redistribution Matrix

Last reviewed: 2026-03-14

This matrix describes how VisionG treats each dependency class for public releases.
It is a release-engineering guide, not legal advice.

## Public release baseline

Default public release posture:

- publish source plus VisionG binary packages
- do not publish staged SDK bundles
- do not publish staged vendor libraries
- keep third-party source under its original license, with local `LICENSE` files in-tree

## Matrix

| Class | Example paths | Public source release | Public binary release | Default handling |
|---|---|---|---|---|
| VisionG project-owned code | `src/`, `include/visiong/`, `scripts/`, `docs/` | Yes | Yes, through `visiong_cpp.zip` and `visiong_python.zip` | Keep in repository |
| In-tree third-party source | `3rdparty/media-server/`, `3rdparty/pybind11/`, `3rdparty/quirc/`, `3rdparty/stb/`, `3rdparty/nuklear/` | Yes | Usually yes for linked outputs, subject to their original licenses and notices | Keep in repository with local `LICENSE` |
| In-tree prebuilt OpenCV subset | `3rdparty/opencv/` | No, excluded from the public source archive | Yes, as part of the prepared VisionG binary packaging flow | Keep in repository for CI and binary packaging |
| Project-owned replacements for SDK helper code | `src/common/internal/dma_alloc.cpp`, `src/modules/internal/ive_memory.cpp` | Yes | Yes | Keep in repository |
| Rockchip/Luckfox SDK headers and runtime libraries | `_stage/vendor/rockchip/include/`, `_stage/lib/` | No | Review separately file-by-file | Fetch/stage only |
| RKNN Toolkit2 headers and runtime libraries | `_stage/vendor/rknpu2/include/`, `_stage/lib/librknnmrt.so` | No | High risk, review separately file-by-file | Fetch/stage only |
| librga headers and runtime libraries | `_stage/vendor/librga/include/`, `_stage/lib/librga.so` | No by default | Review separately before publishing | Fetch/stage only |
| IVE headers and runtime libraries | `_stage/vendor/ive/include/`, `_stage/3rdparty/ive/ive/lib/` | No by default | Review separately before publishing | Fetch/stage only |
| target Python sysroot | `_stage/3rdparty/target_python/` | No | No public binary redistribution | Fetch/stage only |
| optional OpenCV source archive for local regeneration | `_deps/downloads/opencv-source.tar.gz` | No | No direct redistribution | Optional local use only |
| Local build outputs | `build/`, `visiong_cpp.zip`, `visiong_python.zip` | No | Prepared VisionG binary packages only | Ignore/remove before publishing unrelated artifacts |
| Staged dependency bundles | `_stage/`, `_deps/`, `vendor/`, `toolchain/` | No | No public binary redistribution | Ignore/remove before publishing |

## Non-negotiable repository rules

1. `include/` keeps project-owned public headers only.
2. Repository-local copies of SDK/sample helper trees stay out of the public tree.
3. Public GitHub releases may include the VisionG binary packages, but staged vendor libraries are not published and must be present on the target.
4. In-tree third-party directories must carry a local `LICENSE` file.
5. Use `scripts/audit_build_inputs.py`, `scripts/audit_dependency_policy.py`, and `scripts/audit_source_release.py` before publishing.
