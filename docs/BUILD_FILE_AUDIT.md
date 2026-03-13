<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Build File Audit

This repository keeps only files that are either:

1. direct build inputs,
2. release/compliance metadata, or
3. automation support files.

## Kept as direct build inputs

- `CMakeLists.txt`
- `cmake/rv1106_toolchain.cmake`
- `build.sh`
- `release/release_components.cmake`
- `src/core/`, `src/modules/`, `src/c_compat/`, `src/npu/`, `src/python/`
- `src/c_compat/include/`
- `src/{common,core,modules,python}/internal/`
- `src/npu/internal/tracking/`
- `src/modules/internal/gui_nuklear_config.h`
- `src/modules/internal/gui_nuklear_style.*`
- `3rdparty/quirc/`
- `3rdparty/media-server/` subsets listed in `CMakeLists.txt`
- `3rdparty/pybind11/`, `3rdparty/stb/`, `3rdparty/nuklear/`
- `3rdparty/opencv/`
- `3rdparty/target_python/`
- `include/visiong/` public project headers only

## Project-owned internal replacements

These replace earlier SDK-derived helper/source snapshots:

- `src/common/internal/dma_alloc.*`
- `src/modules/internal/ive_memory.*`

## Compliance-critical retained metadata

- `3rdparty/media-server/LICENSE`
- `3rdparty/quirc/LICENSE`
- `3rdparty/pybind11/LICENSE`
- `3rdparty/stb/LICENSE`
- `3rdparty/nuklear/LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `licenses/`
- `scripts/audit_build_inputs.py`
- `scripts/audit_dependency_policy.py`
- `scripts/audit_spdx_headers.py`
- `scripts/audit_binary_release_policy.py`
- `scripts/audit_source_release.py`
- `scripts/dependency_policy.json`
- `scripts/binary_release_policy.json`
- `scripts/prepare_binary_artifact.sh`
- `scripts/create_release_archives.sh`
- `docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md`
- `docs/BINARY_REDISTRIBUTION_POLICY.md`
- `docs/BINARY_ARTIFACT_WORKFLOW.md`
- `docs/GITHUB_ACTIONS_RELEASE.md`
- `.github/workflows/binary-artifact.yml`
- `.github/workflows/release.yml`

## External build inputs

These are intentionally not stored in the source tree. Provide them via `--sdk-root`, `--deps-root`, or explicit CMake paths:

- Rockchip SDK headers
- RKNN headers
- librga headers
- IVE headers
- vendor prebuilt libraries
- IVE shared/static libraries
- optional target Python sysroot overrides provided by external SDK trees
- optional official OpenCV source archive or source tree for local regeneration

## Removed on purpose

- repository-local SDK header snapshots under `include/`
- SDK-derived source/helper trees such as `3rdparty/allocator/`, `3rdparty/ive/`, `3rdparty/librga/`, and `3rdparty/rockchip_samples/`
- project examples and unit-test scaffolding
- upstream demo/test directories from vendored dependencies
- unused upstream build-system files (`Makefile`, `Android.mk`, Visual Studio/Xcode projects)
- legacy `src/*.cc` under project-owned code
- non-API helper headers under `include/visiong/`

## Repeatable checks

```bash
./scripts/audit_build_inputs.py
./scripts/audit_dependency_policy.py
./scripts/audit_binary_release_policy.py
./scripts/audit_spdx_headers.py
./scripts/audit_source_release.py /path/to/visiong-source.zip
```

The scripts verify required paths, forbidden paths, retained third-party license files, and that no repo-local SDK header snapshots or binary artifacts leak into the public source release.
