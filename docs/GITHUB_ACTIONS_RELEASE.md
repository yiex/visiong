<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# GitHub Actions Release Flow

This repository uses `.github/workflows/release.yml` to verify builds against staged external dependencies and publish the VisionG binary packages.

## Artifacts on a tagged release (`v*.*.*`)

- `visiong_cpp.zip`
- `visiong_python.zip`

## Workflow steps

1. Checkout VisionG source.
1. Checkout VisionG source.
2. Fetch Luckfox SDK, RKNN Toolkit2, and librga.
3. Run `scripts/stage_release_deps.sh` to stage dependencies into `_stage/`.
4. Build with `./build.sh --deps-root _stage --no-package`, plus optional `VISIONG_ENABLE_IVE/NPU/GUI` toggles from workflow inputs.
5. Create `visiong_cpp.zip` and `visiong_python.zip` with `scripts/create_release_archives.sh`.
6. Write release notes from the current build outputs.
7. Upload/publish the two binary packages.

## Expected staged layout

```text
_stage/
  lib/
  toolchain/arm-rockchip830-linux-uclibcgnueabihf/
  3rdparty/ive/ive/lib/
  3rdparty/target_python/
  vendor/rockchip/include/
  vendor/librga/include/
  vendor/ive/include/
  vendor/rknpu2/include/
  opencv/
```

`CMakeLists.txt` resolves SDK headers from the staged bundle, not from repository snapshots.

## Workflow dispatch behavior

- Tag push to `v*.*.*`: build, audit, upload artifacts, and publish/update the GitHub release for that tag.
- Manual `workflow_dispatch` without `release_tag`: build and upload artifacts only.
- Manual `workflow_dispatch` with `release_tag`: build the currently selected ref, refresh release notes, and update the GitHub release assets/body for that tag.
- Both workflows also accept `visiong_enable_ive`, `visiong_enable_npu`, and `visiong_enable_gui` so forks can trim components directly from the dispatch form.

## Variables

Optional repository variables for override:

- `LUCKFOX_SDK_REF`
- `RKNN_TOOLKIT2_REF`
- `LIBRGA_REF`
- `visiong_enable_ive`
- `visiong_enable_npu`
- `visiong_enable_gui`

## Compliance note

CI automation does not create redistribution rights.
The workflow publishes VisionG binary packages only. Staged vendor libraries are not published, and the required runtime libraries must be present on the target.
Validate upstream license terms for the exact refs/assets you fetch during CI.

For non-tag preview builds, use `.github/workflows/binary-artifact.yml`, which uploads workflow artifacts only and does not call any GitHub release action.
