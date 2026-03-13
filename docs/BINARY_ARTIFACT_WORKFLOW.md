<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Binary Artifact Workflow

This workflow exists for non-tag preview builds and controlled delivery.
It is not the tag-driven public release workflow.
Its outputs are workflow artifacts only.

## Workflow file

- `.github/workflows/binary-artifact.yml`

## What it does

1. Fetches the external SDK/toolchain inputs.
2. Reuses the vendored minimal OpenCV subset from `3rdparty/opencv/`.
3. Stages the external inputs into `_stage/`.
4. Builds VisionG with `--no-package`.
5. Enables the CI fallback ISP controller implementation to avoid SDK API drift breaking packaging-only builds.
6. Allows `visiong_enable_ive`, `visiong_enable_npu`, and `visiong_enable_gui` workflow inputs for component trimming.
7. Creates `visiong_cpp.zip` and `visiong_python.zip` with `scripts/create_release_archives.sh`.
8. Uploads the result as workflow artifacts only.

## What it does not do

- It does not create a GitHub release.
- It does not publish staged vendor libraries.
- It does not replace `.github/workflows/release.yml`, which creates the tagged public release assets.
- It does not rebuild OpenCV from source during CI.

## Expected artifact contents

- `visiong_cpp.zip`
- `visiong_python.zip`

`visiong_cpp.zip` contains the normal C/C++ release layout, including `include/visiong/`, `lib/`, `cmake/`, notices, and the VisionG component manifest.
It intentionally excludes staged Rockchip SDK header trees.

`visiong_python.zip` keeps `_visiong.so` and `visiong.py` directly at the archive root, alongside notices and the VisionG component manifest.

## When to use it

Use this workflow when you need the same binary packages on a non-tag run for board-side validation or controlled sharing.
It remains workflow artifacts only.

## Relation to the tag release workflow

The tag-driven workflow in `.github/workflows/release.yml` follows the same build path,
then additionally creates the public source archive, runs release audits, writes the release body,
uploads workflow artifacts, and publishes the GitHub release for tags.
