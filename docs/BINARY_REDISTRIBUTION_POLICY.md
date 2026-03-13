<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Binary Redistribution Policy

Last reviewed: 2026-03-12

This document defines the current binary-release posture for VisionG.
It is a release-engineering control, not legal advice.

## Default posture

1. Public GitHub releases publish VisionG binary packages.
2. Staged vendor libraries are not published.
3. The default binary artifacts are `visiong_cpp.zip` and `visiong_python.zip`.
4. `visiong_cpp.zip` contains VisionG public headers under `include/visiong/`, `lib/`, `cmake/`, notices, and the component manifest.
5. `visiong_python.zip` keeps `_visiong.so` and `visiong.py` at the archive root.
6. The minimal OpenCV subset used by VisionG is linked statically into the published binaries.
7. Required external Rockchip runtime libraries must still be present on the target.
8. Public binary packages do not copy staged Rockchip SDK header trees into the release archives.

## Staged external libraries

| Filename | Staged path | Default public release | Classification | Reason |
|---|---|---|---|---|
| `librga.so` | `lib/librga.so` | No | review-required | Public upstream exists, but staged binary provenance and notice set must be reviewed before any redistribution |
| `librga.a` | `lib/librga.a` | No | review-required | Static vendor binary; review provenance and downstream obligations before any redistribution |
| `librkaiq.so` | `lib/librkaiq.so` | No | forbidden-by-default | Vendor SDK binary; keep out of public releases |
| `librkaiq.a` | `lib/librkaiq.a` | No | forbidden-by-default | Vendor SDK static library; keep out of public releases |
| `librknnmrt.so` | `lib/librknnmrt.so` | No | forbidden-by-default | RKNN runtime is distribution-sensitive; do not publish it directly |
| `librknnmrt.a` | `lib/librknnmrt.a` | No | forbidden-by-default | RKNN runtime static library is distribution-sensitive; do not publish it directly |
| `librockchip_mpp.so.1` | `lib/librockchip_mpp.so.1` | No | forbidden-by-default | Vendor media stack binary; keep out of public releases |
| `librockchip_mpp.so.0` | `lib/librockchip_mpp.so.0` | No | forbidden-by-default | Vendor media stack binary; keep out of public releases |
| `librockchip_mpp.a` | `lib/librockchip_mpp.a` | No | forbidden-by-default | Vendor media stack static library; keep out of public releases |
| `librockit.so` | `lib/librockit.so` | No | forbidden-by-default | Vendor middleware binary; keep out of public releases |
| `librockit.a` | `lib/librockit.a` | No | forbidden-by-default | Vendor middleware static library; keep out of public releases |
| `librockiva.so` | `lib/librockiva.so` | No | forbidden-by-default | Vendor IVA runtime; keep out of public releases |
| `librockiva.a` | `lib/librockiva.a` | No | forbidden-by-default | Vendor IVA static library; keep out of public releases |
| `librtsp.a` | `lib/librtsp.a` | No | forbidden-by-default | Prebuilt static RTSP library is unnecessary for release because source is already retained in-tree |
| `librve.so` | `3rdparty/ive/ive/lib/librve.so` | No | forbidden-by-default | IVE vendor runtime; keep out of public releases |
| `librve.a` | `3rdparty/ive/ive/lib/librve.a` | No | forbidden-by-default | IVE vendor static library; keep out of public releases |
| `libivs.so` | `3rdparty/ive/ive/lib/libivs.so` | No | forbidden-by-default | IVE vendor runtime; keep out of public releases |
| `libivs.a` | `3rdparty/ive/ive/lib/libivs.a` | No | forbidden-by-default | IVE vendor static library; keep out of public releases |

## VisionG build outputs

| Filename | Path | Default public release | Classification | Reason |
|---|---|---|---|---|
| `_visiong.so` | `build/_visiong.so` | Yes | public-release | Published inside `visiong_python.zip` as the Python extension; the minimal OpenCV subset is linked statically, while required Rockchip runtime libraries stay external to the release |
| `visiong.py` | `build/visiong.py` | Yes | public-release | Published inside `visiong_python.zip` as the Python loader entry point |
| `libvisiong.so` | `build/libvisiong.so` | Yes | public-release | Published inside `visiong_cpp.zip` as the shared C/C++ library; the minimal OpenCV subset is linked statically, while required Rockchip runtime libraries stay external to the release |
| `libvisiong.a` | `build/libvisiong.a` | Yes | public-release | Published inside `visiong_cpp.zip` as the static C/C++ library for downstream integration; staged vendor libraries are still not bundled |
| `visiong_cpp.zip` | `visiong_cpp.zip` | Yes | public-release | Primary public C/C++ release archive containing headers, libraries, CMake package files, and notices |
| `visiong_python.zip` | `visiong_python.zip` | Yes | public-release | Primary public Python release archive containing `_visiong.so`, `visiong.py`, and notices |

## Allowed default GitHub release artifacts

Only these artifacts are expected on a default public release:

- `visiong_cpp.zip`
- `visiong_python.zip`

## Manual CI artifact path

Use `.github/workflows/binary-artifact.yml` for non-tag preview builds.
That workflow uploads Actions artifacts only and does not create a GitHub release.

## Required checks before any binary publication

1. Confirm exact upstream provenance for each staged binary file.
2. Confirm the license text and any notice obligations for the exact binary package or SDK ref.
3. Confirm the file is not covered by restrictive proprietary wording or SDK-only redistribution terms.
4. Confirm the resulting VisionG artifact does not bundle any staged vendor library still marked forbidden-by-default.
5. Keep `scripts/audit_binary_release_policy.py` passing before changing release posture.
