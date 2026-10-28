<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Third-Party Notices for VisionG

Last updated: 2026-10-28

VisionG project-owned code is distributed under GNU LGPL-3.0-or-later.
Third-party components keep their original licenses and are not relicensed by LGPL.

## In-tree third-party source components

| Component | Path | License | Notes |
|---|---|---|---|
| AprilTag | `3rdparty/apriltag/` | BSD-2-Clause | Fiducial marker detection, official AprilRobotics C implementation v3.4.5 |
| pybind11 | `3rdparty/pybind11/` | BSD-3-Clause | Python binding layer, local `LICENSE` included |
| media-server | `3rdparty/media-server/` | MIT | RTSP/FLV/MOV helpers |
| OpenCV prebuilt subset | `3rdparty/opencv/` | Apache-2.0 plus bundled third-party notices | Repository-local RV1106 OpenCV subset for CI/binary packaging; not part of the public source archive |
| quirc | `3rdparty/quirc/` | ISC | QR code detection/decoding |
| stb | `3rdparty/stb/` | MIT or Public Domain | image load/save helpers, local `LICENSE` included |
| Nuklear | `3rdparty/nuklear/` | MIT or Public Domain | immediate-mode GUI, local `LICENSE` included |
| Rockchip YOLO26 example | `src/npu/yolo26.cpp`, `src/npu/internal/yolo26_decode.h`, `scripts/convert_yolo26.py` | Apache-2.0 | Adapted from airockchip/rknn_model_zoo commit `837db8e992815ff8647eadb709f07067f4bc3a71`; original copyright retained, VisionG modifications identified in each file. License: `licenses/Apache-2.0.txt`. |

The YOLO26 pretrained model is a separate Ultralytics artifact, not Rockchip's
Apache-2.0 code. Its embedded metadata identifies AGPL-3.0 and Ultralytics as
the author. Conversion to RKNN does not change the upstream model's license.
Model origin, hashes, embedded metadata and license text accompany generated
model artifacts. See `docs/YOLO26.md`.

## Project-owned replacements and glue

The following files are project-owned compatibility layers, not vendored SDK/sample source:

- `src/common/internal/dma_alloc.cpp`
- `src/modules/internal/ive_memory.cpp`
- `src/c_compat/http-parser.c`
- `src/python/license_guard.cpp`

IVE interface conventions and background-model parameter defaults are checked
against the Rockchip examples distributed in
[LuckfoxTECH/luckfox-pico](https://github.com/LuckfoxTECH/luckfox-pico/tree/main/media/ive/ive/simulator).
The VisionG wrappers remain project-owned glue. Vendor headers, libraries and
their original copyright and license terms remain external dependencies.

`scripts/patch_ive_lk_runtime.py` creates a separate corrected copy of one
verified external IVE runtime build. It does not distribute that runtime or
change its copyright, license or redistribution terms. Original and corrected
vendor binaries are excluded from the public source archive. The correction
and measured scope are documented in `docs/IVE_LK_RUNTIME.md`.

## External build-time dependencies not bundled in source releases

These inputs may be fetched or staged during CI/local builds, but are intentionally not part of the public source archive:

- Luckfox/Rockchip SDK headers and runtime libraries
- RKNN Toolkit2 headers and runtime libraries
- librga headers and runtime libraries
- IVE headers and runtime libraries
- optional OpenCV source archive for local regeneration
- repository-local OpenCV prebuilt subset under `3rdparty/opencv/`

For CI and convenience builds, this repository also carries a minimal **target Python sysroot subset** for Python 3.11 under `3rdparty/target_python/`.
It is used only for cross-building the Python extension and is excluded from the public source-release archive.

Those components keep their original licenses and redistribution terms.
Do not claim LGPL relicenses them.

## Redistribution boundary

High-risk items that should be reviewed separately before any binary redistribution:

- staged SDK/library bundles under `_stage/`
- local `build/` outputs linked against vendor SDK libraries
- locally produced runtime bundles such as `visiong_runtime_bundle.zip`

Recommended public-release practice:

- publish a source archive created by `./scripts/create_source_release.sh`,
- publish the prepared dynamic-linked VisionG binary package only,
- do not publish staged vendor libraries, and
- use `docs/BINARY_REDISTRIBUTION_POLICY.md` as the allowlist/review list for staged binaries and vendor-linked outputs.
