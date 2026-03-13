<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Include License Audit

Last reviewed: 2026-03-11

This note summarizes a practical header-license audit for VisionG.
It is not legal advice.

## Scope

- `include/**` (project-owned public API headers only)
- `src/c_compat/include/**` (project-owned internal C helper headers)
- staged external SDK headers under `_stage/vendor/**`
- externally provided SDK trees passed by `--sdk-root`

## Current project posture

- `include/` contains project-owned headers only.
- librga and IVE headers are no longer stored in the repository.
- SDK headers are consumed from `--deps-root` / `--sdk-root` during build.
- `scripts/create_source_release.sh` builds public archives from an allowlisted file set and audits the finished zip.

## Generally compatible with LGPL project code

When redistributed with proper notices and without relicensing claims:

- pybind11 headers (`BSD-3-Clause`)
- official OpenCV headers (`Apache-2.0` plus preserved OpenCV notices)
- Python 3.11 headers (`PSF-2.0`)
- many Rockchip MPI/RT headers marked dual-license (`GPL-2.0 WITH Linux-syscall-note OR Apache-2.0`)
- official librga headers when sourced from the public Apache-2.0 repository
- official IVE headers that carry clear Apache-2.0 notices

## High-risk headers (require explicit redistribution confirmation)

### A) RKNN SDK headers with restrictive proprietary wording

Examples:

- `_stage/vendor/rknpu2/include/rknn_api.h`
- `_stage/vendor/rknpu2/include/rknn_custom_op.h`
- `_stage/vendor/rknpu2/include/rknn_matmul_api.h`

### B) Some RKAIQ algorithm headers with restrictive wording

Examples under:

- `_stage/vendor/rockchip/include/rkaiq/algos/**`

Some files include wording such as "without written permission".

## Copyleft-bearing headers to track carefully

These are open source but add obligations when redistributed or modified:

- LGPL-2.1+: `_stage/vendor/rockchip/include/rkaiq/common/mediactl/mediactl.h`
- GPL-2.0+: `_stage/vendor/rockchip/include/rkaiq/iq_parser_v2/j2s/common.h`, `.../j2s.h`

## Recommended release posture

1. Keep project code under LGPL-3.0-or-later.
2. Do not claim LGPL relicenses external SDK headers or binaries.
3. Exclude staged SDK headers from public source archives unless you have explicit permission and provenance ready.
4. Keep `THIRD_PARTY_NOTICES.md` and `licenses/` aligned with what you actually publish.
