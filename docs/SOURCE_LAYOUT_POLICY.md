<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Source Layout Policy

## Rules

1. Project-owned C++ implementation files use `.cpp`.
2. Project-owned C compatibility helpers stay in `src/c_compat/` and use `.c`.
3. Third-party preserved source code remains under `3rdparty/`.
4. NanoTrack internal code stays under `src/npu/internal/tracking/`, with the public wrapper in `src/npu/NanoTrack.cpp`.
5. Public project headers stay under `include/visiong/`.
6. C helper headers used only by bundled sources stay under `src/c_compat/include/`.
7. Third-party SDK headers must not be stored under `include/`.
8. Build-time SDK headers are resolved from `--deps-root` or `--sdk-root`.
9. Internal helper headers live under `src/**/internal/` and are not exported from `include/visiong/`.
10. `Camera` capture/runtime logic and `ImageBuffer` value semantics stay in separate translation units (`src/core/Camera.cpp` and `src/core/ImageBuffer.cpp`).
11. Project-owned replacements for SDK helper/sample code stay under `src/**/internal/`, not under `3rdparty/`.
12. In-tree third-party directories must include a local `LICENSE` file.

## Current structure notes

- `src/npu/internal/tracking/` groups tracker internals under the shared tracking namespace instead of a one-off module folder.
- `src/npu/internal/models/` keeps model-specific internal headers.
- `src/{common,core,modules,python}/internal/` keeps project helper headers and internal-only sources out of the public include tree.
- GUIManager stays split between constructor/impl setup (src/modules/GUIManager.cpp), canonical API forwarding (src/modules/GUIManagerApi.cpp), low-level bridge helpers (src/modules/GUIManagerBridge.cpp), compatibility wrappers (src/modules/GUIManagerCompat.cpp), and internal touch/render/widget support under src/modules/internal/.
- `src/common/internal/dma_alloc.*` and `src/modules/internal/ive_memory.*` are project-owned replacements for earlier SDK-derived helper implementations.
- `include/visiong/common/vision_types.h` keeps reusable public result/value types out of heavyweight container headers.
- Release dependency staging writes SDK headers to `_stage/vendor/rockchip/include`, `_stage/vendor/librga/include`, `_stage/vendor/ive/include`, and `_stage/vendor/rknpu2/include`.
- Repository-local SDK header snapshots and SDK-derived sample source trees are not stored in this tree.

## Release policy

- Public source releases should not bundle staged SDK headers, staged libraries, or SDK-derived helper/source snapshots by default.
- Use `scripts/create_source_release.sh` for source archives.
- Keep `THIRD_PARTY_NOTICES.md`, `licenses/`, `scripts/dependency_policy.json`, and `scripts/binary_release_policy.json` aligned with the released file set.
- Keep project-owned files carrying SPDX headers that match the project license.
- Keep the generated source archive passing `scripts/audit_source_release.py`.

