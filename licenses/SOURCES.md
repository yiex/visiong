# License Source References

Collected on: 2026-03-13

## In-tree license texts

- Apache-2.0: `licenses/Apache-2.0.txt`
- LGPL-2.1 reference text: `licenses/LGPL-2.1.txt`
- media-server MIT: local `3rdparty/media-server/LICENSE`
- quirc ISC: local `3rdparty/quirc/LICENSE`
- pybind11 BSD-3: local `3rdparty/pybind11/LICENSE` (source text mirrored from `licenses/BSD-3-Clause-pybind11.txt`)
- Nuklear dual license: local `3rdparty/nuklear/LICENSE` (source text mirrored from `licenses/Dual-MIT-or-Public-Domain-nuklear.txt`)
- stb dual license: local `3rdparty/stb/LICENSE` (source text mirrored from `licenses/Dual-MIT-or-Public-Domain-stb.txt`)

## Reference-only license texts for external staged dependencies

- official OpenCV 4.x source build: local `licenses/OpenCV-4.x-LICENSE.txt` plus `licenses/BSD-3-Clause-OpenCV-header-notice.txt` for preserved historical header notices
- Python PSF license: local `licenses/PSF-2.0-Python-3.11.txt`
- RKNN SDK license reference: local `licenses/RKNN-SDK-LICENSE.txt`
- Rockchip sample BSD-style notice reference: local `licenses/BSD-3-Clause-rockchip-samples.txt`

## Repository-local build helper content

- Minimal target Python 3.11 sysroot subset: `3rdparty/target_python/`

## Dependency provenance used by the release workflow

The GitHub Actions release workflow is designed around these upstream sources:

- Luckfox SDK: `LuckfoxTECH/luckfox-pico`
- RKNN runtime source tree: `airockchip/rknn-toolkit2` (`rknpu2/`)
- RGA source tree: `airockchip/librga`
- OpenCV package source: `opencv/opencv`

If you redistribute generated dependency bundles, include the corresponding upstream license texts or notices next to the artifact.

## Open-source candidate notes

The legacy product-key path was removed from this candidate, so the following
license texts were also removed together with their no-longer-used dependencies:

- `micro-ecc`
- `tiny-AES-c`
- legacy SHA-256 helper used only by the removed key-verification flow
