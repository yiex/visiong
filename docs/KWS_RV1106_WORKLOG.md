# KWS RV1106 Worklog

## Goal

Port a mature keyword spotting pipeline to `visiong`, validate it on the real `RV1106` board, and keep the final implementation both fast and safe.

## Working Layout

Repository worktree:

- local `visiong` checkout on `D:`

Large data and artifacts:

- `D:\visiong-kws-demo`

Repository:

- `https://github.com/yiex/visiong`

## Model Direction

- dataset: Google Speech Commands v0.02
- task: 12-way keyword spotting
- model: compact DS-CNN
- target runtime: `visiong.LowLevelNPU`
- target platform: `rv1106`

## Major Completed Steps

- downloaded Speech Commands to `D:\visiong-kws-demo\data`
- prepared features, calibration tensors, and a board eval bundle under `D:\visiong-kws-demo\prepared_kws`
- trained DS-CNN variants and selected the 64-channel model
- converted the winning ONNX model to RKNN for `rv1106`
- validated the RKNN model on the real board
- found and fixed a stride-handling bug in `LowLevelNPU` generic tensor input
- added explicit board-side input strategy control:
  - `compat`
  - `array-only`
  - `packed-fallback`
- added CI guards for the stride fix
- added a native `visiong.KwsLogMelFrontend`
- found and fixed a frontend math mismatch between training and C++ inference
- validated the native frontend against the Python reference directly on the board
- documented the migration, performance, and pitfalls

## Important Validation Milestones

### Stride fix validation

The original KWS deployment exposed a real bug in `LowLevelNPU` when the RKNN input tensor had:

- logical width `40`
- runtime `w_stride` `48`

Result:

- bug fixed in the library
- cloud-built `_visiong.so` validated on the board
- plain `set_input_array()` now works correctly for this model on the fixed runtime

### Native frontend validation

After adding the native audio frontend, the first board run was faster but accuracy regressed.

Root cause:

- C++ used `log(max(epsilon, mel))`
- training/reference path used `log(mel + epsilon)`

After fixing that:

- native frontend max feature error dropped to about `7.37e-06`
- board accuracy returned to `0.9083`
- native frontend kept its large speed advantage

## Cloud Build History

### Fixed native frontend artifact

- workflow: `build-binary-artifact`
- run id: `23943474500`
- artifact id: `6258357084`
- local artifact root:
  - `D:\visiong-kws-demo\github_action_build_audio_fix`
- board override directory:
  - `/root/visiong_action_pkg_fix`

Board results from that artifact:

- Python frontend accuracy: `0.9083`
- native frontend accuracy: `0.9083`
- Python post-warmup total mean: `36.297 ms`
- native post-warmup total mean: `13.463 ms`
- native post-warmup total median: `13.345 ms`
- native post-warmup total P95: `13.961 ms`

### Experimental extra NEON attempt

- workflow: `build-binary-artifact`
- run id: `23951414494`
- artifact id: `6261386543`
- local artifact root:
  - `D:\visiong-kws-demo\github_action_build_audio_neon`
- board override directory:
  - `/root/visiong_action_pkg_neon`

What was tested:

- additional NEON expansion for inner mel-projection and normalization loops

Outcome:

- correctness stayed fine
- repeated board runs did not show a stable, reproducible latency win
- the extra code path was reverted

This was intentionally not kept.

### Dedicated `KWS` wrapper validation

- workflow: `build-binary-artifact`
- run id: `23952785354`
- artifact id: `6261923551`
- local artifact root:
  - `D:\visiong-kws-demo\github_action_build_kws_wrapper`
- board override directory:
  - `/root/visiong_action_pkg_kws`

Board validation result:

- summary:
  - `D:\visiong-kws-demo\kws_wrapper_eval.json`
- evaluation samples: `120`
- accuracy: `0.9083`
- post-warmup wall mean: `11.012 ms`
- post-warmup wall median: `10.020 ms`
- post-warmup wall P95: `19.803 ms`

Single-sample smoke result:

- summary:
  - `D:\visiong-kws-demo\kws_wrapper_smoke.json`
- returned type: `KWSResult`
- predicted label: `yes`
- confidence: about `0.9997`

## Current Final Recommendation

Keep:

- the stride-safe `LowLevelNPU` fix
- the native `KwsLogMelFrontend`
- the exact `log(mel + epsilon)` frontend semantics
- explicit runtime mode selection in the board demo

Do not currently keep:

- extra inner-loop NEON expansion that lacks a stable measured win on the board

## Current State

This KWS migration is successful.

The board now has a clean validated path for:

- open-dataset training
- ONNX export
- RKNN conversion
- `LowLevelNPU` inference
- native audio frontend acceleration
- real-board timing and accuracy validation

The next performance work, if needed, should focus on:

- reducing Python-side audio ingestion overhead further
- isolating frontend-only microbenchmarks on the board
- only then considering deeper FFT / mel kernel specialization
