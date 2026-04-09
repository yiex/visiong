# RV1106 KWS Performance And Porting Report

## Executive Summary

This migration is now in a good production-ready state for a compact keyword spotting demo on `RV1106`.

The final kept solution has three important properties:

- it is correct end to end on the real board
- it is materially faster than the original Python frontend path
- it stays conservative about risk and only keeps optimizations that were actually validated

What is now true:

- the `LowLevelNPU` stride bug for generic tensor input was fixed in `visiong`
- `visiong` now exposes a native `KwsLogMelFrontend`
- the native frontend numerically matches the training/reference pipeline
- the board-side KWS demo can choose between the Python frontend and the native frontend explicitly
- the native frontend delivers a large real board speedup without losing accuracy

## Final Kept Configuration

Repository worktree:

- local `visiong` checkout on `D:`

Artifacts and datasets:

- `D:\visiong-kws-demo`

Fork branch used for cloud builds:

- `yiex/visiong`
- branch `codex/kws-rv1106`

Cloud-built artifact that validated the native frontend fix:

- workflow run: `23943474500`
- artifact id: `6258357084`
- local download root: `D:\visiong-kws-demo\github_action_build_audio_fix`
- board override path: `/root/visiong_action_pkg_fix`

Board-side result files:

- Python frontend summary:
  - `D:\visiong-kws-demo\board_summary_action_python_frontend_fix.json`
- Native frontend summary:
  - `D:\visiong-kws-demo\board_summary_action_native_frontend_fix.json`

## Accuracy

Training-side result from `D:\visiong-kws-demo\train_kws_c64\training_summary.json`:

- best validation accuracy: `0.9244`
- test accuracy: `0.9212`
- ONNX accuracy: `0.9210`

Board-side result on the 120-sample evaluation bundle:

- Python frontend accuracy: `0.9083`
- native frontend accuracy: `0.9083`

That means the native frontend preserved accuracy exactly on the evaluation bundle that mattered for board validation.

## Final Performance

The important comparison is no longer "old runtime workaround vs new runtime workaround".

The important comparison is now:

- Python log-mel frontend on CPU
- native `visiong.KwsLogMelFrontend` on CPU, with the same NPU classifier

Both measurements below were taken on the real board with:

- `warmup_count=10`
- `--input-strategy array-only`
- the same `120`-sample board bundle
- the same RKNN model

### Python Frontend Baseline

Source:

- `D:\visiong-kws-demo\board_summary_action_python_frontend_fix.json`

Post-warmup metrics:

- frontend mean: `34.177 ms`
- input prepare mean: `0.484 ms`
- NPU mean: `0.342 ms`
- postprocess mean: `1.128 ms`
- total mean: `36.297 ms`
- total median: `36.261 ms`
- total P95: `37.154 ms`
- effective throughput: about `27.55 FPS`

### Native Frontend Final Result

Source:

- `D:\visiong-kws-demo\board_summary_action_native_frontend_fix.json`

Post-warmup metrics:

- frontend mean: `11.341 ms`
- input prepare mean: `0.488 ms`
- NPU mean: `0.347 ms`
- postprocess mean: `1.125 ms`
- total mean: `13.463 ms`
- total median: `13.345 ms`
- total P95: `13.961 ms`
- effective throughput: about `74.28 FPS`

### Net Gain

Using the native frontend instead of the Python frontend produced:

- about `3.01x` frontend speedup
- about `2.70x` end-to-end speedup
- no measurable accuracy loss on the board evaluation bundle

The NPU stage itself was never the bottleneck.

The real win came from moving audio feature extraction out of Python and into the library.

## Dedicated `KWS` Wrapper Result

After the native frontend and stride-safe `LowLevelNPU` path were both stable, I added a dedicated high-level `KWS` class.

What it keeps inside the library:

- log-mel frontend
- tensor upload
- RKNN execution
- softmax
- label mapping
- WAV parsing for the common PCM16 case

Validated cloud-built artifact:

- workflow run: `23952785354`
- artifact id: `6261923551`
- local artifact root: `D:\visiong-kws-demo\github_action_build_kws_wrapper`
- board override path: `/root/visiong_action_pkg_kws`

Board result from the dedicated wrapper:

- source summary:
  - `D:\visiong-kws-demo\kws_wrapper_eval.json`
- evaluation samples: `120`
- accuracy: `0.9083`
- post-warmup wall mean: `11.012 ms`
- post-warmup wall median: `10.020 ms`
- post-warmup wall P95: `19.803 ms`

Compared with the already-optimized native frontend demo path:

- native demo post-warmup total mean: `13.463 ms`
- dedicated `KWS` wrapper post-warmup wall mean: `11.012 ms`

That improvement is exactly why a dedicated class is worth adding:

- fewer Python-to-C++ crossings
- no Python-side softmax
- no Python-side label lookup
- a smaller and safer API surface for normal use

## Is This The Best Solution?

### Best validated solution kept in the codebase

Yes.

This is the best solution that was both:

- measurably faster on the board
- safe enough to keep

The final kept optimization set is:

- stride-safe `LowLevelNPU` input handling
- native `KwsLogMelFrontend`
- correct `log(mel + epsilon)` semantics
- existing low-risk NEON use where it already proved valuable

### What I tried and intentionally did not keep

I also tried pushing further by extending ARM NEON into more inner loops inside `KwsFrontend.cpp`, specifically:

- mel projection dot-product loops
- feature normalization loops

That experimental version compiled, passed static checks, and still matched the reference numerically.

However, repeated real-board measurements did not show a stable, reproducible end-to-end win:

- median latency stayed roughly flat
- means and P95s were dominated by board-side scheduling spikes
- the extra code complexity was not justified by the measured benefit

Because of that, that extra inner-loop NEON expansion was reverted.

This was the right call for safety and maintainability.

So the honest answer is:

- this is the best validated solution
- it is not the theoretical last word on optimization
- but it is the best point on the speed / safety / maintainability tradeoff curve that was proven on the hardware

## Was NEON Used?

### Yes, but selectively

NEON is used in the kept final solution.

Relevant retained paths:

- `src/audio/KwsFrontend.cpp`
  - window multiplication uses an ARM NEON path when available
- `src/core/RgaHelper.cpp`
  - stride-safe row copies use the existing NEON-aware `copy_data_with_stride()` helper on ARM

This means the final native frontend is not "pure scalar C++ only".

### What is not yet heavily NEON-specialized

The final kept code does not include a large custom NEON rewrite of:

- the FFT core
- the full mel projection stage
- the whole normalization stage

That was a deliberate choice.

A larger NEON rewrite is still possible in the future, but it was not retained because the measured board benefit was not yet convincing enough.

## Main Bugs And Pitfalls

## 1. `LowLevelNPU` generic input was not stride-safe

Symptom:

- host ONNX looked fine
- board inference produced wrong classes
- no obvious exception pointed at the real cause

Root cause:

- the RKNN tensor was `NHWC`
- logical width was `40`
- runtime `w_stride` was `48`
- generic input code wrote rows linearly instead of honoring the padded stride

Fix:

- patch `LowLevelNPU` generic input writes to use stride-aware copies
- keep a demo-side compatibility mode for already-installed old runtimes

Lesson:

- for RKNN pass-through tensors, never assume logical width equals runtime row stride

## 2. Native frontend used the wrong epsilon semantics

Symptom:

- native frontend was much faster
- but accuracy dropped from `0.9083` to `0.8917`
- feature comparison showed early mel bins collapsing to the same constant

Root cause:

- Python/training pipeline used:
  - `log(mel + epsilon)`
- native C++ frontend incorrectly used:
  - `log(max(epsilon, mel))`

Why that mattered:

- low-energy mel bins below `epsilon` were all clamped to one identical value
- after normalization, that created visible feature drift
- the first few feature values became repeated constants, which matched the observed board-side mismatch exactly

Fix:

- change the native frontend to:
  - `std::log(energy + config_.epsilon)`

Measured validation after the fix:

- frontend comparison max absolute error fell to about `7.37e-06`
- board accuracy returned to `0.9083`

Lesson:

- tiny-looking numerical differences in audio preprocessing can absolutely move an embedded classifier
- audio frontend math must match training semantics exactly, not approximately

## 3. Board timing noise can mislead optimization decisions

Symptom:

- repeated board runs showed occasional huge first-sample or background spikes
- those spikes inflated mean latency and P95

Fix:

- always report warmup-aware metrics
- use median and P95, not only a single average
- compare multiple runs before keeping low-level micro-optimizations

Lesson:

- on embedded Linux boards, a micro-optimization that does not survive repeated real-board measurement should not be kept

## 4. GitHub Actions on a fresh fork needed explicit re-registration

Symptom:

- workflow files existed in the fork
- the Actions API initially behaved as if there were no workflows

Fix:

- push the working branch
- touch and push the fork default branch so GitHub re-indexes workflow files
- then dispatch builds from the fork

Lesson:

- on a new fork, do not assume workflow registration is immediate

## 5. WSL2 and proxy setup mattered for reproducibility

What worked best:

- keep all large datasets and artifacts on `D:`
- use WSL2 for downloads and board-side automation glue
- route heavy downloads through the host Clash proxy

Lesson:

- for this machine, WSL2 is the reliable path for both large downloads and board automation

## Validation Checklist

The final kept solution was validated in these ways:

1. training and ONNX metrics were recorded locally
2. RKNN conversion completed successfully for `rv1106`
3. fixed cloud-built library artifact was downloaded from GitHub Actions
4. the artifact was deployed to a dedicated board override directory
5. Python frontend and native frontend were both run against the same board evaluation bundle
6. native frontend features were compared directly against the Python reference frontend on the board

The direct frontend comparison on the board produced only tiny residual float error:

- max absolute error: about `7.37e-06`
- mean absolute error: about `1.2e-07`

That is close enough to treat the two frontend implementations as numerically aligned for this deployment.

## Files Worth Reading

Main implementation:

- `include/visiong/audio/KwsFrontend.h`
- `src/audio/KwsFrontend.cpp`
- `src/python/bind_audio.cpp`
- `scripts/lowlevelnpu_kws/board_kws_demo.py`

Board-side measured summaries:

- `D:\visiong-kws-demo\board_summary_action_python_frontend_fix.json`
- `D:\visiong-kws-demo\board_summary_action_native_frontend_fix.json`

Migration notes:

- `docs/KWS_RV1106_WORKLOG.md`
- `docs/LOWLEVELNPU_KWS_TUTORIAL.md`

## Final Recommendation

For the current project state, the recommended deployment path is:

- keep the `LowLevelNPU` stride fix
- keep the native `KwsLogMelFrontend`
- use the dedicated `KWS` class as the default Python API for keyword spotting
- keep the exact `log(mel + epsilon)` frontend math
- keep the board demo's explicit runtime mode selection
- do not keep extra low-level NEON experiments unless they show a stable real-board win

If a future phase needs more speed than the current `13.463 ms` post-warmup mean, the next sensible targets are:

1. more aggressive native audio capture / PCM16 ingestion, so fewer Python-side copies happen before frontend compute
2. a better-isolated frontend microbenchmark on the board, so future NEON work can be judged without scheduler noise
3. only after that, deeper FFT / mel-specific kernel work

That keeps the codebase clean while still delivering a very meaningful real-world speedup today.
