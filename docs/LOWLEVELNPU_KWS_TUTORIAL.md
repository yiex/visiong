# LowLevelNPU Keyword Spotting Tutorial

This tutorial ports a mature keyword spotting pipeline to `visiong` and validates it on the target board with `LowLevelNPU`.

## What We Built

- Dataset: Google Speech Commands v0.02
- Task: 12-way keyword spotting
- Labels:
  - `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`
- Frontend:
  - 16 kHz mono
  - 1 second clip
  - 30 ms window
  - 20 ms hop
  - 49 frames
  - 40 log-mel bins
- Model:
  - compact DS-CNN-style network
  - `64` channels in the main depthwise-separable stack
- Runtime:
  - `visiong.LowLevelNPU`
  - target platform `rv1106`

## Why This Route

I deliberately chose a mature embedded KWS recipe:

- Speech Commands is a standard public dataset for limited-vocabulary speech recognition.
- DS-CNN is a proven small-footprint architecture for keyword spotting.
- The frontend is simple enough to run on-device with `numpy`, while the classifier itself runs on the NPU.

This gives a practical split:

- CPU does the audio frontend
- NPU does the classifier

That is a very common embedded deployment pattern.

In the final validated version, the CPU frontend can run in two modes:

- Python/`numpy` reference frontend
- native `visiong.KwsLogMelFrontend`

There is now also a dedicated high-level wrapper:

- `visiong.KWS`

That wrapper keeps the same native frontend and NPU path, but hides:

- feature extraction
- tensor upload
- RKNN execution
- softmax
- label mapping

So for normal Python use, this is now the recommended API.

## Repository and Data Layout

Repository worktree:

- local `visiong` checkout on `D:`

Large data and artifacts:

- `D:\visiong-kws-demo`

Main scripts:

- `scripts/lowlevelnpu_kws/common.py`
- `scripts/lowlevelnpu_kws/prepare_speech_commands.py`
- `scripts/lowlevelnpu_kws/train_kws_ds_cnn.py`
- `scripts/lowlevelnpu_kws/convert_kws_ds_cnn_to_rknn.py`
- `scripts/lowlevelnpu_kws/board_kws_demo.py`

Detailed engineering report:

- `docs/KWS_RV1106_PERFORMANCE_AND_PORTING_REPORT.md`

## Recommended Runtime API

For normal Python deployment, prefer the dedicated high-level wrapper:

```python
import numpy as np
import visiong as vg

kws = vg.KWS(
    "/root/visiong_kws_demo_c64/kws_ds_cnn.rknn",
    "/root/visiong_kws_demo_c64/board_bundle/labels.txt",
)

wav_result = kws.infer_wav("/root/visiong_kws_demo_c64/board_bundle/eval/yes/022cd682_nohash_0.wav")
print(wav_result.label, wav_result.score)

audio = np.zeros(kws.clip_samples, dtype=np.int16)
pcm_result = kws.infer(audio)
print(pcm_result.class_id, pcm_result.label, pcm_result.score)
```

Why this is the recommended API:

- it uses the native audio frontend automatically
- it keeps tensor upload and RKNN execution inside the library
- it does softmax and label mapping in C++
- it can read common PCM16 WAV files directly from C++
- it gives one error-handled entry point instead of several Python-side steps

`LowLevelNPU` is still useful for teaching, debugging, and unusual custom models, but `KWS` is the better default for keyword spotting.

## Step 1: Download Speech Commands to D:

WSL2 command:

```bash
export http_proxy=http://172.18.80.1:7890
export https_proxy=http://172.18.80.1:7890

mkdir -p /mnt/d/visiong-kws-demo/data
cd /mnt/d/visiong-kws-demo/data
wget -c -O speech_commands_v0.02.tar.gz \
  https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p speech_commands_v0.02
tar -xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
```

Why `storage.googleapis.com` instead of `download.tensorflow.org`:

- Under the current proxy setup, the latter produced a hostname mismatch certificate issue.
- The Google Storage URL is the same payload and worked correctly through Clash.

## Step 2: Prepare Features, Calibration Tensors, and a Board Eval Bundle

WSL2 command:

```bash
cd /mnt/d/visiong/scripts/lowlevelnpu_kws
python3 prepare_speech_commands.py \
  --raw-root /mnt/d/visiong-kws-demo/data/speech_commands_v0.02 \
  --output-root /mnt/d/visiong-kws-demo/prepared_kws \
  --board-per-label 10 \
  --calibration-count 128 \
  --overwrite
```

This script does all of the boring but necessary work:

- parses `validation_list.txt` and `testing_list.txt`
- groups the 10 target keywords
- samples an `unknown` class from the remaining words
- generates a `silence` class from `_background_noise_`
- computes log-mel features
- writes `train/val/test` feature tensors
- writes RKNN calibration `.npy` tensors
- exports a small `board_bundle` of WAV files for on-device verification

Prepared split counts from the actual run:

- Train:
  - keywords total: `30769`
  - `unknown`: `3692`
  - `silence`: `3692`
- Val:
  - keywords total: `3743`
  - `unknown`: `444`
  - `silence`: `444`
- Test:
  - keywords total: `4074`
  - `unknown`: `489`
  - `silence`: `489`

Relevant outputs:

- `D:\visiong-kws-demo\prepared_kws\features\train.npz`
- `D:\visiong-kws-demo\prepared_kws\features\val.npz`
- `D:\visiong-kws-demo\prepared_kws\features\test.npz`
- `D:\visiong-kws-demo\prepared_kws\calibration\quant_dataset.txt`
- `D:\visiong-kws-demo\prepared_kws\board_bundle`

## Step 3: Train the DS-CNN Model

I trained two variants:

- `48` channels
- `64` channels

The `64`-channel model won clearly, so it became the final one.

WSL2 command:

```bash
cd /mnt/d/visiong/scripts/lowlevelnpu_kws
python3 train_kws_ds_cnn.py \
  --prepared-root /mnt/d/visiong-kws-demo/prepared_kws \
  --output-root /mnt/d/visiong-kws-demo/train_kws_c64 \
  --epochs 22 \
  --batch-size 256 \
  --channels 64 \
  --overwrite
```

Actual result from the winning run:

- Best epoch: `18`
- Best validation accuracy: `0.9244`
- Test accuracy: `0.9212`
- ONNX accuracy: `0.9210`

Artifacts:

- `D:\visiong-kws-demo\train_kws_c64\kws_ds_cnn.pt`
- `D:\visiong-kws-demo\train_kws_c64\kws_ds_cnn.onnx`
- `D:\visiong-kws-demo\train_kws_c64\training_summary.json`

## Step 4: Convert ONNX to RKNN

WSL2 command:

```bash
cd /mnt/d/visiong/scripts/lowlevelnpu_kws
python3 convert_kws_ds_cnn_to_rknn.py \
  --onnx-model /mnt/d/visiong-kws-demo/train_kws_c64/kws_ds_cnn.onnx \
  --dataset /mnt/d/visiong-kws-demo/prepared_kws/calibration/quant_dataset.txt \
  --output-rknn /mnt/d/visiong-kws-demo/rknn_kws_c64/kws_ds_cnn.rknn
```

RKNN build result:

- target platform: `rv1106`
- quantization: enabled
- input dtype after quantization: `INT8`
- output dtype after quantization: `INT8`

Artifact:

- `D:\visiong-kws-demo\rknn_kws_c64\kws_ds_cnn.rknn`

## Step 5: Deploy and Run on the Board

Files needed on the board:

- the RKNN model
- the board bundle with WAV files and labels
- `common.py`
- `board_kws_demo.py`

The board-side demo supports:

- single WAV inference
- manifest-based batch evaluation
- explicit input strategy selection:
  - `compat`
  - `array-only`
  - `packed-fallback`
- warmup-aware latency statistics

The actual board validation used a 120-sample evaluation bundle:

- 10 WAV files per label
- 12 labels total

## Actual Board Runtime Result

Measured on the real board:

- Evaluation samples: `120`
- Board accuracy with the Python frontend: `0.9083`
- Board accuracy with the native frontend: `0.9083`

Fixed cloud-built runtime, Python frontend, `--input-strategy array-only`:

- post-warmup frontend mean: `34.177 ms`
- post-warmup input-prepare mean: `0.484 ms`
- post-warmup NPU mean: `0.342 ms`
- post-warmup end-to-end mean: `36.297 ms`
- post-warmup end-to-end median: `36.261 ms`
- post-warmup end-to-end P95: `37.154 ms`

Fixed cloud-built runtime, native frontend, `--input-strategy array-only`:

- post-warmup frontend mean: `11.341 ms`
- post-warmup input-prepare mean: `0.488 ms`
- post-warmup NPU mean: `0.347 ms`
- post-warmup end-to-end mean: `13.463 ms`
- post-warmup end-to-end median: `13.345 ms`
- post-warmup end-to-end P95: `13.961 ms`

That is about:

- `3.01x` frontend speedup
- `2.70x` end-to-end speedup

Dedicated `visiong.KWS` wrapper on the same board bundle:

- accuracy: `0.9083`
- post-warmup wall mean: `11.012 ms`
- post-warmup wall median: `10.020 ms`
- post-warmup wall P95: `19.803 ms`

This dedicated wrapper is the recommended deployment API because it hides the full frontend-to-NPU path inside the library and trims extra Python overhead.

Same 120-sample bundle on host ONNX:

- Accuracy: `0.9000`

That means the board result is consistent with the host reference for the exact same evaluation pack.

## Cloud-Built Library Artifact Validation

I also validated the fix using the cloud-built `visiong_python.zip` artifact generated from the repository GitHub Actions workflow.

This second validation is important because it proves the fix is really in the library binary, not only hidden behind a demo-side workaround.

Validation setup:

- workflow: `build-binary-artifact`
- workflow run: `23943474500`
- branch: `codex/kws-rv1106`
- downloaded artifact root: `D:\visiong-kws-demo\github_action_build_audio_fix`
- deployed board override path: `/root/visiong_action_pkg_fix`
- board execution path:
  - `PYTHONPATH=/root/visiong_action_pkg_fix:/root/visiong_kws_demo_c64/scripts`
  - model and evaluation WAVs reused from `/root/visiong_kws_demo_c64`

For this run, the validation script used only:

- `LowLevelNPU.set_input_array()`
- `--input-strategy array-only`

It did not use the manual pass-through byte-packing fallback.

Actual cloud-artifact board result with the native frontend:

- Evaluation samples: `120`
- Board accuracy: `0.9083`
- Post-warmup frontend mean: `11.341 ms`
- Post-warmup input-prepare mean: `0.488 ms`
- Post-warmup NPU mean: `0.347 ms`
- Post-warmup end-to-end mean: `13.463 ms`
- Post-warmup end-to-end median: `13.345 ms`
- Post-warmup end-to-end P95: `13.961 ms`

I also compared the native frontend directly against the Python frontend on the board after deploying this artifact.

The measured feature drift was only:

- max absolute error: about `7.37e-06`
- mean absolute error: about `1.2e-07`

That confirms both:

- the stride fix is present in the built `_visiong.so`
- the native frontend is numerically aligned with the training/reference pipeline

## Important Runtime Discovery: `w_stride` Bug in Generic Tensor Input

This project uncovered a real issue in the current `LowLevelNPU` generic array input path.

For this KWS model, the RKNN input tensor reports:

```text
dims = [1, 49, 40, 1]
format = NHWC
type = INT8
pass_through = true
w_stride = 48
size_bytes = 1960
size_with_stride_bytes = 2352
```

The key detail is:

- logical width is `40`
- runtime width stride is `48`

Before the fix, `set_input_array()` and `set_input_from_float()` copied input data linearly into tensor memory.

That is fine only when:

- `size_with_stride_bytes == size_bytes`

It is wrong when the tensor has padding at the end of each row, because the runtime expects each row to start at `w_stride`, not at the logical width.

### What That Broke

Without stride-aware packing:

- host ONNX on the sample predicted `yes`
- board runtime predicted nonsense classes like `off`
- board accuracy on the 120-sample bundle collapsed to `0.1583`

### How I Verified the Root Cause

I manually quantized one sample into an `INT8` buffer with row padding:

- 49 rows
- 48 bytes per row
- the first 40 bytes in each row contained valid feature data
- the remaining 8 bytes were zero padding

Feeding that packed buffer through `set_input_bytes()` immediately restored the correct prediction.

That confirmed the issue was not:

- the trained model
- ONNX export
- RKNN quantization
- the frontend

It was specifically the generic tensor memory write path.

## Library Fix

I patched `src/npu/LowLevelNPU.cpp` so that:

- `set_input_buffer()`
- `set_input_from_float()`

now detect stride-padded tensor layouts and copy row-by-row with the correct destination stride.

This uses the existing SIMD row-copy helper already used elsewhere in the library via `copy_data_with_stride()`.

That is the correct long-term fix.

## Board Demo Compatibility Fallback

Because the board may still be running the previously installed binary, I also added a compatibility fallback in `board_kws_demo.py`.

When the runtime input tensor is:

- quantized
- `pass_through == true`
- stride-padded

the demo now:

- quantizes the float feature map itself
- packs each row with the required stride
- uses `set_input_bytes()` instead of the buggy generic float path

This behavior is selected explicitly with:

- `--input-strategy compat`

This makes the board demo work correctly even before the new fixed `_visiong.so` is deployed.

## Why I Did Both

The fallback solves the immediate deployment problem.

The library patch solves the actual platform bug for all future generic tensor users.

That combination gave the fastest route to success without abandoning the proper fix.

After the GitHub Actions artifact was built, I verified that the new binary no longer needs the fallback for this model, which is the more important long-term result.

## Commands Summary

Prepare:

```bash
python3 prepare_speech_commands.py \
  --raw-root /mnt/d/visiong-kws-demo/data/speech_commands_v0.02 \
  --output-root /mnt/d/visiong-kws-demo/prepared_kws \
  --board-per-label 10 \
  --calibration-count 128 \
  --overwrite
```

Train:

```bash
python3 train_kws_ds_cnn.py \
  --prepared-root /mnt/d/visiong-kws-demo/prepared_kws \
  --output-root /mnt/d/visiong-kws-demo/train_kws_c64 \
  --epochs 22 \
  --batch-size 256 \
  --channels 64 \
  --overwrite
```

Convert:

```bash
python3 convert_kws_ds_cnn_to_rknn.py \
  --onnx-model /mnt/d/visiong-kws-demo/train_kws_c64/kws_ds_cnn.onnx \
  --dataset /mnt/d/visiong-kws-demo/prepared_kws/calibration/quant_dataset.txt \
  --output-rknn /mnt/d/visiong-kws-demo/rknn_kws_c64/kws_ds_cnn.rknn
```

Board eval on older runtime:

```bash
python3 board_kws_demo.py \
  --model /root/visiong_kws_demo_c64/kws_ds_cnn.rknn \
  --labels /root/visiong_kws_demo_c64/board_bundle/labels.txt \
  --manifest /root/visiong_kws_demo_c64/board_bundle/eval_manifest.tsv \
  --input-strategy compat \
  --warmup-count 10 \
  --summary-json /root/visiong_kws_demo_c64/board_summary.json
```

Board eval on fixed runtime:

```bash
PYTHONPATH=/root/visiong_action_pkg_fix:/root/visiong_kws_demo_c64/scripts \
python3 board_kws_demo.py \
  --model /root/visiong_kws_demo_c64/kws_ds_cnn.rknn \
  --labels /root/visiong_kws_demo_c64/board_bundle/labels.txt \
  --manifest /root/visiong_kws_demo_c64/board_bundle/eval_manifest.tsv \
  --input-strategy array-only \
  --warmup-count 10 \
  --summary-json /root/visiong_kws_demo_c64/board_summary_fixed.json
```

Board eval on fixed runtime with the native audio frontend:

```bash
PYTHONPATH=/root/visiong_action_pkg_fix:/root/visiong_kws_demo_c64/scripts \
python3 board_kws_demo.py \
  --model /root/visiong_kws_demo_c64/kws_ds_cnn.rknn \
  --labels /root/visiong_kws_demo_c64/board_bundle/labels.txt \
  --manifest /root/visiong_kws_demo_c64/board_bundle/eval_manifest.tsv \
  --frontend-backend native \
  --input-strategy array-only \
  --warmup-count 10 \
  --summary-json /root/visiong_kws_demo_c64/board_summary_native.json
```

## Result

The migration succeeded.

The board now runs a mature keyword spotting model with:

- open dataset training
- ONNX export
- RKNN conversion
- on-device `LowLevelNPU` inference
- native `visiong` audio frontend acceleration
- measured real-board accuracy and latency

And along the way, the library gained a real fix for stride-padded generic tensor input.
