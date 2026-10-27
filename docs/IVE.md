<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE

## 使用约定 / Interface conventions

`visiong.IVE()` 调用 RV1106 IVE 硬件。图像操作包含输入暂存、DMA 同步、硬件执行和结果复制；Python 调用耗时包含这些步骤。

`visiong.IVE()` uses RV1106 IVE hardware. Python call timings include input staging, DMA synchronization, hardware execution and output copying.

```python
import visiong as vg

ive = vg.IVE()
previous = vg.ImageBuffer.from_numpy(previous_gray, "gray8")
current = vg.ImageBuffer.from_numpy(current_gray, "gray8")
vectors = ive.lk_optical_flow(previous, current, [(100, 80), (140, 120)])
flow = [(v.mv_x / 128, v.mv_y / 128) if v.status == 0 else None for v in vectors]
```

LK 使用四层硬件金字塔；两帧尺寸必须一致且至少为 64×64，最多 500 个整数坐标点。返回 `status=0` 表示成功，`-1` 表示失败；位移为 S9.7，除以 128 得到像素。成功状态只说明局部迭代有效，不保证匹配到真实运动；大位移、遮挡和重复纹理仍需调用方检查。

LK uses four hardware pyramid levels, matching image dimensions of at least 64×64 and up to 500 integer coordinates. Status is 0 on success and -1 on failure; divide S9.7 displacement by 128 for pixels. Successful local convergence does not guarantee the true motion, especially with large displacement, occlusion or repeated texture.

角点工作区按图像步长分配。厂商在 DMA 后进行 CPU 解包时按只写模式同步；封装在提交前使已初始化的结果缓存失效，避免重复坐标。验证脚本包含反复变换尺寸的角点压力检查。

Corner workspace follows image stride. The SDK uses write-only synchronization during CPU unpacking after DMA; the wrapper invalidates initialized output cache lines before submission to avoid duplicate coordinates. Validation includes corner stress tests across repeated dimension changes.

| 格式 / Format | NumPy dtype | 布局 / Layout |
|---|---|---|
| `gray8` | `uint8` | H×W |
| `s8c1` | `int8` | H×W |
| `s16c1` | `int16` | H×W |
| `u16c1` | `uint16` | H×W |
| `u64c1` | `uint64` | H×W |
| `hsv888` | `uint8` | H×W×3, H/S/V ∈ [0,255] |

整数灰度导入要求 dtype 匹配并使用 `copy=True`。默认 NumPy 视图只读；需要修改时使用 `to_numpy(copy=True)`。`integral(..., SUM/SQSUM)` 返回 U64C1；`norm_grad` 返回两个 S16C1，保留硬件 S8 梯度的数值。

Typed grayscale imports require matching dtypes and `copy=True`. NumPy views are read-only; request `to_numpy(copy=True)` for writable copies. Integral SUM/SQSUM results use U64C1. Normalized gradients use S16C1 containers preserving the hardware S8 values.

`rgb_to_yuv` 返回 NV12；`yuv_to_rgb` 和 `yuv_to_hsv` 接受 NV12/NV21。`full_range` 控制输出范围，YUV 输入默认使用视频范围。处理由 `rgb_to_yuv(..., full_range=True)` 生成的图像时，反向转换应显式传入 `input_full_range=True`。HSV 色相使用 0–255 标度，对比 OpenCV 时应使用 `COLOR_RGB2HSV_FULL`。

`rgb_to_yuv` returns NV12; YUV-to-RGB/HSV accepts NV12 or NV21. `full_range` controls output range, while YUV input defaults to video range. Pass `input_full_range=True` when converting full-range YUV back. HSV uses a 0–255 hue scale, comparable to OpenCV `COLOR_RGB2HSV_FULL`.

形态学输入为 0/255 二值 GRAY8，核尺寸为 3 或 5。有符号转换使用 NEON 准备数据，再调用硬件缩放；SDK 的 16 位转换入口会按无符号输入解释数据，不能直接传入带符号位模式。

Morphology expects binary 0/255 GRAY8 and a kernel size of 3 or 5. Signed casts prepare unsigned values with NEON before hardware scaling because the SDK conversion entry point interprets 16-bit input as unsigned.

## 有状态背景模型 / Stateful background models

```python
model = vg.IVEModel(width, height)
foreground, background = ive.gmm(previous, model, first_frame=True)
foreground, background = ive.gmm(current, model)

codebook = vg.IVEModel(width, height)
for frame_num, frame in enumerate(training_frames):
    ive.match_bg_model(frame, codebook, frame_num)
foreground_status, background = ive.update_bg_model(current, codebook, frame_num + 1)
```

GMM/GMM2 使用单模型硬件模式。实机验证的厂商库在多模型新增记录时，输入和输出共用模型内存，可能覆盖尚未读取的数据。独立输出缓冲区的诊断实验可消除此错误，但公开 SDK 接口未提供该参数，因此正式封装不依赖私有 ABI。该限制意味着 GMM/GMM2 不能维护每像素多个背景分布；需要多个背景状态时可评估码本模型。实测库 SHA-256：`2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455`。

GMM/GMM2 use single-model hardware mode. In the tested vendor runtime, growing multi-model records in shared input/output memory can overwrite unread records. A diagnostic using separate model output memory eliminates the error, but the public SDK exposes no such parameter. The production wrapper therefore avoids private ABI dependencies. GMM/GMM2 cannot maintain multiple background distributions per pixel; consider the codebook model for multiple background states. The tested runtime SHA-256 is given above.

码本 API 修正：`match_bg_model(frame, model, frame_num)` 原地训练并返回 `None`；`update_bg_model(frame, model, frame_num)` 返回 `(foreground_status, background_image)`。旧版 `fg_flag` 输入参数已移除；旧版训练函数返回的所谓前景图并非厂商输出。非 16 对齐尺寸内部补齐后裁回原尺寸。不同算法、序列和图像尺寸应各自使用独立模型。

Codebook API correction: training updates the model and returns `None`; updating returns `(foreground_status, background_image)`. The former `fg_flag` input argument is removed. The old training result was not a vendor foreground output. Non-aligned dimensions are padded internally and outputs cropped back. Use separate models for different algorithms, sequences and dimensions.

## 板端验证 / On-board validation

在已配置 VisionG 和厂商运行库的板端，安装包含 `video` 模块的 OpenCV Python 包后执行：

Run on a board with VisionG, vendor runtimes and OpenCV Python including the `video` module:

```sh
python3 scripts/check_ive_components.py --loops 5
python3 scripts/check_ive_formats.py
python3 scripts/check_ive_advanced.py
python3 scripts/check_ive_pipeline.py --repeat 1000 --resize-cycles 100
python3 scripts/compare_ive_optical_flow.py --loops 10 --output ive_flow.json
```

前四个脚本以非零退出码报告失败，并记录硬件中断增量。对比脚本使用固定种子生成图像、已知平移真值和 OpenCV PyrLK；包含大位移和无纹理诊断场景，不将所有场景假定为成功。两种实现的窗口和迭代细节不同，耗时不能视为所有实际视频的固定加速比。

The first four scripts fail with a nonzero exit status and record hardware interrupt deltas. The comparison uses deterministic images, known translations and OpenCV PyrLK, including large-motion and textureless diagnostics. Implementation parameters differ; timings do not establish a universal speedup for real video.

接口参数参考 Rockchip 的 [LKOpticalFlow](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/LKOpticalFlow.c)、[GMM](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/GMM.c)、[BGM](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/BGM.c) 示例。原始厂商版权和许可保持不变，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

Interface parameters are checked against the linked Rockchip samples. Original vendor copyright and licenses remain unchanged; see the third-party notices.
