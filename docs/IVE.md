<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE

## 使用约定 / Interface conventions

`visiong.IVE()` 调用 RV1106 IVE 硬件。图像操作包含输入暂存、DMA 同步、硬件执行和结果复制；Python 调用耗时包含这些步骤。

`visiong.IVE()` uses RV1106 IVE hardware. Python call timings include input staging, DMA synchronization, hardware execution and output copying.

完整接口、组合实现及厂商限制见 [覆盖清单](IVE_COVERAGE.md)。`mag_and_ang` 的幅值结果为 U16C1，角度为 U8C1；需要转为 8 位显示时选择无符号转换模式。

See the coverage list for complete interfaces, compositions and runtime limits. Magnitudes use U16C1 and angles use U8C1; select an unsigned cast mode when preparing magnitudes for display.

积分 COMBINE 的打包、饱和范围，以及角点分数与质量阈值的数值边界见
[数值约定](IVE_NUMERIC.md)。角点分数受整幅图像影响，不能作为绝对特征值使用。

See the [numerical notes](IVE_NUMERIC.md) for combined-integral packing and
saturation, and for context-dependent corner scores and measured quality thresholds.

Canny 的掩码、阈值、边界与受控全图对照范围见 [Canny 说明](IVE_CANNY.md)。
硬件中断次数可以变化，不能把一次 API 调用等同于一次中断。

See the [Canny notes](IVE_CANNY.md) for masks, thresholds, borders and bounded
full-frame comparisons. One API call does not imply one hardware interrupt.

`add(a,b)` 保留原有各 0.5 权重的平均混合行为。需要其他权重时使用 `add_weighted`，系数以 Q0.16 整数表示。`ncc` 使用未减均值的归一化内积；`ncc_raw` 返回三个整数聚合，不等同于 Pearson 相关系数。

`add(a,b)` retains equal half-weight blending. Use `add_weighted` for explicit Q0.16 coefficients. NCC is a normalized dot product without mean subtraction; `ncc_raw` exposes its three integer aggregates.

`add_weighted(a,b,x,y)` 的实测字节结果为 `((a*x+b*y)>>16)&255`：先合并乘积再向下取整，溢出时按 8 位回绕。系数范围为 0～65535；系数和不超过 65536 时不会回绕。`add` 因此等于 `floor((a+b)/2)`。15 组系数各覆盖全部 65,536 个字节输入对，重复两次共 2,304,000 个输出像素与此规则一致；未穷举全部系数组合。

Weighted addition truncates the combined Q0.16 sum and wraps to U8. It does not saturate: `a=128,b=129,x=y=65535` produces 0. All byte pairs passed for 15 coefficient pairs over two repetitions; coefficient combinations were not exhaustively enumerated.

非退化直方图的均衡参考为 `L(v)=clip((255*max(0,C(v)-C0)+D//2)//D,0,255)`，其中 `C` 为累积计数，`C0` 是首个非零 bin 的计数，`D=N-C0>0`。实测半整数向上舍入，精度边界符合整数参考；OpenCV 的浮点计算与半整数规则不能保证逐字节一致。外部直方图按传入分布计算，即使它与源图像不同。单 bin 的退化情形须单独检查。

Nondegenerate equalization subtracts the first occupied cumulative count and uses integer nearest rounding with half ties upward. External mode uses the supplied distribution. Degenerate single-bin histograms require a separate rule.

单 bin 直方图仅在 `k` 处非零时，实测映射为 `v<k` 输出 0，否则输出 `k`。
内部均衡因此保留常量图像；外部均衡作用于其他源时并非恒等映射。全部 256 个 `k`
在 256×256 图像、内部和外部模式、两次重复下均已验证，共 1,024 次硬件调用。

For a histogram occupied only at `k`, the measured map is zero below `k` and
`k` otherwise. Every possible occupied bin passed in internal and external
modes over two repetitions. External mode is not an identity transform on a different source.

`filter_ex`、`sobel_ex` 和 `norm_grad_config` 的受控全图检查符合复制边界、卷积整数和算术右移 `norm` 位，再按目标类型饱和；显式 `morphology` 在图外补零。秩滤波和全方形形态学的部分边界模型数学等价，仅凭这些输出不能确定内部采用哪一种补边实现。

Full-frame convolution checks match replicated edges, arithmetic right shift and destination-type saturation. Explicit morphology uses zero outside the image. Symmetric rank and morphology kernels can make different edge-extension models observationally equivalent.

`mag_and_ang_ex` 在阈值为 0 的受控测试中符合以下整数参考：将卷积分量对称限幅到 `[-32767,32767]`，幅值取绝对值之和，方向使用有符号半开区间。Sobel 的 S16 输出范围仍为 `[-32768,32767]`，不能直接将它的最小负值代入通常的绝对值公式。例如两个分量均为 `-32768` 时，实测幅值为 `65534`；分量为 `(-32768,13572)` 时，幅值为 `46339`、方向为 `3`。

At threshold zero, the captured MagAndAng outputs match symmetric clipping, L1 magnitude and the integer angle reference below. Sobel has an asymmetric S16 range. These rules describe output compatibility with the measured runtime, without identifying internal arithmetic.

```python
x = max(-32767, min(32767, raw_gx))
y = max(-32767, min(32767, raw_gy))
magnitude = abs(x) + abs(y)
if x == 0 and y == 0:
    angle = 0
else:
    offset = 4 if x < 0 else 0
    if x < 0:
        x, y = -x, -y
    s = 8192 * y
    if s >= 19777 * x:
        angle = 2
    elif s >= 3393 * x:
        angle = 1
    elif s >= -3393 * x:
        angle = 0
    elif s >= -19777 * x:
        angle = 7
    else:
        angle = 6
    angle = (angle + offset) & 7
```

软件参考已核验全部 `[-255,255]²` 受控分量、放大梯度、四象限边界等号及邻点、饱和核和 S16 极值图像。没有穷举全部图像和核。该方向量化与浮点 `atan2` 后取最近 45° 在少量边界处不同，不能直接替换。

Validation covers the controlled small-gradient domain, scaled gradients, signed boundary ties and neighbors, saturated kernels and S16 extremes. It does not exhaust every image and kernel. Floating-point `atan2` followed by nearest-angle rounding differs at some boundaries.

`sub(a,b,SubMode.SHIFT)` 保留原始 U8 存储，实测字节值为 `((A-B)>>1)&255`：先进行带符号差及算术右移，再取低 8 位。需要有符号数值时按补码解释；不要将字节值直接当作带偏置的正数差。该映射已枚举全部 65,536 个 U8 输入对验证。

SHIFT subtraction stores `((A-B)>>1)&255` in U8 bytes. Interpret those bytes as two's-complement values when signed differences are needed. This mapping was verified over all 65,536 byte input pairs.

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

已确认的厂商运行库在配置层级 0–2 时取错当前帧步长。封装识别到该故障实现时，会在提交前拒绝这些层级，四层默认路径可继续使用。需要单层或其他层级时，先按 [运行库修正说明](IVE_LK_RUNTIME.md) 创建并显式加载独立修正副本。

The measured vendor runtime selects the wrong current-frame stride at levels 0–2. The wrapper rejects those levels before submission when it recognizes that implementation; the default four-level path remains available. Create and explicitly load a separate corrected copy as described in the runtime note before using the other levels.

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

C++ 自建灰度缓冲区必须提供非空数据、有效尺寸及足够步长，容量至少为 `((height-1)*w_stride+width)*item_size` 字节；末行后的填充不要求实际分配。复制与 16 位转换入口在读取前使用 64 位整数检查容量，并拒绝无法用 `int` 表示的字节步长。仅有 `ImageBuffer.is_valid()` 为真不足以保证容量满足图像布局。

Custom C++ grayscale buffers must contain every active row. Capacity validation uses 64-bit arithmetic and does not require unused padding after the last row. `is_valid()` alone does not establish sufficient storage.

`uint8` 的 `from_numpy(...,copy=True)` 会将奇数宽高补齐到偶数并填零；软件参考应使用导入后的实际尺寸。需要保留奇数 GRAY8 尺寸时使用 `from_bytes(data,width,height,"gray8")`。

The uint8 NumPy importer pads odd dimensions to even sizes with zeros when copying. Compare software references against the imported dimensions. Use `from_bytes` with explicit dimensions to retain odd-sized GRAY8 images.

`create_pyramid` 包含原尺寸第零层，之后每层使用复制边界、5×5 二项式高斯核、整数和右移 8 位，再保留偶数坐标，尺寸逐层向下取整。它采用 Filter 和 DMA 组合，与 LK 内部的原生金字塔入口分开验证。返回的第零层共享输入图像。

The public pyramid includes the source as level zero. Each subsequent level applies a 5×5 binomial Gaussian kernel with replicated edges, shifts the integer sum right by eight and selects even coordinates with floor-sized dimensions. It composes Filter and DMA and is validated separately from LK's native pyramid. Level zero shares the source image.

`rgb_to_yuv` 返回 NV12；`yuv_to_rgb` 和 `yuv_to_hsv` 接受 NV12/NV21。YUV 输入默认使用视频范围。处理由 `rgb_to_yuv(..., full_range=True)` 生成的图像时，反向转换应显式传入 `input_full_range=True`。RGB→HSV 的四个 BT601/BT709、full/limited 模式在实测库中完全一致，输出均为全范围 HSV888；不能用 `full_range=False` 请求视频范围 HSV。其定点规则与 OpenCV `COLOR_RGB2HSV_FULL` 有系统性差异，详见 [HSV 兼容模型](IVE_HSV.md)。

`rgb_to_yuv` returns NV12; YUV-to-RGB/HSV accepts NV12 or NV21. YUV input defaults to video range. Pass `input_full_range=True` when converting full-range YUV back. The four RGB-to-HSV mode flags produce identical full-range HSV888 on the tested runtime. Its fixed-point rules differ from OpenCV; see the linked compatibility model.

形态学输入为 0/255 二值 GRAY8，核尺寸为 3 或 5。有符号转换使用 NEON 准备数据，再调用硬件缩放；SDK 的 16 位转换入口会按无符号输入解释数据，不能直接传入带符号位模式。

Morphology expects binary 0/255 GRAY8 and a kernel size of 3 or 5. Signed casts prepare unsigned values with NEON before hardware scaling because the SDK conversion entry point interprets 16-bit input as unsigned.

## 有状态背景模型 / Stateful background models

`IVEModel(width,height)` 固定绑定原始输入尺寸，内部补齐不会改变这个约定。GMM/GMM2 首次调用或切换算法必须指定 `first_frame=True`，该调用先清空旧状态；码本模型必须先训练再更新。不能将 GMM 状态直接交给码本算法，先调用 `model.reset()` 或创建独立模型。`reset()` 保留尺寸并清空状态；同一模型上的调用会串行访问。

Models bind the exact source dimensions, including when internal padding is needed. Initialize or switch GMM/GMM2 with `first_frame=True`, which clears previous state. Train codebook models before updating them. Call `model.reset()` or create a separate model before using GMM state for codebook training. Reset preserves dimensions and clears state; access to the same model is serialized.

```python
model = vg.IVEModel(width, height)
foreground, background = ive.gmm(previous, model, first_frame=True)
foreground, background = ive.gmm(current, model)

codebook = vg.IVEModel(width, height)
for frame_num, frame in enumerate(training_frames):
    ive.match_bg_model(frame, codebook, frame_num)
foreground_status, background = ive.update_bg_model(current, codebook, frame_num + 1)
```

`gmm`/`gmm2` 保留单模型接口；`gmm_config`/`gmm2_config` 在已核验的运行库中支持每像素 1–5 个模型，返回前景、背景与匹配图。多模型使用独立输出缓冲区，避免记录增长时覆盖未读输入；匹配图也随每个模型独立保存，并按真实紧凑行布局读取。此兼容路径核验完整运行库指纹，不改写厂商库或驱动。初始化、模型数量切换和错误恢复约定见 [GMM 说明](IVE_GMM.md)。

The original helpers retain one model. The configurable helpers provide 1–5 models on fingerprint-checked runtimes, with separate model output and persistent packed matched records. This compatibility path leaves the vendor library and driver unchanged. See the GMM notes for version, initialization and recovery requirements.

单模型的背景图是更新后的状态，不能假定前景出现时仍输出旧背景。受控序列中，80 训练背景切换到 10 或 220 的矩形时，该帧前景为矩形、背景图已等于新输入；重复新输入的下一帧前景归零，恢复 80 时再次产生一帧前景。此结果覆盖当前默认参数及 GMM2 因子 `0x0808`，不等同于多模型背景保持能力。

The single-model background output reflects updated state. In the measured 80-to-10/220 rectangle sequences, the first changed frame reports foreground while its background already equals the new input; the repeat is quiet, and returning to 80 produces foreground once more. This evidence uses the current defaults and GMM2 factor `0x0808`; it does not establish multi-model background retention.

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
python3 scripts/check_ive_extended.py --sizes 320x240 --groups model stateful
python3 scripts/compare_ive_optical_flow.py --loops 10 --output ive_flow.json
```

前四个脚本以非零退出码报告失败，并记录硬件中断增量。对比脚本使用固定种子生成图像、已知平移真值和 OpenCV PyrLK；包含大位移和无纹理诊断场景，不将所有场景假定为成功。两种实现的窗口和迭代细节不同，耗时不能视为所有实际视频的固定加速比。

The first four scripts fail with a nonzero exit status and record hardware interrupt deltas. The comparison uses deterministic images, known translations and OpenCV PyrLK, including large-motion and textureless diagnostics. Implementation parameters differ; timings do not establish a universal speedup for real video.

接口参数参考 Rockchip 的 [LKOpticalFlow](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/LKOpticalFlow.c)、[GMM](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/GMM.c)、[BGM](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/simulator/BGM.c) 示例。原始厂商版权和许可保持不变，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

Interface parameters are checked against the linked Rockchip samples. Original vendor copyright and licenses remain unchanged; see the third-party notices.
