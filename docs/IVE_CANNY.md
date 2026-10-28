<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Canny

## 接口 / Interface

`canny` 和 `canny_ex` 接收 GRAY8 图像，返回相同宽高的独立 GRAY8 边缘图，
边缘值为 255，其余为 0。阈值为 0–65535 的整数，必须满足
`low_thresh <= high_thresh`。阈值对应卷积后的梯度尺度，不能直接按输入灰度范围选择。

Both methods take GRAY8 and return an independent, same-size GRAY8 edge image
with values 0 or 255. Thresholds are unsigned 16-bit integers with low no greater
than high. Choose them for the gradient scale of the selected kernel.

```python
import visiong

ive = visiong.IVE()
edges = ive.canny(gray, high_thresh=100, low_thresh=50)
edges = ive.canny_ex(gray, high_thresh=100, low_thresh=50, mask=[])
```

空 `mask` 使用下列默认核。显式核为按行展开的 25 个有符号 8 位系数。
默认核是嵌入 5×5 布局的普通 3×3 Sobel 核；用于水平分量，转置用于垂直分量。
它与 `sobel` 便利方法的默认 5×5 核不同。比较两个接口时应显式传入相同核。

An empty mask selects this 3×3 Sobel kernel embedded in a 5×5 layout. An explicit
mask contains 25 signed byte coefficients in row order. Its transpose supplies
the other gradient direction. The `sobel` convenience method has a different
default kernel; provide the same explicit mask when comparing results.

```text
 0  0  0  0  0
 0 -1  0  1  0
 0 -2  0  2  0
 0 -1  0  1  0
 0  0  0  0  0
```

## 软件参考 / Software reference

对下述已测输入，完整图像与以下参考逐像素一致：相同核、S16 卷积、复制边界、
不做归一化，再使用 OpenCV 的导数输入 Canny 和 L1 梯度。
直接使用 `cv2.Canny(gray, ...)` 会隐藏导数生成步骤，不适合作为自定义核的比较接口。

The tested full images exactly match this reference: S16 derivatives from the
same kernel with replicated borders and no normalization, followed by OpenCV's
derivative-input Canny with L1 magnitude. Supplying derivatives makes the kernel
and boundary assumptions explicit.

```python
import cv2
import numpy as np

kernel = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 0, 1, 0],
    [0, -2, 0, 2, 0],
    [0, -1, 0, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32)

dx = cv2.filter2D(gray_array, cv2.CV_16S, kernel,
                  borderType=cv2.BORDER_REPLICATE)
dy = cv2.filter2D(gray_array, cv2.CV_16S, kernel.T,
                  borderType=cv2.BORDER_REPLICATE)
reference = cv2.Canny(dx, dy, low_thresh, high_thresh, L2gradient=False)
```

参考环境为 OpenCV 4.11。另用独立整数参考检查方向比较、非极大值抑制、阈值和
滞后连接，确认其在每组已保存导数上也与 OpenCV 一致；没有按最接近的候选选择结果。

The reference used OpenCV 4.11. A separate integer suppression and hysteresis
implementation also matched OpenCV for every saved derivative pair. Diagnostic
alternatives were not selected to obtain a passing result.

## 实测范围 / Measured scope

RV1106 上每个输入重复两次；比较包含最外圈像素，不只比较内侧区域。
每次保存源图、Sobel 两个 S16 分量及原始硬件 Canny 输出，并核验完整字节数、
SHA-256、格式、重复结果、输入未被改写和已返回输出保持不变。

Each input was repeated twice on RV1106. Comparisons include the outermost pixels.
Complete sources, signed derivatives and original hardware edge maps were checked
for size, hashes, type, repeatability, input preservation and output lifetime.

| 图像与输入 / Size and fixtures | Sobel 调用 / calls | Canny 调用 / calls |
| --- | ---: | ---: |
| 64×64：常量、角落脉冲、靠边及内部阶跃 / constants, corner impulse and boundary/interior steps | 24 | 24 |
| 64×64：斜线、22.5°/67.5°附近方向、梯度平顶 / slanted steps, direction boundaries and derivative plateaus | 24 | 24 |
| 64×64：强弱边、连接、不同间隙、斜向接触和独立弱分量 / strong/weak edges, gaps, diagonal contact and separate weak components | 20 | 20 |
| 64×64：幅值为 80 的阶跃，阈值 0、79、80、81、65535 的 11 组组合 / eleven threshold pairs around a magnitude-80 step | 2 | 22 |
| 66×70：非对齐行的相同边界输入 / repeated boundary fixtures with unaligned rows | 24 | 24 |
| 合计 / Total | 94 | 114 |

全部 114 张边缘图和 94 组导数均与固定参考全图一致，349 个源图和输出文件的
完整哈希通过。幅值为 80 的受控阶跃在低阈值为 0 时，高阈值 79 保留边缘，
高阈值 80 不保留边缘。该输入区分了此处的高阈值等号条件；不能单独确定
“等于低阈值且与其他强边连通”的分支。

All 114 edge maps and 94 derivative pairs matched the fixed references over the
full image; all 349 source/output files verified. For the controlled magnitude-80
step with low threshold zero, high 79 retained the edge and high 80 rejected it.
This isolates that high-threshold equality case, but does not independently isolate
a low-threshold equality connected to another strong edge.

实测 Canny 每次 API 调用的中断增量有 1、2、3，重复相同输出也可能不同。
检查要求每次有正增量、前后计数连续、总量等于各次增量之和；不固定为一次或两次。
中断只作为执行证据，数值验收依据仍为完整输出。中断次数变化的原因尚未确认。

Canny interrupt deltas of 1, 2 and 3 were observed, including variation between
identical repeated outputs. Accounting checks positive per-call deltas, a
continuous timeline and exact totals. Interrupt counts supplement numerical
validation; their variability is not explained by these measurements.

这些结果支持上述默认核、尺寸和受控输入的兼容参考，不表示所有自定义核、
奇数尺寸、梯度极值、阈值组合和连接拓扑均已穷举，也不等于重建了硬件内部算法。
原始接口与版权来自 Rockchip，见 [第三方说明](../THIRD_PARTY_NOTICES.md)
及 [IVE 覆盖清单](IVE_COVERAGE.md)。

The result is bounded to the stated kernel, sizes and controlled inputs. Custom
kernels, odd dimensions, gradient extremes, all threshold pairs and every
connectivity topology were not exhaustively tested. It does not identify the
complete internal algorithm. See the third-party notices for the original
Rockchip interface and copyright information.
