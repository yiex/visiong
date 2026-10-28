<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE 数值约定 / Numerical behavior

本文记录已测 RV1106 运行库的输出兼容规则及证据边界。成功返回、硬件中断和内存边界完整均不能代替数值比较。以下规则没有修改 Rockchip SDK 的原始代码或许可信息，也不表示其他芯片或运行库采用相同内部算法。

This note describes output compatibility with the measured RV1106 runtime.
Successful status, hardware interrupts and intact allocation guards do not
establish numerical correctness. The observations do not identify the internal
algorithm or establish behavior on other devices and runtimes.

实测 `librve.so` 的 SHA-256 为：

```text
2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455
```

## 积分 COMBINE / Combined integral

`integral(src, IntegOutCtrl.COMBINE)` 返回同宽、同高的 U64C1 图像，每个位置对应包含该位置的左上矩形；没有额外的零首行、零首列。它将普通和与平方和存入同一个 64 位字，不能将整个字直接当作其中任一积分值。

COMBINE returns an H×W U64C1 image of inclusive upper-left prefixes. There is
no additional zero row or column. Each word packs two fields rather than one
ordinary integral value.

设 `S` 为该矩形内像素和，`Q` 为像素平方和，已测兼容规则为：

```python
sum_field = min(S, (1 << 28) - 1)
square_field = min(Q, (1 << 36) - 1)
word = (square_field << 28) | sum_field
```

低 28 位保存饱和后的 `S`，高 36 位保存饱和后的 `Q`。越界字段饱和，不按字段位宽回绕；一旦饱和，原始积分不能从 COMBINE 结果恢复。需要更大数值范围时分别选择 `SUM`、`SQSUM`，并按所用模式校验输出。

The low 28 bits hold saturated S and the high 36 bits hold saturated Q.
Overflow saturates each field rather than wrapping it. Values beyond a field's
limit cannot be recovered from COMBINE; use the separate SUM/SQSUM modes when
the required range exceeds the packed fields and validate the selected mode.

```python
import numpy as np
import visiong as vg

ive = vg.IVE()
packed = ive.integral(src, vg.IntegOutCtrl.COMBINE).to_numpy(copy=True)
sum_field = packed & np.uint64((1 << 28) - 1)
square_field = packed >> np.uint64(28)
```

边界验证使用 1056×1024 的常量 254、255 图像，并重复一次 255。每次检查全部 1,081,344 个原始小端 U64 字，核对输入、输出摘要、源字节保留、缓冲区保护和一次硬件中断；重复 255 的完整输出摘要相同。独立整数参考同时比较了字段截断、单字段饱和、双字段饱和及未掩码拼接，只有上述双字段饱和规则与三次采集均逐字相同。

Boundary captures used constant 254/255 images of 1056×1024 pixels, with a
repeated 255 capture. Every one of the 1,081,344 little-endian U64 words was
checked per capture. Source preservation, guards, hashes and one hardware
interrupt were verified. The repeat was byte-identical. Among truncation,
one-field saturation, two-field saturation and unmasked-packing candidates,
only saturation of both fields matched all three captures.

例如，255 常量图在 `(x=1055,y=996)` 的 `S=268472160` 已超过低字段上限，而 `Q=68460400800` 尚未超过高字段上限；在 `(1055,1000)` 两字段均超过上限，输出为 `2**64-1`。255 常量图的低、高字段分别从矩形面积 1,052,689 和 1,056,817 起饱和。

For a constant 255 image, the low and high fields first saturate at prefix
areas 1,052,689 and 1,056,817 respectively. The measured capture includes both
the low-field-only and both-fields-overflow regions.

这些结果明确扩展了未越界的小图打包检查，但尚未覆盖任意空间分布的溢出输入或精确上限附近的每个整数。输出规则吻合不证明硬件累加器位宽、饱和发生阶段或内部存储布局。

These captures extend the earlier non-overflow packing checks. They do not
cover every spatial overflow pattern or every integer immediately around a
field limit, and do not establish internal accumulator widths or saturation
stages.

## 角点 lambda / Corner scores

`st_corner_candidates` 和 `st_corner_sorted` 返回 `(x,y,lambda)`，其中 `lambda` 是厂商输出的 8 位分数。候选原始字按 `x=word&2047`、`y=(word>>11)&2047`、`lambda=(word>>22)&255` 解码，与 SDK 角点后处理实际读取的字段一致。

The candidate and sorted interfaces return a raw vendor 8-bit score with each
coordinate. Candidate words use x bits 0–10, y bits 11–21 and lambda bits 22–29,
matching the SDK postprocessor's field extraction.

该分数依赖图像上下文，不能作为固定增益的局部最小特征值。64×64 受控采集中，单独象限角点在对比度 4～255 时的 252 个中心分数全部为 255；对比度 0～3 没有返回候选。相同弱 8×8 方块独处时分数为 255，在远处加入强度 255 的方块后，弱对比度 31、63、127 的分数分别变为 4、16、63。弱角点周围 17×17 输入逐字节保持一致。

The score depends on image context. An isolated quadrant corner scored 255
for all 252 contrasts from 4 to 255, while contrasts 0–3 produced no candidate.
An unchanged weak 8×8 box scored 255 alone, but adding a remote strong box
changed its scores at contrasts 31/63/127 to 4/16/63. The local 17×17 input
patch was identical before and after the remote insertion.

上述弱方块分数兼容按全图强响应归一化的假设；255 倍和 256 倍平方对比度比的最近整数都能解释这三个分数，因此不能据此确定归一化常数、整数舍入顺序、导数核或张量窗口。固定尺度的软件结构张量候选即使在单角点样本中全匹配，也可能只是所有输出均已饱和，不能据此宣布算法相同。

The three weak-box scores are compatible with image-wide normalization, but
both 255-times and 256-times squared-contrast ratios round to these values.
They therefore do not identify a normalization constant, rounding sequence,
derivative kernel or tensor window. A fixed-scale tensor model matching the
isolated corners may only be matching their saturated scores.

`quality_level=1` 时，高对比度直线边缘还会产生低分候选。例如对比度 191 的 `(45,31)` 分数为 3，对比度 255 的 `(45,30)` 分数为 2；30 种独立导数核与张量窗口组合在对应位置的理想最小特征值均为 0。对比度 255 的匹配采集中，质量阈值 25 去掉了这些低分边缘候选。这属于已观察到的数值差异，具体定点或查表机制尚未确定。

At quality 1, high-contrast straight edges can produce low-score candidates
despite zero ideal minimum eigenvalue for all 30 tested derivative/window
models. Quality 25 removed these low-score edges in the matching contrast-255
capture. The internal fixed-point or lookup mechanism remains unknown.

补充的固定远处参考采集完成 96 次调用，核验了全部原始输入、候选缓冲和源文件摘要；80 次有近受控位置的分数，16 次按缺失观测保留。归一化候选以偶数对比度的 24 个观测拟合，并将奇数对比度及其他形状的 56 个观测留出。Sobel 3×3、box 3×3 张量、全图最大值归一化、255 倍和最近整数的组合在拟合中无差异，在留出中仍有 1 个分数不同；因此它只是接近观测的候选，不是已确立的精确算法。

A further 96-call capture with a fixed remote reference was audited from raw
inputs and candidate buffers. It yielded 80 nearby scores and 16 censored
observations. Fitting used 24 even-contrast observations, with 56 odd-contrast
and other-shape observations withheld. A Sobel-3/box-3, image-maximum-normalized,
255-times nearest-rounded candidate had zero fitting mismatches but one
holdout mismatch. It remains an approximate candidate rather than an exact
algorithm.

该差异有可解析的反例：对比度 91、远处参考对比度 255 的 8×8 方块，四角理想张量均有 `A=C=430612`、`|B|=132496`，最小特征值均为 `298116`。上述归一化参考四角均为 32，实际原始分数按左上、右上、左下、右下为 `33,32,32,33`。同图相同理想特征值出现不同分数，不能只调整一个全图增益就得到逐点精确参考。后续两次原样采集的完整原始候选缓冲与首采相同；方块中心平移至 `(11,11)`、`(13,13)` 后，对齐坐标的完整候选集合仍相同。内部近似发生在哪个阶段仍未知。

An analytic counterexample occurs at contrast 91: all four box corners have
equal ideal minimum eigenvalue 298116, with A=C=430612 and |B|=132496. The
normalized reference rounds all four to 32, but raw scores are 33/32/32/33
in top-left/top-right/bottom-left/bottom-right order. One image-wide gain
cannot map equal ideal eigenvalues to these unequal scores. Two repeat raw
buffers were identical to the first capture; translations to centers (11,11)
and (13,13) preserved the complete candidate sets after coordinate alignment.
The internal approximation stage remains unknown.

进一步检验在采集前固定了三组预测，再采集 8 幅新相对对比度图及上述 4 次重复、平移。带符号张量算术右移 9 位的候选符合已知方向差异，但在新对比度 146、190、221、225 各有两个角点差 1；新图的 32 个角点仅 24 个与冻结预测相同。向零截断及未量化的理想张量候选也未覆盖全部观测。没有据新结果重新拟合参数；这些反例否定了候选作为精确参考的适用性，未解释的低分直线边缘仍保留为独立限制。

Three predictions were frozen before a further eight unseen relative-contrast
images and four repeat/translation captures. An arithmetic-shift-by-nine
tensor candidate reproduced the known directional difference but missed two
corners by one at each new contrast 146, 190, 221 and 225: only 24 of 32 new
corner scores matched. Truncation toward zero and the unquantized ideal tensor
also failed. Parameters were not refitted. These counterexamples reject the
candidates as exact references; the low-score straight-edge behavior remains
separately unresolved.

同一输入的质量阈值关系已补验：以质量值 1 的完整候选列表为基线，4 幅受控图像上的 11 个非基线设置均逐项等于筛选 `lambda > quality_level` 后的列表，涵盖分数 4、16、63、255 的等号及邻近阈值。分数等于阈值时被排除。这建立了受测输入和阈值上的严格大于关系，未穷举所有图像、分数和阈值；这些采集均使用质量值 1～255，没有建立质量值 0 的数值约定。

For four fixed inputs, all 11 non-baseline quality settings exactly reproduced
the complete quality-1 candidate list filtered by `lambda > quality_level`.
The checks cover equality and neighboring thresholds at scores 4, 16, 63 and
255: a score equal to the threshold is excluded. This establishes the relation
for the tested inputs and settings, not every image, score or threshold.
These captures do not establish numerical behavior at quality zero.

首轮 381 次调用的原始候选缓冲均已逐文件核验，并重新解码计数、坐标和分数；每次状态成功、中断增量为 1、源字节保留且保护区完整。345 次有靠近受控角点的分数，36 次没有；没有候选的情形按缺失观测保留，不能擅自赋值为零。低分边缘、缺失检测及重复候选应由调用方结合场景检查，不能仅因 SDK 成功返回就视为角点正确。

The initial 381 calls were audited from raw buffers and re-decoded independently,
including counts, coordinates and scores. Of these calls, 345 had a nearby
controlled score and 36 did not. Missing detections remain censored and are
not assigned zero. Applications should assess low-score edges, missing
detections and duplicate candidates in their intended scene.
