<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE HSV 定点兼容模型 / Fixed-point HSV compatibility

RV1106 的 RGB→HSV 可以在软件中精确复现。这里的模型来自受控输入和实机输出拟合，不代表已知内部电路或 RTL。验证使用的厂商 `librve.so` SHA-256 为 `2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455`。

RGB-to-HSV is reproducible in software for the tested RV1106 runtime. This is an experimentally validated compatibility model, not a claim about its internal circuit.

## 整数规则 / Integer rules

输入为 R、G、B 三个无符号字节，令 `V=max(R,G,B)`、`D=V-min(R,G,B)`。`D=0` 时 H=S=0。其余情况使用两个倒数表，除法只用于生成表项：

```text
Srec[V] = floor((65280 + floor(V/2)) / V)
Hrec[D] = floor((65536 + 3*D) / (6*D))

N = G-B          if V == R
    B-R+2*D      else if V == G
    R-G+4*D      otherwise

S = (D*Srec[V] + 128) >> 8
H = ((N*Hrec[D] + 128) >> 8) & 255
```

右移为带符号算术右移；H 的负分子先参与定点乘法，最后才按 256 回绕。不要先把负分子加上 `6*D`，该顺序会改变舍入结果。最大通道并列时按 R、G、B 的顺序选择分支。

Use signed arithmetic right shift and wrap hue only after multiplication and rounding. Adding `6*D` before multiplication changes quantization. Break maximum-channel ties in R, G, B order.

[`ive_hsv_reference.py`](../scripts/ive_hsv_reference.py) 提供 NumPy 和单像素整数实现。它明确运行在 CPU 上，不调用 IVE。

## 全色域验证 / Exhaustive validation

采集按 R 分层、G 为行、B 为列，分 256 帧覆盖全部 `256³=16,777,216` 种输入。四个 RGB2HSV 模式各运行一遍，并在 R=0、127、255 时重复采集。共观察到 1060 次硬件中断，12 组重复输出均一致。

| 检查 / Check | 结果 / Result |
|---|---|
| BT601 full、BT709 full、BT601 limited、BT709 limited | 四个完整数据流逐字节相同 / all four streams identical |
| 上述整数模型 / Integer model | H、S、V 各 0 个不同像素 / zero channel mismatches |
| HSV 字节流 SHA-256 | `504d46de486781d5b90b25d89efaa7c3fbc78103f116b2bcaa96ab9c629d1125` |
| V=max(R,G,B) | 全色域精确相同 / exact |
| OpenCV 4.11 `COLOR_RGB2HSV_FULL` H | 4,146,270 个不同像素，最大圆周误差 3 / differing pixels, maximum circular error 3 |
| OpenCV S | 1,898,634 个不同像素，最大误差 1 / differing pixels, maximum error 1 |
| OpenCV V | 0 个不同像素 / zero mismatches |

差异来自倒数表精度和舍入顺序，而不是 RGB/BGR 通道互换，也不能只用 255 与 256 的色相比例解释。例如纯绿输出 H=86，纯蓝 H=171；RGB=(125,126,127) 输出 HSV=(149,4,127)。

The reciprocal precision and rounding order explain the measured differences. Merely rescaling hue between 255 and 256 does not reproduce the output. OpenCV's implementation uses 12 fractional bits; this compatibility model uses 8. See the [OpenCV implementation](https://github.com/opencv/opencv/blob/4.12.0/modules/imgproc/src/color_hsv.simd.hpp).

## 复现 / Reproduction

在板端已安装运行库和 VisionG 的目录执行，然后在具有完整 NumPy 的机器检查采集文件：

Run the capture on a configured board, then validate each stream with a full NumPy installation:

```sh
python3 capture_ive_hsv.py --full --output hsv_cube
python3 check_ive_hsv_reference.py hsv_cube/full601.hsv --output hsv_reference.json
```

原始采集每个模式占 48 MiB，不随源码分发。检查脚本的非零退出码表示不匹配；输出包含像素数、逐通道错误及预测字节流哈希。当前结论针对 RGB→HSV；YUV→HSV 还涉及 YUV 矩阵、视频范围和色度采样，必须单独验证。

Each raw stream is 48 MiB and is not included in source distribution. The checker fails on any mismatch and reports channel errors and the predicted stream hash. The exhaustive results above cover RGB-to-HSV.

## YUV 输入 / YUV input

另以 2×2 独立色度单元采集 Y、U 全部 256 个值和 V 的 14 个分层值，共 917,504 组 YUV 输入。四种 YUV2HSV 模式各自与同模式 YUV2RGB 的输出接上述整数模型逐字节一致。这验证了该样本中的色彩转换顺序；它不是全部 YUV 色域的穷举结论。

An additional grid covers all Y/U byte values and 14 V strata, or 917,504 distinct YUV inputs. For each of the four modes, direct YUV-to-HSV exactly matches its corresponding IVE YUV-to-RGB output passed through this integer HSV model. This is a sampled result, not an exhaustive YUV-domain claim.

在该网格上，BT601/BT709 的 full RGB 输出分别与下列 Q8 矩阵零差异。输入列向量为 `[Y-16,U-128,V-128]`，每个分量先计算 `(dot+128)>>8`，再截断至 0–255：

```text
BT601 full RGB             BT709 full RGB
[ 298,    0,  409 ]         [ 298,    0,  459 ]
[ 298, -100, -208 ]         [ 298,  -55, -136 ]
[ 298,  516,    0 ]         [ 298,  541,    0 ]
```

limited RGB 输出在同一网格上与以下 Q8 矩阵也逐字节一致。其输入为 `[Y,U-128,V-128]`，按 `(dot+128)>>8` 舍入，最终截断至 16–235：

```text
BT601 limited RGB          BT709 limited RGB
[ 256,    0,  351 ]         [ 256,    0,  394 ]
[ 256,  -86, -179 ]         [ 256,  -47, -117 ]
[ 256,  444,    0 ]         [ 256,  464,    0 ]
```

这两个模型各覆盖 917,504 个已采集输入，三个通道均为零差异。limited 使用自己的系数及末级截断，不能简单地把 full RGB 缩放到 16–235。以上不是对内部系数寄存器或电路实现的断言。

The limited-output models also match all 917,504 captured inputs per mode. They use different coefficients and final clipping to 16–235; scaling a clipped full-RGB result does not reproduce them. These are compatibility models, not claims about coefficient registers or circuitry.

随后取回的全 YUV 色域采集覆盖每模式 16,777,216 组输入。先按采集清单校验文件长度及 SHA-256，再逐字节比较参考输出，结果如下：

| 全域检查 / Full-domain check | 结果 / Result |
|---|---|
| BT601 full RGB | 全部像素和三个通道零差异 / exact |
| BT709 full、limited RGB | 全部像素和三个通道零差异 / exact |
| BT601/BT709 full、limited HSV | 四种模式的 H/S/V 全部零差异 / all four exact |
| BT601 limited RGB | 内存目录分块重采后，全部像素和三个通道零差异 / exact after verified recapture |

原采集中的 BT601 limited RGB 文件因存储错误而哈希不符，已排除并保留。随后按连续的 32 个 V 层分八块在内存目录重新采集，每块完成同步、读回及传输哈希检查；合并流的 SHA-256 为 `8111e6739d20e2a95983e524263d6b76ada0e6239a2174cef10200699f0cbf97`，与原采集写入前的哈希一致。重新采集另观察到 256 次硬件中断。

最终八个完整数据流的参考输出 SHA-256 均等于实测文件，共 134,217,728 个输出像素、402,653,184 个通道值，差异均为零。这些是先后两次采集的合并证据；不能据此声称所有模式都在同一个新版封装上重新运行。

The corrupted BT601 limited RGB file remains excluded. A fresh capture used eight consecutive 32-V-layer pieces in volatile storage, with synchronization, readback and transfer hashes checked separately. Its merged hash matches the original pre-storage capture hash. All eight accepted streams now match their predicted hashes: 134,217,728 output pixels and 402,653,184 channel values, with zero differences. The evidence combines separate capture runs; it does not establish regression acceptance of every mode on one new wrapper version.

[`check_ive_csc.py`](../scripts/check_ive_csc.py) 包含这些独立整数参考，并通过采集文件哈希检查后逐字节比对：

```sh
python3 scripts/check_ive_csc.py --self-check --capture-root yuv_capture --output yuv_reference.json
```

需要比较 HSV 时，应先对齐 YUV 排列、输入范围、矩阵与输出范围，最后比较 HSV 的定点结果。不能套用 RGB→HSV 四模式完全相同的结论。

Align YUV layout, input range, matrix and output range before comparing HSV quantization. Do not extend the RGB-to-HSV flag equivalence to YUV modes.

## RGB 转 YUV 的整数兼容模型 / RGB-to-YUV compatibility

四种模式各自覆盖全部 16,777,216 种 RGB 输入。颜色重复成 2×2 单元，按 R 分层在
512×512 图像中提交，共 32 个内存分块、1,024 次硬件调用。四个 Y 副本、源图、紧凑
数据和重建完整 NV12 的哈希均经检查；三个通道共 201,326,592 个值与以下模型零差异。

All four modes were measured over every RGB triple using repeated 2×2 cells.
Source, compact capture and reconstructed NV12 hashes were checked across 32
chunks and 1,024 hardware calls. All 201,326,592 channel values match this model.

各通道计算 `clip(((aR*R+aG*G+aB*B+128)>>8)+bias)`，使用有符号算术右移：

| Mode | Y coefficients | U coefficients | V coefficients | bias |
|---|---|---|---|---|
| BT601 limited | `(66,129,25)` | `(-38,-75,112)` | `(112,-94,-18)` | `(16,128,128)` |
| BT709 limited | `(47,157,16)` | `(-26,-87,112)` | `(112,-102,-10)` | `(16,128,128)` |
| BT601 full | `(77,150,29)` | `(-43,-85,128)` | `(128,-107,-21)` | `(0,128,128)` |
| BT709 full | `(54,183,18)` | `(-29,-99,128)` | `(128,-116,-12)` | `(0,128,128)` |

limited 裁剪范围为 Y `[16,235]`、U/V `[16,240]`；full 为 `[0,255]`。
这些范围给出已测输入域上的等价函数；部分通道的自然范围不会触及某些裁剪边界，
不能据此确定内部裁剪寄存器。成比例的 Q10 系数也可能等价，不能将兼容模型解释为
已经确定硬件内部位宽、计算顺序或 RTL。

The compatible clipping ranges are Y `[16,235]` and U/V `[16,240]` for limited
output, and `[0,255]` for full output. Some bounds cannot be distinguished from
the natural range. Equivalent scaled coefficients do not identify internal precision.

BT601 limited 的 U 通道 G 系数为 `-75`，不能替换为常见的 `-74`。
BT709 full 的 Y 系数和为 255，纯白的 Y 因此为 254；limited 两模式的纯白 U 为 127。

BT601 limited uses `-75` for the green contribution to U. BT709 full maps white
to Y=254; both limited modes map white to U=127. Ideal floating-point matrices
and ideal neutral-chroma assumptions do not reproduce these values.

受控空间图案和独立留出图符合左上像素取样：420 色度取 `[2*y,2*x]`，Y 保留逐像素
结果；不进行四像素平均。422 组合路径重复行后调用 420 硬件，色度因此取 `[y,2*x]`。
空间证据未穷举全部 2×2 颜色组合。布局及 BGR/VU 控制由 CSC 脚本另行检查。

Spatial fixtures and held-out patterns match top-left chroma selection for 420,
and per-row left-pixel selection for the row-composed 422 path. This evidence
covers the controlled patterns rather than every possible four-color cell.

全流按 R、G、B 升序保存 Y/U/V；采集与参考 SHA-256 相同：

| Mode | SHA-256 |
|---|---|
| BT601 limited | `b717d635983add2231699089ccdeaa784c136873a7b983006c30485bb749461d` |
| BT709 limited | `b385edfa5eb4efcd6587889c85a20bd20bda7ce5cd2f3c5ff2522dedaf0b868e` |
| BT601 full | `9e8f5a4ce65d43c02c95e0a83a2de2cb7282a4b61f5b52ba036505f8fb4b144e` |
| BT709 full | `fb7aa659e864b9581cf2a9b45ea25b89a169a1608ee9b74322c526fe54df7251` |
