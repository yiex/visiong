<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE LK runtime stride correction / IVE LK 运行库步长修正

The measured RV1106 IVE runtime has a current-frame pyramid descriptor indexing
error in `rve_hal_lk_params_to_regs`. This document describes a correction for
that exact runtime build. Pyramid levels 0–2 are supported by the hardware; the
observed failure comes from the supplied runtime's parameter preparation.

实测的 RV1106 IVE 运行库在 `rve_hal_lk_params_to_regs` 中存在当前帧金字塔描述符
索引错误。本文的局部修正仅适用于该精确版本。硬件支持金字塔层级 0–2；观测到的
异常来自该运行库的参数准备。

## Evidence and scope / 证据与范围

`RK_MPI_IVE_LKOpticalFlowPyr` places the current-frame descriptors at index 6.
The register preparation code instead calculates their stride base as
`max_level + 3`. These values coincide only when `max_level` is 3. At levels
0–2, current-frame strides are read from the wrong descriptors.

`RK_MPI_IVE_LKOpticalFlowPyr` 将当前帧描述符固定放在索引 6；寄存器参数准备代码
却按 `max_level + 3` 计算其步长基址，仅在 `max_level` 为 3 时吻合。层级 0–2
会从错误描述符读取当前帧步长。

Controlled same-frame tests through both the direct SDK API and VisionG showed
incorrect nonzero motion at levels 0–2 with the original runtime. Level 3 gave
zero motion. With the four-byte correction applied to a private copy, the same
test gave zero motion at every requested level from 0 to 3 through both paths.
The system runtime was unchanged during these tests.

同帧受控测试同时覆盖直接 SDK API 和 VisionG：原运行库在层级 0–2 输出错误的
非零位移，层级 3 输出零位移。对独立副本实施四字节修正后，两条调用路径在所请求
的层级 0–3 均得到零位移。测试期间未修改系统运行库。

This establishes the descriptor-stride fix for that test and version. It does
not establish general tracking accuracy, convergence, occlusion handling or
equivalence to OpenCV on arbitrary image sequences. Those require independent
motion fixtures and software comparisons.

该结果证明此版本在上述测试中的描述符步长修正有效。一般场景的跟踪精度、收敛、
遮挡处理及与 OpenCV 的一致性，仍须通过独立运动样例和软件对照验证。

Motion regression with the corrected copy covered 40 points, five repetitions
and image sizes 320×240, 322×242 and 640×480. Zero motion and a (1, −1) pixel
translation passed at levels 0–3; the maximum displacement error was 0.012 pixel.
A (4, −3) translation passed at levels 2 and 3, but level 1 still produced
incorrect matches marked valid, with maximum errors of 9.50, 17.95 and 19.52
pixels for those sizes. The original failed results are retained. The stride
correction does not enlarge the algorithm's local convergence range or make
its valid flag a ground-truth guarantee.

修正副本的运动回归覆盖 40 个点、5 次重复及 320×240、322×242、640×480 三种尺寸。
层级 0–3 的静止与 (1, −1) 像素平移通过，最大位移误差小于 0.012 像素。
(4, −3) 平移在层级 2、3 通过，但层级 1 仍将错误匹配标记为有效，三种尺寸的最大
误差分别为 9.50、17.95、19.52 像素。原失败结果保留。步长修正不会扩大算法的局部
收敛范围，也不能使有效标志成为真实运动的保证。

An additional level-1 diagnostic supplied the exact translated absolute points
as the initial flow. All 40 points then met the unchanged 0.2-pixel limit at
each size, with maximum errors of 0.011049, 0.0078125 and 0.011049 pixels.
Two calls per configuration had identical coordinates, status and residuals,
unchanged inputs and seven IVE interrupts per call. Small-window OpenCV
comparisons showed similar sensitivity to initialization. These observations
support a local convergence limitation; they do not identify the hardware
window or make an exact initial guess available for unknown motion. The
unseeded failures remain failures.

额外的层级 1 诊断将真实平移后的绝对坐标作为初始点，三个尺寸的全部 40 个点均
满足原有 0.2 像素界限，最大误差分别为 0.011049、0.0078125、0.011049 像素。
每种配置两次调用的坐标、状态和残差一致，输入未变，每次记录到七个 IVE 中断。
小窗口 OpenCV 对照也表现出相近的初值敏感性。这些观测支持局部收敛限制的解释，
不能据此确定硬件窗口，也不能将真实初值用于估计未知运动。无初值失败项仍保留
为失败。

An additional guarded tracking matrix explicitly loaded the corrected copy with
matching bindings in an isolated application directory. The module SHA256 was
`57a7902d9decf9cc45522e05609be71f17f45e6039498a37ee6f32776a32a8ef`.
Each requested level 0–3 ran the same 35-frame 320×240 sequence with at most
40 points. Each process reported only the corrected IVE mapping. Corrected level 3
matched the original-runtime level-3 track values, seeds, losses and hardware
arguments exactly. Across each level, the four trackable scenes retained
200/200, 238/240, 191/200 and 200/200 correct correspondences, with no wrong or
occluded-core correspondence accepted. Repeated and flat scenes remained
unverified. Maximum accepted error was 0.17259 pixel, with seeds derived only from
previous measured motion. This tests the guarded wrapper; it does not overturn
the raw unseeded convergence failures above. Timing and residual limits are in
[LK Tracker](IVE_LK_TRACKER.md).

另一次带保护的跟踪矩阵在独立应用目录中使用匹配绑定，显式加载了修正副本，模块哈希见上。
所请求的层级 0–3 各运行相同的 35 帧 320×240 序列，最多 40 点；每个进程仅报告映射了修正
IVE 副本。修正副本层级 3 的轨迹值、初值、丢点及硬件参数与原库层级 3 精确相同。
每个层级的四个可跟踪场景分别保留 200/200、238/240、191/200、200/200 个正确对应，没有
错误对应或遮挡核心内对应被接受。重复及无纹理场景仍为未验证。接受点最大误差为
0.17259 像素，初值仅由过去实测运动产生。这验证的是带保护的封装，不推翻上述原始无初值
收敛失败。耗时及残差边界见 [LK 跟踪封装](IVE_LK_TRACKER.md)。

## Exact supported build / 精确支持的版本

| Item / 项目 | Value / 值 |
|---|---|
| Original SHA256 / 原文件 SHA256 | `2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455` |
| Corrected SHA256 / 修正后 SHA256 | `80c5c104f5bf02b88dafd1ed0867d90f2439fded2464174080d30ff97bd03409` |
| Format / 格式 | ELF32, ARM, little-endian / 小端 |
| Virtual address / 虚拟地址 | `0x1d650` |
| File offset / 文件偏移 | `120400` (`0x1d650`) |
| Original instruction / 原指令 | `03 30 83 e2`: `add r3, r3, #3` |
| Corrected instruction / 修正指令 | `06 30 a0 e3`: `mov r3, #6` |

The tool validates the complete original SHA256, ELF headers, executable
`PT_LOAD` mapping and original instruction. It changes only the four instruction
bytes and requires the complete corrected SHA256 to match the measured copy.
Unknown versions and already corrected inputs are rejected.

工具校验原文件完整 SHA256、ELF 头、可执行 `PT_LOAD` 映射及原指令，仅替换该指令
的四个字节，并要求修正后的完整 SHA256 与实测副本一致。未知版本和已修正输入均
会被拒绝。

## Create and load a separate copy / 创建并加载独立副本

Check an independently obtained vendor runtime without writing a file:

对自行取得的厂商运行库进行只读检查：

```sh
python3 scripts/patch_ive_lk_runtime.py librve.so --check
```

Create a new application runtime directory and corrected copy:

创建新的应用运行库目录及修正副本：

```sh
mkdir ive_runtime
python3 scripts/patch_ive_lk_runtime.py librve.so --output ive_runtime/librve.so
```

The output must be a new file distinct from the input. Existing files, including
links, are not overwritten. The tool flushes, synchronizes and reads back the
output before reporting success; a failed write or verification removes the new
output. The source is opened only for reading.

输出必须是与输入不同的新文件，既有文件及链接均不覆盖。工具在报告成功前刷新、
同步并读回校验输出；写入或校验失败时删除新输出。原文件仅以只读方式打开。

Load the corrected copy explicitly in a new Python process, before importing
VisionG. The measured IVE runtime uses Rockit MMZ symbols without declaring that
dependency, so load Rockit globally first. Keep the application's existing RKNN
runtime selection and library search environment.

在新的 Python 进程中先显式加载修正副本，再导入 VisionG。实测 IVE 运行库使用了
未声明依赖的 Rockit MMZ 符号，因此须先将 Rockit 全局加载。保留应用原有的 RKNN
运行库选择及库搜索环境。

Use a separate application module directory containing the matching `visiong.py`,
`_visiong.so` and corrected `librve.so`, and select it for application imports.
VisionG's existing preloader searches its module directory before the vendor
directory. A corrected copy loaded elsewhere does not prevent an import from a
different binding directory from opening another IVE runtime. Keep the selected
RKNN/runtime configuration; do not replace files in a system or stable install.
The measured matrix used this isolated-directory layout and checked the mapped
runtime's complete hash.

使用独立应用模块目录，放入匹配的 `visiong.py`、`_visiong.so` 与修正 `librve.so`，并使应用
从该目录导入。VisionG 的既有预加载先搜索模块目录，再搜索厂商目录；在别处加载修正副本，
不能阻止从其他绑定目录导入时打开另一 IVE 运行库。保留所选 RKNN 及运行库配置，不替换
系统或稳定安装中的文件。实测矩阵使用了该独立目录布局，并核验映射运行库的完整哈希。

```python
import ctypes
import sys
from pathlib import Path

runtime_directory = Path("ive_runtime").resolve()
sys.path.insert(0, str(runtime_directory))
rockit_runtime = ctypes.CDLL("librockit.so", mode=ctypes.RTLD_GLOBAL)
ive_runtime = ctypes.CDLL(
    str(runtime_directory / "librve.so"), mode=ctypes.RTLD_GLOBAL
)

import visiong
```

Confirm that the application actually loads this copy. A runtime already loaded
by the process is not replaced by loading another file. Standalone
`LD_PRELOAD` of this IVE library does not satisfy its undeclared MMZ dependency.
The correction tool itself does not load the vendor runtime, modify process
memory, replace system files or change firmware.

应确认应用实际加载的是该副本；加载另一文件不会替换进程中已经加载的运行库。
单独对该 IVE 库使用 `LD_PRELOAD` 无法满足其未声明的 MMZ 依赖。修正工具自身不加载
厂商运行库、不修改进程内存、不替换系统文件，也不修改固件。

## Copyright and redistribution / 版权与分发

The correction tool is project-owned source under LGPL-3.0-or-later. The input
and corrected vendor runtime retain their original copyright, notices and
license terms. The tool's license does not relicense that runtime or authorize
its redistribution. Obtain the runtime separately and preserve its accompanying
notices. No original or corrected vendor binary belongs in the public source
archive.

修正工具为项目自有源码，采用 LGPL-3.0-or-later。原厂商运行库及修正副本保留其
原版权、声明和许可条款；工具的许可不改变运行库许可，也不授予其再分发权限。
运行库须另行取得，并保留随附声明。公开源码包不包含原始或修正后的厂商二进制。

See [Third-Party Notices](../THIRD_PARTY_NOTICES.md) and
[Binary Redistribution Policy](BINARY_REDISTRIBUTION_POLICY.md) for the external
runtime boundary.

外部运行库边界参见[第三方声明](../THIRD_PARTY_NOTICES.md)及
[二进制分发策略](BINARY_REDISTRIBUTION_POLICY.md)。
