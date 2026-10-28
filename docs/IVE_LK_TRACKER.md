<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Optional hardware LK tracker / 可选硬件 LK 跟踪封装

`LKTracker` is a Python policy layer over the public IVE corner and pyramidal LK
interfaces. It has no NumPy dependency. Create it explicitly; importing VisionG
does not create a tracker, patch a runtime, or select a corrected runtime copy.

`LKTracker` 在公开的 IVE 角点及金字塔 LK 接口之上增加 Python 跟踪策略，无需 NumPy。
须显式创建；导入 VisionG 不会创建跟踪器、修正运行库或选择修正副本。

```python
import visiong

tracker = visiong.LKTracker(max_points=100)
first = tracker.update(gray_frame, timestamp=0.0)
next_result = tracker.update(next_gray_frame, timestamp=1.0 / 30.0)
for track in next_result["tracks"]:
    print(track["id"], track["x"], track["y"], track["fb_error"])
tracker.reset()
```

Pass valid GRAY8 `ImageBuffer` frames of at least 64×64 pixels. The tracker keeps
a deep copy of the current frame for the next call. The default `max_level=3`
uses four pyramid levels. Explicit levels 0–2 retain the native rejection for
the known defective runtime; there is no automatic patch or fallback.

输入必须是至少 64×64 的有效 GRAY8 `ImageBuffer`。跟踪器深拷贝当前帧供下次使用。
默认 `max_level=3` 使用四层金字塔；显式选择层级 0–2 时仍保留原生接口针对已知故障
运行库的拒绝行为，不会自动修正或回退。

## API and results / API 与结果

`update(frame, timestamp=None, points=None)` accepts optional integer `(x,y)`
points only on the first call after `reset()`. Otherwise hardware Shi-Tomasi
corners supply the initial points. `timestamp` is a finite, nonnegative acquisition
time in seconds and must strictly increase. Omitting it uses `time.monotonic()`;
do not mix the two clock modes within one sequence. A gap over `max_interval=0.5`
seconds or a frame-size change discards the old identities and selects new points.
Use `max_interval=None` when that policy is inappropriate.

`update(frame, timestamp=None, points=None)` 的可选整数 `(x,y)` 起点仅限 `reset()` 后
首帧，否则由硬件 Shi-Tomasi 提供。`timestamp` 为有限、非负、严格递增的采集时间，
单位秒；省略时使用 `time.monotonic()`，同一序列不能混用时钟。超过默认
`max_interval=0.5` 秒或尺寸变化时，丢弃旧编号并重新选点；可用 `None` 禁用时间间隔限制。

The returned dictionary owns independent track and loss values. `tracks` contains
`id`, fractional `x/y`, last `dx/dy`, measured `vx/vy` in pixels per second, `age`,
forward residual `error`, round-trip `fb_error`, and zero-mean photometric
`patch_error`. Newly selected points have
`age=1`, zero motion and no verified residuals. `retained` counts accepted existing
tracks; `added` counts new points. `lost` records identities and rejection reasons.
New points are not successful correspondences. `reset()` releases the retained
frame and restarts the identity sequence.

返回字典的轨迹及丢点值与内部状态独立。`tracks` 包含编号、浮点坐标、最近位移、
实测速度（像素/秒）、年龄、正向残差、前后向误差及零均值光度 `patch_error`。
新点 `age=1`，位移为零且残差未验证。
`retained` 是通过筛选的旧轨迹数，`added` 是新点数；`lost` 记录编号和原因。
新增点不能视为成功匹配。`reset()` 释放保留帧并重新开始编号。

Useful constructor controls are `max_points=100`, `min_distance=10`,
`quality_level=25`, `max_level=3`, `iterations=20`, `min_eigenvalue=100`,
`epsilon=32`, `max_step=32.0`, `fb_threshold=1.0`, `border=16`,
`max_error=None`, `max_interval=0.5`, and optional `ive` for sharing an IVE instance.
Step and forward/backward bounds are in pixels; `max_error` uses the native
residual divided by 128. No default residual bound is claimed to work universally.

`appearance_backend="auto"` uses the internal native batch helper when available,
and otherwise retains the Python checks. `"python"` explicitly selects the reference
policy; `"native"` requires the native helper and rejects an unavailable helper.
Native errors propagate. This option does not select an IVE runtime or disable
any appearance check.

`appearance_backend="auto"` 在内部原生批量函数可用时使用它，否则保留 Python 检查。
`"python"` 显式选择参考策略；`"native"` 要求原生函数可用，否则明确拒绝。
原生异常直接上报。该选项不会选择 IVE 运行库，也不会关闭任何外观检查。

构造参数如上。`max_step` 和 `fb_threshold` 单位为像素；`max_error` 使用原生残差除以
128 的数值。默认不设置尚未普适标定的残差阈值。`ive` 可用于复用现有 IVE 实例。

Image guards default to `appearance=True`, `normalize_brightness=True`,
`max_patch_error=12.0`, `ambiguity_ratio=0.2` and `ambiguity_radius=16`.
Set the first two controls to `False` for the original wrapper policy. The raw
`IVE.lk_optical_flow_config` interface remains available independently.

图像保护默认启用，参数如上。将前两个选项设为 `False` 可保留原封装策略；
原始 `IVE.lk_optical_flow_config` 接口仍可独立调用。

## Prediction and checks / 预测与检查

Initial flow uses only previously accepted velocities and the current timestamp
interval. New points can use the median accepted velocity. The first motion has
no motion history and uses the unseeded four-level operation. Forward candidates
must satisfy status, optional residual, border and maximum-step checks. A reverse
hardware LK call checks displacement consistency before acceptance. Lost points
are replenished with separated interior corners and receive new identities.

初值只使用此前已接受的速度及当前时间间隔；新点可使用已接受速度的中位数。
首次运动没有历史，采用无初值的四层跟踪。正向候选需通过状态、可选残差、边界及
最大步长检查，再通过反向硬件 LK 的位移一致性校验。丢点后补选分离的内部角点并赋予新编号。

The native API accepts integer source points and integer absolute initial points.
The wrapper rounds with half ties upward, estimates flow at that integer source,
and applies its Q7 displacement to the stored fractional coordinate. Its reverse
displacement comparison removes the known coordinate-rounding offset. This avoids
accumulated half-pixel coordinate rounding on uniform motion; it does not add
fractional-source support to the native API or prove accuracy on spatially varying
motion. The border is a configurable policy, not a measured hardware-window size.

原生 API 仅接受整数源点和整数绝对初值。封装按半整数向上取整，在该整数源点估计光流，
将 Q7 位移累计到保留的小数坐标；反向位移比较消除已知坐标整数化偏移。这避免匀速运动中
半像素反复取整的累计偏移，但不增加原生小数源点支持，也不证明空间变化运动的精度。
边界距离是可配置策略，不代表已确定的硬件窗口大小。

A valid status and small round-trip error do not prove a correct correspondence.
Repeated textures can produce consistent wrong cycles; occlusion, lighting changes,
acceleration and the first large displacement can still lose or misidentify tracks.
Report geometric errors and correct retention separately from point counts. Current
host checks validate policy and state handling; target motion results must be
evaluated independently before claiming a useful motion range.

有效状态及较小往返误差不能证明真实对应关系。重复纹理可能形成自洽的错误循环；遮挡、
亮度变化、加速及首次大位移仍可能造成丢点或误匹配。应将几何错误、正确保留与点数分别统计。
主机检查验证策略和状态管理，实机运动结果须独立评估后才能声明可用位移范围。

## Image evidence and cost / 图像证据及开销

Brightness alignment estimates an additive offset from the midpoint of sampled
25th and 75th percentiles, relative to the first frame. An odd sampling stride
avoids locking the sample grid to common even texture periods. Offsets within two
gray levels are ignored; larger offsets use `bytes.translate` on a private LK
working image. Original frames and the corner detector input remain unchanged.
This does not correct arbitrary gain, clipping or local illumination changes.

亮度对齐用采样 25% 与 75% 分位数的中点估计相对首帧的加性偏移，奇数采样步长避免
与常见偶数纹理周期锁定。两灰度级以内忽略，更大偏移用 `bytes.translate` 作用于独立的
LK 工作图像。原图和角点输入保持不变；不能据此校正任意增益、饱和或局部光照。

Appearance checks compare zero-mean 7×7 patches, with bilinear sampling at the
fractional forward endpoint. Low-texture patches are rejected. A separate local
self-correlation guard tests nearby repeated appearances: the default uses 160
sparse offsets, nine-pixel descriptors first, and at most eight 7×7 refinements
per point. A measured similar neighbor rejects the point as `ambiguous`; exhausting
the refinement budget rejects it as `appearance_budget`. The sparse bounded test
does not certify uniqueness at every displacement or outside its search radius.

外观检查比较零均值 7×7 块，并在正向小数终点双线性采样；低纹理块会被拒绝。
局部自相关另检查近邻重复外观：默认 160 个稀疏偏移，先比较九像素描述子，每点最多
细检八个 7×7 块。测得相似近邻时记为 `ambiguous`，耗尽细检预算时记为
`appearance_budget`。有界稀疏检查不能证明所有位移及搜索范围外都唯一。

`checks` reports normalization, photometric and ambiguity milliseconds, candidate
and refinement counts, measured ambiguity examples, and selection rejection reasons.
`update_ms` reports total wrapper time. Appearance screening can trade retention
and CPU time for fewer unsafe correspondences. A pure repeated scene may produce
no trusted points; that is not recovered tracking accuracy.

`checks` 返回归一化、光度及歧义检查毫秒数，候选与细检次数、实测歧义示例及选点拒绝原因；
`update_ms` 为总封装耗时。外观筛选会以保留量及 CPU 时间换取较少不可信对应。
纯重复场景可能没有可信点，这不能称为恢复了跟踪精度。

The native helper preserves the same patch statistics, ordered sparse offsets and
refinement budget. Integer statistics use NEON when compiled for it; bilinear
photometric calculations retain double precision. It borrows immutable bytes
without copying the planes. Forward screening uses one batch; a second candidate
batch is used only when the original policy needs new corners. Candidate screening
may precompute points later skipped by NMS or the point limit; counters describe
the actual work. Explicit initial points retain the original validation order.
`checks.backend`, `native_calls` and `native_ms` identify the executed path and cost.

原生函数保留相同的块统计、稀疏偏移顺序和细检预算；编译支持时整数统计使用 NEON，
双线性光度计算保留双精度。它借用不可变 bytes，不复制图像平面。正向筛选使用一批；
仅在原策略需要新角点时，才使用第二批候选检查。候选批次可能预计算随后被 NMS 或数量
上限跳过的点，计数反映实际工作。显式初始点保留原校验顺序。
`checks.backend`、`native_calls` 和 `native_ms` 标识实际执行路径与耗时。

## Bounded target measurements / 有界目标机实测

The 2026-10-28 RV1106 comparison used the original OEM IVE runtime, `max_level=3`,
320×240 inputs, at most 40 points, the same 35 frames and a 0.5-pixel truth bound.
The native module SHA256 was
`1c3b5f318e4b768194928b369ec88d4eaf079899e821d9c386766592abb6a61e`.
All 35 source hashes, track IDs, coordinates, residuals, losses, prediction seeds
and hardware submission arguments matched the guarded Python run exactly.
Seven separate bounded fixtures also matched the target Python reference exactly,
including periodic noise, fractional patches, boundaries and the refinement budget.

2026-10-28 的 RV1106 对照使用原始 OEM IVE 运行库、`max_level=3`、320×240 输入、
最多 40 点、相同的 35 帧和 0.5 像素真值界限。原生模块哈希见上。
全部源图哈希、轨迹编号、坐标、残差、丢点、预测初值及硬件提交参数均与带保护的
Python 测试精确相同。另七个有界输入与目标机 Python 参考精确相符，包含周期噪声、
小数块、边界和细检预算。

The table measures `update()` wall time in milliseconds, including each scene's
initial selection. Input rendering/construction, capture and application work are
excluded. P50 and P95 use linear interpolation at sorted index `(N-1)*q`; with only
4–7 samples per scene they describe this short run, not a steady-state latency bound.

下表为 `update()` 毫秒耗时，包含各场景首帧选点，不含输入生成及构造、采集和应用处理。
P50、P95 在排序索引 `(N-1)*q` 处线性插值；每场景仅 4–7 个样本，只描述此次短测试，
不能作为稳定延迟上限。

| Scene / 场景 | N | Python mean / 均值 | Native mean / 均值 | Native P50 | Native P95 | Native min–max / 范围 | Mean speedup / 均值加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Translation / 匀速 | 6 | 681.10 | 34.87 | 34.89 | 38.32 | 31.96–39.10 | 19.53× |
| Acceleration / 加速 | 7 | 682.96 | 47.82 | 39.58 | 78.40 | 31.66–81.41 | 14.28× |
| Occlusion / 遮挡 | 6 | 695.90 | 39.72 | 33.33 | 63.13 | 31.52–72.06 | 17.52× |
| Repetition / 重复纹理 | 6 | 197.52 | 48.62 | 48.50 | 49.44 | 47.76–49.59 | 4.06× |
| Lighting / 亮度变化 | 6 | 696.19 | 36.66 | 36.04 | 41.27 | 32.72–42.01 | 18.99× |
| Flat / 无纹理 | 4 | 6.04 | 6.04 | 6.05 | 6.08 | 5.99–6.08 | 1.00× |

The four trackable scenes retained respectively 200/200, 238/240, 191/200 and
200/200 correct correspondences with no wrong or occluded-core correspondences
accepted. Repeated and flat scenes retained no trusted correspondences and remained
unverified. All 59 hardware submissions retained their order and arguments;
18 updates retaining all 40 points submitted no extra corner operation.
The native helper ran 34 batches across 31 frames with `backend="native_neon"`;
the four flat frames had no appearance candidates or native batches.

四个可跟踪场景分别保留 200/200、238/240、191/200、200/200 个正确对应，没有错误对应或
遮挡核心内对应被接受。重复及无纹理场景没有可信对应，仍为未验证。59 次硬件提交的顺序和
参数保持相同；18 个保留全部 40 点的更新未额外提交角点操作。
31 帧执行了 34 批 `native_neon` 检查；四个无纹理帧没有外观候选，也未执行原生批次。

For the four trackable scenes, mean ambiguity work fell from 610–626 ms to
1.84–2.50 ms, and the complete native body averaged 2.47–3.26 ms per update.
Whole-wrapper mean time fell by 93.0–94.9%. This does not establish sustained
30-fps operation: the acceleration sample reached 81.41 ms. No 640-pixel input,
camera pipeline, other pyramid-level or universal ambiguity-rejection performance
is inferred from this run.

四个可跟踪场景的歧义检查平均耗时由 610–626 ms 降至 1.84–2.50 ms，整个原生函数体
平均每次更新 2.47–3.26 ms；封装总平均耗时降低 93.0–94.9%。这不能证明稳定 30 帧每秒：
加速场景曾达 81.41 ms。本测试不推断 640 像素输入、相机链路、其他金字塔层级的性能，
也不证明能拒绝所有重复纹理错误对应。

## Corrected-runtime level matrix / 修正运行库层级矩阵

A separate run used the explicitly loaded corrected runtime from
[IVE LK Runtime](IVE_LK_RUNTIME.md), module SHA256
`57a7902d9decf9cc45522e05609be71f17f45e6039498a37ee6f32776a32a8ef`,
and the same 35-frame 320×240 sequence at each level 3, 0, 1 and 2. The only mapped
IVE runtime reported by each process had the corrected complete SHA256. The system
runtime and stable installation were unchanged. Corrected level 3 matched the
original-runtime level-3 run's track values, prediction seeds, losses and hardware
arguments exactly.

另一轮按 [IVE LK 运行库说明](IVE_LK_RUNTIME.md) 显式加载修正副本，使用上述模块哈希，
分别在层级 3、0、1、2 运行相同的 35 帧 320×240 序列。各进程仅报告映射了完整哈希匹配的
修正 IVE 副本，系统运行库与稳定安装保持不变。修正副本层级 3 的轨迹值、预测初值、丢点及
硬件参数均与原运行库的层级 3 测试精确相同。

Each level retained 200/200 correct translation correspondences, 238/240 under
acceleration, 191/200 under occlusion, and 200/200 under lighting changes. No wrong
or occluded-core correspondence was accepted. Repeated and flat scenes remained
unverified at every level. Maximum accepted step error was 0.17259 pixel at level 0
and 0.16573 pixel at levels 1–3. Forward initial guesses were independently
recomputed from previous measured velocities; reverse guesses were the known old
source points. Motion truth was used for scoring only.

各层级均保留匀速 200/200、加速 238/240、遮挡 191/200、亮度变化 200/200 个正确对应，
没有错误对应或遮挡核心内对应被接受。每个层级的重复及无纹理场景仍为未验证。
接受点的最大步长误差在层级 0 为 0.17259 像素，层级 1–3 为 0.16573 像素。
正向初值已独立按过去实测速度重算，反向初值为已知旧起点；运动真值仅用于评分。

| Maximum level / 最大层级 | Translation mean / 匀速均值 ms | Acceleration mean / 加速均值 ms | Occlusion mean / 遮挡均值 ms | Lighting mean / 亮度均值 ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32.50 | 41.36 | 36.68 | 31.95 |
| 1 | 33.87 | 43.36 | 38.35 | 33.51 |
| 2 | 34.46 | 43.19 | 38.74 | 34.20 |
| 3 | 38.77 | 45.47 | 40.12 | 35.23 |

These are short whole-wrapper measurements, including initial selection, with
the same sampling limits as the original-runtime table. They do not isolate the
runtime patch's timing cost or establish a universal faster level. The unseeded
level-1 convergence failures documented in the runtime report remain failures;
this tracker adds prediction and rejection checks around native LK.

这些是包含首帧选点的短时封装测量，样本限制与原运行库表格相同，不能单独归因于运行库
补丁的耗时，也不能证明某个层级始终更快。运行库说明中的无初值层级 1 收敛失败仍为失败；
本跟踪器是在原生 LK 外增加速度预测与拒绝检查。

## Native residual units / 原生残差单位

`IVE.lk_optical_flow_config` returns a native unsigned Q9.7 residual field.
`LKTracker` reports `error = raw / 128.0` and compares `max_error` in those units.
The field is not a displacement error in pixels: a controlled corrected-runtime
level-0 diagnostic produced zero displacement at every point while uniform
brightness changes produced nonzero residuals. A ±1 gray-level change gave raw
127; +2, +4, +8 and +16 gave respectively 255, 511, 1023 and 2047. Identical inputs
gave zero. Twelve configurations, two calls each and 16 equivalent interior points
had identical repeats and unchanged inputs.

`IVE.lk_optical_flow_config` 返回原生无符号 Q9.7 残差字段，`LKTracker` 按
`error = raw / 128.0` 上报，`max_error` 使用相同单位。该字段不是像素位移误差：修正运行库
层级 0 的受控诊断中，全部点位移为零，但统一亮度变化仍产生非零残差。
±1 灰度变化得到原始值 127，+2、+4、+8、+16 分别得到 255、511、1023、2047；相同输入为零。
12 种配置各调用两次，16 个等价内部点的重复输出一致，输入未变。

The local-change observations were raw 20 for an eight-level center impulse,
501 for an eight-level radius-three square ring, and 355 for an eight-level sparse
symmetric change. On these stationary symmetric fixtures only, the observations
satisfied `0 <= SAD7/49 - raw/128 <= 1/128`, where `SAD7` is the independently
computed 7×7 absolute-difference sum. Exact truncation of `128*SAD7/49` failed the
uniform-change cases by one raw unit. Several reciprocal precisions and a different
integer boundary rule matched all captured values but disagree on unmeasured
inputs. The internal precision, rounding and actual weighted support are therefore
unidentified; the inequality is an observation for these fixtures, not an arbitrary
image reference implementation.

局部变化的原始值为：中心增加八灰度得 20，半径三的方环增加八灰度得 501，稀疏对称变化
增加八灰度得 355。仅在这些静止对称输入上，结果满足
`0 <= SAD7/49 - raw/128 <= 1/128`，其中 `SAD7` 为独立计算的 7×7 绝对差之和。
直接截断 `128*SAD7/49` 在统一亮度变化样例中多一个原始单位。多种倒数精度以及不同的整数
边界规则均能吻合已采值，却会在未测输入上分歧，因此内部精度、舍入和实际加权范围仍未确定。
上述不等式仅是这些样例的观察，不能作为任意图像的硬件参考实现。

Keep status, geometry and forward/backward checks separate from this residual.
It responds to additive brightness and differs from the wrapper's explicitly
zero-mean 7×7 RMS `patch_error`. No universal `max_error` accuracy threshold is
established, so its default remains `None`. Other levels, nonzero motion, gain,
clipping and arbitrary texture were not calibrated by this residual diagnostic.

状态、几何及前后向检查须与该残差分别判断。它响应加性亮度变化，与封装明确计算的
零均值 7×7 RMS `patch_error` 不同。尚未建立通用的 `max_error` 精度阈值，默认仍为 `None`。
该残差诊断未校准其他层级、非零运动、增益、饱和或任意纹理。

The preserved first hardware run accepted 158 wrong periodic-texture matches.
Its lighting sequence retained only six correct correspondences out of 200 nominal
tracking slots, despite no wrong accepted points. Those observations remain the
baseline; the bounded guarded comparison above does not remove those failures.

保留的首轮硬件测试接受了 158 个周期纹理错误对应；亮度序列虽然没有错误接受，却只在
200 个名义跟踪位置中保留六个正确对应。原始结果保持为基线；上述有界保护对照不抹去
这些失败记录，主机筛选结果也不能替代实机验收。

## Explicit corrected runtime / 显式修正运行库

First create a separate corrected copy using the procedure in
[IVE LK Runtime](IVE_LK_RUNTIME.md). The original and copied vendor libraries keep
their original copyright and license; this source does not redistribute either.

先按 [IVE LK 运行库说明](IVE_LK_RUNTIME.md) 创建独立修正副本。厂商原库及副本保留原版权
和许可，本源码不分发任何一种二进制。

```sh
python3 scripts/run_with_ive_runtime.py --runtime ive_runtime/librve.so --check
PYTHONPATH=ive_runtime python3 scripts/run_with_ive_runtime.py --runtime ive_runtime/librve.so -- examples/ive_lk_tracker.py frame01.png frame02.png
```

For the loading example, `ive_runtime` must be a separate application module
directory containing matching `visiong.py`, `_visiong.so` and the corrected
`librve.so`. Arrange application imports to select that directory and preserve
the existing required RKNN/runtime configuration. VisionG's existing preloader
checks the selected module's directory before the vendor directory. Merely loading
a copy elsewhere does not prevent an import from another binding directory from
opening a second IVE runtime. Confirm the actual mapping after importing VisionG;
the measured level matrix used the isolated-directory arrangement.

加载示例中的 `ive_runtime` 须是独立应用模块目录，包含相互匹配的 `visiong.py`、
`_visiong.so` 及修正 `librve.so`。应用导入须选中该目录，保留既有 RKNN 及所需运行库配置。
VisionG 的既有预加载先检查所选模块目录，再检查厂商目录；只在别处加载副本，不能阻止从
其他绑定目录导入时打开第二个 IVE 运行库。导入 VisionG 后须核对实际映射；上述层级矩阵
使用的是独立目录方案。

The explicit runner verifies the complete corrected SHA256 and executable ELF
mapping before any load. It rejects imported VisionG bindings, mapped IVE runtimes
and existing loader registrations. It loads Rockit globally before the selected
IVE copy, verifies that copy's mapping, then runs the application. It leaves the
library search environment and RKNN selection unchanged and preserves this runner
if VisionG performs its existing Python restart. It never modifies system files,
patches memory or silently switches an already running process.

显式入口在任何加载前校验完整修正哈希及 ELF 可执行映射；拒绝已导入 VisionG、已经映射
IVE 或已注册加载器句柄的进程。随后先全局加载 Rockit，再加载并核验所选 IVE 副本，最后运行
应用。它保留库搜索环境及 RKNN 选择，并在 VisionG 按既有流程重启 Python 时保留显式入口；
不会修改系统文件、修补内存或静默切换已运行进程的库。
