<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# IVE 接口覆盖与验证 / Coverage and validation

公开头文件 `rk_mpi_ive.h` 包含 38 个图像执行入口和 7 个辅助入口。本地源码直接调用 37 个执行入口；单层 LK 功能通过可配置金字塔入口的第零层实现，避开旧入口固定四层的问题。这个清单表示静态覆盖和替代关系，不能用作全部功能已通过实机验收的证明。

The public header declares 38 execution entries and 7 helpers. Source calls cover 37 entries directly. Single-level LK uses level zero of the configurable entry because the older entry fixes four levels. Static coverage and documented replacements do not establish hardware correctness.

## 接口范围 / Interfaces

| 类别 / Category | 接口 / Interfaces | 主要约定 / Contract |
|---|---|---|
| DMA | `dma`, `dma_transfer` | copy、interval、3/8 字节非零填充；GRAY8 字节平面 / byte planes |
| 卷积 / Convolution | `filter`, `filter_ex`, `sobel`, `sobel_ex` | 25 系数，显式 norm，U8/S8/U16/S16 输出 / coefficients, norm and output type |
| 边缘和梯度 / Edges and gradients | `canny`, `canny_ex`, `mag_and_ang`, `mag_and_ang_ex`, `norm_grad`, `norm_grad_config` | 核、输出方向和打包规则明确 / explicit kernel, direction and packing |
| 形态学及秩 / Morphology and rank | `dilate`, `erode`, `morphology`, `ordered_stat_filter` | 形态学为 0/255 二值；任意 5×5 掩码 / binary morphology and configurable mask |
| 逻辑和算术 / Logic and arithmetic | `logic_op`, `add`, `add_weighted`, `sub` | AND/OR/XOR、Q0.16 加权、ABS/SHIFT；GRAY8 / weighted addition and subtraction |
| 阈值 / Threshold | `threshold`, `threshold_u8`, `threshold_s16`, `threshold_u16` | U8 全 8 模式、S16 全 4 模式、U16 全 2 模式 / all declared modes |
| 转换和查表 / Cast and lookup | `cast_8bit_ex`, `cast_16bit_ex`, `map_typed` | 有符号模式、比例和偏置；U8/S16/U16 LUT / signed modes, ratio, bias and typed LUT |
| 统计 / Statistics | `hist`, `equalize_hist_ex`, `integral`, `ncc`, `ncc_raw` | 三种均衡模式、三种积分模式、原始 NCC 聚合 / complete modes and raw aggregates |
| 连通区域 / Connected components | `ccl`, `ccl_config` | 4/8 邻域、非连续标签、面积等号包含；中心为 bbox 中心 / connectivity, sparse labels and inclusive area |
| 色彩 / Color | `csc`, `csc_bytes`, RGB/YUV 便利方法 / helpers | 16 模式及紧凑 packed/planar、420/422 布局 / modes and explicit layouts |
| 局部纹理 / Local texture | `lbp` | NORMAL 有符号阈值、ABS 无符号阈值，显式 0 保留 / full threshold ranges |
| 光流和金字塔 / Flow and pyramid | `lk_optical_flow`, `lk_optical_flow_config`, `lk_optical_flow_single`, `create_pyramid` | 配置、状态、原始误差和单层入口 / controls, status, raw error and single-level entry |
| 角点 / Corners | `st_corner`, `st_corner_candidates`, `st_corner_sorted` | 两阶段、候选和原生排序入口 / two-stage, candidates and sorted entry |
| 背景 / Background | `gmm`, `gmm2`, `gmm_config`, `gmm2_config`, `match_bg_model`, `update_bg_model`, `IVEModel.reset` | 固定尺寸、显式初始化；已核验运行库支持 1–5 模型 / bound dimensions, explicit lifecycle and version-checked multi-model GMM |

## 原生与组合 / Native and composed operations

该运行库中部分公开封装存在已定位的问题。组合实现仍调用真实 IVE 算子；以下 CPU 步骤是接口的一部分，应纳入端到端计时：

Some vendor wrappers have identified defects. Composed paths still call real IVE operators; the CPU work below is part of end-to-end timing:

- U8 两个三段阈值模式和完整 8 位转换使用 256 项硬件 Map。CPU 生成查表，不逐像素计算结果。Two U8 threshold modes and complete byte casts use hardware Map.
- 有符号 16 位转换在无符号硬件缩放前后进行绝对值和符号恢复。Signed word casts prepare and restore signs around unsigned hardware scaling.
- SAD 使用原生 SAD-only 输出，再用原生 U16 阈值组合出 BOTH/THRESH。SDK 的 BOTH 路径会覆盖输出描述符；THRESH-only 在诊断中未写目标。SAD composes SAD-only and U16 threshold.
- NormGrad 始终计算双分量，再选择或完整分辨率打包；不使用 SDK 的缩小 COMBINE 路径。Normalized gradients select or pack full-resolution components.
- MagAndAng 的原生阈值字段未传入硬件。公开 `threshold` 定义为包含等号的最小幅值，幅值不足时由 CPU 将幅值和角度同时清零。MagAndAng applies its explicit inclusive minimum on the CPU after hardware gradients.
- 单层 LK 使用 `LKOpticalFlowPyr(max_level=0)`；旧 `LKOpticalFlow` 在上层地址为空时仍配置 level=3，不能据此实现单层计算。Single-level LK uses the public configurable level-zero path.
- U16 LUT 使用 S16 硬件查表保持全部 16 位，再按 U16 读取，避开原生 U16 分支遗漏的步长换算。U16 LUTs preserve raw word bits through the S16 hardware path.
- 任意二值形态学掩码使用 U16 滤波求和和硬件阈值，图外像素按零补齐。Configurable binary morphology uses U16 filter sums and threshold, with zero outside the image.
- CSC planar 布局通过 CPU 打包/拆包。422 通过重复行与抽取行组合 420 CSC；高图像分块避开已测原生高度异常，支持公开的 4096 行边界，见 [CSC 分块](IVE_CSC_422_CHUNKS.md)。Planar layouts use CPU packing; tall CSC images use bounded native blocks.

SDK 的 Init/Deinit、描述符转换、查询与容量计算不是独立图像算子。当前接口采用同步调用并持有缓冲区，不开放缺少生命周期约束的原始异步句柄。

Lifecycle, descriptor and query helpers are not image operators. The wrapper uses synchronous calls and retains its buffers.

## 不支持项与边界 / Unsupported entries and limits

- `GradFg` 虽存在导出，但目标库固定返回 72，即 NOT_SUPPORT。`GradFg` is an unsupported stub in the tested runtime.
- SDK 中有 Resize 结构和枚举，目标库没有对应执行导出或 HAL 单元。There is no callable IVE Resize entry in the tested runtime.
- 额外 Math/Warp 导出不在公开 API 头文件中。反汇编显示部分是内部排队辅助，Warp 是硬件坐标计算加 CPU 采样，且存在工作区、步长和坐标约定风险。它们不作为可用公开接口发布。Undocumented Math/Warp exports are not exposed as supported public APIs.
- GMM/GMM2 多模型使用已核验版本的独立输出兼容路径；其他运行库仍只提供单模型。匹配图随模型保存，并使用紧凑行布局，见 [GMM 说明](IVE_GMM.md)。Multi-model GMM uses a version-checked separate-output path.
- `Pyramid_GetSize` 对部分步长的容量估算不足，分配容量按实际各层步长计算。Pyramid capacity follows actual aligned levels.
- 已测运行库的 LK 层级 0–2 有当前帧步长索引错误；封装拒绝已知故障实现中的这些调用，独立修正副本及范围见 [LK 运行库说明](IVE_LK_RUNTIME.md)。The measured runtime needs the documented current-stride correction for LK levels 0–2.

## 验证入口 / Validation

```sh
python3 scripts/audit_ive_coverage.py sdk/include/rk_mpi_ive.h
python3 scripts/check_ive_modes.py --sizes 320x240 --loops 2 --groups threshold,dma
python3 scripts/check_ive_modes.py --sizes 320x240,322x242 --loops 2 --groups extended
python3 scripts/check_ive_modes.py --sizes 320x240,322x242 --loops 2 --groups arithmetic,equalize_degenerate
python3 scripts/check_ive_modes.py --sizes 320x240 --loops 2 --groups border,gradient_quantization
python3 scripts/check_ive_extended.py --sizes 320x240 --loops 2 --groups norm sad flow corners
python3 scripts/check_ive_extended.py --sizes 320x240 --loops 1 --groups lbp
python3 scripts/check_ive_extended.py --sizes 320x240 --loops 1 --groups model stateful
python3 scripts/check_ive_gmm.py --sizes 64x64,66x70 --loops 2 --output gmm.json
python3 scripts/check_ive_extended.py --sizes 320x240,322x242,323x243 --loops 2 --groups pyramid
python3 scripts/check_ive_edges_logic.py --groups logic,invalid --loops 2 --output logic.json
python3 scripts/check_ive_edges_logic.py --groups canny --edge-sizes 320x240 --quick --output canny.json
python3 scripts/check_ive_csc.py --quick --sizes 320x240,322x242 --output csc.json
python3 scripts/check_ive_csc.py --sizes 64x66,322x3 --loops 2 --require-complete --output csc_complete.json
```

检查结果区分 `passed`、`failed`、`unverified`。`--require-complete` 还会将未确认的数值约定视为未完成。硬件中断只证明发生提交，不能替代逐像素真值、边界、格式和状态序列检查。

Checks distinguish passed, failed and unverified. `--require-complete` also rejects unresolved conventions. Interrupt counts do not replace value, boundary, format or state-sequence validation.

部分脚本保留早期浮点、取整或补边候选的观测行。候选被独立固定参考取代后，它们仍作为历史证据保留，不代表另一个算子验收失败；CSC 的 `capability_counts` 将能力检查与历史观测分开。其他脚本中的 `unverified` 必须结合该行的原因和严格数值检查判断，不能简单累计为已通过项。

Some reports retain exploratory alternatives after a strict independent reference is established. CSC separates these historical observations from capability counts. Unverified rows in other scripts must be read with their stated reason; they are not automatically accepted.

逻辑检查枚举全部 65,536 个字节输入对。SHIFT 同时保留原始候选观测和独立的严格检查：输出为 `((A-B)>>1)&255`，同一输入别名相减为零。候选观测不替代固定规则的逐字节验收。Canny 使用相同核生成有符号导数，再与 OpenCV 的导数输入接口和独立整数 NMS/滞后参考比较；五组受控边界、方向、连接、阈值与非对齐尺寸全图一致，具体范围及未隔离的条件见 [Canny 说明](IVE_CANNY.md)。

Logic checks enumerate all 65,536 byte pairs. SHIFT retains exploratory captures and separately checks the fixed byte contract `((A-B)>>1)&255`, including zero output for identical aliases. Five Canny fixture groups match full-frame OpenCV derivative-input and independent integer references; the linked notes retain their bounded scope and unresolved conditions.

## 实测范围 / Measured scope

本地 RV1106 验证覆盖上述全部公开算子类别，包含原有入口、补齐模式和组合路径。参考包括整数真值、OpenCV、源数据及保留输出校验、重复运行、尺寸切换和硬件中断。不同组件记录各自的构建、脚本和运行库指纹；以下范围不表示所有参数和所有图像已被穷举。

Local RV1106 checks cover every public operator category above using integer truth, OpenCV, source preservation, retained outputs, repetition, dimension changes and submission evidence. Each capture retains its build, script and runtime provenance. This does not exhaust every parameter or image.

| 范围 / Scope | 独立数值依据 / Numeric evidence |
|---|---|
| 色彩 / Color | RGB→HSV 四模式、RGB→YUV 四模式及 YUV→RGB/HSV 八条路径，分别穷举全部 24 位输入；整数兼容参考逐字节一致。布局和 420/422 采样相位另测。 / Exhaustive per-color checks plus separate layout and sampling fixtures. |
| 阈值、DMA、Map、转换 / Threshold, DMA, Map and casts | 所有公开模式；完整 U8、S16/U16 值域；非对齐行、填充相位、比例/偏置极值。 / Complete scalar domains and declared modes with stride and control fixtures. |
| 逻辑及加权 / Logic and weighted addition | 全部 65,536 个字节输入对；加权另测 15 对系数；SHIFT 为带符号右移后取低字节。 / Full byte pairs with selected coefficient pairs. |
| 均衡 / Equalization | 非退化整数 CDF 舍入、不同外部直方图；全部 256 个单 bin 退化分布。 / Integer CDF and every single-bin distribution. |
| 卷积、梯度、形态学 / Convolution, gradients and morphology | 全图复制边界、算术右移和类型饱和；完整受控小梯度及大梯度方向边界；另测 S16 极值对称限幅。 / Full-frame references, controlled gradient domain and signed extremes. |
| 积分、NCC、SAD、CCL / Integral, NCC, SAD and CCL | 整数和、平方和、原始聚合、块归约、4/8 邻域与稀疏标签；COMBINE 的 28/36 位饱和打包见 [数值约定](IVE_NUMERIC.md)。 / Independent sums, reductions and labeling fixtures; measured saturating COMBINE packing. |
| LBP、秩、角点 / LBP, rank and corners | LBP 全部 512 个邻域；秩参考；角点受控位置、候选打包、排序和尺寸切换。 / Exhaustive binary neighborhoods and controlled rank/corner fixtures. |
| 光流、金字塔 / Flow and pyramid | 相同帧与已知平移；OpenCV 对照；金字塔逐层整数滤波/抽样参考。 / Motion truth and exact pyramid references. |
| GMM/GMM2、码本 / Background models | 初始化、训练、切换、重置、1–5 模型、模型/其他算子交错及非对齐行；长期序列确认默认参数可能淘汰组件，见 [GMM 说明](IVE_GMM.md)。 / Lifecycle, bounded state sequences, independent state and observed forgetting. |

HSV、YUV 和 RGB 转换的完整范围、公式与哈希见 [IVE_HSV.md](IVE_HSV.md)；梯度极值与量化见 [IVE.md](IVE.md)。

下列结果仍有明确边界：修正步长后的 LK 在低层级较大位移时仍可能返回有效但错误的局部收敛结果；residual 是无符号 Q9.7 灰度残差量，完整归一化与舍入仍未被识别，不能当作像素位移误差，见 [跟踪说明](IVE_LK_TRACKER.md)。Canny 的五组全图 OpenCV 对照通过，不据此宣称全部图像、核、阈值和连接规则完全等同 OpenCV。角点 lambda 是受图像上下文影响的原始分数，见 [数值约定](IVE_NUMERIC.md)。GMM 的运行库版本约束及默认遗忘、CSC 分块范围和未公开入口限制见上文。

Correcting LK stride does not guarantee global correspondence. Residual is an unsigned Q9.7 photometric quantity with incompletely identified normalization, rather than pixel displacement error. Controlled full-frame Canny agreement does not establish universal OpenCV equivalence. Corner lambda remains context-dependent; runtime, forgetting and layout limits remain applicable.

公开定义来自 Rockchip 的 [API 头文件](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/include/rk_mpi_ive.h)、[类型与模式](https://github.com/LuckfoxTECH/luckfox-pico/blob/main/media/ive/ive/include/rk_ive.h) 和 [官方示例](https://github.com/LuckfoxTECH/luckfox-pico/tree/main/media/ive/ive/simulator)。原始版权和许可保持不变，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
