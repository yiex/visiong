<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# GMM 与 GMM2 / GMM and GMM2

`gmm` 和 `gmm2` 保留单模型和双返回值接口。`gmm_config`、`gmm2_config`
可选择每像素 1–5 个模型，返回三个独立的 GRAY8 图像：前景、背景、匹配记录。

The original helpers retain one model and two outputs. The configurable helpers
accept 1–5 models per pixel and return independent foreground, background and
matched-record images.

```python
import visiong

ive = visiong.IVE()
model = visiong.IVEModel(width, height)
foreground, background, matched = ive.gmm_config(
    gray, model, max_models=3, first_frame=True
)
# Continue with new same-size GRAY8 frames. / 后续传入相同尺寸的灰度帧。
foreground, background, matched = ive.gmm_config(
    next_gray, model, max_models=3
)
```

GMM2 另需相同尺寸的 U16C1 factor 图。当前控制项与原有辅助接口保持一致；
`max_models` 是模型数量上限。首次调用、重置后、切换算法或改变数量时，
必须指定 `first_frame=True`，同时清空模型记录与匹配状态。

GMM2 additionally requires a same-size U16C1 factor image. Other hardware controls
retain the existing helper defaults. Initialize with `first_frame=True` after a
reset or when changing algorithm or model count.

同一个 `IVEModel` 的调用串行执行，匹配图随模型独立保存。不同模型之间、
与其他 IVE 算子之间可以交错调用。返回图像是快照，后续运算不会改写已返回结果。
参数错误在清空旧状态和硬件提交前拒绝；状态访问、硬件执行或结果读取失败后，
需重新初始化该模型。

Each model serializes access and owns its matched-record state. Models and other
IVE operators can be interleaved without sharing that state. Returned images are
snapshots. Invalid arguments preserve existing state; execution or state-access
errors require reinitialization.

匹配记录的硬件行布局是 `width` 字节紧凑行，不随灰度源图的 16 字节步长补齐。
模型内部为此使用独立描述符；返回的匹配图保持原始宽高。

Hardware matched records use packed `width`-byte rows independently of source
stride alignment. A dedicated descriptor preserves the original output dimensions.

默认内存容量覆盖五模型与码本需求，第二份模型缓冲区按需分配。
自定义 `model_size` 必须为 4 的倍数，且 GMM 至少需要
`align16(width) * height * max_models * 4 + 16` 字节。尺寸仍绑定创建模型时的
原始宽高，不因内部行步长补齐而改变。

Automatic capacity covers five-model GMM and codebook storage. A second model
buffer is allocated when needed. An explicit capacity must be four-byte aligned
and large enough for the selected number of models. The model remains bound to
its original image dimensions.

## 运行库兼容 / Runtime compatibility

已验证的 RV1106 运行库把公开 GMM 接口的模型输入、输出设为同一地址。
多模型新增记录会移动后续记录，实测可覆盖尚未读取的输入。多模型路径通过
该版本导出的调度入口分离输出，保留厂商锁、调度器与后处理；成功后交换两份
缓冲区。厂商库、驱动和固件均不需要改写。

The verified RV1106 runtime aliases public GMM model input and output. Inserting
records can overwrite unread input. The multi-model path separates output through
the runtime's exported scheduler, retains its lock and postprocessor, and swaps
buffers after success. It does not modify the vendor library, driver or firmware.

此路径依赖特定运行库 ABI，调用前同时核验完整文件 SHA-256、函数位置和关键
指令。当前接受原始版本及仅修正 LK 步长的已知副本：

This path uses a version-specific ABI and checks the complete library fingerprint
and relevant loaded functions before submission:

- 原始版本 / original: `2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455`
- LK 修正副本 / LK-corrected copy: `80c5c104f5bf02b88dafd1ed0867d90f2439fded2464174080d30ff97bd03409`

其他版本的多模型调用在提交前拒绝。原有单模型路径继续使用公开 SDK 接口。
需要 LK 修正副本时见 [LK 运行库说明](IVE_LK_RUNTIME.md)，加载选择必须显式进行。

Other library versions are rejected for multi-model calls. Single-model helpers
continue to use the public SDK. See the LK runtime notes for explicit selection of
the optional corrected copy.

## 匹配图解释与验证边界 / Matched records and validation scope

在上述版本的已测灰度序列中，匹配字节高 4 位为当前像素的记录数量，最低位
表示本帧是否匹配已有记录，位 1–3 为匹配记录的排序索引。它与紧凑模型记录共同
构成持久状态，不能将共享工作图代替该状态。

On the tested versions and grayscale sequences, the high nibble is the per-pixel
record count, bit zero indicates an existing-record match, and bits 1–3 identify
the matched sorted record. Both this image and the compact records are stateful.

实机验证包括 2–5 模型的不同灰度背景建立与交错复现、逐像素记录分组、独立
输出对照、重置、不同容量、模型交错和其他算子干扰。该范围证明了状态维护与
对应背景复现，不意味着已重建厂商的全部高斯参数更新、替换策略或 factor 作用。
OpenCV MOG2 的参数和数值结果不应直接视为此硬件的逐字节参考。

Hardware checks cover distinct background establishment and recall, compact
record grouping, independent output comparisons, reset, capacity and interleaving.
They establish the tested state behavior, not a complete reconstruction of the
Gaussian update rules. OpenCV MOG2 is not a byte-exact reference for this hardware.

连续 180 帧的五背景交替序列进一步确认：默认参数不保证一直保留全部五个组件。
两种算法均在序列下标 45（从零计数）由五个组件减少为四个，下标 46 再次出现
对应灰度时重建该组件。不经过 VisionG 的独立调度调用逐帧得到完全相同的三份
输出，并核对了 31 份完整模型记录，因此该淘汰不能归因于封装的输入输出覆盖。
另一组 114 帧稀有背景序列也发生淘汰，紧接重现帧的重复输入能够匹配。
这些有界观测不建立通用遗忘时长，也没有重建全部权重更新、删除和替换公式。

In a continuous 180-frame five-background sequence, both algorithms dropped a
component at zero-based frame 45 and recreated it on frame 46. Independent vendor
scheduling without VisionG reproduced every output and verified 31 complete model
checkpoints. A separate 114-frame rare-background sequence also showed forgetting
and successful matching on the immediate repeated return. The defaults do not
guarantee permanent retention, and these runs establish no general forgetting time.

可在目标机重跑以下有界检查，覆盖 1–5 模型、两种算法、对齐与非对齐尺寸、输入
保留、其他硬件算子干扰、快照保留及重置复现。该脚本只检查建立与两轮背景复现，
不会把短测试通过当作长期保留保证。

The portable check exercises establishment and two recall cycles, input and
snapshot preservation, unrelated hardware work, and reset/replay consistency:

```sh
python3 scripts/check_ive_gmm.py --sizes 64x64,66x70 --loops 2 --output gmm.json
```
