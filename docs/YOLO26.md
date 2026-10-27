<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# YOLO26

[中文](#中文) | [English](#english)

## 中文

VisionG 通过现有 `NPU` 接口支持 YOLO26 one2one 检测模型，后处理适配自
[Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo/tree/837db8e992815ff8647eadb709f07067f4bc3a71/examples/yolo26)。
RV1106 上已验证 INT8 模型；转换脚本同时提供 RV1103 目标选项。

### 使用

准备 RKNN 模型、对应标签文件和测试图片后：

```python
import visiong as vg

detector = vg.NPU("yolo26", "yolo26n.rknn", "coco_80_labels_list.txt", box=0.25)
image = vg.ImageBuffer.load("bus.jpg")

for detection in detector.infer(image):
    print(detection.label, detection.score, detection.box)
    image.draw_rectangle(detection.box, (0, 255, 0), 2)

image.save("result.png")
```

- 模型类型也可以写为 `vg.ModelType.YOLO26`，已有枚举值保持兼容。
- 预处理使用 RGB、归一化除以255、letterbox 填充值114。
- `box` 控制置信度阈值。one2one 后处理不执行 NMS 或 DFL，共用的 `nms` 参数对此模型无效。
- 返回 `Detection` 列表，框为源图坐标 `(x, y, width, height)`，支持现有 ROI 偏移和边界裁切。
- 沿用 VisionG 的128框容量；Rockchip 独立示例的容量为256框。
- 适配器只接受两路 INT8 量化输出，并校验形状、标签数量、量化参数和缓冲区大小。

### 模型转换

在安装 RKNN Toolkit2、ONNX 的 Linux 环境中运行；opset 降级验证还需要 ONNX Runtime 和 NumPy。
校准图片应覆盖实际部署场景。

```sh
python scripts/convert_yolo26.py yolo26n.onnx \
  --dataset dataset.txt --output yolo26n.rknn
```

脚本沿用 Rockchip 的输出截断方式：

| 输出节点 | 含义 | 640×640、80类时的形状 |
|---|---|---|
| `/model.23/Sigmoid_output_0` | 类别概率 | `[1, 80, 8400]` |
| `/model.23/Mul_2_output_0` | 解码后的 xyxy 坐标 | `[1, 4, 8400]` |

候选数量与类别数量由张量元数据确定。支持三维 NCHW、NHWC、保持导出顺序的 UNDEFINED，
以及高度为1的四维 NHWC 输出。其他布局会明确报错。

对于 opset 高于19的输入，脚本提取上述输出后执行 ONNX 语义版本转换，检查计算图，
并通过固定输入比较转换前后的 ONNX Runtime 输出。生成的 JSON 记录原模型元数据、
来源 URL（可通过 `--source-url` 提供）、工具链版本与模型 SHA256。

验证环境使用 Toolkit2 2.5.0 / runtime 2.5.0。目标固件需提供兼容的 RKNN 运行库。
若在独立目录测试运行库，可将 `run_yolo26.sh` 与 `visiong.py`、`_visiong.so`、可选的
`librknnmrt.so` 放在一起，再通过脚本运行 Python；脚本会优先加载同目录运行库。
公共源码和 Python 包不随此功能新增厂商运行库二进制。

### 验证与性能

后处理使用 INT8 NEON 最大值归约、阈值筛选和部分 TopK 排序，仅对保留的候选计算类别和坐标。
同分候选按索引排序；工作内存在模型实例中复用，并由现有 NPU 互斥锁保护。

```sh
g++ -std=c++17 -O3 -fno-fast-math -Isrc \
  scripts/test_yolo26_decode.cpp -o test_yolo26_decode
./test_yolo26_decode
```

ARM 交叉编译时使用目标工具链，并增加 `-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard`。
参考测试覆盖标量/NEON、平面/交错布局、带填充步长、尾部元素、不同类别数及量化阈值边界。

板端可使用 `scripts/bench_yolo26.py` 测量完整 `infer()`，使用 `scripts/smoke_yolo26.py`
检查固定图像一致性、ROI、重复加载及摄像头输入。功能检查默认使用上游 `bus.jpg` 和 COCO 标签。
低层接口与完整接口可能采用不同的输出布局，不能直接将两个接口的计时相减作为后处理耗时。
后处理微基准也不等同于模型端到端 FPS；模型、分辨率、运行库和硬件频率均会影响结果。

### 开源来源

Rockchip 衍生文件保留原始版权与 Apache-2.0 许可，并标明 VisionG 的适配修改。
新写的集成接口、绑定和测试沿用项目 LGPL-3.0-or-later。
来源和具体版本见 [licenses/SOURCES.md](../licenses/SOURCES.md)，
许可正文见 [licenses/Apache-2.0.txt](../licenses/Apache-2.0.txt)。

Ultralytics 预训练权重属于独立产物，原始 ONNX 元数据标明 AGPL-3.0。
转换为 RKNN 不改变其来源或许可；分发模型时应同时保留来源、元数据、哈希及对应许可文本。

## English

### YOLO26 on RV1103 / RV1106

`NPU("yolo26", model_path, label_path, box=0.25)` uses the Rockchip Model Zoo
one2one detection path: maximum class score, confidence filter, score TopK,
decoded xyxy coordinates and inverse letterbox. It does not perform NMS or
DFL. The shared `nms` constructor argument is ignored for this model.

The existing VisionG result capacity is 128 detections (Rockchip's standalone
example uses 256). Existing ModelType numeric values are preserved; YOLO26 is
appended. Python also accepts `ModelType.YOLO26`. Returned boxes follow VisionG's
`(x, y, width, height)` API, with ROI offsets and clipping to the source image.
YOLO26 preprocessing uses RGB / 255 and letterbox padding 114.

Only quantized INT8 scores and decoded INT8 boxes are accepted by this RV1106
adapter. It validates two output tensors, batch size, native layouts, buffer
bounds, positive quantization scales, equal candidate counts and label count.
Unsupported layouts fail explicitly. It accepts planar rank-3 NCHW or native
interleaved rank-3 NHWC, rank-3 UNDEFINED in exported `[1,C,N]` order, and NHWC
rank-4 with singleton H. Candidate counts and
class counts come from tensor metadata rather than hardcoded 8400/80.

### Conversion

Use RKNN Toolkit2 in a Linux environment, with an RV1106-compatible runtime on
the target. Use representative calibration images for the deployed scene.
The tested build uses Toolkit2 2.5.0 and runtime 2.5.0. Firmware vendor libraries
can load an older runtime first: when testing a side-by-side `librknnmrt.so`,
use the supplied `run_yolo26.sh` with the runtime placed beside the Python module.
The Python library archive does not redistribute vendor runtime binaries.

```sh
python scripts/convert_yolo26.py yolo26n.onnx \
  --dataset dataset.txt --output yolo26n.rknn
```

The script truncates to `/model.23/Sigmoid_output_0` (class probabilities) and
`/model.23/Mul_2_output_0` (decoded boxes), as in Rockchip's example. For upstream
opset-20 assets it extracts those outputs and uses ONNX's version converter to
opset 19, checks the graph and compares ONNX Runtime outputs before conversion.
The generated JSON records model hashes, toolkit version and original metadata.

### Performance and correctness

INT8 NEON maximum reduction avoids per-class floating-point dequantization;
only TopK survivors have their class IDs and box coordinates decoded. TopK
uses `nth_element` then sorts the survivors, with candidate index breaking ties.
Scratch storage is reused per model under the existing NPU mutex.

`scripts/test_yolo26_decode.cpp` checks scalar and NEON against an independent
float/full-sort reference, including alternate layouts, padded strides, tails,
one/four/80-class cases and exact quantization thresholds. Cross-compile it with
`-std=c++17 -O3 -fno-fast-math -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -Isrc`
and run on the board. The tests also reject non-finite or non-positive quantization scales.
Its microbenchmark is not an end-to-end model/FPS measurement.

Use `scripts/bench_yolo26.py` for full `infer()` timing and `scripts/smoke_yolo26.py`
for repeatability, ROI, repeated loading and optional camera checks with the upstream
bus image and COCO labels. Low-level and high-level interfaces may use different output
layouts; subtracting their timings does not isolate post-processing time.

### Attribution

Adapted algorithm/conversion code retains Rockchip's 2026 copyright and
Apache-2.0 license, with modification notices. See `licenses/SOURCES.md` and
`licenses/Apache-2.0.txt`. Ultralytics pretrained weights are separate artifacts;
their original metadata states AGPL-3.0. Retain their source, license and hashes
when distributing converted models. VisionG's license does not relicense them.
