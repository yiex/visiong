# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
# Source / 来源：airockchip/rknn_model_zoo, examples/yolo26/python/convert.py.
# Revision / 版本：837db8e992815ff8647eadb709f07067f4bc3a71.
# Modified for VisionG: argument validation, verified opset conversion and provenance metadata.
# VisionG 修改：校验参数、验证 opset 转换、记录来源元数据并确保释放工具链资源。
# License / 许可全文：licenses/Apache-2.0.txt.
"""Convert the YOLO26 one2one head to RV1106 INT8 without NMS or TopK in RKNN.
将 YOLO26 one2one 检测头转换为 RV1106 INT8 模型，RKNN 图中不包含 NMS 或 TopK。
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import onnx
from rknn.api import RKNN

OUTPUTS = ["/model.23/Sigmoid_output_0", "/model.23/Mul_2_output_0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="rv1106", choices=["rv1106", "rv1103"])
    parser.add_argument("--source-url", help="Optional provenance URL for this specific ONNX artifact.")
    return parser.parse_args()


def prepare_onnx(onnx_path: Path, output_path: Path):
    model = onnx.load(str(onnx_path))
    nodes = {out for node in model.graph.node for out in node.output}
    if not all(name in nodes for name in OUTPUTS):
        raise ValueError("Expected Rockchip YOLO26 one2one nodes are missing; check ONNX export version.")
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or any(x <= 0 for x in shape):
        raise ValueError("Expected fixed [1,3,H,W] input, got " + str(shape))
    # Convert the graph semantically; changing only the opset label is insufficient.
    # 使用 ONNX 版本转换器转换图语义，不能只改写 opset 版本号。
    load_path = onnx_path
    if max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")) > 19:
        extracted = output_path.with_suffix(".one2one.onnx")
        onnx.utils.extract_model(str(onnx_path), str(extracted), [model.graph.input[0].name], OUTPUTS)
        truncated = onnx.load(str(extracted))
        converted = onnx.version_converter.convert_version(truncated, 19)
        onnx.checker.check_model(converted)
        load_path = output_path.with_suffix(".opset19.onnx")
        onnx.save(converted, str(load_path))
        # Verify with a deterministic normalized RGB input. / 使用固定种子的归一化 RGB 输入验证图转换。
        import numpy as np
        import onnxruntime as ort

        x = np.random.default_rng(26).random(shape, dtype=np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        before = ort.InferenceSession(str(extracted), options, providers=["CPUExecutionProvider"])
        after = ort.InferenceSession(str(load_path), options, providers=["CPUExecutionProvider"])
        a = before.run(None, {before.get_inputs()[0].name: x})
        b = after.run(None, {after.get_inputs()[0].name: x})
        for lhs, rhs in zip(a, b):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)
        print(
            "ONNX opset conversion verified:",
            [float(np.max(np.abs(lhs - rhs))) for lhs, rhs in zip(a, b)],
            flush=True,
        )
    return model, shape, load_path


def checked(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError("{} failed with code {}".format(operation, result))


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, shape, load_path = prepare_onnx(args.onnx, args.output)
    rknn = RKNN(verbose=False)
    try:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.target)
        checked(rknn.load_onnx(model=str(load_path), outputs=OUTPUTS), "rknn.load_onnx")
        checked(rknn.build(do_quantization=True, dataset=str(args.dataset)), "rknn.build")
        checked(rknn.export_rknn(str(args.output)), "rknn.export_rknn")
    finally:
        rknn.release()
    metadata = {
        "target": args.target,
        "input_shape": shape,
        "outputs": OUTPUTS,
        "quantization": "INT8",
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "rknn_input_onnx_sha256": hashlib.sha256(load_path.read_bytes()).hexdigest(),
        "source_url": args.source_url,
        "rockchip_reference_commit": "837db8e992815ff8647eadb709f07067f4bc3a71",
        "onnx_opset_conversion_verified": load_path != args.onnx,
        "rknn_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "toolkit_version": importlib.metadata.version("rknn-toolkit2"),
        "onnx_metadata": {p.key: p.value for p in model.metadata_props},
        "calibration_images": args.dataset.read_text().splitlines(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
