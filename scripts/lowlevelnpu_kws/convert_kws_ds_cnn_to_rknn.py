#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
from pathlib import Path

from rknn.api import RKNN

from common import NUM_FRAMES, NUM_MEL_BINS, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the DS-CNN KWS ONNX model to RKNN format.")
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-rknn", type=Path, required=True)
    parser.add_argument("--target-platform", default="rv1106")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def checked(ret: int, step_name: str) -> None:
    if ret != 0:
        raise RuntimeError("{} failed with code {}".format(step_name, ret))


def main() -> int:
    args = parse_args()
    onnx_model = args.onnx_model.resolve()
    dataset = args.dataset.resolve()
    output_rknn = args.output_rknn.resolve()
    output_rknn.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=args.verbose)
    try:
        print("--> Config RKNN")
        checked(
            rknn.config(
                target_platform=args.target_platform,
                optimization_level=args.optimization_level,
            ),
            "rknn.config",
        )

        print("--> Load ONNX")
        load_ret = rknn.load_onnx(model=str(onnx_model))
        if load_ret != 0:
            print("Embedded-shape load failed, retrying with explicit shape metadata.")
            load_ret = rknn.load_onnx(
                model=str(onnx_model),
                inputs=["input"],
                input_size_list=[[1, NUM_FRAMES, NUM_MEL_BINS]],
                outputs=["logits"],
            )
        checked(load_ret, "rknn.load_onnx")

        print("--> Build RKNN")
        checked(
            rknn.build(do_quantization=True, dataset=str(dataset)),
            "rknn.build",
        )

        print("--> Export RKNN")
        checked(rknn.export_rknn(str(output_rknn)), "rknn.export_rknn")
    finally:
        rknn.release()

    write_json(
        output_rknn.with_suffix(".conversion.json"),
        {
            "onnx_model": str(onnx_model),
            "dataset": str(dataset),
            "output_rknn": str(output_rknn),
            "target_platform": args.target_platform,
            "optimization_level": args.optimization_level,
        },
    )
    print("RKNN model written to {}".format(output_rknn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
