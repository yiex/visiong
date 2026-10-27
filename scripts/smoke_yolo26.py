# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check RV1106 inference, ROI, repeated loading and optional camera input.
验证 RV1106 推理、ROI、重复加载以及可选的摄像头输入。
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
import visiong as vg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.rknn")
    parser.add_argument("--labels", default="coco_80_labels_list.txt")
    parser.add_argument("--image", default="bus.jpg")
    parser.add_argument("--output", default="smoke.json")
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    image = vg.ImageBuffer.load(args.image)
    report = {"library": vg.__version__, "checks": []}
    model = vg.NPU(vg.ModelType.YOLO26, args.model, args.labels)
    expected = None
    for i in range(200):
        detections = model.infer(image)
        data = [(d.class_id, d.score, d.box) for d in detections]
        if expected is None:
            expected = data
        if data != expected:
            raise AssertionError("static-frame outputs changed")
    assert detections and any(d.label == "bus" for d in detections)
    report["checks"].append("200 identical-frame inferences stable; bus detected")
    roi = (32, 64, image.width - 64, image.height - 128)
    detections = model.infer(image, roi=roi)
    for d in detections:
        x, y, w, h = d.box
        assert 0 <= x < x + w <= image.width - 1 and 0 <= y < y + h <= image.height - 1
    report["roi_detections"] = [{"label": d.label, "score": d.score, "box": d.box} for d in detections]
    report["checks"].append("ROI inference and source-coordinate bounds")
    # A non-square input exercises inverse letterbox. / 使用非方形输入验证 letterbox 坐标还原。
    cropped = image.crop((0, 80, image.width, image.height - 160))
    for d in model.infer(cropped):
        x, y, w, h = d.box
        assert 0 <= x < x + w <= cropped.width - 1 and 0 <= y < y + h <= cropped.height - 1
    report["checks"].append("non-square input letterbox")
    del model
    gc.collect()
    for _ in range(4):
        model = vg.NPU("yolo26", args.model, args.labels)
        assert model.infer(image)
        del model
        gc.collect()
    report["checks"].append("four load/infer/release cycles")
    with tempfile.TemporaryDirectory(prefix="visiong_labels_") as directory:
        label = Path(directory) / "labels.txt"
        label.write_text("wrong\n", encoding="utf-8")
        try:
            vg.NPU("yolo26", args.model, str(label))
        except RuntimeError:
            report["checks"].append("label-count mismatch rejected")
        else:
            raise AssertionError("wrong label count was accepted")
    if args.camera:
        model = vg.NPU("yolo26", args.model, args.labels)
        camera = vg.Camera(640, 480, "yuv")
        try:
            camera.skip(10)
            times = []
            counts = []
            for _ in range(60):
                start = time.perf_counter()
                frame = camera.snapshot()
                detections = model.infer(frame)
                times.append((time.perf_counter() - start) * 1000)
                counts.append(len(detections))
            report["camera"] = {
                "frames": len(times),
                "mean_ms": sum(times) / len(times),
                "fps": len(times) * 1000 / sum(times),
                "detection_counts": counts,
            }
            for d in detections:
                frame.draw_rectangle(d.box, (0, 255, 0), 2)
            frame.save("camera_yolo26.png")
        finally:
            camera.release()
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("SMOKE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
