// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include <cstring>

void bind_io_devices(py::module_& m) {
    py::class_<DisplayUDP>(m, "DisplayUDP")
        .def(py::init<const std::string&, int, int>(), "udp_ip"_a = "172.32.0.100", "udp_port"_a = 8000, "jpeg_quality"_a = 75,
             "Creates DisplayUDP. Optionally call init(ip_address, port, jpeg_quality) to set or change target.\n"
             "DisplayUDP keeps its own JPEG session lock to reduce VENC re-init: color priority is BGR > RGB > YUV > other, and size only grows while smaller frames are black-padded to the locked canvas.\n"
             "创建 DisplayUDP。也可之后调用 init(ip_address, port, jpeg_quality) 修改目标。\n"
             "DisplayUDP 为了减少 VENC 反复重初始化，会维持自己的 JPEG 会话锁：颜色优先级为 BGR > RGB > YUV > 其他；尺寸只会向更大方向扩张，较小帧会补黑边到锁定画布。")
        .def("init", &DisplayUDP::init, "ip_address"_a, "port"_a, "jpeg_quality"_a = 75,
             py::call_guard<py::gil_scoped_release>(),
             "Initializes or re-initializes the UDP sender (target IP, port, JPEG quality 1-100).")
        .def("display", &DisplayUDP::display, "img_buf"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Encodes the ImageBuffer to JPEG (VENC) and sends it via UDP to the configured address. The DisplayUDP-local lock may convert color format and black-pad smaller frames before encoding.\n"
             "把 ImageBuffer 编码成 JPEG（VENC）并通过 UDP 发到目标地址。编码前可能会根据 DisplayUDP 的本地锁定策略做颜色格式转换和黑边补齐。")
        .def("release", &DisplayUDP::release, py::call_guard<py::gil_scoped_release>(), "Releases DisplayUDP resources.")
        .def("is_initialized", &DisplayUDP::is_initialized, "Checks if DisplayUDP is initialized.");

    py::class_<TouchPoint>(m, "TouchPoint", "Represents a single touch coordinate.")
        .def_readonly("x", &TouchPoint::x)
        .def_readonly("y", &TouchPoint::y)
        .def("__repr__", [](const TouchPoint &p) {
            return "<TouchPoint x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) + ">";
        });

    py::class_<TouchDevice, std::unique_ptr<TouchDevice>>(m, "Touch", "Interface for I2C touch screen devices.")
        .def(py::init([](const std::string& chip_model, 
                          const std::string& i2c_bus, 
                          const py::object& width, 
                          const py::object& height, 
                          const py::object& rotation) 
        {
            auto device = create_touch_device(chip_model, i2c_bus);
            if (!device) {
                throw py::value_error("Failed to create touch device for model: " + chip_model);
            }

            const bool custom_geom = !width.is_none() && !height.is_none() && !rotation.is_none();
            if (custom_geom) {
                device->configure_geometry(py::cast<int>(width), py::cast<int>(height), py::cast<int>(rotation));
            }

            return device;
        }), 
        "chip_model"_a = "FT6336U",
        "i2c_bus"_a = "/dev/i2c-3",
        "original_width"_a = py::none(),
        "original_height"_a = py::none(),
        "rotation_degrees"_a = py::none(),
        "Initializes the touch controller. This constructor performs all necessary hardware setup.\n"
        "Raises a RuntimeError if initialization fails (e.g., device not found).\n\n"
        "By default, initializes for an FT6336U on /dev/i2c-3, configured for a\n"
        "240x320 screen rotated 270 degrees to a 320x240 landscape view.")

        .def("release", &TouchDevice::release, py::call_guard<py::gil_scoped_release>(), "Releases the touch device (closes I2C).")
        .def("is_pressed", &TouchDevice::is_pressed, "Returns True if at least one finger is on the screen.")
        
        .def("read", &TouchDevice::read, py::return_value_policy::move, 
             py::call_guard<py::gil_scoped_release>(),
             "Reads all active touch points and returns a list of TouchPoint objects.")

        .def("configure_geometry", &TouchDevice::configure_geometry,
             "original_width"_a, "original_height"_a, "rotation_degrees"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Re-configures the screen geometry and coordinate rotation at runtime.");
    
    py::class_<DisplayFB>(m, "DisplayFB", "Framebuffer display. mode: 'high' (default) or 'low' refresh.")
        .def(py::init([](py::args args, py::kwargs kwargs) {
            std::string mode_str = "high";
            if (args.size() > 0) {
                if (args.size() > 1) {
                    throw std::invalid_argument("DisplayFB takes at most 1 positional argument ('mode').");
                }
                mode_str = py::cast<std::string>(args[0]);
            }
            if (kwargs && kwargs.contains("mode")) {
                mode_str = py::cast<std::string>(kwargs["mode"]);
            }
            
            return std::make_unique<DisplayFB>(parse_displayfb_mode(mode_str));
        }))
        .def("display", static_cast<bool (DisplayFB::*)(const ImageBuffer&)>(&DisplayFB::display),
            "img_buf"_a, py::call_guard<py::gil_scoped_release>(),
            "Displays the full image on the framebuffer (non-blocking). Returns True on success.")
        .def("display", static_cast<bool (DisplayFB::*)(const ImageBuffer&, const std::tuple<int, int, int, int>&)>(&DisplayFB::display),
            "img_buf"_a, "roi"_a, py::call_guard<py::gil_scoped_release>(),
            "Displays the image region (x, y, w, h) on the framebuffer. Returns True on success.")
        .def("release", &DisplayFB::release, py::call_guard<py::gil_scoped_release>(), "Releases framebuffer resources.")
        .def("is_initialized", &DisplayFB::is_initialized, "Returns True if the framebuffer is initialized.")
        .def_property_readonly("screen_width", &DisplayFB::screen_width, "Framebuffer width in pixels.")
        .def_property_readonly("screen_height", &DisplayFB::screen_height, "Framebuffer height in pixels.")
        .def("__repr__", [](const DisplayFB& self) {
            return py::str("DisplayFB(screen_width={}, screen_height={})")
                .format(self.screen_width(), self.screen_height());
        });
}

#if VISIONG_WITH_NPU
void bind_npu(py::module_& m) {
    py::enum_<ModelType>(m, "ModelType")
        .value("YOLOV5", ModelType::YOLOV5)
        .value("RETINAFACE", ModelType::RETINAFACE)
        .value("FACENET", ModelType::FACENET)
        .value("YOLO11", ModelType::YOLO11)
        .value("YOLO11_SEG", ModelType::YOLO11_SEG)
        .value("YOLO11_POSE", ModelType::YOLO11_POSE)
        .value("LPRNET", ModelType::LPRNET)
        .export_values();

    py::class_<Detection>(m, "Detection", "Single detection from YOLOv5/RetinaFace/YOLO11/YOLO11_SEG/YOLO11_POSE inference.")
        .def_readonly("box", &Detection::box, "Bounding box as (x, y, w, h).")
        .def_readonly("score", &Detection::score, "Confidence in [0, 1] (normalized from model output).")
        .def_readonly("class_id", &Detection::class_id, "Class index.")
        .def_readonly("label", &Detection::label, "Class label string.")
        .def_readonly("landmarks", &Detection::landmarks, "RetinaFace only: list of 5 (x, y) float coords: [left_eye, right_eye, nose, left_mouth, right_mouth]; empty for other models.")
        .def_readonly("keypoints", &Detection::keypoints, "Pose model (YOLO11_POSE): list of (x, y, score); empty for non-pose models.")
        .def_readonly("mask_points", &Detection::mask_points, "YOLO11_SEG only: list of (x, y) contour points; empty for other models.")
        .def("__repr__", [](const Detection &d) {
            return "<Detection label='" + d.label + "', score=" + std::to_string(d.score) + ">";
        });

    py::class_<NPU>(m, "NPU")
        .def(py::init([](const std::string& model_type_str,
                         const std::string& model_path,
                         const std::string& label_path,
                         float box,
                         float nms) {
            const ModelType model_type_enum = parse_model_type(model_type_str);
            return std::make_unique<NPU>(model_type_enum, model_path, label_path, box, nms);
        }),
        "model_type"_a, "model_path"_a, "label_path"_a = "",
        "box"_a = 0.25f, "nms"_a = 0.45f,
	"Initializes NPU with model_type ('yolov5', 'retinaface', 'facenet', 'yolo11', 'yolo11_seg', 'yolo11_pose', 'lprnet'), model_path, optional label_path, box_thresh (default 0.25), nms_thresh (default 0.45).")

        .def(py::init([](ModelType model_type,
                         const std::string& model_path,
                         const std::string& label_path,
                         float box,
                         float nms) {
            return std::make_unique<NPU>(model_type, model_path, label_path, box, nms);
        }),
        "model_type"_a, "model_path"_a, "label_path"_a = "",
        "box"_a = 0.25f, "nms"_a = 0.45f,
        "Initializes NPU with ModelType enum, model_path, optional label_path, box_thresh (default 0.25), nms_thresh (default 0.45).")

        .def("infer", &NPU::infer,
             "img_buf"_a, "roi"_a = std::make_tuple(0, 0, 0, 0),
             "model_format"_a = "rgb",
             py::call_guard<py::gil_scoped_release>(),
             "Runs inference. For detection/pose models (YOLOv5, RetinaFace, YOLO11, YOLO11_SEG, YOLO11_POSE) returns a list of Detection. "
             "For FACENET use get_face_feature(); for LPRNET use recognize_plate(). model_format: 'rgb' or 'bgr'. "
             "Does not support grayscale input.")

        .def("get_face_feature", &NPU::get_face_feature, "face_image"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Extracts a 128-dimensional feature vector from a cropped face image. Requires FACENET model_type; raises RuntimeError otherwise.")

        .def("recognize_plate", &NPU::recognize_plate, "plate_image"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Recognizes a license plate from a cropped image. For use with LPRNET models. Returns a string.")

	.def_static("get_feature_distance", &NPU::get_feature_distance, "feature1"_a, "feature2"_a,
             "Euclidean distance between two 128-D face feature vectors. Returns 100.0 if either length is not 128.")
	.def("is_initialized", &NPU::is_initialized, "Checks if the NPU is initialized.")
        .def_property_readonly("model_width", &NPU::model_width, "Input width required by the loaded model.")
        .def_property_readonly("model_height", &NPU::model_height, "Input height required by the loaded model.");
    py::class_<LowLevelTensorInfo>(m, "LowLevelTensorInfo", "Low-level RKNN tensor metadata.")
        .def_readonly("index", &LowLevelTensorInfo::index)
        .def_readonly("name", &LowLevelTensorInfo::name)
        .def_readonly("dims", &LowLevelTensorInfo::dims)
        .def_readonly("format", &LowLevelTensorInfo::format)
        .def_readonly("type", &LowLevelTensorInfo::type)
        .def_readonly("quant_type", &LowLevelTensorInfo::quant_type)
        .def_readonly("zero_point", &LowLevelTensorInfo::zero_point)
        .def_readonly("scale", &LowLevelTensorInfo::scale)
        .def_readonly("num_elements", &LowLevelTensorInfo::num_elements)
        .def_readonly("size_bytes", &LowLevelTensorInfo::size_bytes)
        .def_readonly("size_with_stride_bytes", &LowLevelTensorInfo::size_with_stride_bytes)
        .def_readonly("w_stride", &LowLevelTensorInfo::w_stride)
        .def_readonly("h_stride", &LowLevelTensorInfo::h_stride)
        .def_readonly("pass_through", &LowLevelTensorInfo::pass_through);

    py::class_<LowLevelNPU>(m, "LowLevelNPU", "Low-level RKNN runtime wrapper for teaching and custom tensor IO.")
        .def(py::init<const std::string&, uint32_t>(),
             "model_path"_a,
             "init_flags"_a = 0,
             "Initializes RKNN from model file path and allocates tensor memories for all inputs/outputs.")
        .def("is_initialized", &LowLevelNPU::is_initialized)
        .def("num_inputs", &LowLevelNPU::num_inputs)
        .def("num_outputs", &LowLevelNPU::num_outputs)
        .def("input_tensors", &LowLevelNPU::input_tensors)
        .def("output_tensors", &LowLevelNPU::output_tensors)
        .def("input_tensor", &LowLevelNPU::input_tensor, "index"_a)
        .def("output_tensor", &LowLevelNPU::output_tensor, "index"_a)
        .def("input_shape", &LowLevelNPU::input_shape, "index"_a)
        .def("output_shape", &LowLevelNPU::output_shape, "index"_a)
        .def("sdk_versions", [](const LowLevelNPU& self) {
            auto versions = self.sdk_versions();
            py::dict out;
            out["api"] = versions.first;
            out["driver"] = versions.second;
            return out;
        })
        .def_property_readonly("last_run_us", &LowLevelNPU::last_run_us)
        .def("set_core_mask", &LowLevelNPU::set_core_mask, "core_mask"_a,
             "Sets RKNN core mask. Accepts: 'auto', '0', '1', '2', '0_1', '0_1_2'.")
        .def("set_input_attr", &LowLevelNPU::set_input_attr,
             "index"_a, "tensor_type"_a, "tensor_format"_a, "pass_through"_a,
             "Rebinds one input tensor attr. Example: set_input_attr(0, 'uint8', 'nhwc', False).")
        .def("reset_input_attr", &LowLevelNPU::reset_input_attr, "index"_a,
             "Resets an input tensor attr to its startup value.")
        .def("set_input_bytes", [](LowLevelNPU& self, int index, py::bytes payload,
                                     bool zero_pad, bool sync_to_device) {
            std::string raw = payload;
            {
                py::gil_scoped_release release;
                self.set_input_buffer(index, raw.data(), raw.size(), zero_pad, sync_to_device);
            }
        },
             "index"_a, "payload"_a,
             "zero_pad"_a = true,
             "sync_to_device"_a = true,
             "Writes raw bytes into an input tensor buffer.")
        .def("set_input_array", [](LowLevelNPU& self, int index, py::array arr,
                                     bool quantize_if_needed, bool zero_pad, bool sync_to_device) {
            py::array c_array = py::array::ensure(arr, py::array::c_style);
            if (!c_array) {
                throw std::invalid_argument("set_input_array: expected a contiguous numpy array.");
            }
            py::buffer_info info = c_array.request();
            if (info.format == py::format_descriptor<float>::format() &&
                info.itemsize == static_cast<ssize_t>(sizeof(float))) {
                py::gil_scoped_release release;
                self.set_input_from_float(index,
                                          static_cast<const float*>(info.ptr),
                                          static_cast<size_t>(info.size),
                                          quantize_if_needed,
                                          zero_pad,
                                          sync_to_device);
            } else {
                const size_t bytes = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
                py::gil_scoped_release release;
                self.set_input_buffer(index, info.ptr, bytes, zero_pad, sync_to_device);
            }
        },
             "index"_a, "array"_a,
             "quantize_if_needed"_a = true,
             "zero_pad"_a = true,
             "sync_to_device"_a = true,
             "Writes a numpy array into input tensor memory. Float arrays can be auto-quantized.")
        .def("set_input_image", &LowLevelNPU::set_input_image,
             "index"_a, "image"_a,
             "color_order"_a = "rgb",
             "keep_aspect"_a = true,
             "pad_value"_a = 114,
             "driver_convert"_a = true,
             py::call_guard<py::gil_scoped_release>(),
             "Writes ImageBuffer into input tensor memory using RGA path when possible.")
        .def("sync_input_to_device", &LowLevelNPU::sync_input_to_device, "index"_a,
             py::call_guard<py::gil_scoped_release>())
        .def("sync_output_from_device", &LowLevelNPU::sync_output_from_device, "index"_a,
             py::call_guard<py::gil_scoped_release>())
        .def("sync_all_outputs_from_device", &LowLevelNPU::sync_all_outputs_from_device,
             py::call_guard<py::gil_scoped_release>())
        .def("run", &LowLevelNPU::run,
             "sync_outputs"_a = true,
             "non_block"_a = false,
             "timeout_ms"_a = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Runs RKNN. Optional sync_outputs controls output cache sync.")
        .def("wait", &LowLevelNPU::wait,
             "timeout_ms"_a = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("output_bytes", [](LowLevelNPU& self, int index, bool with_stride, bool sync_from_device) {
            std::vector<uint8_t> raw;
            {
                py::gil_scoped_release release;
                raw = self.output_bytes(index, with_stride, sync_from_device);
            }
            return py::bytes(reinterpret_cast<const char*>(raw.data()), raw.size());
        },
             "index"_a,
             "with_stride"_a = false,
             "sync_from_device"_a = true,
             "Returns raw output bytes.")
        .def("output_float", [](LowLevelNPU& self, int index, bool dequantize_if_needed, bool sync_from_device) {
            std::vector<float> values;
            std::vector<int64_t> shape64;
            {
                py::gil_scoped_release release;
                values = self.output_float(index, dequantize_if_needed, sync_from_device);
                shape64 = self.output_shape(index);
            }
            std::vector<ssize_t> shape;
            shape.reserve(shape64.size());
            size_t expected = 1;
            for (int64_t d : shape64) {
                const ssize_t dim = static_cast<ssize_t>(d);
                shape.push_back(dim);
                if (d > 0) {
                    expected *= static_cast<size_t>(d);
                }
            }
            if (shape.empty() || expected != values.size()) {
                shape = {static_cast<ssize_t>(values.size())};
            }
            py::array_t<float> out(shape);
            if (!values.empty()) {
                std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(float));
            }
            return out;
        },
             "index"_a,
             "dequantize_if_needed"_a = true,
             "sync_from_device"_a = true,
             "Returns output tensor as float32 numpy array (dequantized when possible).")
        .def("output_array", [](LowLevelNPU& self, int index, bool dequantize_if_needed, bool sync_from_device) {
            if (dequantize_if_needed) {
                std::vector<float> values;
                std::vector<int64_t> shape64;
                {
                    py::gil_scoped_release release;
                    values = self.output_float(index, true, sync_from_device);
                    shape64 = self.output_shape(index);
                }
                std::vector<ssize_t> shape;
                shape.reserve(shape64.size());
                size_t expected = 1;
                for (int64_t d : shape64) {
                    shape.push_back(static_cast<ssize_t>(d));
                    if (d > 0) {
                        expected *= static_cast<size_t>(d);
                    }
                }
                if (shape.empty() || expected != values.size()) {
                    shape = {static_cast<ssize_t>(values.size())};
                }
                py::array_t<float> out(shape);
                if (!values.empty()) {
                    std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(float));
                }
                return py::object(out);
            }
            std::vector<uint8_t> raw;
            {
                py::gil_scoped_release release;
                raw = self.output_bytes(index, false, sync_from_device);
            }
            py::array_t<uint8_t> out(raw.size());
            if (!raw.empty()) {
                std::memcpy(out.mutable_data(), raw.data(), raw.size());
            }
            return py::object(out);
        },
             "index"_a,
             "dequantize_if_needed"_a = true,
             "sync_from_device"_a = true,
             "Returns output as float array (default) or raw uint8 vector when dequantize_if_needed=False.")
        .def("input_dma_fd", &LowLevelNPU::input_dma_fd, "index"_a)
        .def("output_dma_fd", &LowLevelNPU::output_dma_fd, "index"_a);

    py::class_<OCRResult>(m, "OCRResult", "Single OCR result with text and quadrilateral location.")
        .def_readonly("quad", &OCRResult::quad, "Text box corners as [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] clockwise.")
        .def_readonly("rect", &OCRResult::rect, "Axis-aligned bounding box as (x, y, w, h).")
        .def_readonly("det_score", &OCRResult::det_score, "Detection confidence score.")
        .def_readonly("text", &OCRResult::text, "Recognized UTF-8 text.")
        .def_readonly("text_score", &OCRResult::text_score, "Recognition confidence score.")
        .def("__repr__", [](const OCRResult &r) {
            return "<OCRResult text='" + r.text + "', det_score=" + std::to_string(r.det_score) + ">";
        });

    py::class_<PPOCR>(m, "PPOCR")
        .def(py::init<const std::string&, const std::string&, const std::string&, float, float, bool,
                      const std::string&, float, bool, float, const std::string&, float>(),
             "det_model_path"_a, "rec_model_path"_a, "dict_path"_a = "",
             "det_threshold"_a = 0.3f, "box_threshold"_a = 0.5f, "use_dilate"_a = true,
             "rec_fast_model_path"_a = "", "rec_fast_max_ratio"_a = 2.4f,
             "rec_fast_enable_fallback"_a = true, "rec_fast_fallback_score_thresh"_a = 0.2f,
             "model_input_format"_a = "rgb", "det_unclip_ratio"_a = 1.6f,
             "Initializes PPOCR with DET model, main REC model, optional fast REC model, dictionary path, postprocess thresholds, and model input color order (rgb/bgr).")
        .def("infer", &PPOCR::infer, "img_buf"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Runs DET+REC OCR on one image and returns a list of OCRResult.")
        .def("is_initialized", &PPOCR::is_initialized, "Checks whether PPOCR runtime is initialized.")
        .def_property_readonly("det_model_width", &PPOCR::det_model_width, "DET model input width.")
        .def_property_readonly("det_model_height", &PPOCR::det_model_height, "DET model input height.")
        .def_property_readonly("rec_model_width", &PPOCR::rec_model_width, "REC model input width.")
        .def_property_readonly("rec_model_height", &PPOCR::rec_model_height, "REC model input height.");

    py::class_<NanoTrackResult>(m, "NanoTrackResult", "Single frame tracking output from NanoTrack.")
        .def_readonly("box", &NanoTrackResult::box, "Tracked bounding box as (x, y, w, h).")
        .def_readonly("score", &NanoTrackResult::score, "Tracker confidence score in [0, 1].");

    py::class_<NanoTrack>(m, "NanoTrack")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             "template_model"_a, "search_model"_a, "head_model"_a,
             "Loads three RKNN NanoTrack models: template backbone, search backbone, and head.")
        .def("init", &NanoTrack::init,
             "img_buf"_a, "bbox"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Initializes tracker state with the first frame and initial bbox (x, y, w, h).")
        .def("track", &NanoTrack::track,
             "img_buf"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Tracks target on a new frame and returns NanoTrackResult.")
        .def("is_initialized", &NanoTrack::is_initialized,
             "Returns True after init() succeeds.")
        .def("reset", &NanoTrack::reset,
             "Clears tracker state. You must call init() again before track().")
        .def_property_readonly("bbox", &NanoTrack::bbox,
             "Latest tracked bbox as (x, y, w, h).")
        .def_property_readonly("score", &NanoTrack::score,
             "Latest confidence score.");

    py::class_<KWSResult>(m, "KWSResult", "Single keyword spotting classification result.")
        .def_readonly("class_id", &KWSResult::class_id, "Predicted class index.")
        .def_readonly("label", &KWSResult::label, "Predicted class label.")
        .def_readonly("score", &KWSResult::score, "Top-1 probability after softmax.")
        .def_readonly("scores", &KWSResult::scores, "Full per-class softmax probabilities.")
        .def("__repr__", [](const KWSResult& result) {
            return "<KWSResult label='" + result.label + "', score=" + std::to_string(result.score) + ">";
        });

    py::class_<KWS>(m, "KWS")
        .def(py::init<const std::string&, const std::string&, int, int, int, int, int, int, float, float, float, bool, uint32_t>(),
             "model_path"_a,
             "labels_path"_a = "",
             "sample_rate"_a = 16000,
             "clip_samples"_a = 16000,
             "window_size_ms"_a = 30,
             "window_stride_ms"_a = 20,
             "fft_size"_a = 512,
             "num_mel_bins"_a = 40,
             "lower_edge_hertz"_a = 20.0f,
             "upper_edge_hertz"_a = 4000.0f,
             "epsilon"_a = 1e-6f,
             "normalize"_a = true,
             "init_flags"_a = 0,
             "Initializes a dedicated keyword spotting pipeline with native audio frontend and LowLevelNPU backend.")
        .def("infer_pcm16",
             [](KWS& self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> audio) {
                 const py::buffer_info info = audio.request();
                 KWSResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.infer_pcm16(static_cast<const int16_t*>(info.ptr),
                                               static_cast<size_t>(info.size));
                 }
                 return result;
             },
             "audio"_a,
             "Runs keyword spotting on a contiguous int16 PCM numpy array.")
        .def("infer_float",
             [](KWS& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> audio) {
                 const py::buffer_info info = audio.request();
                 KWSResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.infer_float(static_cast<const float*>(info.ptr),
                                               static_cast<size_t>(info.size));
                 }
                 return result;
             },
             "audio"_a,
             "Runs keyword spotting on a contiguous float32 numpy array.")
        .def("infer",
             [](KWS& self, py::array audio) {
                 py::array c_array = py::array::ensure(audio, py::array::c_style);
                 if (!c_array) {
                     throw std::invalid_argument("KWS.infer expects a contiguous numpy array.");
                 }
                 const py::buffer_info info = c_array.request();
                 if (info.format == py::format_descriptor<int16_t>::format() &&
                     info.itemsize == static_cast<ssize_t>(sizeof(int16_t))) {
                     KWSResult result;
                     {
                         py::gil_scoped_release release;
                         result = self.infer_pcm16(static_cast<const int16_t*>(info.ptr),
                                                   static_cast<size_t>(info.size));
                     }
                     return result;
                 }
                 if (info.format == py::format_descriptor<float>::format() &&
                     info.itemsize == static_cast<ssize_t>(sizeof(float))) {
                     KWSResult result;
                     {
                         py::gil_scoped_release release;
                         result = self.infer_float(static_cast<const float*>(info.ptr),
                                                   static_cast<size_t>(info.size));
                     }
                     return result;
                 }
                 throw std::invalid_argument("KWS.infer expects a contiguous int16 or float32 numpy array.");
             },
             "audio"_a,
             "Dispatches to infer_pcm16() or infer_float() based on numpy dtype.")
        .def("infer_wav", &KWS::infer_wav,
             "wav_path"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Loads a PCM16 WAV file, converts it to mono if needed, and runs keyword spotting.")
        .def("is_initialized", &KWS::is_initialized,
             "Checks whether the keyword spotting pipeline is initialized.")
        .def_property_readonly("sample_rate", &KWS::sample_rate,
             "Frontend sample rate in Hz.")
        .def_property_readonly("clip_samples", &KWS::clip_samples,
             "Expected clip length in samples.")
        .def_property_readonly("num_frames", &KWS::num_frames,
             "Frontend frame count.")
        .def_property_readonly("num_mel_bins", &KWS::num_mel_bins,
             "Frontend mel bin count.")
        .def_property_readonly("num_classes", &KWS::num_classes,
             "Model output class count.")
        .def_property_readonly("labels", &KWS::labels,
             "Loaded class labels.")
        .def_property_readonly("last_run_us", &KWS::last_run_us,
             "Last measured RKNN run time in microseconds.");

    // Register nk_command_buffer as an opaque pybind handle.
    // 将 nk_command_buffer 注册为 pybind 使用的不透明句柄。
}
#endif

#if VISIONG_WITH_GUI
void bind_gui(py::module_& m) {
    py::class_<nk_command_buffer>(m, "Canvas", "An opaque handle to a Nuklear drawing canvas.");

    py::class_<GUIManager>(m, "GUI", "Nuklear GUI Manager")
        .def(py::init([](int width, int height, const py::object& font_path_obj, const py::object& pre_chars_obj) {
            std::string font_path = "";
            if (!font_path_obj.is_none()) {
                font_path = font_path_obj.cast<std::string>();
            }
            std::string pre_chars = "";
            if (!pre_chars_obj.is_none()) {
                pre_chars = pre_chars_obj.cast<std::string>();
            }
            return std::make_unique<GUIManager>(width, height, font_path, pre_chars);
        }), "width"_a, "height"_a, "font"_a = py::none(), "predefine_chars"_a = py::none(),
        R"(Initializes the GUI manager. 
    Font configuration must be provided here.

    Args:
        width (int): Width of the GUI canvas.
        height (int): Height of the GUI canvas.
        font (str, optional): Path to a custom TTF font file. Defaults to 'assets/font.ttf' if it exists, otherwise uses a built-in font.
        predefine_chars (str, optional): A string of characters to bake into the font atlas to save memory (e.g., '?????23').
    )")
        .def("begin_frame", &GUIManager::begin_frame, "touch_device"_a,
             "Starts a new frame. Pass Touch device or None. Call before any widgets.")
        .def("begin_window", &GUIManager::begin_window, "title"_a, "x"_a, "y"_a, "w"_a, "h"_a, "flags"_a = "border|movable|scalable|minimizable|title",
             "Begins a window. x,y,w,h are floats. Returns True if the window is visible.")
        .def("end_window", &GUIManager::end_window, "Ends the current window.")
        .def("end_frame", &GUIManager::end_frame, "target_image"_a,
             "Ends the frame and renders the GUI into the given ImageBuffer.")
        .def("layout_row_dynamic", &GUIManager::layout_row_dynamic, "height"_a, "cols"_a = 1,
             "Starts a row with dynamic column widths. cols: number of widgets in the row.")
        .def("layout_row_static", &GUIManager::layout_row_static, "height"_a, "item_width"_a, "cols"_a,
             "Starts a row with fixed widget width.")
        .def("layout_row_begin", &GUIManager::layout_row_begin, "format"_a, "row_height"_a, "cols"_a,
             "Begins a row with custom format (e.g. percentage). Use layout_row_push then layout_row_end.")
        .def("layout_row_push", &GUIManager::layout_row_push, "value"_a, "Pushes a column size (after layout_row_begin).")
        .def("layout_row_end", &GUIManager::layout_row_end, "Ends the row started by layout_row_begin.")
        .def("group_begin", &GUIManager::group_begin, "title"_a, "flags"_a = "border",
             "Begins a group. Returns True if the group is visible. Must call group_end after.")
        .def("group_end", &GUIManager::group_end, "Ends the current group.")
        .def("label", &GUIManager::label, "text"_a, "align"_a = "left", "Draws a label. align: 'left', 'right', 'center'.")
        .def("label_wrap", &GUIManager::label_wrap, "text"_a, "Draws a label with word wrapping.")
        .def("button", &GUIManager::button, "label"_a, "Returns True if the button was clicked.")
        .def("slider", &GUIManager::slider, "label"_a, "value"_a, "min"_a, "max"_a, "step"_a,
             "Slider widget. Touch interaction is relative-drag based: press anywhere in range, then slide horizontally to adjust.")
        .def("checkbox", &GUIManager::checkbox, "label"_a, "is_checked"_a,
             "Checkbox. Returns the new checked state (bool).")
        .def("option", &GUIManager::option, "label"_a, "is_active"_a,
             "Radio option. Returns True if this option is selected.")
        .def("edit_string", &GUIManager::edit_string, "text"_a, "max_len"_a = 256,
             "Single-line text edit. Returns (changed: bool, text: str).")
        .def("progress", &GUIManager::progress, "current"_a, "max"_a, "is_modifyable"_a = true,
             "Progress bar. If modifiable, touch interaction is relative-drag based rather than absolute jump-to-position.")
        .def("button_image", &GUIManager::button_image, "image"_a, "Creates a clickable button from an ImageBuffer. Returns True if clicked.")
        .def("tree_node", &GUIManager::tree_node, "title"_a, "is_expanded"_a,
             "Begins a tree node. Returns True if expanded. Call tree_pop when done with children.")
        .def("tree_pop", &GUIManager::tree_pop, "Ends the current tree node.")
        .def("property_int", &GUIManager::property_int, "name"_a, "value"_a, "min"_a, "max"_a, "step"_a, "inc_per_pixel"_a = 1.0f,
             "Integer property control. Touch drag adjusts by horizontal delta instead of jumping to an absolute position.")
        .def("property_float", &GUIManager::property_float, "name"_a, "value"_a, "min"_a, "max"_a, "step"_a, "inc_per_pixel"_a = 1.0f,
             "Float property control. Touch drag adjusts by horizontal delta instead of jumping to an absolute position.")
        .def("combo_begin", &GUIManager::combo_begin, "text"_a, "width"_a, "height"_a,
             "Begins a combo box. Popup placement is touch-first and constrained within the parent window.")
        .def("combo_item", &GUIManager::combo_item, "text"_a, "Adds an item to the current combo. Returns True if selected.")
        .def("combo_end", &GUIManager::combo_end, "Ends the combo box.")
        .def("contextual_begin", &GUIManager::contextual_begin, "width"_a, "height"_a,
             "Begins a contextual menu. On touch devices this is typically opened via long press, with popup placement constrained to the parent window.")
        .def("contextual_item", &GUIManager::contextual_item, "text"_a, "Adds an item to the contextual menu. Returns True if clicked.")
        .def("contextual_end", &GUIManager::contextual_end, "Ends the contextual menu.")

        .def("chart_begin", &GUIManager::chart_begin, "type"_a, "count"_a, "min_val"_a, "max_val"_a, "Begins a chart section. Type can be 'lines' or 'columns'.")
        .def("chart_push", &GUIManager::chart_push, "value"_a, "Pushes a new value to the active chart.")
        .def("chart_end", &GUIManager::chart_end, "Ends the chart section.")
        .def("menubar_begin", &GUIManager::menubar_begin, "Begins a menubar at the top of the current window.")
        .def("menubar_end", &GUIManager::menubar_end, "Ends the menubar section.")
        .def("menu_begin", &GUIManager::menu_begin, "label"_a, "width"_a, "height"_a, "Begins a dropdown menu.")
        .def("input_is_pointer_down_in_rect", &GUIManager::input_is_pointer_down_in_rect, "rect"_a, "primary_pointer"_a = true,
             "Touch-first alias of input_is_mouse_down_in_rect. Returns True if the primary pointer is held inside the rect.")
        .def("input_is_pointer_dragging_in_rect", &GUIManager::input_is_pointer_dragging_in_rect,
             "Touch-first alias of input_is_mouse_dragging_in_rect. Returns drag state and delta, including momentum when active.")
        .def("is_title_bar_active", &GUIManager::is_title_bar_active,
             "Touch-first alias of is_title_bar_pressed. Returns True while the title bar is held or being dragged.")
        .def("get_scroll_delta_y", &GUIManager::get_scroll_delta_y,
             "Touch-first alias of get_smart_scroll_dy. Returns current scroll delta, including momentum when active.")
        .def("menu_item", &GUIManager::menu_item, "label"_a, "Adds a clickable item to a menu.")
        .def("menu_end", &GUIManager::menu_end, "Ends the menu section.")
        .def("tooltip", &GUIManager::tooltip, "text"_a, "Shows a tooltip for the previously declared widget. On touch devices it appears on long press.")

        .def("get_canvas", &GUIManager::get_canvas, py::return_value_policy::reference,
             "Returns the current window's Canvas for custom drawing (stroke_line, fill_rect, draw_text, etc.).")
        .def("widget_bounds", &GUIManager::widget_bounds, "canvas"_a,
             "Returns (x, y, w, h) of the last laid-out widget. Pass the canvas from get_canvas.")
        .def("input_is_mouse_down_in_rect", &GUIManager::input_is_mouse_down_in_rect, "rect"_a, "left_mouse"_a = true,
             "Returns True if the primary pointer is held inside the given (x,y,w,h) rect. On touch devices this follows finger contact.")
        .def("window_set_focus", &GUIManager::window_set_focus, "name"_a,
             "Sets focus to the window with the given name.")
        .def("window_drag_from_pos", &GUIManager::window_drag_from_pos, "canvas"_a,
             "Direct-manipulation window drag. On touch devices this moves the current window while dragging its title bar.")
        .def("window_set_scroll", &GUIManager::window_set_scroll, "scroll_y"_a,
             "Sets the current window's vertical scroll offset.")
        .def("input_is_mouse_dragging_in_rect", &GUIManager::input_is_mouse_dragging_in_rect,
             "Returns (is_dragging, scroll_dy, (x,y,w,h) content_rect). On touch devices scroll_dy includes locked-axis drag and fling momentum.")
        .def("is_title_bar_pressed", &GUIManager::is_title_bar_pressed,
             "Returns True if the current window title bar is actively held or dragged.")
        .def("get_content_height", &GUIManager::get_content_height,
             "Returns the content area height of the current layout.")
        .def("push_style_vec2", &GUIManager::push_style_vec2, "name"_a, "x"_a, "y"_a,
             "Pushes a vec2 style. name: 'padding' or 'spacing'. Must be popped with pop_style.")
        .def("pop_style", &GUIManager::pop_style, "Pops the last pushed style (e.g. vec2).")
        .def("get_smart_scroll_dy", &GUIManager::get_smart_scroll_dy,
             "Returns touch-first scroll delta for the current window, including momentum when active.")
        .def("stroke_line", &GUIManager::stroke_line, "canvas"_a, "x0"_a, "y0"_a, "x1"_a, "y1"_a, "thickness"_a, "color"_a,
             "Draws a line on the canvas. color: (R,G,B,A).")
        .def("stroke_rect", &GUIManager::stroke_rect, "canvas"_a, "x"_a, "y"_a, "w"_a, "h"_a, "rounding"_a, "thickness"_a, "color"_a,
             "Draws a rectangle outline. color: (R,G,B,A).")
        .def("fill_rect", &GUIManager::fill_rect, "canvas"_a, "x"_a, "y"_a, "w"_a, "h"_a, "rounding"_a, "color"_a,
             "Fills a rectangle. color: (R,G,B,A).")
        .def("draw_text", &GUIManager::draw_text, "canvas"_a, "x"_a, "y"_a, "text"_a, "color"_a,
             "Draws text at (x,y). color: (R,G,B,A).")
        .def("set_style_color", &GUIManager::set_style_color, "property_name"_a, "color"_a,
             "Sets a theme color. property_name: 'text', 'header_bg', 'button_normal', 'button_hover', 'button_active'. color: (R,G,B,A).")
        .def("set_style_button_rounding", &GUIManager::set_style_button_rounding, "rounding"_a,
             "Sets button corner rounding radius.")
        .def("set_style_window_rounding", &GUIManager::set_style_window_rounding, "rounding"_a,
             "Sets window corner rounding radius.")
        .def("set_window_background_color", &GUIManager::set_window_background_color, "color"_a,
             "Sets the current window background color. color: (R,G,B,A).");
}
#endif
