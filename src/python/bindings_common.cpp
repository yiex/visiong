// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/core/BufferStateMachine.h"
#include "core/internal/rga_utils.h"

std::string get_image_buffer_compact_bytes(const ImageBuffer& img) {
    if (!img.is_valid()) {
        return std::string();
    }
    const void* src_ptr = img.get_data();
    const int bpp = get_bpp_for_format(img.format);
    const size_t compact_size = img.width * img.height * bpp / 8;
    std::string user_data(compact_size, '\0');

    visiong::bufstate::prepare_cpu_read(img);
    copy_data_from_stride(user_data.data(), src_ptr,
                          img.width * bpp / 8, img.height,
                          img.w_stride * bpp / 8);

    return user_data;
}

py::bytes get_image_buffer_bytes(const ImageBuffer& img) {
    const std::string raw = get_image_buffer_compact_bytes(img);
    return py::bytes(raw);
}

py::array make_image_buffer_numpy_view(ImageBuffer& img) {
    if (!img.is_valid()) {
        throw std::runtime_error("Cannot create a NumPy view from an invalid ImageBuffer.");
    }

    const ImageBuffer* storage = &img;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (img.format == kGray8) {
        const ImageBuffer& gray_img = img.get_gray_version();
        visiong::bufstate::prepare_cpu_read(gray_img);
        storage = &gray_img;
        shape = { static_cast<ssize_t>(gray_img.height), static_cast<ssize_t>(gray_img.width) };
        strides = { static_cast<ssize_t>(gray_img.w_stride), static_cast<ssize_t>(1) };
    } else {
        const ImageBuffer& bgr_img = img.get_bgr_version();
        visiong::bufstate::prepare_cpu_read(bgr_img);
        storage = &bgr_img;
        shape = { static_cast<ssize_t>(bgr_img.height), static_cast<ssize_t>(bgr_img.width), static_cast<ssize_t>(3) };
        strides = { static_cast<ssize_t>(bgr_img.w_stride * 3), static_cast<ssize_t>(3), static_cast<ssize_t>(1) };
    }

    py::object base = py::cast(&img, py::return_value_policy::reference);
    py::array arr(py::dtype::of<unsigned char>(),
                  shape,
                  strides,
                  static_cast<const unsigned char*>(storage->get_data()),
                  base);
    // Expose ImageBuffer -> NumPy views as read-only. Mutating these views would / Expose 图像缓冲区 -> NumPy 视图 作为 read-only. Mutating 这些 视图 would
    // bypass the BufferStateMachine CPU-write boundary accounting and make the
    // Python-visible semantics differ from the documentation.
    // 将 ImageBuffer -> NumPy 视图保持为只读；若允许就地修改，会绕过
    // BufferStateMachine 的 CPU 写边界统计，并破坏文档声明的语义。
    arr.attr("setflags")(py::bool_(false));
    return arr;
}

namespace {

std::string normalize_token(const std::string& value) {
    return visiong::to_lower_copy(value);
}

bool is_h264_alias(const std::string& codec) {
    return codec == "h264" || codec == "264";
}

bool is_h265_alias(const std::string& codec) {
    return codec == "h265" || codec == "265" || codec == "hevc";
}

}  // namespace

DisplayFB::Mode parse_displayfb_mode(const std::string& mode_str) {
    const std::string mode = normalize_token(mode_str);
    if (mode == "low") {
        return DisplayFB::Mode::LOW_REFRESH;
    }
    if (mode == "high") {
        return DisplayFB::Mode::HIGH_REFRESH;
    }
    throw std::invalid_argument("Invalid mode string. Use 'low' or 'high'.");
}

#if VISIONG_WITH_NPU
ModelType parse_model_type(const std::string& model_type_str) {
    const std::string model_type = normalize_token(model_type_str);
    if (model_type == "yolov5") {
        return ModelType::YOLOV5;
    }
    if (model_type == "retinaface") {
        return ModelType::RETINAFACE;
    }
    if (model_type == "facenet") {
        return ModelType::FACENET;
    }
    if (model_type == "yolo11") {
        return ModelType::YOLO11;
    }
    if (model_type == "yolo11_seg" || model_type == "yolo11-seg") {
        return ModelType::YOLO11_SEG;
    }
    if (model_type == "yolo11_pose" || model_type == "yolo11-pose") {
        return ModelType::YOLO11_POSE;
    }
    if (model_type == "lprnet") {
        return ModelType::LPRNET;
    }
    throw std::invalid_argument(
        "Unsupported model_type: '" + model_type_str +
        "'. Use 'yolov5', 'retinaface', 'facenet', 'yolo11', 'yolo11_seg', 'yolo11_pose', or 'lprnet'.");
}
#endif

DisplayRTSP::Codec parse_rtsp_codec(const std::string& codec_str) {
    const std::string codec = normalize_token(codec_str);
    if (is_h264_alias(codec)) {
        return DisplayRTSP::Codec::H264;
    }
    if (is_h265_alias(codec)) {
        return DisplayRTSP::Codec::H265;
    }
    throw std::invalid_argument("Invalid codec. Use 'h264' or 'h265'.");
}

DisplayRTSP::RcMode parse_rtsp_rc_mode(const std::string& rc_mode_str) {
    const std::string rc_mode = normalize_token(rc_mode_str);
    if (rc_mode == "cbr") {
        return DisplayRTSP::RcMode::CBR;
    }
    if (rc_mode == "vbr") {
        return DisplayRTSP::RcMode::VBR;
    }
    throw std::invalid_argument("Invalid rc_mode. Use 'cbr' or 'vbr'.");
}

VencRecorder::Codec parse_venc_codec(const std::string& codec_str) {
    const std::string codec = normalize_token(codec_str);
    if (is_h264_alias(codec)) {
        return VencRecorder::Codec::H264;
    }
    if (is_h265_alias(codec)) {
        return VencRecorder::Codec::H265;
    }
    throw std::invalid_argument("Invalid codec. Use 'h264' or 'h265'.");
}

VencRecorder::Container parse_venc_container(const std::string& container_str) {
    const std::string container = normalize_token(container_str);
    if (container == "mp4") {
        return VencRecorder::Container::MP4;
    }
    if (container == "annexb" || container == "raw" || container == "h264" || container == "h265") {
        return VencRecorder::Container::ANNEXB;
    }
    throw std::invalid_argument("Invalid container. Use 'mp4' or 'annexb'.");
}

DisplayHTTPFLV::Codec parse_httpflv_codec(const std::string& codec_str) {
    const std::string codec = normalize_token(codec_str);
    if (is_h264_alias(codec)) {
        return DisplayHTTPFLV::Codec::H264;
    }
    if (is_h265_alias(codec)) {
        return DisplayHTTPFLV::Codec::H265;
    }
    throw std::invalid_argument("Invalid codec. Use 'h264' or 'h265'.");
}

DisplayHTTPFLV::RcMode parse_httpflv_rc_mode(const std::string& rc_mode_str) {
    const std::string rc_mode = normalize_token(rc_mode_str);
    if (rc_mode == "cbr") {
        return DisplayHTTPFLV::RcMode::CBR;
    }
    if (rc_mode == "vbr") {
        return DisplayHTTPFLV::RcMode::VBR;
    }
    throw std::invalid_argument("Invalid rc_mode. Use 'cbr' or 'vbr'.");
}

