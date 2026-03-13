// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/NanoTrack.h"

#include "visiong/core/ImageBuffer.h"
#include "core/internal/logger.h"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <tuple>

#include "internal/tracking/core.h"

namespace {

struct RgbMatInput {
    cv::Mat mat;
    ImageBuffer converted_owner;
};

RgbMatInput to_rgb_mat(const ImageBuffer& image) {
    if (!image.is_valid()) {
        throw std::runtime_error("NanoTrack: invalid image input.");
    }

    RgbMatInput out;
    const ImageBuffer* rgb = &image;

    // Tracker internal pipeline uses RGB888 only. / Tracker 内部 pipeline uses RGB888 仅.
    // Non-RGB888 sources are converted via ImageBuffer::to_format (RGA path).
    if (image.format != RK_FMT_RGB888) {
        out.converted_owner = image.to_format(RK_FMT_RGB888);
        rgb = &out.converted_owner;
    }

    if (!rgb->is_valid()) {
        throw std::runtime_error("NanoTrack: failed to prepare RGB888 input.");
    }

    cv::Mat rgb_view(rgb->height, rgb->w_stride, CV_8UC3, const_cast<void*>(rgb->get_data()));
    out.mat = rgb_view(cv::Rect(0, 0, rgb->width, rgb->height));
    return out;
}

cv::Rect sanitize_bbox(const std::tuple<int, int, int, int>& bbox_tuple, int image_w, int image_h) {
    cv::Rect bbox(std::get<0>(bbox_tuple), std::get<1>(bbox_tuple), std::get<2>(bbox_tuple), std::get<3>(bbox_tuple));
    cv::Rect valid_rect(0, 0, image_w, image_h);
    bbox = bbox & valid_rect;
    if (bbox.width <= 1 || bbox.height <= 1) {
        throw std::invalid_argument("NanoTrack: bbox is out of bounds or too small.");
    }
    return bbox;
}

cv::Rect clamp_state_bbox(const cv::Rect& bbox, int image_w, int image_h) {
    cv::Rect valid_rect(0, 0, image_w, image_h);
    cv::Rect clamped = bbox & valid_rect;
    if (clamped.width <= 0 || clamped.height <= 0) {
        return cv::Rect(0, 0, 0, 0);
    }
    return clamped;
}

}  // namespace

struct NanoTrack::Impl {
    NanoTrackCore core;
    bool initialized = false;
};

NanoTrack::NanoTrack(const std::string& template_model_path,
                     const std::string& search_model_path,
                     const std::string& head_model_path)
    : m_impl(std::make_unique<Impl>()) {
    const nn_error_e ret = m_impl->core.LoadModel(template_model_path.c_str(), search_model_path.c_str(),
                                                  head_model_path.c_str());
    if (ret != NN_SUCCESS) {
        throw std::runtime_error("NanoTrack: model load failed, ret=" + std::to_string(static_cast<int>(ret)));
    }
    VISIONG_LOG_INFO("NanoTrack", "Model trio loaded.");
}

NanoTrack::~NanoTrack() = default;

void NanoTrack::init(const ImageBuffer& image, const std::tuple<int, int, int, int>& bbox) {
    std::lock_guard<std::mutex> lock(m_mutex);

    RgbMatInput rgb_input = to_rgb_mat(image);
    cv::Mat& rgb = rgb_input.mat;
    if (rgb.empty()) {
        throw std::runtime_error("NanoTrack: init image is empty.");
    }

    const cv::Rect init_box = sanitize_bbox(bbox, rgb.cols, rgb.rows);
    m_impl->core.init(rgb, init_box);
    m_impl->initialized = true;
}

NanoTrackResult NanoTrack::track(const ImageBuffer& image) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_impl->initialized) {
        throw std::runtime_error("NanoTrack: call init() before track().");
    }

    RgbMatInput rgb_input = to_rgb_mat(image);
    cv::Mat& rgb = rgb_input.mat;
    if (rgb.empty()) {
        throw std::runtime_error("NanoTrack: track image is empty.");
    }

    const float score = m_impl->core.track(rgb);
    cv::Rect bbox = clamp_state_bbox(m_impl->core.state.bbox, rgb.cols, rgb.rows);
    if (bbox.width <= 0 || bbox.height <= 0) {
        bbox = cv::Rect(0, 0, 0, 0);
    }

    m_impl->core.state.bbox = bbox;

    NanoTrackResult result;
    result.box = std::make_tuple(bbox.x, bbox.y, bbox.width, bbox.height);
    result.score = score;
    return result;
}

bool NanoTrack::is_initialized() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_impl->initialized;
}

std::tuple<int, int, int, int> NanoTrack::get_bbox() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const cv::Rect& bbox = m_impl->core.state.bbox;
    return std::make_tuple(bbox.x, bbox.y, bbox.width, bbox.height);
}

float NanoTrack::get_score() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_impl->core.state.cls_score_max;
}

void NanoTrack::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_impl->core.state = State{};
    m_impl->initialized = false;
}
