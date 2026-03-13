// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/PPOCR.h"
#include "npu/internal/npu_common.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/RgaHelper.h"
#include "im2d.hpp"
#include "core/internal/rga_utils.h"
#include "common/internal/dma_alloc.h"
#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"
#include "core/internal/logger.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <cctype>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

constexpr int kMaxOcrResults = 256;

struct QuadBox {
    std::array<cv::Point2f, 4> pts;
    float score = 0.0f;
};

struct DecodedText {
    std::string text;
    float score = 0.0f;
};

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;

    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        f = sign | (exp << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

float clamp_float(float value, float min_value, float max_value) {
    return std::max(min_value, std::min(value, max_value));
}

int clamp_int(int value, int min_value, int max_value) {
    return std::max(min_value, std::min(value, max_value));
}

float tensor_value_as_f32(const rknn_tensor_attr& attr, const void* data, int index) {
    switch (attr.type) {
        case RKNN_TENSOR_INT8:
            return visiong::npu::yolo::dequantize_from_i8(static_cast<const int8_t*>(data)[index], attr.zp, attr.scale);
        case RKNN_TENSOR_UINT8:
            return (static_cast<float>(static_cast<const uint8_t*>(data)[index]) - static_cast<float>(attr.zp)) * attr.scale;
        case RKNN_TENSOR_FLOAT16:
            return fp16_to_fp32(static_cast<const uint16_t*>(data)[index]);
        case RKNN_TENSOR_FLOAT32:
            return static_cast<const float*>(data)[index];
        default:
            return 0.0f;
    }
}

std::vector<float> tensor_to_f32(const rknn_tensor_attr& attr, const void* data) {
    std::vector<float> values(attr.n_elems, 0.0f);
    for (uint32_t i = 0; i < attr.n_elems; ++i) {
        values[i] = tensor_value_as_f32(attr, data, i);
    }
    return values;
}

std::array<cv::Point2f, 4> order_quad(std::array<cv::Point2f, 4> pts) {
    std::array<cv::Point2f, 4> ordered{};

    float min_sum = std::numeric_limits<float>::max();
    float max_sum = std::numeric_limits<float>::lowest();
    float min_diff = std::numeric_limits<float>::max();
    float max_diff = std::numeric_limits<float>::lowest();

    for (const auto& p : pts) {
        const float sum = p.x + p.y;
        const float diff = p.x - p.y;
        if (sum < min_sum) {
            min_sum = sum;
            ordered[0] = p;  // top-left
        }
        if (sum > max_sum) {
            max_sum = sum;
            ordered[2] = p;  // bottom-right
        }
        if (diff < min_diff) {
            min_diff = diff;
            ordered[3] = p;  // bottom-left
        }
        if (diff > max_diff) {
            max_diff = diff;
            ordered[1] = p;  // top-right
        }
    }
    return ordered;
}

std::vector<std::tuple<int, int>> quad_to_tuples(const std::array<cv::Point2f, 4>& quad) {
    std::vector<std::tuple<int, int>> out;
    out.reserve(4);
    for (const auto& p : quad) {
        out.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    }
    return out;
}

std::tuple<int, int, int, int> quad_to_rect(const std::array<cv::Point2f, 4>& quad, int max_w, int max_h) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& p : quad) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    const int x = clamp_int(static_cast<int>(std::floor(min_x)), 0, std::max(0, max_w - 1));
    const int y = clamp_int(static_cast<int>(std::floor(min_y)), 0, std::max(0, max_h - 1));
    const int r = clamp_int(static_cast<int>(std::ceil(max_x)), 0, std::max(0, max_w - 1));
    const int b = clamp_int(static_cast<int>(std::ceil(max_y)), 0, std::max(0, max_h - 1));
    return std::make_tuple(x, y, std::max(0, r - x), std::max(0, b - y));
}

struct PreparedInputMat {
    cv::Mat mat;
    ImageBuffer converted_owner;
    const ImageBuffer* source = nullptr;
};

PIXEL_FORMAT_E parse_model_input_format(const std::string& format) {
    std::string lower;
    lower.reserve(format.size());
    for (const char ch : format) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lower == "rgb" || lower == "rgb888") {
        return RK_FMT_RGB888;
    }
    if (lower == "bgr" || lower == "bgr888") {
        return RK_FMT_BGR888;
    }
    throw std::invalid_argument("Unsupported PPOCR model_input_format: '" + format + "'. Use 'rgb' or 'bgr'.");
}

PreparedInputMat prepare_model_input_mat(const ImageBuffer& input, PIXEL_FORMAT_E expected_format) {
    if (!input.is_valid()) {
        return {};
    }

    PreparedInputMat out;
    const ImageBuffer* src = &input;

    // Match model color order explicitly. Conversion is handled by ImageBuffer::to_format (RGA when available). / 匹配 模式l color order explicitly. Conversion 为 handled 由 图像缓冲区::to_格式 (RGA 当 available).
    if (input.format != expected_format) {
        out.converted_owner = input.to_format(expected_format);
        src = &out.converted_owner;
    }

    cv::Mat wrapped(src->height,
                    src->width,
                    CV_8UC3,
                    const_cast<void*>(src->get_data()),
                    static_cast<size_t>(src->w_stride) * 3U);
    out.mat = wrapped(cv::Rect(0, 0, src->width, src->height));
    out.source = src;
    return out;
}

bool try_write_image_to_npu_input_rga(const ImageBuffer& src_img, rknn_app_context_t* ctx) {
    if (!src_img.is_valid() || ctx == nullptr || ctx->input_mems[0] == nullptr || ctx->input_attrs == nullptr) {
        return false;
    }

    const rknn_tensor_attr& input_attr = ctx->input_attrs[0];
    if (input_attr.fmt != RKNN_TENSOR_NHWC) {
        return false;
    }
    if (input_attr.type != RKNN_TENSOR_INT8 && input_attr.type != RKNN_TENSOR_UINT8) {
        return false;
    }
    if (ctx->input_mems[0]->fd < 0 || ctx->input_mems[0]->virt_addr == nullptr) {
        return false;
    }
    if (src_img.format != RK_FMT_RGB888 && src_img.format != RK_FMT_BGR888) {
        return false;
    }

    const int model_w = ctx->model_width;
    const int model_h = ctx->model_height;
    if (model_w <= 0 || model_h <= 0) {
        return false;
    }

    const int w_stride = (input_attr.w_stride > 0) ? input_attr.w_stride : model_w;
    const int h_stride = (input_attr.h_stride > 0) ? input_attr.h_stride : model_h;

    try {
        std::unique_ptr<RgaDmaBuffer> src_dma;
        if (src_img.is_zero_copy() && src_img.get_dma_fd() >= 0) {
            src_dma = std::make_unique<RgaDmaBuffer>(src_img.get_dma_fd(),
                                                     const_cast<void*>(src_img.get_data()),
                                                     src_img.get_size(),
                                                     src_img.width,
                                                     src_img.height,
                                                     static_cast<int>(src_img.format),
                                                     src_img.w_stride,
                                                     src_img.h_stride);
            dma_sync_cpu_to_device(src_dma->get_fd());
        } else {
            src_dma = std::make_unique<RgaDmaBuffer>(src_img.width, src_img.height, static_cast<int>(src_img.format));
            const int bpp = get_bpp_for_format(src_img.format);
            copy_data_with_stride(src_dma->get_vir_addr(),
                                  src_dma->get_wstride() * bpp / 8,
                                  src_img.get_data(),
                                  src_img.w_stride * bpp / 8,
                                  src_img.height,
                                  src_img.width * bpp / 8);
            dma_sync_cpu_to_device(src_dma->get_fd());
        }

        RgaDmaBuffer dst_wrapper(ctx->input_mems[0]->fd,
                                 ctx->input_mems[0]->virt_addr,
                                 ctx->input_mems[0]->size,
                                 model_w,
                                 model_h,
                                 static_cast<int>(src_img.format),
                                 w_stride,
                                 h_stride);

        if (imresize(src_dma->get_buffer(), dst_wrapper.get_buffer()) != IM_STATUS_SUCCESS) {
            return false;
        }

        dma_sync_cpu_to_device(ctx->input_mems[0]->fd);
        return true;
    } catch (...) {
        return false;
    }
}

void write_mat_to_npu_input(const cv::Mat& mat_rgb,
                            rknn_app_context_t* ctx,
                            bool normalize_to_neg1_pos1) {
    if (ctx == nullptr || ctx->input_mems[0] == nullptr) {
        throw std::runtime_error("Invalid RKNN context input memory.");
    }
    if (mat_rgb.empty() || mat_rgb.type() != CV_8UC3) {
        throw std::invalid_argument("Input matrix for NPU must be non-empty CV_8UC3.");
    }

    const int model_w = ctx->model_width;
    const int model_h = ctx->model_height;

    cv::Mat resized;
    if (mat_rgb.cols != model_w || mat_rgb.rows != model_h) {
        cv::resize(mat_rgb, resized, cv::Size(model_w, model_h), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = mat_rgb;
    }

    const int w_stride = (ctx->input_attrs != nullptr && ctx->input_attrs[0].w_stride > 0)
                             ? ctx->input_attrs[0].w_stride
                             : model_w;

    const rknn_tensor_attr* input_attr = (ctx->input_attrs != nullptr) ? &ctx->input_attrs[0] : nullptr;
    if (normalize_to_neg1_pos1 && input_attr != nullptr && input_attr->type == RKNN_TENSOR_INT8) {
        int8_t* dst = static_cast<int8_t*>(ctx->input_mems[0]->virt_addr);
        const float scale = (input_attr->scale == 0.0f) ? 1.0f : input_attr->scale;
        const int zp = input_attr->zp;
        std::array<int8_t, 256> lut{};
        for (int i = 0; i < 256; ++i) {
            const float pix = static_cast<float>(i);
            const float norm = (pix / 255.0f - 0.5f) / 0.5f;
            int q = static_cast<int>(std::lround(norm / scale + static_cast<float>(zp)));
            q = clamp_int(q, -128, 127);
            lut[static_cast<size_t>(i)] = static_cast<int8_t>(q);
        }

        for (int y = 0; y < model_h; ++y) {
            const uint8_t* src_row = resized.ptr<uint8_t>(y);
            int8_t* dst_row = dst + y * w_stride * 3;
            for (int x = 0; x < model_w; ++x) {
                for (int c = 0; c < 3; ++c) {
                    dst_row[x * 3 + c] = lut[static_cast<size_t>(src_row[x * 3 + c])];
                }
            }
        }
    } else {
        copy_data_with_stride(ctx->input_mems[0]->virt_addr,
                              w_stride * 3,
                              resized.data,
                              static_cast<int>(resized.step),
                              model_h,
                              model_w * 3);
    }

    dma_sync_cpu_to_device(ctx->input_mems[0]->fd);
}

bool parse_hwc(const rknn_tensor_attr& attr, int* h, int* w, int* c) {
    if (h == nullptr || w == nullptr || c == nullptr) {
        return false;
    }

    if (attr.n_dims >= 4) {
        if (attr.fmt == RKNN_TENSOR_NCHW) {
            *c = attr.dims[1];
            *h = attr.dims[2];
            *w = attr.dims[3];
        } else {
            *h = attr.dims[1];
            *w = attr.dims[2];
            *c = attr.dims[3];
        }
        return true;
    }

    if (attr.n_dims == 3) {
        if (attr.fmt == RKNN_TENSOR_NCHW) {
            *c = attr.dims[0];
            *h = attr.dims[1];
            *w = attr.dims[2];
        } else {
            *h = attr.dims[0];
            *w = attr.dims[1];
            *c = attr.dims[2];
        }
        return true;
    }

    return false;
}

float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour) {
    cv::Mat mask = cv::Mat::zeros(prob.rows, prob.cols, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours{contour};
    cv::fillPoly(mask, contours, cv::Scalar(255));
    return static_cast<float>(cv::mean(prob, mask)[0]);
}

cv::RotatedRect expand_db_box(const cv::RotatedRect& box,
                              const std::vector<cv::Point>& contour,
                              float unclip_ratio) {
    if (unclip_ratio <= 1.0f) {
        return box;
    }

    const float area = std::fabs(static_cast<float>(cv::contourArea(contour)));
    const float perimeter = static_cast<float>(cv::arcLength(contour, true));
    if (area <= 1e-3f || perimeter <= 1e-3f) {
        return box;
    }

    // DB postprocess expands shrinked text region by distance ~= area * ratio / perimeter. / DB postprocess expands shrinked text region 由 distance ~= area * ratio / perimeter.
    const float distance = area * unclip_ratio / perimeter;
    cv::Size2f expanded_size(box.size.width + 2.0f * distance,
                             box.size.height + 2.0f * distance);
    expanded_size.width = std::max(expanded_size.width, 1.0f);
    expanded_size.height = std::max(expanded_size.height, 1.0f);
    return cv::RotatedRect(box.center, expanded_size, box.angle);
}

std::vector<QuadBox> db_postprocess(const cv::Mat& prob,
                                    float threshold,
                                    float box_threshold,
                                    bool use_dilate,
                                    float unclip_ratio,
                                    float scale_x,
                                    float scale_y,
                                    int max_w,
                                    int max_h) {
    cv::Mat bitmap;
    cv::threshold(prob, bitmap, threshold, 255.0, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8UC1);

    if (use_dilate) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bitmap, bitmap, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<QuadBox> boxes;
    boxes.reserve(std::min<int>(static_cast<int>(contours.size()), kMaxOcrResults));

    for (const auto& contour : contours) {
        if (static_cast<int>(contour.size()) < 4) {
            continue;
        }
        if (std::fabs(cv::contourArea(contour)) < 16.0) {
            continue;
        }

        const float score = contour_score(prob, contour);
        if (score < box_threshold) {
            continue;
        }

        const cv::RotatedRect box = cv::minAreaRect(contour);
        if (std::min(box.size.width, box.size.height) < 3.0f) {
            continue;
        }

        const cv::RotatedRect expanded_box = expand_db_box(box, contour, unclip_ratio);

        cv::Point2f pts[4];
        expanded_box.points(pts);

        std::array<cv::Point2f, 4> quad{};
        for (int i = 0; i < 4; ++i) {
            quad[i].x = clamp_float(pts[i].x * scale_x, 0.0f, static_cast<float>(max_w - 1));
            quad[i].y = clamp_float(pts[i].y * scale_y, 0.0f, static_cast<float>(max_h - 1));
        }

        boxes.push_back({order_quad(quad), score});
        if (static_cast<int>(boxes.size()) >= kMaxOcrResults) {
            break;
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const QuadBox& a, const QuadBox& b) {
        const float ay = a.pts[0].y;
        const float by = b.pts[0].y;
        if (std::fabs(ay - by) < 10.0f) {
            return a.pts[0].x < b.pts[0].x;
        }
        return ay < by;
    });

    return boxes;
}

cv::Mat crop_quad(const cv::Mat& rgb, const std::array<cv::Point2f, 4>& quad) {
    const float width_a = std::hypot(quad[2].x - quad[3].x, quad[2].y - quad[3].y);
    const float width_b = std::hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
    const float height_a = std::hypot(quad[1].x - quad[2].x, quad[1].y - quad[2].y);
    const float height_b = std::hypot(quad[0].x - quad[3].x, quad[0].y - quad[3].y);

    const int crop_w = std::max(1, static_cast<int>(std::round(std::max(width_a, width_b))));
    const int crop_h = std::max(1, static_cast<int>(std::round(std::max(height_a, height_b))));

    const std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(crop_w - 1), 0.0f),
        cv::Point2f(static_cast<float>(crop_w - 1), static_cast<float>(crop_h - 1)),
        cv::Point2f(0.0f, static_cast<float>(crop_h - 1)),
    };

    cv::Mat transform = cv::getPerspectiveTransform(quad.data(), dst.data());
    cv::Mat cropped;
    cv::warpPerspective(rgb, cropped, transform, cv::Size(crop_w, crop_h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    if (!cropped.empty() && static_cast<float>(cropped.rows) >= static_cast<float>(cropped.cols) * 1.5f) {
        cv::Mat rotated;
        cv::transpose(cropped, rotated);
        cv::flip(rotated, rotated, 0);
        return rotated;
    }
    return cropped;
}

struct DecodeCandidate {
    DecodedText out;
    float mean_timestep_max = 0.0f;
};

inline int ctc_linear_index(int t, int j, int seq_len, int vocab, bool vocab_major) {
    return vocab_major ? (j * seq_len + t) : (t * vocab + j);
}

template <typename T>
inline int argmax_row_scalar(const T* row, int vocab, T* best_raw) {
    int best_idx = 0;
    T best = row[0];
    for (int j = 1; j < vocab; ++j) {
        const T v = row[j];
        if (v > best) {
            best = v;
            best_idx = j;
        }
    }
    *best_raw = best;
    return best_idx;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline int argmax_row_neon(const int8_t* row, int vocab, int8_t* best_raw) {
    int j = 0;
    int8x16_t vmax = vdupq_n_s8(std::numeric_limits<int8_t>::min());
    for (; j + 16 <= vocab; j += 16) {
        vmax = vmaxq_s8(vmax, vld1q_s8(row + j));
    }

    int8x8_t vmax8 = vmax_s8(vget_low_s8(vmax), vget_high_s8(vmax));
    vmax8 = vpmax_s8(vmax8, vmax8);
    vmax8 = vpmax_s8(vmax8, vmax8);
    vmax8 = vpmax_s8(vmax8, vmax8);
    int8_t maxv = vget_lane_s8(vmax8, 0);

    for (; j < vocab; ++j) {
        if (row[j] > maxv) {
            maxv = row[j];
        }
    }

    int best_idx = 0;
    for (int i = 0; i < vocab; ++i) {
        if (row[i] == maxv) {
            best_idx = i;
            break;
        }
    }
    *best_raw = maxv;
    return best_idx;
}

inline int argmax_row_neon(const uint8_t* row, int vocab, uint8_t* best_raw) {
    int j = 0;
    uint8x16_t vmax = vdupq_n_u8(0);
    for (; j + 16 <= vocab; j += 16) {
        vmax = vmaxq_u8(vmax, vld1q_u8(row + j));
    }

    uint8x8_t vmax8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));
    vmax8 = vpmax_u8(vmax8, vmax8);
    vmax8 = vpmax_u8(vmax8, vmax8);
    vmax8 = vpmax_u8(vmax8, vmax8);
    uint8_t maxv = vget_lane_u8(vmax8, 0);

    for (; j < vocab; ++j) {
        if (row[j] > maxv) {
            maxv = row[j];
        }
    }

    int best_idx = 0;
    for (int i = 0; i < vocab; ++i) {
        if (row[i] == maxv) {
            best_idx = i;
            break;
        }
    }
    *best_raw = maxv;
    return best_idx;
}
#endif

template <typename T>
inline int argmax_row(const T* row, int vocab, T* best_raw) {
    return argmax_row_scalar(row, vocab, best_raw);
}

template <>
inline int argmax_row<int8_t>(const int8_t* row, int vocab, int8_t* best_raw) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return argmax_row_neon(row, vocab, best_raw);
#else
    return argmax_row_scalar(row, vocab, best_raw);
#endif
}

template <>
inline int argmax_row<uint8_t>(const uint8_t* row, int vocab, uint8_t* best_raw) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return argmax_row_neon(row, vocab, best_raw);
#else
    return argmax_row_scalar(row, vocab, best_raw);
#endif
}

inline float fast_sigmoid(float x) {
    const float clamped = clamp_float(x, -10.0f, 10.0f);
    return 1.0f / (1.0f + std::exp(-clamped));
}

DecodeCandidate decode_ctc_layout_fp16(const uint16_t* values,
                                       int seq_len,
                                       int vocab,
                                       const std::vector<std::string>& dict,
                                       bool vocab_major,
                                       bool values_are_prob) {
    DecodeCandidate candidate;
    if (values == nullptr || seq_len <= 0 || vocab <= 1) {
        return candidate;
    }

    int prev_idx = 0;
    float score_sum = 0.0f;
    int kept = 0;
    float timestep_max_sum = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int best_idx = 0;
        uint16_t best_raw = values[ctc_linear_index(t, 0, seq_len, vocab, vocab_major)];
        float best_val = 0.0f;
        if (values_are_prob) {
            for (int j = 1; j < vocab; ++j) {
                const uint16_t raw = values[ctc_linear_index(t, j, seq_len, vocab, vocab_major)];
                if (raw > best_raw) {
                    best_raw = raw;
                    best_idx = j;
                }
            }
            best_val = fp16_to_fp32(best_raw);
        } else {
            best_val = fp16_to_fp32(best_raw);
            for (int j = 1; j < vocab; ++j) {
                const uint16_t raw = values[ctc_linear_index(t, j, seq_len, vocab, vocab_major)];
                const float v = fp16_to_fp32(raw);
                if (v > best_val) {
                    best_raw = raw;
                    best_val = v;
                    best_idx = j;
                }
            }
        }

        const float best_prob = values_are_prob ? std::max(best_val, 0.0f) : fast_sigmoid(best_val);
        timestep_max_sum += best_prob;

        if (best_idx > 0 && best_idx != prev_idx) {
            const int dict_idx = best_idx - 1;
            if (dict_idx >= 0 && dict_idx < static_cast<int>(dict.size())) {
                candidate.out.text += dict[dict_idx];
                score_sum += best_prob;
                ++kept;
            }
        }
        prev_idx = best_idx;
    }

    candidate.mean_timestep_max = timestep_max_sum / static_cast<float>(std::max(1, seq_len));
    candidate.out.score = (kept > 0) ? (score_sum / static_cast<float>(kept)) : 0.0f;
    return candidate;
}

template <typename T>
DecodeCandidate decode_ctc_layout_quantized(const T* values,
                                            int seq_len,
                                            int vocab,
                                            const std::vector<std::string>& dict,
                                            bool vocab_major,
                                            float scale,
                                            int zp,
                                            bool values_are_prob) {
    DecodeCandidate candidate;
    if (values == nullptr || seq_len <= 0 || vocab <= 1) {
        return candidate;
    }

    const float safe_scale = (scale == 0.0f) ? 1.0f : scale;

    int prev_idx = 0;
    float score_sum = 0.0f;
    int kept = 0;
    float timestep_max_sum = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int best_idx = 0;
        T best_raw = 0;
        if (!vocab_major) {
            const T* row = values + t * vocab;
            best_idx = argmax_row<T>(row, vocab, &best_raw);
        } else {
            best_raw = values[ctc_linear_index(t, 0, seq_len, vocab, vocab_major)];
            for (int j = 1; j < vocab; ++j) {
                const T raw = values[ctc_linear_index(t, j, seq_len, vocab, vocab_major)];
                if (raw > best_raw) {
                    best_raw = raw;
                    best_idx = j;
                }
            }
        }

        const float best_val = (static_cast<float>(best_raw) - static_cast<float>(zp)) * safe_scale;
        const float best_prob = values_are_prob ? clamp_float(best_val, 0.0f, 1.0f) : fast_sigmoid(best_val);
        timestep_max_sum += best_prob;

        if (best_idx > 0 && best_idx != prev_idx) {
            const int dict_idx = best_idx - 1;
            if (dict_idx >= 0 && dict_idx < static_cast<int>(dict.size())) {
                candidate.out.text += dict[dict_idx];
                score_sum += best_prob;
                ++kept;
            }
        }
        prev_idx = best_idx;
    }

    candidate.mean_timestep_max = timestep_max_sum / static_cast<float>(std::max(1, seq_len));
    candidate.out.score = (kept > 0) ? (score_sum / static_cast<float>(kept)) : 0.0f;
    return candidate;
}

DecodeCandidate decode_ctc_layout_fp32(const float* values,
                                       int seq_len,
                                       int vocab,
                                       const std::vector<std::string>& dict,
                                       bool vocab_major,
                                       bool values_are_prob) {
    DecodeCandidate candidate;
    if (values == nullptr || seq_len <= 0 || vocab <= 1) {
        return candidate;
    }

    int prev_idx = 0;
    float score_sum = 0.0f;
    int kept = 0;
    float timestep_max_sum = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int best_idx = 0;
        float best_val = values[ctc_linear_index(t, 0, seq_len, vocab, vocab_major)];
        for (int j = 1; j < vocab; ++j) {
            const float v = values[ctc_linear_index(t, j, seq_len, vocab, vocab_major)];
            if (v > best_val) {
                best_val = v;
                best_idx = j;
            }
        }

        const float best_prob = values_are_prob ? std::max(best_val, 0.0f) : fast_sigmoid(best_val);
        timestep_max_sum += best_prob;

        if (best_idx > 0 && best_idx != prev_idx) {
            const int dict_idx = best_idx - 1;
            if (dict_idx >= 0 && dict_idx < static_cast<int>(dict.size())) {
                candidate.out.text += dict[dict_idx];
                score_sum += best_prob;
                ++kept;
            }
        }
        prev_idx = best_idx;
    }

    candidate.mean_timestep_max = timestep_max_sum / static_cast<float>(std::max(1, seq_len));
    candidate.out.score = (kept > 0) ? (score_sum / static_cast<float>(kept)) : 0.0f;
    return candidate;
}

DecodeCandidate decode_ctc_layout_generic(const rknn_tensor_attr& attr,
                                          const void* raw,
                                          int seq_len,
                                          int vocab,
                                          const std::vector<std::string>& dict,
                                          bool vocab_major) {
    DecodeCandidate candidate;
    if (raw == nullptr || seq_len <= 0 || vocab <= 1) {
        return candidate;
    }

    const bool is_softmax_prob = std::string(attr.name).find("softmax") != std::string::npos;

    int prev_idx = 0;
    float score_sum = 0.0f;
    int kept = 0;
    float timestep_max_sum = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        int best_idx = 0;
        float best_val = tensor_value_as_f32(attr, raw, ctc_linear_index(t, 0, seq_len, vocab, vocab_major));
        for (int j = 1; j < vocab; ++j) {
            const float v = tensor_value_as_f32(attr, raw, ctc_linear_index(t, j, seq_len, vocab, vocab_major));
            if (v > best_val) {
                best_val = v;
                best_idx = j;
            }
        }

        const float best_prob = is_softmax_prob ? std::max(best_val, 0.0f) : fast_sigmoid(best_val);
        timestep_max_sum += best_prob;

        if (best_idx > 0 && best_idx != prev_idx) {
            const int dict_idx = best_idx - 1;
            if (dict_idx >= 0 && dict_idx < static_cast<int>(dict.size())) {
                candidate.out.text += dict[dict_idx];
                score_sum += best_prob;
                ++kept;
            }
        }
        prev_idx = best_idx;
    }

    candidate.mean_timestep_max = timestep_max_sum / static_cast<float>(std::max(1, seq_len));
    candidate.out.score = (kept > 0) ? (score_sum / static_cast<float>(kept)) : 0.0f;
    return candidate;
}

DecodedText decode_ctc_output(const rknn_tensor_attr& attr, const void* raw, const std::vector<std::string>& dict) {
    if (dict.empty() || raw == nullptr || attr.n_elems <= 0) {
        return {};
    }

    const int vocab = static_cast<int>(dict.size()) + 1;
    if (vocab <= 1 || attr.n_elems % vocab != 0) {
        return {};
    }

    const int seq_len = static_cast<int>(attr.n_elems) / vocab;

    bool layout_known = false;
    bool vocab_major = false;
    if (attr.n_dims == 3) {
        const int d1 = attr.dims[1];
        const int d2 = attr.dims[2];
        if (d1 == seq_len && d2 == vocab) {
            layout_known = true;
            vocab_major = false;
        } else if (d1 == vocab && d2 == seq_len) {
            layout_known = true;
            vocab_major = true;
        }
    } else if (attr.n_dims >= 4) {
        const int d1 = attr.dims[attr.n_dims - 2];
        const int d2 = attr.dims[attr.n_dims - 1];
        if (d1 == seq_len && d2 == vocab) {
            layout_known = true;
            vocab_major = false;
        } else if (d1 == vocab && d2 == seq_len) {
            layout_known = true;
            vocab_major = true;
        }
    }

    const bool output_is_prob = std::string(attr.name).find("softmax") != std::string::npos;

    if (layout_known) {
        switch (attr.type) {
            case RKNN_TENSOR_FLOAT16:
                return decode_ctc_layout_fp16(static_cast<const uint16_t*>(raw),
                                              seq_len,
                                              vocab,
                                              dict,
                                              vocab_major,
                                              output_is_prob)
                    .out;
            case RKNN_TENSOR_FLOAT32:
                return decode_ctc_layout_fp32(static_cast<const float*>(raw),
                                              seq_len,
                                              vocab,
                                              dict,
                                              vocab_major,
                                              output_is_prob)
                    .out;
            case RKNN_TENSOR_INT8:
                return decode_ctc_layout_quantized(static_cast<const int8_t*>(raw),
                                                   seq_len,
                                                   vocab,
                                                   dict,
                                                   vocab_major,
                                                   attr.scale,
                                                   attr.zp,
                                                   output_is_prob)
                    .out;
            case RKNN_TENSOR_UINT8:
                return decode_ctc_layout_quantized(static_cast<const uint8_t*>(raw),
                                                   seq_len,
                                                   vocab,
                                                   dict,
                                                   vocab_major,
                                                   attr.scale,
                                                   attr.zp,
                                                   output_is_prob)
                    .out;
            default:
                break;
        }
        return decode_ctc_layout_generic(attr, raw, seq_len, vocab, dict, vocab_major).out;
    }

    DecodeCandidate a;
    DecodeCandidate b;
    switch (attr.type) {
        case RKNN_TENSOR_FLOAT16:
            a = decode_ctc_layout_fp16(static_cast<const uint16_t*>(raw), seq_len, vocab, dict, false, output_is_prob);
            b = decode_ctc_layout_fp16(static_cast<const uint16_t*>(raw), seq_len, vocab, dict, true, output_is_prob);
            break;
        case RKNN_TENSOR_FLOAT32:
            a = decode_ctc_layout_fp32(static_cast<const float*>(raw), seq_len, vocab, dict, false, output_is_prob);
            b = decode_ctc_layout_fp32(static_cast<const float*>(raw), seq_len, vocab, dict, true, output_is_prob);
            break;
        case RKNN_TENSOR_INT8:
            a = decode_ctc_layout_quantized(static_cast<const int8_t*>(raw),
                                            seq_len,
                                            vocab,
                                            dict,
                                            false,
                                            attr.scale,
                                            attr.zp,
                                            output_is_prob);
            b = decode_ctc_layout_quantized(static_cast<const int8_t*>(raw),
                                            seq_len,
                                            vocab,
                                            dict,
                                            true,
                                            attr.scale,
                                            attr.zp,
                                            output_is_prob);
            break;
        case RKNN_TENSOR_UINT8:
            a = decode_ctc_layout_quantized(static_cast<const uint8_t*>(raw),
                                            seq_len,
                                            vocab,
                                            dict,
                                            false,
                                            attr.scale,
                                            attr.zp,
                                            output_is_prob);
            b = decode_ctc_layout_quantized(static_cast<const uint8_t*>(raw),
                                            seq_len,
                                            vocab,
                                            dict,
                                            true,
                                            attr.scale,
                                            attr.zp,
                                            output_is_prob);
            break;
        default:
            a = decode_ctc_layout_generic(attr, raw, seq_len, vocab, dict, false);
            b = decode_ctc_layout_generic(attr, raw, seq_len, vocab, dict, true);
            break;
    }

    if (a.mean_timestep_max > b.mean_timestep_max + 1e-5f) {
        return a.out;
    }
    if (b.mean_timestep_max > a.mean_timestep_max + 1e-5f) {
        return b.out;
    }
    if (a.out.score > b.out.score) {
        return a.out;
    }
    if (b.out.score > a.out.score) {
        return b.out;
    }
    return (a.out.text.size() >= b.out.text.size()) ? a.out : b.out;
}

cv::Mat preprocess_rec_crop(const cv::Mat& crop, int target_w, int target_h) {
    if (crop.empty()) {
        return {};
    }

    const float ratio = static_cast<float>(crop.cols) / static_cast<float>(std::max(1, crop.rows));
    int resized_w = static_cast<int>(std::ceil(target_h * ratio));
    resized_w = clamp_int(resized_w, 1, target_w);

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(resized_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, resized_w, target_h)));
    return padded;
}

}  // namespace

PPOCR::PPOCR(const std::string& det_model_path,
             const std::string& rec_model_path,
             const std::string& dict_path,
             float det_threshold,
             float box_threshold,
             bool use_dilate,
             const std::string& rec_fast_model_path,
             float rec_fast_max_ratio,
             bool rec_fast_enable_fallback,
             float rec_fast_fallback_score_thresh,
             const std::string& model_input_format,
             float det_unclip_ratio)
    : m_det_ctx(std::make_unique<rknn_app_context_t>()),
      m_rec_ctx(std::make_unique<rknn_app_context_t>()),
      m_rec_fast_ctx(std::make_unique<rknn_app_context_t>()),
      m_det_model_path(det_model_path),
      m_rec_model_path(rec_model_path),
      m_rec_fast_model_path(rec_fast_model_path),
      m_dict_path(dict_path),
      m_model_input_format(model_input_format),
      m_rec_fast_max_ratio(std::max(0.5f, rec_fast_max_ratio)),
      m_rec_fast_enable_fallback(rec_fast_enable_fallback),
      m_rec_fast_fallback_score_thresh(clamp_float(rec_fast_fallback_score_thresh, 0.0f, 1.0f)),
      m_det_threshold(det_threshold),
      m_box_threshold(box_threshold),
      m_det_unclip_ratio(std::max(1.0f, det_unclip_ratio)),
      m_use_dilate(use_dilate) {
    std::memset(m_det_ctx.get(), 0, sizeof(rknn_app_context_t));
    std::memset(m_rec_ctx.get(), 0, sizeof(rknn_app_context_t));
    std::memset(m_rec_fast_ctx.get(), 0, sizeof(rknn_app_context_t));

    // Validate once so runtime path does not need branching checks. / Validate 一次 so 运行时 路径 does 不 need branching 检查.
    (void)parse_model_input_format(m_model_input_format);

    const int ret = initialize_runtime();
    if (ret != 0) {
        throw std::runtime_error("Failed to initialize PPOCR runtime. ret=" + std::to_string(ret));
    }
}

PPOCR::~PPOCR() {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    release_runtime();
}

int PPOCR::load_dictionary(const std::string& dict_path) {
    std::filesystem::path actual_path(dict_path);
    if (actual_path.empty()) {
        actual_path = std::filesystem::path(m_rec_model_path).parent_path() / "ppocr_keys_v1.txt";
    }

    std::ifstream ifs(actual_path, std::ios::binary);
    if (!ifs.is_open()) {
        VISIONG_LOG_ERROR("PPOCR", "Failed to open dict file: " << actual_path.string());
        return -1;
    }

    m_dict.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            m_dict.push_back(line);
        }
    }

    if (m_dict.empty()) {
        VISIONG_LOG_ERROR("PPOCR", "Dictionary is empty: " << actual_path.string());
        return -1;
    }

    m_dict_path = actual_path.string();
    VISIONG_LOG_INFO("PPOCR", "Loaded dictionary entries: " << m_dict.size());
    return 0;
}

int PPOCR::initialize_runtime() {
    if (visiong::npu::rknn::init_zero_copy_model(m_det_model_path.c_str(), m_det_ctx.get()) != 0) {
        return -1;
    }

    if (visiong::npu::rknn::init_zero_copy_model(m_rec_model_path.c_str(), m_rec_ctx.get()) != 0) {
        visiong::npu::rknn::release_zero_copy_model(m_det_ctx.get());
        std::memset(m_det_ctx.get(), 0, sizeof(rknn_app_context_t));
        return -1;
    }

    if (!m_rec_fast_model_path.empty()) {
        if (visiong::npu::rknn::init_zero_copy_model(m_rec_fast_model_path.c_str(), m_rec_fast_ctx.get()) != 0) {
            VISIONG_LOG_WARN("PPOCR",
                             "Failed to init fast REC model (" << m_rec_fast_model_path
                                                               << "), fallback to single REC.");
            std::memset(m_rec_fast_ctx.get(), 0, sizeof(rknn_app_context_t));
        }
    }

    if (load_dictionary(m_dict_path) != 0) {
        if (m_rec_fast_ctx != nullptr && m_rec_fast_ctx->rknn_ctx != 0) {
            visiong::npu::rknn::release_zero_copy_model(m_rec_fast_ctx.get());
            std::memset(m_rec_fast_ctx.get(), 0, sizeof(rknn_app_context_t));
        }
        visiong::npu::rknn::release_zero_copy_model(m_rec_ctx.get());
        visiong::npu::rknn::release_zero_copy_model(m_det_ctx.get());
        std::memset(m_det_ctx.get(), 0, sizeof(rknn_app_context_t));
        std::memset(m_rec_ctx.get(), 0, sizeof(rknn_app_context_t));
        return -1;
    }

    auto configure_rec_ctx = [&](rknn_app_context_t* rec_ctx, const char* tag) {
        if (rec_ctx == nullptr || rec_ctx->rknn_ctx == 0) {
            return;
        }

        if (rec_ctx->input_attrs != nullptr && rec_ctx->input_mems[0] != nullptr) {
            rknn_tensor_attr rec_native_attr;
            std::memset(&rec_native_attr, 0, sizeof(rec_native_attr));
            rec_native_attr.index = 0;
            if (rknn_query(rec_ctx->rknn_ctx,
                           RKNN_QUERY_NATIVE_INPUT_ATTR,
                           &rec_native_attr,
                           sizeof(rec_native_attr)) == RKNN_SUCC) {
                VISIONG_LOG_INFO("PPOCR",
                                 tag << " native input type=" << rec_native_attr.type
                                     << ", fmt=" << rec_native_attr.fmt);
                if (rec_native_attr.type == RKNN_TENSOR_INT8) {
                    rec_ctx->input_attrs[0].type = RKNN_TENSOR_INT8;
                    rec_ctx->input_attrs[0].fmt = RKNN_TENSOR_NHWC;
                    const int bind_ret = rknn_set_io_mem(rec_ctx->rknn_ctx,
                                                         rec_ctx->input_mems[0],
                                                         &rec_ctx->input_attrs[0]);
                    if (bind_ret != RKNN_SUCC) {
                        VISIONG_LOG_WARN("PPOCR", "Failed to rebind " << tag << " input as INT8, ret=" << bind_ret);
                    } else {
                        VISIONG_LOG_INFO("PPOCR", tag << " input rebound to INT8 for normalized preprocessing.");
                    }
                }
            }
        }

        if (rec_ctx->output_attrs != nullptr && rec_ctx->output_mems[0] != nullptr) {
            const bool try_int8_out = (std::getenv("VISIONG_PPOCR_REC_INT8_OUT") != nullptr);
            if (try_int8_out) {
                rknn_tensor_attr rec_quant_out;
                std::memset(&rec_quant_out, 0, sizeof(rec_quant_out));
                rec_quant_out.index = 0;
                if (rknn_query(rec_ctx->rknn_ctx,
                               RKNN_QUERY_OUTPUT_ATTR,
                               &rec_quant_out,
                               sizeof(rec_quant_out)) == RKNN_SUCC) {
                    if (rec_quant_out.type == RKNN_TENSOR_INT8 || rec_quant_out.type == RKNN_TENSOR_UINT8) {
                        rec_ctx->output_attrs[0] = rec_quant_out;
                        const int bind_ret = rknn_set_io_mem(rec_ctx->rknn_ctx,
                                                             rec_ctx->output_mems[0],
                                                             &rec_ctx->output_attrs[0]);
                        if (bind_ret != RKNN_SUCC) {
                            VISIONG_LOG_WARN("PPOCR",
                                             "Failed to rebind " << tag << " output to quantized type, ret="
                                                                 << bind_ret);
                        } else {
                            VISIONG_LOG_INFO("PPOCR",
                                             tag << " output rebound to quantized type="
                                                 << rec_ctx->output_attrs[0].type);
                        }
                    } else {
                        VISIONG_LOG_WARN("PPOCR", tag << " quantized output type unavailable, keep native FP16.");
                    }
                }
            }
        }

        const int rec_out_elems =
            (rec_ctx->output_attrs != nullptr && rec_ctx->io_num.n_output > 0)
                ? static_cast<int>(rec_ctx->output_attrs[0].n_elems)
                : 0;
        if (rec_out_elems > 0) {
            const int expect_vocab = static_cast<int>(m_dict.size()) + 1;
            if (expect_vocab > 1 && rec_out_elems % expect_vocab != 0) {
                const int expect_vocab_with_space = expect_vocab + 1;
                if (rec_out_elems % expect_vocab_with_space == 0) {
                    if (m_dict.empty() || m_dict.back() != " ") {
                        m_dict.emplace_back(" ");
                    }
                    VISIONG_LOG_WARN("PPOCR", tag << " vocab mismatch resolved by appending space token. dict size="
                                                   << m_dict.size());
                } else {
                    VISIONG_LOG_WARN("PPOCR", tag << " vocab mismatch remains. output_elems=" << rec_out_elems
                                                   << ", dict size=" << m_dict.size());
                }
            }
        }
    };

    configure_rec_ctx(m_rec_ctx.get(), "REC");
    configure_rec_ctx(m_rec_fast_ctx.get(), "REC_FAST");

    m_initialized = true;
    if (m_rec_fast_ctx != nullptr && m_rec_fast_ctx->rknn_ctx != 0) {
        VISIONG_LOG_INFO("PPOCR",
                         "Initialized DET(" << m_det_ctx->model_width << "x" << m_det_ctx->model_height
                                            << ") REC_MAIN(" << m_rec_ctx->model_width << "x"
                                            << m_rec_ctx->model_height << ") REC_FAST("
                                            << m_rec_fast_ctx->model_width << "x"
                                            << m_rec_fast_ctx->model_height << "), fast_ratio<="
                                            << m_rec_fast_max_ratio);
    } else {
        VISIONG_LOG_INFO("PPOCR",
                         "Initialized DET(" << m_det_ctx->model_width << "x" << m_det_ctx->model_height
                                            << ") REC(" << m_rec_ctx->model_width << "x"
                                            << m_rec_ctx->model_height << ")");
    }
    return 0;
}

void PPOCR::release_runtime() {
    if (m_det_ctx != nullptr && m_det_ctx->rknn_ctx != 0) {
        visiong::npu::rknn::release_zero_copy_model(m_det_ctx.get());
        std::memset(m_det_ctx.get(), 0, sizeof(rknn_app_context_t));
    }
    if (m_rec_ctx != nullptr && m_rec_ctx->rknn_ctx != 0) {
        visiong::npu::rknn::release_zero_copy_model(m_rec_ctx.get());
        std::memset(m_rec_ctx.get(), 0, sizeof(rknn_app_context_t));
    }
    if (m_rec_fast_ctx != nullptr && m_rec_fast_ctx->rknn_ctx != 0) {
        visiong::npu::rknn::release_zero_copy_model(m_rec_fast_ctx.get());
        std::memset(m_rec_fast_ctx.get(), 0, sizeof(rknn_app_context_t));
    }
    m_initialized = false;
}

std::vector<OCRResult> PPOCR::infer(const ImageBuffer& image) {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);

    if (!m_initialized) {
        throw std::runtime_error("PPOCR runtime is not initialized.");
    }
    if (!image.is_valid()) {
        throw std::invalid_argument("Input image is invalid.");
    }

    using Clock = std::chrono::steady_clock;
    const bool enable_profile = (std::getenv("VISIONG_PPOCR_PROFILE") != nullptr);
    const auto t_begin = Clock::now();

    const PIXEL_FORMAT_E model_input_format = parse_model_input_format(m_model_input_format);
    PreparedInputMat prepared = prepare_model_input_mat(image, model_input_format);
    cv::Mat& model_input = prepared.mat;
    if (model_input.empty()) {
        throw std::runtime_error("Failed to prepare PPOCR input image.");
    }

    const auto t_det_begin = Clock::now();
    bool det_input_ready = false;
    if (prepared.source != nullptr) {
        det_input_ready = try_write_image_to_npu_input_rga(*prepared.source, m_det_ctx.get());
    }
    if (!det_input_ready) {
        write_mat_to_npu_input(model_input, m_det_ctx.get(), false);
    }
    if (visiong::npu::rknn::run_and_sync_outputs(m_det_ctx.get(), "PPOCR-DET") != 0) {
        throw std::runtime_error("PPOCR DET inference failed.");
    }
    const auto t_det_end = Clock::now();

    const rknn_tensor_attr& det_attr = m_det_ctx->output_attrs[0];
    int out_h = 0;
    int out_w = 0;
    int out_c = 0;
    if (!parse_hwc(det_attr, &out_h, &out_w, &out_c)) {
        throw std::runtime_error("Unsupported DET output tensor shape.");
    }

    if (out_h <= 0 || out_w <= 0 || out_c <= 0) {
        throw std::runtime_error("Invalid DET output tensor dimensions.");
    }

    const auto t_post_begin = Clock::now();
    const void* det_raw = m_det_ctx->output_mems[0]->virt_addr;
    std::vector<float> det_f32 = tensor_to_f32(det_attr, det_raw);

    cv::Mat prob(out_h, out_w, CV_32FC1);
    for (int y = 0; y < out_h; ++y) {
        float* row = prob.ptr<float>(y);
        for (int x = 0; x < out_w; ++x) {
            const int idx = (y * out_w + x) * out_c;
            row[x] = det_f32[idx];
        }
    }

    const float scale_x = static_cast<float>(model_input.cols) / static_cast<float>(out_w);
    const float scale_y = static_cast<float>(model_input.rows) / static_cast<float>(out_h);

    const std::vector<QuadBox> boxes = db_postprocess(prob,
                                                      m_det_threshold,
                                                      m_box_threshold,
                                                      m_use_dilate,
                                                      m_det_unclip_ratio,
                                                      scale_x,
                                                      scale_y,
                                                      model_input.cols,
                                                      model_input.rows);
    const auto t_post_end = Clock::now();

    double crop_prep_ms = 0.0;
    double rec_npu_ms = 0.0;
    double decode_ms = 0.0;
    size_t fast_route_count = 0;
    size_t fallback_to_main_count = 0;
    const bool has_fast_rec = (m_rec_fast_ctx != nullptr && m_rec_fast_ctx->rknn_ctx != 0);

    std::vector<OCRResult> results;
    results.reserve(boxes.size());

    for (const auto& box : boxes) {
        const auto t_crop_begin = Clock::now();
        cv::Mat crop = crop_quad(model_input, box.pts);
        if (crop.empty()) {
            continue;
        }

        const float crop_ratio = static_cast<float>(crop.cols) / static_cast<float>(std::max(1, crop.rows));
        rknn_app_context_t* selected_rec_ctx = m_rec_ctx.get();
        bool used_fast = false;
        if (has_fast_rec && crop_ratio <= m_rec_fast_max_ratio) {
            selected_rec_ctx = m_rec_fast_ctx.get();
            used_fast = true;
            ++fast_route_count;
        }

        cv::Mat rec_input = preprocess_rec_crop(crop, selected_rec_ctx->model_width, selected_rec_ctx->model_height);
        if (rec_input.empty()) {
            continue;
        }
        const auto t_crop_end = Clock::now();
        crop_prep_ms += std::chrono::duration<double, std::milli>(t_crop_end - t_crop_begin).count();

        const auto t_rec_begin = Clock::now();
        write_mat_to_npu_input(rec_input, selected_rec_ctx, true);
        if (visiong::npu::rknn::run_and_sync_outputs(selected_rec_ctx, used_fast ? "PPOCR-REC-FAST" : "PPOCR-REC")
            != 0) {
            continue;
        }
        const auto t_rec_end = Clock::now();
        rec_npu_ms += std::chrono::duration<double, std::milli>(t_rec_end - t_rec_begin).count();

        const auto t_decode_begin = Clock::now();
        const rknn_tensor_attr& rec_attr = selected_rec_ctx->output_attrs[0];
        const void* rec_raw = selected_rec_ctx->output_mems[0]->virt_addr;
        DecodedText decoded = decode_ctc_output(rec_attr, rec_raw, m_dict);
        const auto t_decode_end = Clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_begin).count();

        if (used_fast && m_rec_ctx != nullptr && m_rec_ctx->rknn_ctx != 0) {
            const bool should_fallback = m_rec_fast_enable_fallback &&
                                         (decoded.text.empty() || decoded.score < m_rec_fast_fallback_score_thresh);
            if (should_fallback) {
                cv::Mat rec_main_input = preprocess_rec_crop(crop, m_rec_ctx->model_width, m_rec_ctx->model_height);
                if (!rec_main_input.empty()) {
                    const auto t_rec_fb_begin = Clock::now();
                    write_mat_to_npu_input(rec_main_input, m_rec_ctx.get(), true);
                    if (visiong::npu::rknn::run_and_sync_outputs(m_rec_ctx.get(), "PPOCR-REC-FALLBACK") == 0) {
                        const auto t_rec_fb_end = Clock::now();
                        rec_npu_ms += std::chrono::duration<double, std::milli>(t_rec_fb_end - t_rec_fb_begin).count();

                        const auto t_dec_fb_begin = Clock::now();
                        const rknn_tensor_attr& rec_fb_attr = m_rec_ctx->output_attrs[0];
                        const void* rec_fb_raw = m_rec_ctx->output_mems[0]->virt_addr;
                        const DecodedText decoded_fb = decode_ctc_output(rec_fb_attr, rec_fb_raw, m_dict);
                        const auto t_dec_fb_end = Clock::now();
                        decode_ms += std::chrono::duration<double, std::milli>(t_dec_fb_end - t_dec_fb_begin).count();

                        if (!decoded_fb.text.empty() || decoded_fb.score > decoded.score + 1e-3f) {
                            decoded = decoded_fb;
                        }
                        ++fallback_to_main_count;
                    }
                }
            }
        }

        OCRResult item;
        item.quad = quad_to_tuples(box.pts);
        item.rect = quad_to_rect(box.pts, model_input.cols, model_input.rows);
        item.det_score = box.score;
        item.text = decoded.text;
        item.text_score = decoded.score;
        results.push_back(std::move(item));
    }

    if (enable_profile) {
        const auto t_end = Clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
        const double det_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_begin).count();
        const double post_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_begin).count();
        VISIONG_LOG_INFO("PPOCR",
                         "profile total_ms=" << total_ms << ", det_ms=" << det_ms << ", post_ms=" << post_ms
                                             << ", crop_prep_ms=" << crop_prep_ms << ", rec_npu_ms=" << rec_npu_ms
                                             << ", decode_ms=" << decode_ms << ", boxes=" << boxes.size()
                                             << ", results=" << results.size() << ", fast_routes="
                                             << fast_route_count << ", rec_fallbacks=" << fallback_to_main_count);
    }

    return results;
}

bool PPOCR::is_initialized() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return m_initialized;
}

int PPOCR::det_model_width() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return (m_initialized && m_det_ctx != nullptr) ? m_det_ctx->model_width : 0;
}

int PPOCR::det_model_height() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return (m_initialized && m_det_ctx != nullptr) ? m_det_ctx->model_height : 0;
}

int PPOCR::rec_model_width() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return (m_initialized && m_rec_ctx != nullptr) ? m_rec_ctx->model_width : 0;
}

int PPOCR::rec_model_height() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return (m_initialized && m_rec_ctx != nullptr) ? m_rec_ctx->model_height : 0;
}
