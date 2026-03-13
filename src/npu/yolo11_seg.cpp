// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/yolo11_seg.h"

#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"
#include "internal/yolo_neon_opt.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

struct Yolo11SegPostProcessCtx {
    int class_count = 80;
    int mask_coeff_count = 32;
    char** labels = nullptr;
    float nms_threshold = 0.45f;
    float box_threshold = 0.25f;
    float mask_threshold = 0.5f;

    ~Yolo11SegPostProcessCtx() { visiong::npu::yolo::free_c_labels(&labels, class_count); }
};

namespace {

constexpr int kDefaultClassCount = 80;
constexpr int kDefaultMaskCoeffCount = 32;
constexpr float kDefaultNmsThreshold = 0.45f;
constexpr float kDefaultBoxThreshold = 0.25f;
constexpr float kDefaultMaskThreshold = 0.5f;
constexpr size_t kMaxInputDetections = 8400;
constexpr size_t kMaxNmsCandidates = 1024;
constexpr int kDefaultMaskTopK = 3;
constexpr int kDefaultYoloStrides[] = {8, 16, 32, 64, 128};

struct DenseTensorView {
    size_t det_count = 0;
    size_t field_count = 0;
    bool channel_first = true;
};

struct ProtoLayout {
    int channels = 0;
    int h = 0;
    int w = 0;
    bool channel_first = true;
};

float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float dot_product_f32(const float* lhs, const float* rhs, int n) {
    if (lhs == nullptr || rhs == nullptr || n <= 0) {
        return 0.0f;
    }
#if defined(__ARM_NEON)
    int i = 0;
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(lhs + i);
        const float32x4_t vb = vld1q_f32(rhs + i);
        vacc = vmlaq_f32(vacc, va, vb);
    }
    float32x2_t vpair = vadd_f32(vget_low_f32(vacc), vget_high_f32(vacc));
    vpair = vpadd_f32(vpair, vpair);
    float sum = vget_lane_f32(vpair, 0);
    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#endif
}

float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = static_cast<uint32_t>(value & 0x7C00U) >> 10;
    const uint32_t mantissa = static_cast<uint32_t>(value & 0x03FFU);

    uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            uint32_t normalized_mantissa = mantissa;
            int32_t exp = -14;
            while ((normalized_mantissa & 0x0400U) == 0U) {
                normalized_mantissa <<= 1;
                --exp;
            }
            normalized_mantissa &= 0x03FFU;
            const uint32_t exp_bits = static_cast<uint32_t>(exp + 127) << 23;
            bits = sign | exp_bits | (normalized_mantissa << 13);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        const uint32_t exp_bits = (exponent + 112U) << 23;
        bits = sign | exp_bits | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float decode_value(const void* base,
                   rknn_tensor_type type,
                   size_t index,
                   bool is_quant,
                   int32_t zero_point,
                   float scale) {
    switch (type) {
        case RKNN_TENSOR_INT8: {
            const int8_t value = static_cast<const int8_t*>(base)[index];
            return is_quant ? visiong::npu::yolo::dequantize_from_i8(value, zero_point, scale)
                            : static_cast<float>(value);
        }
        case RKNN_TENSOR_UINT8: {
            const uint8_t value = static_cast<const uint8_t*>(base)[index];
            if (is_quant) {
                return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
            }
            return static_cast<float>(value);
        }
        case RKNN_TENSOR_FLOAT16: {
            const uint16_t value = static_cast<const uint16_t*>(base)[index];
            return fp16_to_fp32(value);
        }
        case RKNN_TENSOR_FLOAT32:
            return static_cast<const float*>(base)[index];
        default:
            return 0.0f;
    }
}

bool initialize_labels(const char* label_path, int required_num_classes, Yolo11SegPostProcessCtx* ctx) {
    if (ctx == nullptr) {
        return false;
    }
    if (required_num_classes > 0) {
        ctx->class_count = required_num_classes;
    }

    if (label_path == nullptr || label_path[0] == '\0') {
        return true;
    }

    std::vector<std::string> labels;
    bool has_empty_line_after_data = false;
    if (visiong::npu::yolo::load_non_empty_lines(label_path, &labels, &has_empty_line_after_data) < 0) {
        std::printf("ERROR: Open label file %s fail!\n", label_path);
        return false;
    }

    if (required_num_classes > 0 && static_cast<int>(labels.size()) != required_num_classes) {
        std::printf("ERROR: label count mismatch: expected %d classes but loaded %d.\n",
                    required_num_classes,
                    static_cast<int>(labels.size()));
        return false;
    }
    if (labels.empty()) {
        std::printf("ERROR: No labels loaded from %s\n", label_path);
        return false;
    }

    if (visiong::npu::yolo::assign_c_labels(labels, &ctx->labels) < 0) {
        std::printf("ERROR: Malloc yolo11_seg labels failed!\n");
        return false;
    }

    ctx->class_count = static_cast<int>(labels.size());
    if (has_empty_line_after_data) {
        std::printf("Warning: label file contains empty line(s) in the middle or end; those lines were skipped.\n");
    }
    return true;
}

bool resolve_raw_view(const rknn_tensor_attr& attr,
                      size_t min_field_count,
                      size_t max_field_count,
                      DenseTensorView* view) {
    if (view == nullptr || attr.n_dims <= 0) {
        return false;
    }

    size_t total_count = 1;
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (attr.dims[i] <= 0) {
            return false;
        }
        total_count *= static_cast<size_t>(attr.dims[i]);
    }

    if (attr.n_dims >= 2U) {
        const size_t field_count_cf = static_cast<size_t>(attr.dims[1]);
        if (field_count_cf >= min_field_count && field_count_cf <= max_field_count &&
            total_count % field_count_cf == 0U) {
            view->field_count = field_count_cf;
            view->det_count = total_count / field_count_cf;
            view->channel_first = true;
            return true;
        }
    }

    const size_t field_count_cl = static_cast<size_t>(attr.dims[attr.n_dims - 1]);
    if (field_count_cl >= min_field_count && field_count_cl <= max_field_count &&
        total_count % field_count_cl == 0U) {
        view->field_count = field_count_cl;
        view->det_count = total_count / field_count_cl;
        view->channel_first = false;
        return true;
    }

    return false;
}

size_t dense_index(const DenseTensorView& view, size_t det_index, size_t field_index) {
    if (view.channel_first) {
        return field_index * view.det_count + det_index;
    }
    return det_index * view.field_count + field_index;
}

void compute_dfl_from_logits(const std::vector<float>& logits, int dfl_len, float out_box[4]) {
    for (int side = 0; side < 4; ++side) {
        const size_t base = static_cast<size_t>(side) * static_cast<size_t>(dfl_len);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < dfl_len; ++i) {
            max_logit = std::max(max_logit, logits[base + static_cast<size_t>(i)]);
        }

        float exp_sum = 0.0f;
        float weighted_sum = 0.0f;
        for (int i = 0; i < dfl_len; ++i) {
            const float exp_val = std::exp(logits[base + static_cast<size_t>(i)] - max_logit);
            exp_sum += exp_val;
            weighted_sum += exp_val * static_cast<float>(i);
        }
        out_box[side] = (exp_sum <= 0.0f) ? 0.0f : (weighted_sum / exp_sum);
    }
}

bool build_anchor_grid(int model_width,
                       int model_height,
                       size_t det_count,
                       std::vector<float>* anchor_x,
                       std::vector<float>* anchor_y,
                       std::vector<float>* stride_values) {
    if (anchor_x == nullptr || anchor_y == nullptr || stride_values == nullptr || model_width <= 0 || model_height <= 0 ||
        det_count == 0) {
        return false;
    }

    anchor_x->clear();
    anchor_y->clear();
    stride_values->clear();
    anchor_x->reserve(det_count);
    anchor_y->reserve(det_count);
    stride_values->reserve(det_count);

    size_t produced = 0;
    for (int stride : kDefaultYoloStrides) {
        if (produced >= det_count) {
            break;
        }
        const int grid_w = model_width / stride;
        const int grid_h = model_height / stride;
        if (grid_w <= 0 || grid_h <= 0) {
            continue;
        }

        const size_t level_count = static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h);
        const size_t consume = std::min(level_count, det_count - produced);
        for (size_t i = 0; i < consume; ++i) {
            const int gy = static_cast<int>(i / static_cast<size_t>(grid_w));
            const int gx = static_cast<int>(i % static_cast<size_t>(grid_w));
            anchor_x->push_back((static_cast<float>(gx) + 0.5f) * static_cast<float>(stride));
            anchor_y->push_back((static_cast<float>(gy) + 0.5f) * static_cast<float>(stride));
            stride_values->push_back(static_cast<float>(stride));
        }
        produced += consume;
    }

    if (produced != det_count) {
        anchor_x->clear();
        anchor_y->clear();
        stride_values->clear();
        return false;
    }
    return true;
}

int resolve_mask_topk() {
    const char* env = std::getenv("VISIONG_YOLO11_SEG_MASK_TOPK");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultMaskTopK;
    }
    const long v = std::strtol(env, nullptr, 10);
    if (v <= 0) {
        return 0;
    }
    if (v > OBJ_NUMB_MAX_SIZE) {
        return OBJ_NUMB_MAX_SIZE;
    }
    return static_cast<int>(v);
}

bool choose_proto_candidate(int expected_channels,
                            int c0,
                            int h0,
                            int w0,
                            bool channel_first,
                            ProtoLayout* best,
                            bool* has_best,
                            int* best_score,
                            int* best_area) {
    if (c0 <= 0 || h0 <= 0 || w0 <= 0) {
        return false;
    }

    int score = 0;
    if (expected_channels > 0) {
        score = std::abs(c0 - expected_channels);
    } else if (c0 < 8 || c0 > 256) {
        score = 1000;
    }
    const int area = h0 * w0;

    const bool replace = (!*has_best) || (score < *best_score) || (score == *best_score && area > *best_area);
    if (!replace) {
        return true;
    }

    best->channels = c0;
    best->h = h0;
    best->w = w0;
    best->channel_first = channel_first;
    *has_best = true;
    *best_score = score;
    *best_area = area;
    return true;
}

bool resolve_proto_layout(const rknn_tensor_attr& attr, int expected_channels, ProtoLayout* layout) {
    if (layout == nullptr || attr.n_dims < 3) {
        return false;
    }

    ProtoLayout best;
    bool has_best = false;
    int best_score = std::numeric_limits<int>::max();
    int best_area = -1;

    if (attr.n_dims == 4) {
        choose_proto_candidate(expected_channels,
                               static_cast<int>(attr.dims[1]),
                               static_cast<int>(attr.dims[2]),
                               static_cast<int>(attr.dims[3]),
                               true,
                               &best,
                               &has_best,
                               &best_score,
                               &best_area);
        choose_proto_candidate(expected_channels,
                               static_cast<int>(attr.dims[3]),
                               static_cast<int>(attr.dims[1]),
                               static_cast<int>(attr.dims[2]),
                               false,
                               &best,
                               &has_best,
                               &best_score,
                               &best_area);
    } else if (attr.n_dims == 3) {
        choose_proto_candidate(expected_channels,
                               static_cast<int>(attr.dims[0]),
                               static_cast<int>(attr.dims[1]),
                               static_cast<int>(attr.dims[2]),
                               true,
                               &best,
                               &has_best,
                               &best_score,
                               &best_area);
        choose_proto_candidate(expected_channels,
                               static_cast<int>(attr.dims[2]),
                               static_cast<int>(attr.dims[0]),
                               static_cast<int>(attr.dims[1]),
                               false,
                               &best,
                               &has_best,
                               &best_score,
                               &best_area);
    }

    if (!has_best || best.channels <= 0 || best.h <= 0 || best.w <= 0) {
        return false;
    }

    *layout = best;
    return true;
}

size_t proto_index(const ProtoLayout& layout, int c, int y, int x) {
    if (layout.channel_first) {
        return (static_cast<size_t>(c) * static_cast<size_t>(layout.h) + static_cast<size_t>(y)) *
                   static_cast<size_t>(layout.w) +
               static_cast<size_t>(x);
    }
    return (static_cast<size_t>(y) * static_cast<size_t>(layout.w) + static_cast<size_t>(x)) *
               static_cast<size_t>(layout.channels) +
           static_cast<size_t>(c);
}

void append_mask_contour_points(const float* coeff,
                                int coeff_count,
                                const std::vector<float>& proto_values,
                                const ProtoLayout& proto_layout,
                                int model_width,
                                int model_height,
                                float box_x1_model,
                                float box_y1_model,
                                float box_x2_model,
                                float box_y2_model,
                                float letterbox_scale,
                                int letterbox_pad_x,
                                int letterbox_pad_y,
                                object_detect_result* out,
                                float mask_threshold) {
    if (out == nullptr) {
        return;
    }
    out->mask_point_count = 0;

    if (coeff == nullptr || coeff_count <= 0 || proto_values.empty() || model_width <= 0 || model_height <= 0 ||
        letterbox_scale <= 0.0f) {
        return;
    }

    const float sx = static_cast<float>(proto_layout.w) / static_cast<float>(model_width);
    const float sy = static_cast<float>(proto_layout.h) / static_cast<float>(model_height);

    int px1 = static_cast<int>(std::floor(box_x1_model * sx));
    int py1 = static_cast<int>(std::floor(box_y1_model * sy));
    int px2 = static_cast<int>(std::ceil(box_x2_model * sx));
    int py2 = static_cast<int>(std::ceil(box_y2_model * sy));

    px1 = std::max(0, std::min(px1, proto_layout.w - 1));
    py1 = std::max(0, std::min(py1, proto_layout.h - 1));
    px2 = std::max(0, std::min(px2, proto_layout.w));
    py2 = std::max(0, std::min(py2, proto_layout.h));
    if (px2 <= px1 || py2 <= py1) {
        return;
    }

    cv::Mat binary(py2 - py1, px2 - px1, CV_8UC1, cv::Scalar(0));
    for (int py = py1; py < py2; ++py) {
        uint8_t* row = binary.ptr<uint8_t>(py - py1);
        for (int px = px1; px < px2; ++px) {
            float logit = 0.0f;
            if (proto_layout.channel_first) {
                const size_t hw_offset = static_cast<size_t>(py) * static_cast<size_t>(proto_layout.w) +
                                         static_cast<size_t>(px);
                for (int k = 0; k < coeff_count; ++k) {
                    const size_t idx = static_cast<size_t>(k) * static_cast<size_t>(proto_layout.h) *
                                           static_cast<size_t>(proto_layout.w) +
                                       hw_offset;
                    logit += coeff[k] * proto_values[idx];
                }
            } else {
                const size_t base = (static_cast<size_t>(py) * static_cast<size_t>(proto_layout.w) +
                                     static_cast<size_t>(px)) *
                                    static_cast<size_t>(proto_layout.channels);
                logit = dot_product_f32(coeff,
                                        &proto_values[base],
                                        coeff_count);
            }
            row[px - px1] = (sigmoid(logit) > mask_threshold) ? 255U : 0U;
        }
    }

    if (cv::countNonZero(binary) <= 0) {
        return;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return;
    }

    size_t best_idx = 0;
    double best_area = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best_idx = i;
        }
    }

    std::vector<cv::Point> approx;
    cv::approxPolyDP(contours[best_idx], approx, 1.5, true);
    const std::vector<cv::Point>& contour = approx.empty() ? contours[best_idx] : approx;
    if (contour.empty()) {
        return;
    }

    const int total_points = static_cast<int>(contour.size());
    const int step = std::max(1, total_points / SEG_MASK_POINT_MAX_SIZE);
    const float inv_sx = static_cast<float>(model_width) / static_cast<float>(proto_layout.w);
    const float inv_sy = static_cast<float>(model_height) / static_cast<float>(proto_layout.h);

    int point_count = 0;
    for (int i = 0; i < total_points && point_count < SEG_MASK_POINT_MAX_SIZE; i += step) {
        const int proto_x = contour[static_cast<size_t>(i)].x + px1;
        const int proto_y = contour[static_cast<size_t>(i)].y + py1;
        const float model_x = (static_cast<float>(proto_x) + 0.5f) * inv_sx;
        const float model_y = (static_cast<float>(proto_y) + 0.5f) * inv_sy;
        const float image_x = (model_x - static_cast<float>(letterbox_pad_x)) / letterbox_scale;
        const float image_y = (model_y - static_cast<float>(letterbox_pad_y)) / letterbox_scale;

        out->mask_points[point_count][0] = image_x;
        out->mask_points[point_count][1] = image_y;
        ++point_count;
    }

    out->mask_point_count = point_count;
}

int post_process_yolo11_seg(rknn_app_context_t* app_ctx,
                            rknn_tensor_mem** outputs,
                            const Yolo11SegPostProcessCtx* ctx,
                            float letterbox_scale,
                            int letterbox_pad_x,
                            int letterbox_pad_y,
                            object_detect_result_list* od_results) {
    if (app_ctx == nullptr || outputs == nullptr || ctx == nullptr || od_results == nullptr) {
        return -1;
    }
    if (letterbox_scale <= 0.0f) {
        std::printf("ERROR: invalid letterbox_scale=%f\n", letterbox_scale);
        return -1;
    }
    if (app_ctx->io_num.n_output < 2) {
        std::printf("ERROR: unexpected YOLO11-seg output count: %u\n", app_ctx->io_num.n_output);
        return -1;
    }

    const rknn_tensor_attr& det_attr = app_ctx->output_attrs[0];
    const rknn_tensor_attr& proto_attr = app_ctx->output_attrs[1];
    if ((det_attr.type != RKNN_TENSOR_INT8 && det_attr.type != RKNN_TENSOR_UINT8 &&
         det_attr.type != RKNN_TENSOR_FLOAT16 && det_attr.type != RKNN_TENSOR_FLOAT32) ||
        (proto_attr.type != RKNN_TENSOR_INT8 && proto_attr.type != RKNN_TENSOR_UINT8 &&
         proto_attr.type != RKNN_TENSOR_FLOAT16 && proto_attr.type != RKNN_TENSOR_FLOAT32)) {
        std::printf("ERROR: unsupported YOLO11-seg output tensor type(s).\n");
        return -1;
    }

    DenseTensorView det_view;
    if (!resolve_raw_view(det_attr, 6, 512, &det_view)) {
        std::printf("ERROR: unexpected YOLO11-seg detection output shape.\n");
        return -1;
    }

    ProtoLayout proto_layout;
    if (!resolve_proto_layout(proto_attr, ctx->mask_coeff_count, &proto_layout)) {
        std::printf("ERROR: unexpected YOLO11-seg prototype output shape.\n");
        return -1;
    }

    const int mask_coeff_count = proto_layout.channels;
    const int hinted_class_count = (ctx->class_count > 0) ? ctx->class_count : kDefaultClassCount;

    bool raw_dfl_mode = false;
    int dfl_len = 1;
    int class_count = -1;

    const int remain_with_hint = static_cast<int>(det_view.field_count) - mask_coeff_count - hinted_class_count;
    if (remain_with_hint == 4) {
        class_count = hinted_class_count;
    } else if (remain_with_hint > 4 && (remain_with_hint % 4) == 0) {
        class_count = hinted_class_count;
        dfl_len = remain_with_hint / 4;
        raw_dfl_mode = (dfl_len > 1);
    }

    if (class_count <= 0) {
        class_count = static_cast<int>(det_view.field_count) - 4 - mask_coeff_count;
        dfl_len = 1;
        raw_dfl_mode = false;
    }
    if (class_count <= 0) {
        std::printf("ERROR: invalid YOLO11-seg class_count=%d (field_count=%zu, mask_coeff_count=%d).\n",
                    class_count,
                    det_view.field_count,
                    mask_coeff_count);
        return -1;
    }

    if (raw_dfl_mode && dfl_len <= 1) {
        std::printf("ERROR: invalid YOLO11-seg DFL layout, dfl_len=%d\n", dfl_len);
        return -1;
    }

    const size_t box_field_count = raw_dfl_mode ? static_cast<size_t>(4 * dfl_len) : 4U;
    const size_t cls_field_start = box_field_count;
    const size_t mask_field_start = cls_field_start + static_cast<size_t>(class_count);
    if (mask_field_start + static_cast<size_t>(mask_coeff_count) > det_view.field_count) {
        std::printf("ERROR: YOLO11-seg field mapping exceeds output field count (%zu > %zu).\n",
                    mask_field_start + static_cast<size_t>(mask_coeff_count),
                    det_view.field_count);
        return -1;
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));

    const size_t candidate_count = std::min(det_view.det_count, kMaxInputDetections);
    const void* det_base = outputs[0]->virt_addr;

    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> class_ids;
    static thread_local std::vector<size_t> source_det_indices;
    boxes.clear();
    scores.clear();
    class_ids.clear();
    source_det_indices.clear();

    boxes.reserve(candidate_count * 4U);
    scores.reserve(candidate_count);
    class_ids.reserve(candidate_count);
    source_det_indices.reserve(candidate_count);

    const bool use_dfl_neon_i8 = raw_dfl_mode && app_ctx->is_quant && det_attr.type == RKNN_TENSOR_INT8;
    const bool use_quant_cls_i8 = raw_dfl_mode && app_ctx->is_quant && det_attr.type == RKNN_TENSOR_INT8;
    const int8_t* det_base_i8 = (use_dfl_neon_i8 || use_quant_cls_i8) ? static_cast<const int8_t*>(det_base) : nullptr;

    static thread_local std::vector<float> anchor_x;
    static thread_local std::vector<float> anchor_y;
    static thread_local std::vector<float> anchor_stride;
    static thread_local std::vector<float> dfl_logits;
    static thread_local float dfl_exp_delta_lut[511];
    static thread_local float dfl_lut_scale = 0.0f;
    static thread_local uint8_t dfl_lut_ready = 0;

    if (raw_dfl_mode) {
        if (!build_anchor_grid(app_ctx->model_width, app_ctx->model_height, candidate_count, &anchor_x, &anchor_y,
                               &anchor_stride)) {
            std::printf("ERROR: failed to build YOLO11-seg anchor grid for det_count=%zu model=%dx%d\n",
                        candidate_count,
                        app_ctx->model_width,
                        app_ctx->model_height);
            return -1;
        }
        if (use_dfl_neon_i8) {
            if (!dfl_lut_ready || dfl_lut_scale != det_attr.scale) {
                for (int d = -255; d <= 255; ++d) {
                    dfl_exp_delta_lut[d + 255] = std::exp(static_cast<float>(d) * det_attr.scale);
                }
                dfl_lut_scale = det_attr.scale;
                dfl_lut_ready = 1;
            }
        } else {
            dfl_logits.assign(static_cast<size_t>(4 * dfl_len), 0.0f);
        }
    }

    for (size_t i = 0; i < candidate_count; ++i) {
        float box_x1 = 0.0f;
        float box_y1 = 0.0f;
        float box_w = 0.0f;
        float box_h = 0.0f;

        if (raw_dfl_mode) {
            float dfl_box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (use_dfl_neon_i8) {
                if (det_view.channel_first) {
                    const size_t det_stride = det_view.det_count;
                    for (int side = 0; side < 4; ++side) {
                        const size_t side_base = static_cast<size_t>(side * dfl_len);
                        const int8_t* side_ptr = det_base_i8 + i + side_base * det_stride;

                        int8_t qmax = std::numeric_limits<int8_t>::min();
                        for (int b = 0; b < dfl_len; ++b) {
                            const int8_t q = side_ptr[static_cast<size_t>(b) * det_stride];
                            if (q > qmax) {
                                qmax = q;
                            }
                        }

                        float sum = 0.0f;
                        float wsum = 0.0f;
                        for (int b = 0; b < dfl_len; ++b) {
                            const int8_t q = side_ptr[static_cast<size_t>(b) * det_stride];
                            const int delta = static_cast<int>(q) - static_cast<int>(qmax);
                            const float e = dfl_exp_delta_lut[delta + 255];
                            sum += e;
                            wsum += e * static_cast<float>(b);
                        }
                        dfl_box[side] = (sum > 0.0f) ? (wsum / sum) : 0.0f;
                    }
                } else {
                    const size_t det_base_offset = i * det_view.field_count;
                    for (int side = 0; side < 4; ++side) {
                        const int8_t* side_ptr = det_base_i8 + det_base_offset + static_cast<size_t>(side * dfl_len);

                        int8_t qmax = std::numeric_limits<int8_t>::min();
                        for (int b = 0; b < dfl_len; ++b) {
                            const int8_t q = side_ptr[b];
                            if (q > qmax) {
                                qmax = q;
                            }
                        }

                        float sum = 0.0f;
                        float wsum = 0.0f;
                        for (int b = 0; b < dfl_len; ++b) {
                            const int8_t q = side_ptr[b];
                            const int delta = static_cast<int>(q) - static_cast<int>(qmax);
                            const float e = dfl_exp_delta_lut[delta + 255];
                            sum += e;
                            wsum += e * static_cast<float>(b);
                        }
                        dfl_box[side] = (sum > 0.0f) ? (wsum / sum) : 0.0f;
                    }
                }
            } else {
                for (int side = 0; side < 4; ++side) {
                    for (int b = 0; b < dfl_len; ++b) {
                        const size_t field_idx = static_cast<size_t>(side * dfl_len + b);
                        dfl_logits[field_idx] = decode_value(det_base,
                                                             det_attr.type,
                                                             dense_index(det_view, i, field_idx),
                                                             app_ctx->is_quant,
                                                             det_attr.zp,
                                                             det_attr.scale);
                    }
                }
                compute_dfl_from_logits(dfl_logits, dfl_len, dfl_box);
            }

            const float stride = anchor_stride[i];
            const float ax = anchor_x[i];
            const float ay = anchor_y[i];
            const float x1 = ax - dfl_box[0] * stride;
            const float y1 = ay - dfl_box[1] * stride;
            const float x2 = ax + dfl_box[2] * stride;
            const float y2 = ay + dfl_box[3] * stride;

            box_x1 = x1;
            box_y1 = y1;
            box_w = x2 - x1;
            box_h = y2 - y1;
        } else {
            const float cx = decode_value(det_base, det_attr.type, dense_index(det_view, i, 0), app_ctx->is_quant,
                                          det_attr.zp, det_attr.scale);
            const float cy = decode_value(det_base, det_attr.type, dense_index(det_view, i, 1), app_ctx->is_quant,
                                          det_attr.zp, det_attr.scale);
            const float w = decode_value(det_base, det_attr.type, dense_index(det_view, i, 2), app_ctx->is_quant,
                                         det_attr.zp, det_attr.scale);
            const float h = decode_value(det_base, det_attr.type, dense_index(det_view, i, 3), app_ctx->is_quant,
                                         det_attr.zp, det_attr.scale);

            box_x1 = cx - 0.5f * w;
            box_y1 = cy - 0.5f * h;
            box_w = w;
            box_h = h;
        }

        if (!(box_w > 0.0f) || !(box_h > 0.0f)) {
            continue;
        }

        float best_score = -std::numeric_limits<float>::infinity();
        int best_class = -1;
        if (use_quant_cls_i8) {
            int8_t best_q = std::numeric_limits<int8_t>::min();
            int best_q_class = -1;

            if (!det_view.channel_first) {
                const size_t cls_offset = dense_index(det_view, i, cls_field_start);
                best_q = ::visiong::npu::yolo::neonopt::argmax_i8(det_base_i8 + cls_offset, class_count, &best_q_class);
            } else {
                for (int c = 0; c < class_count; ++c) {
                    const int8_t q = det_base_i8[dense_index(det_view, i, cls_field_start + static_cast<size_t>(c))];
                    if (q > best_q) {
                        best_q = q;
                        best_q_class = c;
                    }
                }
            }

            if (best_q_class >= 0) {
                best_class = best_q_class;
                const float raw = ::visiong::npu::yolo::dequantize_from_i8(best_q, det_attr.zp, det_attr.scale);
                best_score = sigmoid(raw);
            }
        } else {
            for (int c = 0; c < class_count; ++c) {
                const float raw = decode_value(det_base,
                                               det_attr.type,
                                               dense_index(det_view, i, cls_field_start + static_cast<size_t>(c)),
                                               app_ctx->is_quant,
                                               det_attr.zp,
                                               det_attr.scale);
                const float score = raw_dfl_mode ? sigmoid(raw) : ((raw < 0.0f || raw > 1.0f) ? sigmoid(raw) : raw);
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }
        }

        if (best_class < 0 || best_score < ctx->box_threshold) {
            continue;
        }

        boxes.push_back(box_x1);
        boxes.push_back(box_y1);
        boxes.push_back(box_w);
        boxes.push_back(box_h);
        scores.push_back(best_score);
        class_ids.push_back(best_class);
        source_det_indices.push_back(i);
    }

    if (scores.empty()) {
        return 0;
    }

    int keep_indices[OBJ_NUMB_MAX_SIZE];
    const int keep_count = ::visiong::npu::yolo::neonopt::nms_topk_classwise_xywh_neon(
        boxes,
        scores,
        class_ids,
        class_count,
        ctx->nms_threshold,
        static_cast<int>(kMaxNmsCandidates),
        keep_indices,
        OBJ_NUMB_MAX_SIZE);
    if (keep_count <= 0) {
        return 0;
    }

    static thread_local std::vector<int> kept_order;
    kept_order.clear();
    kept_order.reserve(static_cast<size_t>(keep_count));
    for (int ki = 0; ki < keep_count; ++ki) {
        const int idx = keep_indices[ki];
        if (idx >= 0 && idx < static_cast<int>(scores.size())) {
            kept_order.push_back(idx);
        }
    }
    if (kept_order.empty()) {
        return 0;
    }
    std::stable_sort(kept_order.begin(), kept_order.end(), [](int lhs, int rhs) {
        return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
    });

    const int mask_topk = resolve_mask_topk();
    const bool need_mask_decode = (mask_topk > 0);

    static thread_local std::vector<float> proto_values;
    if (need_mask_decode) {
        proto_values.resize(static_cast<size_t>(proto_layout.channels) * static_cast<size_t>(proto_layout.h) *
                            static_cast<size_t>(proto_layout.w));

        const void* proto_base = outputs[1]->virt_addr;
        for (int c = 0; c < proto_layout.channels; ++c) {
            for (int y = 0; y < proto_layout.h; ++y) {
                for (int x = 0; x < proto_layout.w; ++x) {
                    const size_t pidx = proto_index(proto_layout, c, y, x);
                    proto_values[pidx] = decode_value(proto_base,
                                                      proto_attr.type,
                                                      pidx,
                                                      app_ctx->is_quant,
                                                      proto_attr.zp,
                                                      proto_attr.scale);
                }
            }
        }
    }

    static thread_local std::vector<float> coeff_buffer;
    coeff_buffer.resize(static_cast<size_t>(mask_coeff_count));

    int out_count = 0;
    int mask_rank = 0;
    for (size_t ki = 0; ki < kept_order.size() && out_count < OBJ_NUMB_MAX_SIZE; ++ki) {
        const int idx = kept_order[ki];
        if (idx < 0 || idx >= static_cast<int>(scores.size())) {
            continue;
        }

        const size_t bi = static_cast<size_t>(idx) * 4U;
        const float box_x1_model = boxes[bi + 0U];
        const float box_y1_model = boxes[bi + 1U];
        const float box_x2_model = boxes[bi + 0U] + boxes[bi + 2U];
        const float box_y2_model = boxes[bi + 1U] + boxes[bi + 3U];

        const float x1 = (box_x1_model - static_cast<float>(letterbox_pad_x)) / letterbox_scale;
        const float y1 = (box_y1_model - static_cast<float>(letterbox_pad_y)) / letterbox_scale;
        const float x2 = (box_x2_model - static_cast<float>(letterbox_pad_x)) / letterbox_scale;
        const float y2 = (box_y2_model - static_cast<float>(letterbox_pad_y)) / letterbox_scale;

        object_detect_result* out = &od_results->results[out_count];
        out->box.left = static_cast<int>(x1);
        out->box.top = static_cast<int>(y1);
        out->box.right = static_cast<int>(x2);
        out->box.bottom = static_cast<int>(y2);
        out->prop = std::max(0.0f, std::min(1.0f, scores[static_cast<size_t>(idx)]));
        out->cls_id = class_ids[static_cast<size_t>(idx)];
        out->mask_point_count = 0;

        if (need_mask_decode && mask_rank < mask_topk && idx < static_cast<int>(source_det_indices.size())) {
            const size_t src_det_index = source_det_indices[static_cast<size_t>(idx)];
            for (int k = 0; k < mask_coeff_count; ++k) {
                coeff_buffer[static_cast<size_t>(k)] =
                    decode_value(det_base,
                                 det_attr.type,
                                 dense_index(det_view, src_det_index, mask_field_start + static_cast<size_t>(k)),
                                 app_ctx->is_quant,
                                 det_attr.zp,
                                 det_attr.scale);
            }

            append_mask_contour_points(coeff_buffer.data(),
                                       mask_coeff_count,
                                       proto_values,
                                       proto_layout,
                                       app_ctx->model_width,
                                       app_ctx->model_height,
                                       box_x1_model,
                                       box_y1_model,
                                       box_x2_model,
                                       box_y2_model,
                                       letterbox_scale,
                                       letterbox_pad_x,
                                       letterbox_pad_y,
                                       out,
                                       ctx->mask_threshold);
            ++mask_rank;
        }

        ++out_count;
    }

    od_results->count = out_count;
    return 0;
}

}  // namespace

int init_yolo11_seg_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_yolo11_seg_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

int inference_yolo11_seg_model(rknn_app_context_t* app_ctx,
                               const Yolo11SegPostProcessCtx* ctx,
                               float letterbox_scale,
                               int letterbox_pad_x,
                               int letterbox_pad_y,
                               object_detect_result_list* od_results) {
    if (app_ctx == nullptr || od_results == nullptr || ctx == nullptr) {
        return -1;
    }

    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "YOLO11-seg") != 0) {
        return -1;
    }

    return post_process_yolo11_seg(app_ctx,
                                   app_ctx->output_mems,
                                   ctx,
                                   letterbox_scale,
                                   letterbox_pad_x,
                                   letterbox_pad_y,
                                   od_results);
}

int get_yolo11_seg_model_num_classes(rknn_app_context_t* app_ctx) {
    if (app_ctx == nullptr || app_ctx->output_attrs == nullptr || app_ctx->io_num.n_output < 2) {
        return -1;
    }

    DenseTensorView det_view;
    if (!resolve_raw_view(app_ctx->output_attrs[0], 6, 512, &det_view)) {
        return -1;
    }

    ProtoLayout proto_layout;
    if (!resolve_proto_layout(app_ctx->output_attrs[1], -1, &proto_layout)) {
        return -1;
    }

    const int class_count = static_cast<int>(det_view.field_count) - 4 - proto_layout.channels;
    return class_count > 0 ? class_count : -1;
}

Yolo11SegPostProcessCtx* create_yolo11_seg_post_process_ctx(const char* label_path,
                                                             float box_thresh,
                                                             float nms_thresh,
                                                             int required_num_classes) {
    std::unique_ptr<Yolo11SegPostProcessCtx> ctx(new Yolo11SegPostProcessCtx());
    ctx->class_count = (required_num_classes > 0) ? required_num_classes : kDefaultClassCount;
    ctx->mask_coeff_count = kDefaultMaskCoeffCount;
    ctx->box_threshold = box_thresh > 0.0f ? box_thresh : kDefaultBoxThreshold;
    ctx->nms_threshold = nms_thresh > 0.0f ? nms_thresh : kDefaultNmsThreshold;
    ctx->mask_threshold = kDefaultMaskThreshold;

    if (!initialize_labels(label_path, required_num_classes, ctx.get())) {
        return nullptr;
    }

    if (ctx->class_count <= 0) {
        ctx->class_count = (required_num_classes > 0) ? required_num_classes : kDefaultClassCount;
    }

    std::printf("YOLO11-seg post-process initialized: %d classes loaded.\n", ctx->class_count);
    return ctx.release();
}

void destroy_yolo11_seg_post_process_ctx(Yolo11SegPostProcessCtx* ctx) {
    delete ctx;
}

const char* coco_cls_to_name_yolo11_seg(const Yolo11SegPostProcessCtx* ctx, int cls_id) {
    if (!ctx || !ctx->labels || cls_id < 0 || cls_id >= ctx->class_count) {
        return "unknown";
    }
    return ctx->labels[cls_id];
}

