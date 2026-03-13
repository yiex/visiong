// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/yolo11_pose.h"

#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"
#include "internal/yolo_neon_opt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

struct Yolo11PosePostProcessCtx {
    int class_count = 1;
    int keypoint_count = POSE_KEYPOINT_MAX_SIZE;
    char** labels = nullptr;
    float nms_threshold = 0.45f;
    float box_threshold = 0.25f;

    ~Yolo11PosePostProcessCtx() { visiong::npu::yolo::free_c_labels(&labels, class_count); }
};

namespace {

constexpr int kDefaultClassCount = 1;
constexpr int kDefaultKeypointCount = POSE_KEYPOINT_MAX_SIZE;
constexpr float kDefaultNmsThreshold = 0.45f;
constexpr float kDefaultBoxThreshold = 0.25f;
constexpr size_t kMaxInputDetections = 8400;
constexpr size_t kMaxNmsCandidates = 1024;

struct DenseTensorView {
    size_t det_count = 0;
    size_t field_count = 0;
    bool channel_first = true;
};

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

bool resolve_raw_view(const rknn_tensor_attr& attr, DenseTensorView* view) {
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
        if (field_count_cf >= 8 && field_count_cf <= 256 && total_count % field_count_cf == 0U) {
            view->field_count = field_count_cf;
            view->det_count = total_count / field_count_cf;
            view->channel_first = true;
            return true;
        }
    }

    const size_t field_count_cl = static_cast<size_t>(attr.dims[attr.n_dims - 1]);
    if (field_count_cl >= 8 && field_count_cl <= 256 && total_count % field_count_cl == 0U) {
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

float normalize_probability(float value) {
    if (value < 0.0f || value > 1.0f) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
    return std::max(0.0f, std::min(1.0f, value));
}

bool infer_model_layout(const rknn_tensor_attr& output_attr,
                        int preferred_class_count,
                        int preferred_keypoint_count,
                        DenseTensorView* view,
                        int* class_count,
                        int* keypoint_count) {
    if (!resolve_raw_view(output_attr, view)) {
        return false;
    }

    const int field_count = static_cast<int>(view->field_count);
    if (field_count <= 7) {
        return false;
    }

    int resolved_class_count = preferred_class_count > 0 ? preferred_class_count : kDefaultClassCount;
    int remaining = field_count - 4 - resolved_class_count;
    if (remaining < 3 || (remaining % 3) != 0) {
        bool found = false;
        for (int c = 1; c <= 8 && c < field_count - 4; ++c) {
            const int rem = field_count - 4 - c;
            if (rem >= 3 && (rem % 3) == 0) {
                resolved_class_count = c;
                remaining = rem;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    int resolved_keypoint_count = remaining / 3;
    if (preferred_keypoint_count > 0 && preferred_keypoint_count <= resolved_keypoint_count) {
        resolved_keypoint_count = preferred_keypoint_count;
    }
    if (resolved_keypoint_count <= 0) {
        return false;
    }

    *class_count = resolved_class_count;
    *keypoint_count = std::min(resolved_keypoint_count, POSE_KEYPOINT_MAX_SIZE);
    return true;
}

bool initialize_labels(const char* label_path, int required_num_classes, Yolo11PosePostProcessCtx* ctx) {
    if (ctx == nullptr) {
        return false;
    }
    if (required_num_classes > 0) {
        ctx->class_count = required_num_classes;
    }

    std::vector<std::string> labels;
    if (label_path == nullptr || label_path[0] == '\0') {
        labels.reserve(static_cast<size_t>(ctx->class_count));
        for (int i = 0; i < ctx->class_count; ++i) {
            labels.emplace_back(i == 0 ? "person" : ("class_" + std::to_string(i)));
        }
    } else {
        bool has_empty_line_after_data = false;
        if (visiong::npu::yolo::load_non_empty_lines(label_path, &labels, &has_empty_line_after_data) < 0) {
            std::printf("ERROR: Open label file %s fail!\n", label_path);
            return false;
        }
        if (required_num_classes > 0 && static_cast<int>(labels.size()) != required_num_classes) {
            std::printf("ERROR: label count mismatch: expected %d classes but loaded %d.\n",
                        required_num_classes, static_cast<int>(labels.size()));
            return false;
        }
        if (has_empty_line_after_data) {
            std::printf("Warning: label file contains empty line(s); those lines were skipped.\n");
        }
    }

    if (labels.empty()) {
        std::printf("ERROR: no labels available for yolo11 pose\n");
        return false;
    }
    if (visiong::npu::yolo::assign_c_labels(labels, &ctx->labels) < 0) {
        std::printf("ERROR: Malloc yolo11_pose labels failed!\n");
        return false;
    }
    ctx->class_count = static_cast<int>(labels.size());
    return true;
}


int post_process_yolo11_pose_split(rknn_app_context_t* app_ctx,
                                   rknn_tensor_mem** outputs,
                                   const Yolo11PosePostProcessCtx* ctx,
                                   object_detect_result_list* od_results) {
    if (app_ctx == nullptr || outputs == nullptr || ctx == nullptr || od_results == nullptr) {
        return -1;
    }
    if (app_ctx->io_num.n_output < 3) {
        return -1;
    }
    const bool split4_mode = (app_ctx->io_num.n_output >= 4);

    const rknn_tensor_attr& box_attr = app_ctx->output_attrs[0];
    const rknn_tensor_attr& score_attr = app_ctx->output_attrs[1];
    const rknn_tensor_attr* kpt_attr = split4_mode ? nullptr : &app_ctx->output_attrs[2];
    const rknn_tensor_attr* kpt_xy_attr = split4_mode ? &app_ctx->output_attrs[2] : nullptr;
    const rknn_tensor_attr* kpt_conf_attr = split4_mode ? &app_ctx->output_attrs[3] : nullptr;

    auto resolve_split_view = [](const rknn_tensor_attr& attr, DenseTensorView* view) -> bool {
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
            const size_t field_cf = static_cast<size_t>(attr.dims[1]);
            if (field_cf > 0 && total_count % field_cf == 0U) {
                view->field_count = field_cf;
                view->det_count = total_count / field_cf;
                view->channel_first = true;
                return true;
            }
        }
        const size_t field_cl = static_cast<size_t>(attr.dims[attr.n_dims - 1]);
        if (field_cl > 0 && total_count % field_cl == 0U) {
            view->field_count = field_cl;
            view->det_count = total_count / field_cl;
            view->channel_first = false;
            return true;
        }
        return false;
    };

    DenseTensorView box_view;
    DenseTensorView score_view;
    DenseTensorView kpt_view;
    DenseTensorView kpt_xy_view;
    DenseTensorView kpt_conf_view;
    if (!resolve_split_view(box_attr, &box_view) || !resolve_split_view(score_attr, &score_view)) {
        std::printf("ERROR: unexpected YOLO11-pose split output shape.\n");
        return -1;
    }
    if (!split4_mode && (kpt_attr == nullptr || !resolve_split_view(*kpt_attr, &kpt_view))) {
        std::printf("ERROR: unexpected YOLO11-pose split output shape.\n");
        return -1;
    }
    if (split4_mode &&
        (kpt_xy_attr == nullptr || kpt_conf_attr == nullptr || !resolve_split_view(*kpt_xy_attr, &kpt_xy_view) ||
         !resolve_split_view(*kpt_conf_attr, &kpt_conf_view))) {
        std::printf("ERROR: unexpected YOLO11-pose split output shape.\n");
        return -1;
    }
    if (box_view.field_count < 4 || score_view.field_count < 1) {
        std::printf("ERROR: invalid YOLO11-pose split field count.\n");
        return -1;
    }
    if (!split4_mode && kpt_view.field_count < 3) {
        std::printf("ERROR: invalid YOLO11-pose split field count.\n");
        return -1;
    }
    if (split4_mode && (kpt_xy_view.field_count < 2 || kpt_conf_view.field_count < 1)) {
        std::printf("ERROR: invalid YOLO11-pose split field count.\n");
        return -1;
    }

    size_t candidate_count = 0;
    int keypoint_count = 0;
    if (split4_mode) {
        candidate_count = std::min(
            {box_view.det_count, score_view.det_count, kpt_xy_view.det_count, kpt_conf_view.det_count, kMaxInputDetections});
        keypoint_count =
            std::min(std::min(static_cast<int>(kpt_xy_view.field_count / 2), static_cast<int>(kpt_conf_view.field_count)),
                     ctx->keypoint_count);
    } else {
        candidate_count = std::min({box_view.det_count, score_view.det_count, kpt_view.det_count, kMaxInputDetections});
        keypoint_count = std::min(static_cast<int>(kpt_view.field_count / 3), ctx->keypoint_count);
    }
    if (keypoint_count <= 0) {
        std::printf("ERROR: invalid YOLO11-pose keypoint count in split output.\n");
        return -1;
    }

    const void* box_base = outputs[0]->virt_addr;
    const void* score_base = outputs[1]->virt_addr;
    const void* kpt_base = split4_mode ? nullptr : outputs[2]->virt_addr;
    const void* kpt_xy_base = split4_mode ? outputs[2]->virt_addr : nullptr;
    const void* kpt_conf_base = split4_mode ? outputs[3]->virt_addr : nullptr;
    const bool box_quant = (box_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const bool score_quant = (score_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const bool kpt_quant = (!split4_mode && kpt_attr != nullptr &&
                            kpt_attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const bool kpt_xy_quant = (split4_mode && kpt_xy_attr != nullptr &&
                               kpt_xy_attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const bool kpt_conf_quant = (split4_mode && kpt_conf_attr != nullptr &&
                                 kpt_conf_attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> class_ids;
    static thread_local std::vector<int> candidate_src_indices;
    boxes.clear();
    scores.clear();
    class_ids.clear();
    candidate_src_indices.clear();

    boxes.reserve(candidate_count * 4U);
    scores.reserve(candidate_count);
    class_ids.reserve(candidate_count);
    candidate_src_indices.reserve(candidate_count);

    for (size_t i = 0; i < candidate_count; ++i) {
        const float cx = decode_value(box_base, box_attr.type, dense_index(box_view, i, 0), box_quant, box_attr.zp, box_attr.scale);
        const float cy = decode_value(box_base, box_attr.type, dense_index(box_view, i, 1), box_quant, box_attr.zp, box_attr.scale);
        const float w = decode_value(box_base, box_attr.type, dense_index(box_view, i, 2), box_quant, box_attr.zp, box_attr.scale);
        const float h = decode_value(box_base, box_attr.type, dense_index(box_view, i, 3), box_quant, box_attr.zp, box_attr.scale);
        if (!(w > 0.0f) || !(h > 0.0f)) {
            continue;
        }

        float score =
            decode_value(score_base, score_attr.type, dense_index(score_view, i, 0), score_quant, score_attr.zp, score_attr.scale);
        score = normalize_probability(score);
        if (score < ctx->box_threshold) {
            continue;
        }

        boxes.push_back(cx - 0.5f * w);
        boxes.push_back(cy - 0.5f * h);
        boxes.push_back(w);
        boxes.push_back(h);
        scores.push_back(score);
        class_ids.push_back(0);
        candidate_src_indices.push_back(static_cast<int>(i));
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));
    if (scores.empty()) {
        return 0;
    }

    int keep_indices[OBJ_NUMB_MAX_SIZE];
    const int keep_count = ::visiong::npu::yolo::neonopt::nms_topk_classwise_xywh_neon(
        boxes, scores, class_ids, 1, ctx->nms_threshold, static_cast<int>(kMaxNmsCandidates), keep_indices, OBJ_NUMB_MAX_SIZE);

    int out_count = 0;
    for (int ki = 0; ki < keep_count; ++ki) {
        const int idx = keep_indices[ki];
        if (idx < 0 || out_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }

        const size_t bi = static_cast<size_t>(idx) * 4U;
        const float x1 = boxes[bi + 0U];
        const float y1 = boxes[bi + 1U];
        const float x2 = boxes[bi + 0U] + boxes[bi + 2U];
        const float y2 = boxes[bi + 1U] + boxes[bi + 3U];

        od_results->results[out_count].box.left = static_cast<int>(x1);
        od_results->results[out_count].box.top = static_cast<int>(y1);
        od_results->results[out_count].box.right = static_cast<int>(x2);
        od_results->results[out_count].box.bottom = static_cast<int>(y2);
        od_results->results[out_count].prop = std::max(0.0f, std::min(1.0f, scores[static_cast<size_t>(idx)]));
        od_results->results[out_count].cls_id = 0;
        od_results->results[out_count].keypoint_count = keypoint_count;

        const int src_det_idx = candidate_src_indices[static_cast<size_t>(idx)];
        const size_t src_det_index = static_cast<size_t>(src_det_idx);
        for (int k = 0; k < keypoint_count; ++k) {
            if (split4_mode) {
                const size_t xy_field = static_cast<size_t>(k) * 2U;
                od_results->results[out_count].keypoints[k][0] =
                    decode_value(kpt_xy_base,
                                 kpt_xy_attr->type,
                                 dense_index(kpt_xy_view, src_det_index, xy_field + 0U),
                                 kpt_xy_quant,
                                 kpt_xy_attr->zp,
                                 kpt_xy_attr->scale);
                od_results->results[out_count].keypoints[k][1] =
                    decode_value(kpt_xy_base,
                                 kpt_xy_attr->type,
                                 dense_index(kpt_xy_view, src_det_index, xy_field + 1U),
                                 kpt_xy_quant,
                                 kpt_xy_attr->zp,
                                 kpt_xy_attr->scale);
                const float ks_raw =
                    decode_value(kpt_conf_base,
                                 kpt_conf_attr->type,
                                 dense_index(kpt_conf_view, src_det_index, static_cast<size_t>(k)),
                                 kpt_conf_quant,
                                 kpt_conf_attr->zp,
                                 kpt_conf_attr->scale);
                od_results->results[out_count].keypoints[k][2] = normalize_probability(ks_raw);
            } else {
                const size_t base_field = static_cast<size_t>(k) * 3U;
                od_results->results[out_count].keypoints[k][0] =
                    decode_value(kpt_base,
                                 kpt_attr->type,
                                 dense_index(kpt_view, src_det_index, base_field + 0U),
                                 kpt_quant,
                                 kpt_attr->zp,
                                 kpt_attr->scale);
                od_results->results[out_count].keypoints[k][1] =
                    decode_value(kpt_base,
                                 kpt_attr->type,
                                 dense_index(kpt_view, src_det_index, base_field + 1U),
                                 kpt_quant,
                                 kpt_attr->zp,
                                 kpt_attr->scale);
                const float ks_raw =
                    decode_value(kpt_base,
                                 kpt_attr->type,
                                 dense_index(kpt_view, src_det_index, base_field + 2U),
                                 kpt_quant,
                                 kpt_attr->zp,
                                 kpt_attr->scale);
                od_results->results[out_count].keypoints[k][2] = normalize_probability(ks_raw);
            }
        }
        ++out_count;
    }

    od_results->count = out_count;
    return 0;
}

int post_process_yolo11_pose(rknn_app_context_t* app_ctx,
                             rknn_tensor_mem** outputs,
                             const Yolo11PosePostProcessCtx* ctx,
                             object_detect_result_list* od_results) {
    if (app_ctx == nullptr || outputs == nullptr || ctx == nullptr || od_results == nullptr) {
        return -1;
    }
    if (app_ctx->io_num.n_output < 1) {
        std::printf("ERROR: unexpected YOLO11-pose output count: %u\n", app_ctx->io_num.n_output);
        return -1;
    }
    if (app_ctx->io_num.n_output >= 3) {
        return post_process_yolo11_pose_split(app_ctx, outputs, ctx, od_results);
    }

    const rknn_tensor_attr& output_attr = app_ctx->output_attrs[0];
    if (output_attr.type != RKNN_TENSOR_INT8 && output_attr.type != RKNN_TENSOR_UINT8 &&
        output_attr.type != RKNN_TENSOR_FLOAT16 && output_attr.type != RKNN_TENSOR_FLOAT32) {
        std::printf("ERROR: unsupported YOLO11-pose output type: %d\n", output_attr.type);
        return -1;
    }

    DenseTensorView view;
    int class_count = ctx->class_count;
    int keypoint_count = ctx->keypoint_count;
    if (!infer_model_layout(output_attr, ctx->class_count, ctx->keypoint_count, &view, &class_count, &keypoint_count)) {
        std::printf("ERROR: unexpected YOLO11-pose output shape.\n");
        return -1;
    }

    const size_t candidate_count = std::min(view.det_count, kMaxInputDetections);
    const void* base = outputs[0]->virt_addr;
    const size_t cls_start = 4U;
    const size_t kpt_start = cls_start + static_cast<size_t>(class_count);
    const bool output_quant = (output_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> class_ids;
    static thread_local std::vector<int> candidate_src_indices;
    boxes.clear();
    scores.clear();
    class_ids.clear();
    candidate_src_indices.clear();

    boxes.reserve(candidate_count * 4U);
    scores.reserve(candidate_count);
    class_ids.reserve(candidate_count);
    candidate_src_indices.reserve(candidate_count);

    for (size_t i = 0; i < candidate_count; ++i) {
        const float cx = decode_value(base, output_attr.type, dense_index(view, i, 0), output_quant,
                                      output_attr.zp, output_attr.scale);
        const float cy = decode_value(base, output_attr.type, dense_index(view, i, 1), output_quant,
                                      output_attr.zp, output_attr.scale);
        const float w = decode_value(base, output_attr.type, dense_index(view, i, 2), output_quant,
                                     output_attr.zp, output_attr.scale);
        const float h = decode_value(base, output_attr.type, dense_index(view, i, 3), output_quant,
                                     output_attr.zp, output_attr.scale);
        if (!(w > 0.0f) || !(h > 0.0f)) {
            continue;
        }

        float best_score = -std::numeric_limits<float>::infinity();
        int best_class = -1;
        for (int c = 0; c < class_count; ++c) {
            const float cls_score = decode_value(base,
                                                 output_attr.type,
                                                 dense_index(view, i, cls_start + static_cast<size_t>(c)),
                                                 output_quant,
                                                 output_attr.zp,
                                                 output_attr.scale);
            if (cls_score > best_score) {
                best_score = cls_score;
                best_class = c;
            }
        }

        const float best_prob = normalize_probability(best_score);
        if (best_class < 0 || best_prob < ctx->box_threshold) {
            continue;
        }

        boxes.push_back(cx - 0.5f * w);
        boxes.push_back(cy - 0.5f * h);
        boxes.push_back(w);
        boxes.push_back(h);
        scores.push_back(best_prob);
        class_ids.push_back(best_class);
        candidate_src_indices.push_back(static_cast<int>(i));
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));
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

    int out_count = 0;
    for (int ki = 0; ki < keep_count; ++ki) {
        const int idx = keep_indices[ki];
        if (idx < 0 || out_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }

        const size_t bi = static_cast<size_t>(idx) * 4U;
        const float x1 = boxes[bi + 0U];
        const float y1 = boxes[bi + 1U];
        const float x2 = boxes[bi + 0U] + boxes[bi + 2U];
        const float y2 = boxes[bi + 1U] + boxes[bi + 3U];

        od_results->results[out_count].box.left = static_cast<int>(x1);
        od_results->results[out_count].box.top = static_cast<int>(y1);
        od_results->results[out_count].box.right = static_cast<int>(x2);
        od_results->results[out_count].box.bottom = static_cast<int>(y2);
        od_results->results[out_count].prop = std::max(0.0f, std::min(1.0f, scores[static_cast<size_t>(idx)]));
        od_results->results[out_count].cls_id = class_ids[static_cast<size_t>(idx)];
        od_results->results[out_count].keypoint_count = keypoint_count;

        const int src_det_idx = candidate_src_indices[static_cast<size_t>(idx)];
        const size_t src_det_index = static_cast<size_t>(src_det_idx);
        for (int k = 0; k < keypoint_count; ++k) {
            const size_t base_field = kpt_start + static_cast<size_t>(k) * 3U;
            od_results->results[out_count].keypoints[k][0] =
                decode_value(base,
                             output_attr.type,
                             dense_index(view, src_det_index, base_field + 0U),
                             output_quant,
                             output_attr.zp,
                             output_attr.scale);
            od_results->results[out_count].keypoints[k][1] =
                decode_value(base,
                             output_attr.type,
                             dense_index(view, src_det_index, base_field + 1U),
                             output_quant,
                             output_attr.zp,
                             output_attr.scale);
            const float ks_raw =
                decode_value(base,
                             output_attr.type,
                             dense_index(view, src_det_index, base_field + 2U),
                             output_quant,
                             output_attr.zp,
                             output_attr.scale);
            od_results->results[out_count].keypoints[k][2] = normalize_probability(ks_raw);
        }

        ++out_count;
    }

    od_results->count = out_count;
    return 0;
}

}  // namespace

int init_yolo11_pose_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_yolo11_pose_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

int inference_yolo11_pose_model(rknn_app_context_t* app_ctx,
                                const Yolo11PosePostProcessCtx* ctx,
                                object_detect_result_list* od_results) {
    if (app_ctx == nullptr || ctx == nullptr || od_results == nullptr) {
        return -1;
    }

    if (visiong::npu::rknn::run_model(app_ctx, "YOLO11-pose") != 0) {
        return -1;
    }

    if (app_ctx->io_num.n_output == 1) {
    // Pose head packs box coordinates and confidence/keypoints into a single tensor. / Pose head packs box coordinates 与 confidence/keypoints into single tensor.
    // On RV1106 int8 quantization, per-tensor scales can heavily quantize confidence fields.
    // Requesting float outputs from runtime preserves usable values for post-process. / Requesting float 输出s from 运行时 preserves usable values 用于 post-process.
    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;

    int ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, nullptr);
    if (ret == RKNN_SUCC && outputs[0].buf != nullptr) {
        const rknn_tensor_type saved_type = app_ctx->output_attrs[0].type;
        const bool saved_is_quant = app_ctx->is_quant;

        app_ctx->output_attrs[0].type = RKNN_TENSOR_FLOAT32;
        app_ctx->is_quant = false;

        rknn_tensor_mem fake_output_mem;
        std::memset(&fake_output_mem, 0, sizeof(fake_output_mem));
        fake_output_mem.virt_addr = outputs[0].buf;
        rknn_tensor_mem* fake_outputs[1] = {&fake_output_mem};

        const int pp_ret = post_process_yolo11_pose(app_ctx, fake_outputs, ctx, od_results);

        app_ctx->output_attrs[0].type = saved_type;
        app_ctx->is_quant = saved_is_quant;
        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        return pp_ret;
    }

    std::printf("WARNING: rknn_outputs_get(want_float=1) failed ret=%d, fallback to output mem dequant path.\n", ret);
    if (ret == RKNN_SUCC) {
        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
    }

    }

    if (visiong::npu::rknn::sync_output_tensors_to_cpu(app_ctx, "YOLO11-pose") != 0) {
        return -1;
    }
    return post_process_yolo11_pose(app_ctx, app_ctx->output_mems, ctx, od_results);
}

int get_yolo11_pose_model_num_classes(rknn_app_context_t* app_ctx) {
    if (app_ctx == nullptr || app_ctx->output_attrs == nullptr || app_ctx->io_num.n_output < 1) {
        return -1;
    }
    if (app_ctx->io_num.n_output >= 3) {
        return 1;
    }

    DenseTensorView view;
    if (!resolve_raw_view(app_ctx->output_attrs[0], &view)) {
        return -1;
    }

    const int field_count = static_cast<int>(view.field_count);
    for (int c = 1; c <= 8 && c < field_count - 4; ++c) {
        const int rem = field_count - 4 - c;
        if (rem >= 3 && (rem % 3) == 0) {
            return c;
        }
    }
    return -1;
}

Yolo11PosePostProcessCtx* create_yolo11_pose_post_process_ctx(const char* label_path,
                                                              float box_thresh,
                                                              float nms_thresh,
                                                              int required_num_classes) {
    std::unique_ptr<Yolo11PosePostProcessCtx> ctx(new Yolo11PosePostProcessCtx());
    ctx->class_count = required_num_classes > 0 ? required_num_classes : kDefaultClassCount;
    ctx->keypoint_count = kDefaultKeypointCount;
    ctx->box_threshold = box_thresh > 0.0f ? box_thresh : kDefaultBoxThreshold;
    ctx->nms_threshold = nms_thresh > 0.0f ? nms_thresh : kDefaultNmsThreshold;

    if (!initialize_labels(label_path, required_num_classes, ctx.get())) {
        return nullptr;
    }

    std::printf("YOLO11-pose post-process initialized: %d classes, %d keypoints.\n",
                ctx->class_count,
                ctx->keypoint_count);
    return ctx.release();
}

void destroy_yolo11_pose_post_process_ctx(Yolo11PosePostProcessCtx* ctx) {
    delete ctx;
}

const char* coco_cls_to_name_yolo11_pose(const Yolo11PosePostProcessCtx* ctx, int cls_id) {
    if (!ctx || !ctx->labels || cls_id < 0 || cls_id >= ctx->class_count) {
        return "unknown";
    }
    return ctx->labels[cls_id];
}

