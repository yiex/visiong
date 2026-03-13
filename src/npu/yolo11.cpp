// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/yolo11.h"

#include "common/internal/dma_alloc.h"
#include "internal/rknn_model_utils.h"
#include "internal/yolo_neon_opt.h"
#include "internal/yolo_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

struct Yolo11PostProcessCtx {
    int class_count = 80;
    char** labels = nullptr;
    float nms_threshold = 0.45f;
    float box_threshold = 0.25f;

    ~Yolo11PostProcessCtx() { visiong::npu::yolo::free_c_labels(&labels, class_count); }
};

namespace {

constexpr int kDefaultClassCount = 80;
constexpr float kDefaultNmsThreshold = 0.45f;
constexpr float kDefaultBoxThreshold = 0.25f;
constexpr int kBranchCount = 3;
constexpr size_t kMaxNmsCandidates = 512;

[[maybe_unused]] void compute_dfl(const float* tensor, int dfl_len, float box[4]) {
    for (int b = 0; b < 4; ++b) {
        float exp_sum = 0.0f;
        float weighted_sum = 0.0f;
        for (int i = 0; i < dfl_len; ++i) {
            const float exp_val = std::exp(tensor[b * dfl_len + i]);
            exp_sum += exp_val;
            weighted_sum += exp_val * static_cast<float>(i);
        }
        box[b] = (exp_sum <= 0.0f) ? 0.0f : (weighted_sum / exp_sum);
    }
}

[[maybe_unused]] void nms_by_class(const std::vector<float>& boxes, const std::vector<int>& class_ids, std::vector<int>* order,
                                   int filter_class_id, float threshold) {
    for (size_t i = 0; i < order->size(); ++i) {
        const int n = (*order)[i];
        if (n < 0 || class_ids[n] != filter_class_id) {
            continue;
        }

        for (size_t j = i + 1; j < order->size(); ++j) {
            const int m = (*order)[j];
            if (m < 0 || class_ids[m] != filter_class_id) {
                continue;
            }

            const float iou = visiong::npu::yolo::calculate_iou_xywh(
                boxes[n * 4 + 0], boxes[n * 4 + 1], boxes[n * 4 + 2], boxes[n * 4 + 3], boxes[m * 4 + 0],
                boxes[m * 4 + 1], boxes[m * 4 + 2], boxes[m * 4 + 3]);
            if (iou > threshold) {
                (*order)[j] = -1;
            }
        }
    }
}

[[maybe_unused]] void build_topk_indices_desc(const std::vector<float>& scores, std::vector<int>* order, size_t max_count) {
    order->resize(scores.size());
    std::iota(order->begin(), order->end(), 0);
    auto desc_by_score = [&scores](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    };
    if (order->size() > max_count) {
        std::partial_sort(order->begin(), order->begin() + static_cast<std::ptrdiff_t>(max_count), order->end(),
                          desc_by_score);
        order->resize(max_count);
        return;
    }
    std::stable_sort(order->begin(), order->end(), desc_by_score);
}

int process_i8_rv1106(int8_t* box_tensor, int32_t box_zp, float box_scale, int8_t* score_tensor, int32_t score_zp,
                      float score_scale, int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len, int class_count, const float* box_exp_lut256,
                      std::vector<float>* boxes, std::vector<float>* obj_probs, std::vector<int>* class_ids,
                      float threshold) {
    (void)box_zp;
    (void)box_scale;
    int valid_count = 0;
    const int8_t score_threshold_i8 = visiong::npu::yolo::quantize_to_i8(threshold, score_zp, score_scale);
    const int8_t score_sum_threshold_i8 =
        visiong::npu::yolo::quantize_to_i8(threshold, score_sum_zp, score_sum_scale);

    float box[4];

    for (int grid_y = 0; grid_y < grid_h; ++grid_y) {
        for (int grid_x = 0; grid_x < grid_w; ++grid_x) {
            const int spatial_offset = grid_y * grid_w + grid_x;
            if (score_sum_tensor != nullptr && score_sum_tensor[spatial_offset] < score_sum_threshold_i8) {
                continue;
            }

            const int score_base_offset = spatial_offset * class_count;
            int max_class_id = 0;
            const int8_t max_score = ::visiong::npu::yolo::neonopt::argmax_i8(score_tensor + score_base_offset,
                                                                            class_count, &max_class_id);
            if (max_score <= score_threshold_i8) {
                continue;
            }

            const int box_base_offset = spatial_offset * 4 * dfl_len;
            const int8_t* __restrict logits = box_tensor + box_base_offset;
            ::visiong::npu::yolo::neonopt::dfl_expect_i8_lut(logits, dfl_len, box_exp_lut256, box);

            const float x1 = (-box[0] + static_cast<float>(grid_x) + 0.5f) * stride;
            const float y1 = (-box[1] + static_cast<float>(grid_y) + 0.5f) * stride;
            const float x2 = (box[2] + static_cast<float>(grid_x) + 0.5f) * stride;
            const float y2 = (box[3] + static_cast<float>(grid_y) + 0.5f) * stride;

            {
                const size_t base = boxes->size();
                boxes->resize(base + 4U);
                float* __restrict bdst = boxes->data() + base;
                bdst[0] = x1;
                bdst[1] = y1;
                bdst[2] = x2 - x1;
                bdst[3] = y2 - y1;
            }
            obj_probs->push_back(visiong::npu::yolo::dequantize_from_i8(max_score, score_zp, score_scale));
            class_ids->push_back(max_class_id);
            ++valid_count;
        }
    }

    return valid_count;
}

bool initialize_labels(const char* label_path, int required_num_classes, Yolo11PostProcessCtx* ctx) {
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
        printf("ERROR: Open label file %s fail!\n", label_path);
        return false;
    }

    if (required_num_classes >= 0 && static_cast<int>(labels.size()) != required_num_classes) {
        printf("ERROR: label count mismatch: model requires %d classes, label file has %d.\n", required_num_classes,
               static_cast<int>(labels.size()));
        return false;
    }
    if (labels.empty()) {
        printf("ERROR: No labels loaded from %s\n", label_path);
        return false;
    }

    if (visiong::npu::yolo::assign_c_labels(labels, &ctx->labels) < 0) {
        printf("ERROR: Malloc yolo11_labels failed!\n");
        return false;
    }

    ctx->class_count = static_cast<int>(labels.size());
    if (has_empty_line_after_data) {
        printf("Warning: label file contains empty line(s) in the middle or end; those lines were skipped.\n");
    }
    return true;
}

int post_process_yolo11(rknn_app_context_t* app_ctx, rknn_tensor_mem** outputs, const Yolo11PostProcessCtx* ctx,
                        float letterbox_scale, int letterbox_pad_x, int letterbox_pad_y,
                        object_detect_result_list* od_results) {
    if (app_ctx == nullptr || outputs == nullptr || od_results == nullptr || ctx == nullptr) {
        return -1;
    }
    if (letterbox_scale <= 0.0f) {
        printf("ERROR: invalid letterbox_scale=%f\n", letterbox_scale);
        return -1;
    }
    if (!app_ctx->is_quant) {
        printf("ERROR: YOLOv11 for RV1106/1103 only supports quantization mode\n");
        return -1;
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));

    if (app_ctx->io_num.n_output < kBranchCount * 2) {
        printf("ERROR: unexpected YOLOv11 output count: %u\n", app_ctx->io_num.n_output);
        return -1;
    }
    const int output_per_branch = app_ctx->io_num.n_output / kBranchCount;
    if (output_per_branch < 2) {
        printf("ERROR: unexpected YOLOv11 branch layout: output_per_branch=%d\n", output_per_branch);
        return -1;
    }
    if (app_ctx->output_attrs[0].n_dims < 4) {
        printf("ERROR: unexpected YOLOv11 box tensor dims\n");
        return -1;
    }
    const int dfl_len = app_ctx->output_attrs[0].dims[3] / 4;
    if (dfl_len <= 0) {
        printf("ERROR: invalid YOLOv11 DFL length=%d\n", dfl_len);
        return -1;
    }
    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> obj_probs;
    static thread_local std::vector<int> class_ids;
    boxes.clear();
    obj_probs.clear();
    class_ids.clear();
    size_t reserve_candidates = 0;
    for (int branch = 0; branch < kBranchCount; ++branch) {
        const int box_idx = branch * output_per_branch;
        if (box_idx >= static_cast<int>(app_ctx->io_num.n_output)) {
            break;
        }
        const int grid_h = app_ctx->output_attrs[box_idx].dims[1];
        const int grid_w = app_ctx->output_attrs[box_idx].dims[2];
        if (grid_h > 0 && grid_w > 0) {
            reserve_candidates += static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w);
        }
    }
    boxes.reserve(reserve_candidates * 4);
    obj_probs.reserve(reserve_candidates);
    class_ids.reserve(reserve_candidates);
    int valid_count = 0;

    static thread_local int32_t box_lut_zp[kBranchCount];
    static thread_local float box_lut_scale[kBranchCount];
    static thread_local uint8_t box_lut_init[kBranchCount];
    static thread_local float box_exp_lut[kBranchCount][256];

    for (int branch = 0; branch < kBranchCount; ++branch) {
        const int box_idx = branch * output_per_branch;
        const int score_idx = box_idx + 1;
        if (score_idx >= static_cast<int>(app_ctx->io_num.n_output)) {
            break;
        }

        const int grid_h = app_ctx->output_attrs[box_idx].dims[1];
        const int grid_w = app_ctx->output_attrs[box_idx].dims[2];
        if (grid_h <= 0 || grid_w <= 0) {
            continue;
        }
        const int stride = app_ctx->model_height / grid_h;

        const int32_t bz = app_ctx->output_attrs[box_idx].zp;
        const float bs = app_ctx->output_attrs[box_idx].scale;
        float* exp_lut256 = box_exp_lut[branch];
        if (!box_lut_init[branch] || box_lut_zp[branch] != bz || box_lut_scale[branch] != bs) {
            ::visiong::npu::yolo::neonopt::build_exp_lut_i8(exp_lut256, bz, bs);
            box_lut_zp[branch] = bz;
            box_lut_scale[branch] = bs;
            box_lut_init[branch] = 1;
        }

        int8_t* score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0f;
        if (output_per_branch >= 3) {
            const int score_sum_idx = box_idx + 2;
            score_sum = static_cast<int8_t*>(outputs[score_sum_idx]->virt_addr);
            score_sum_zp = app_ctx->output_attrs[score_sum_idx].zp;
            score_sum_scale = app_ctx->output_attrs[score_sum_idx].scale;
        }

        valid_count += process_i8_rv1106(
            static_cast<int8_t*>(outputs[box_idx]->virt_addr), bz, bs, static_cast<int8_t*>(outputs[score_idx]->virt_addr),
            app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale, score_sum, score_sum_zp,
            score_sum_scale, grid_h, grid_w, stride, dfl_len, ctx->class_count, exp_lut256, &boxes, &obj_probs, &class_ids,
            ctx->box_threshold);
    }

    if (valid_count <= 0) {
        return 0;
    }
    int keep_indices[OBJ_NUMB_MAX_SIZE];
    const int keep_count = ::visiong::npu::yolo::neonopt::nms_topk_classwise_xywh_neon(
        boxes, obj_probs, class_ids, ctx->class_count, ctx->nms_threshold, (int)kMaxNmsCandidates,
        keep_indices, OBJ_NUMB_MAX_SIZE);

    int out_count = 0;
    for (int ki = 0; ki < keep_count; ++ki) {
        const int idx = keep_indices[ki];
        if (idx < 0) {
            continue;
        }
        if (out_count >= OBJ_NUMB_MAX_SIZE) {
            break;
        }

        const float x1 = (boxes[(size_t)idx * 4U + 0U] - letterbox_pad_x) / letterbox_scale;
        const float y1 = (boxes[(size_t)idx * 4U + 1U] - letterbox_pad_y) / letterbox_scale;
        const float x2 = (boxes[(size_t)idx * 4U + 0U] + boxes[(size_t)idx * 4U + 2U] - letterbox_pad_x) / letterbox_scale;
        const float y2 = (boxes[(size_t)idx * 4U + 1U] + boxes[(size_t)idx * 4U + 3U] - letterbox_pad_y) / letterbox_scale;

        od_results->results[out_count].box.left = static_cast<int>(x1);
        od_results->results[out_count].box.top = static_cast<int>(y1);
        od_results->results[out_count].box.right = static_cast<int>(x2);
        od_results->results[out_count].box.bottom = static_cast<int>(y2);
        od_results->results[out_count].prop = obj_probs[(size_t)idx];
        od_results->results[out_count].cls_id = class_ids[(size_t)idx];
        ++out_count;
    }

    od_results->count = out_count;
    return 0;
}

}  // namespace

int init_yolo11_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_yolo11_model(rknn_app_context_t* app_ctx) { return visiong::npu::rknn::release_zero_copy_model(app_ctx); }

int inference_yolo11_model(rknn_app_context_t* app_ctx, const Yolo11PostProcessCtx* ctx, float letterbox_scale,
                           int letterbox_pad_x, int letterbox_pad_y, object_detect_result_list* od_results) {
    if (app_ctx == nullptr || od_results == nullptr || ctx == nullptr) {
        return -1;
    }

    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "YOLOv11") != 0) {
        return -1;
    }

    return post_process_yolo11(app_ctx, app_ctx->output_mems, ctx, letterbox_scale, letterbox_pad_x, letterbox_pad_y,
                               od_results);
}

int get_yolo11_model_num_classes(rknn_app_context_t* app_ctx) {
    if (!app_ctx || !app_ctx->output_attrs || app_ctx->io_num.n_output < 2) {
        return -1;
    }
    const rknn_tensor_attr& score_attr = app_ctx->output_attrs[1];
    if (score_attr.n_dims >= 4) {
        return static_cast<int>(score_attr.dims[3]);
    }
    if (score_attr.n_dims >= 2) {
        return static_cast<int>(score_attr.dims[1]);
    }
    return -1;
}

Yolo11PostProcessCtx* create_yolo11_post_process_ctx(const char* label_path, float box_thresh, float nms_thresh,
                                                     int required_num_classes) {
    std::unique_ptr<Yolo11PostProcessCtx> ctx(new Yolo11PostProcessCtx());
    ctx->class_count = kDefaultClassCount;
    ctx->box_threshold = box_thresh;
    ctx->nms_threshold = nms_thresh;

    if (!initialize_labels(label_path, required_num_classes, ctx.get())) {
        return nullptr;
    }

    if (ctx->class_count <= 0) {
        ctx->class_count = (required_num_classes > 0) ? required_num_classes : kDefaultClassCount;
    }
    if (ctx->box_threshold <= 0.0f) {
        ctx->box_threshold = kDefaultBoxThreshold;
    }
    if (ctx->nms_threshold <= 0.0f) {
        ctx->nms_threshold = kDefaultNmsThreshold;
    }

    printf("YOLOv11 post-process initialized: %d classes loaded.\n", ctx->class_count);
    return ctx.release();
}

void destroy_yolo11_post_process_ctx(Yolo11PostProcessCtx* ctx) { delete ctx; }

const char* coco_cls_to_name_yolo11(const Yolo11PostProcessCtx* ctx, int cls_id) {
    if (!ctx || !ctx->labels || cls_id < 0 || cls_id >= ctx->class_count) {
        return "unknown";
    }
    return ctx->labels[cls_id];
}

