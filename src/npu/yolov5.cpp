// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (c) 2021-2023 by Rockchip Electronics Co., Ltd. All Rights Reserved. / Copyright (c) 2021-2023 由 Rockchip Electronics Co., Ltd. 全部 Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); / 详见英文原注释。
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at / You 可能 obtain 复制 的 License at
//
//     http://www.apache.org/licenses/LICENSE-2.0 / 详见英文原注释。
//
// Unless required by applicable law or agreed to in writing, software / Unless required 由 applicable law 或 agreed 以 在 writing, 软件
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. / WITHOUT WARRANTIES 或 CONDITIONS 的 ANY KIND, either express 或 implied.
// See the License for the specific language governing permissions and
// limitations under the License. / 详见英文原注释。

#include "internal/models/yolov5.h"

#include "common/internal/dma_alloc.h"
#include "internal/rknn_model_utils.h"
#include "internal/yolo_neon_opt.h"
#include "internal/yolo_common.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

struct YoloV5PostProcessCtx {
    int class_num = 80;
    int prop_box_size = 85;
    float nms_threshold = 0.45f;
    float box_threshold = 0.25f;
    char** labels = nullptr;

    ~YoloV5PostProcessCtx() { visiong::npu::yolo::free_c_labels(&labels, class_num); }
};

namespace {

constexpr int kDefaultClassCount = 80;
constexpr int kDefaultPropBoxSize = 85;
constexpr float kDefaultNmsThreshold = 0.45f;
constexpr float kDefaultBoxThreshold = 0.25f;
constexpr int kBranchCount = 3;
constexpr size_t kMaxNmsCandidates = 512;

const int kAnchors[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326},
};

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

int process_i8_rv1106(int8_t* input, const int* anchor, int grid_h, int grid_w, int stride, int class_num,
                      int prop_box_size, std::vector<float>* boxes, std::vector<float>* box_scores,
                      std::vector<int>* class_ids, float threshold, int32_t zp, float scale) {
    int valid_count = 0;
    const int8_t threshold_i8 = visiong::npu::yolo::quantize_to_i8(threshold, zp, scale);

    constexpr int anchor_per_branch = 3;
    const int align_c = prop_box_size * anchor_per_branch;

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            for (int a = 0; a < anchor_per_branch; ++a) {
                const int hw_offset = h * grid_w * align_c + w * align_c + a * prop_box_size;
                int8_t* hw_ptr = input + hw_offset;
                const int8_t box_confidence = hw_ptr[4];
                if (box_confidence < threshold_i8) {
                    continue;
                }
                int max_class_id = 0;
                const int8_t max_class_score = ::visiong::npu::yolo::neonopt::argmax_i8(hw_ptr + 5, class_num, &max_class_id);

                const float box_conf_f32 = visiong::npu::yolo::dequantize_from_i8(box_confidence, zp, scale);
                const float class_prob_f32 = visiong::npu::yolo::dequantize_from_i8(max_class_score, zp, scale);
                const float score = box_conf_f32 * class_prob_f32;
                if (score <= threshold) {
                    continue;
                }

                float box_x = visiong::npu::yolo::dequantize_from_i8(hw_ptr[0], zp, scale) * 2.0f - 0.5f;
                float box_y = visiong::npu::yolo::dequantize_from_i8(hw_ptr[1], zp, scale) * 2.0f - 0.5f;
                float box_w = visiong::npu::yolo::dequantize_from_i8(hw_ptr[2], zp, scale) * 2.0f;
                float box_h = visiong::npu::yolo::dequantize_from_i8(hw_ptr[3], zp, scale) * 2.0f;

                box_w = box_w * box_w * static_cast<float>(anchor[a * 2]);
                box_h = box_h * box_h * static_cast<float>(anchor[a * 2 + 1]);
                box_x = (box_x + static_cast<float>(w)) * static_cast<float>(stride) - box_w / 2.0f;
                box_y = (box_y + static_cast<float>(h)) * static_cast<float>(stride) - box_h / 2.0f;                {
                    const size_t base = boxes->size();
                    boxes->resize(base + 4U);
                    float* __restrict bdst = boxes->data() + base;
                    bdst[0] = box_x;
                    bdst[1] = box_y;
                    bdst[2] = box_w;
                    bdst[3] = box_h;
                }
                box_scores->push_back(score);
                class_ids->push_back(max_class_id);
                ++valid_count;
            }
        }
    }
    return valid_count;
}

bool initialize_labels(const char* label_path, int required_num_classes, YoloV5PostProcessCtx* ctx) {
    if (ctx == nullptr) {
        return false;
    }
    if (required_num_classes > 0) {
        ctx->class_num = required_num_classes;
        ctx->prop_box_size = ctx->class_num + 5;
    }

    if (label_path == nullptr || label_path[0] == '\0') {
        return true;
    }

    std::vector<std::string> loaded_labels;
    bool has_empty_line_after_data = false;
    if (visiong::npu::yolo::load_non_empty_lines(label_path, &loaded_labels, &has_empty_line_after_data) < 0) {
        printf("Open %s fail!\n", label_path);
        return false;
    }

    if (required_num_classes >= 0 && static_cast<int>(loaded_labels.size()) != required_num_classes) {
        printf("ERROR: label count mismatch: model requires %d classes, label file has %d.\n", required_num_classes,
               static_cast<int>(loaded_labels.size()));
        return false;
    }
    if (loaded_labels.empty()) {
        printf("ERROR: no labels loaded from %s\n", label_path);
        return false;
    }

    if (visiong::npu::yolo::assign_c_labels(loaded_labels, &ctx->labels) < 0) {
        printf("Malloc labels failed!\n");
        return false;
    }

    ctx->class_num = static_cast<int>(loaded_labels.size());
    ctx->prop_box_size = ctx->class_num + 5;

    if (has_empty_line_after_data) {
        printf("Warning: label file contains empty line(s) in the middle or end; those lines were skipped.\n");
    }
    printf("Loaded %d labels\n", ctx->class_num);
    return true;
}

}  // namespace

int get_yolov5_model_num_classes(rknn_app_context_t* app_ctx) {
    if (!app_ctx || !app_ctx->output_attrs || app_ctx->io_num.n_output < 1) {
        return -1;
    }

    uint32_t channels = 0;
    const rknn_tensor_attr& first_output = app_ctx->output_attrs[0];
    if (first_output.n_dims >= 4) {
        channels = first_output.dims[3];
    } else if (first_output.n_dims >= 2) {
        channels = first_output.dims[1];
    } else {
        return -1;
    }

    if (channels < 15 || (channels % 3) != 0) {
        return -1;
    }
    return static_cast<int>(channels / 3) - 5;
}

YoloV5PostProcessCtx* create_yolov5_post_process_ctx(const char* label_path, float box_thresh, float nms_thresh,
                                                     int required_num_classes) {
    std::unique_ptr<YoloV5PostProcessCtx> ctx(new YoloV5PostProcessCtx());
    ctx->class_num = kDefaultClassCount;
    ctx->prop_box_size = kDefaultPropBoxSize;
    ctx->box_threshold = box_thresh;
    ctx->nms_threshold = nms_thresh;

    if (!initialize_labels(label_path, required_num_classes, ctx.get())) {
        return nullptr;
    }

    if (ctx->class_num <= 0) {
        ctx->class_num = kDefaultClassCount;
    }
    if (ctx->prop_box_size <= 0) {
        ctx->prop_box_size = ctx->class_num + 5;
    }

    if (ctx->box_threshold <= 0.0f) {
        ctx->box_threshold = kDefaultBoxThreshold;
    }
    if (ctx->nms_threshold <= 0.0f) {
        ctx->nms_threshold = kDefaultNmsThreshold;
    }

    printf("Init post process: OBJ_CLASS_NUM=%d, BOX_THRESH=%.2f, NMS_THRESH=%.2f\n", ctx->class_num,
           ctx->box_threshold, ctx->nms_threshold);
    return ctx.release();
}

void destroy_yolov5_post_process_ctx(YoloV5PostProcessCtx* ctx) { delete ctx; }

const char* coco_cls_to_name(const YoloV5PostProcessCtx* ctx, int cls_id) {
    if (!ctx || !ctx->labels || cls_id < 0 || cls_id >= ctx->class_num) {
        return "unknown";
    }
    return ctx->labels[cls_id];
}

int post_process(rknn_app_context_t* app_ctx, void* outputs, const YoloV5PostProcessCtx* ctx,
                 object_detect_result_list* od_results) {
    if (app_ctx == nullptr || outputs == nullptr || od_results == nullptr || ctx == nullptr) {
        return -1;
    }
    if (!app_ctx->is_quant) {
        printf("ERROR: YOLOv5 on RV1106/1103 requires quantized model outputs\n");
        return -1;
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));

    auto** output_mems = static_cast<rknn_tensor_mem**>(outputs);
    static thread_local std::vector<float> boxes;
    static thread_local std::vector<float> obj_probs;
    static thread_local std::vector<int> class_ids;
    boxes.clear();
    obj_probs.clear();
    class_ids.clear();
    size_t reserve_candidates = 0;
    for (int i = 0; i < kBranchCount; ++i) {
        const int grid_h = app_ctx->output_attrs[i].dims[2];
        const int grid_w = app_ctx->output_attrs[i].dims[1];
        if (grid_h > 0 && grid_w > 0) {
            reserve_candidates += static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w) * 3;
        }
    }
    boxes.reserve(reserve_candidates * 4);
    obj_probs.reserve(reserve_candidates);
    class_ids.reserve(reserve_candidates);
    int valid_count = 0;

    const int model_in_w = app_ctx->model_width;
    const int model_in_h = app_ctx->model_height;

    for (int i = 0; i < kBranchCount; ++i) {
        const int grid_h = app_ctx->output_attrs[i].dims[2];
        const int grid_w = app_ctx->output_attrs[i].dims[1];
        if (grid_h <= 0 || grid_w <= 0) {
            continue;
        }
        const int stride = model_in_h / grid_h;
        valid_count += process_i8_rv1106(static_cast<int8_t*>(output_mems[i]->virt_addr), kAnchors[i], grid_h, grid_w,
                                         stride, ctx->class_num, ctx->prop_box_size, &boxes, &obj_probs, &class_ids,
                                         ctx->box_threshold, app_ctx->output_attrs[i].zp,
                                         app_ctx->output_attrs[i].scale);
    }

    if (valid_count <= 0) {
        return 0;
    }
    int keep_indices[OBJ_NUMB_MAX_SIZE];
    const int keep_count = ::visiong::npu::yolo::neonopt::nms_topk_classwise_xywh_neon(
        boxes, obj_probs, class_ids, ctx->class_num, ctx->nms_threshold, (int)kMaxNmsCandidates,
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

        const float x1 = boxes[(size_t)idx * 4U + 0U];
        const float y1 = boxes[(size_t)idx * 4U + 1U];
        const float x2 = x1 + boxes[(size_t)idx * 4U + 2U];
        const float y2 = y1 + boxes[(size_t)idx * 4U + 3U];

        od_results->results[out_count].box.left = visiong::npu::yolo::clamp_to_int(x1, 0, model_in_w - 1);
        od_results->results[out_count].box.top = visiong::npu::yolo::clamp_to_int(y1, 0, model_in_h - 1);
        od_results->results[out_count].box.right = visiong::npu::yolo::clamp_to_int(x2, 0, model_in_w - 1);
        od_results->results[out_count].box.bottom = visiong::npu::yolo::clamp_to_int(y2, 0, model_in_h - 1);
        od_results->results[out_count].prop = obj_probs[(size_t)idx];
        od_results->results[out_count].cls_id = class_ids[(size_t)idx];
        ++out_count;
    }

    od_results->count = out_count;
    return 0;
}

int init_yolov5_model(const char* model_path, rknn_app_context_t* app_ctx) {
    const int ret = visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
    if (ret == 0) {
        printf("model input height=%d, width=%d, channel=%d\n", app_ctx->model_height, app_ctx->model_width,
               app_ctx->model_channel);
    }
    return ret;
}

int release_yolov5_model(rknn_app_context_t* app_ctx) {
    const int ret = visiong::npu::rknn::release_zero_copy_model(app_ctx);
    printf("Release success\n");
    return ret;
}

int inference_yolov5_model(rknn_app_context_t* app_ctx, const YoloV5PostProcessCtx* ctx,
                           object_detect_result_list* od_results) {
    if (app_ctx == nullptr || od_results == nullptr || ctx == nullptr) {
        return -1;
    }

    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "YOLOv5") != 0) {
        return -1;
    }

    return post_process(app_ctx, app_ctx->output_mems, ctx, od_results);
}

