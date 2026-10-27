// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Modified for VisionG: native tensor validation, INT8/NEON TopK and runtime integration.
// VisionG 修改：校验原生张量布局，使用 INT8/NEON TopK，并接入现有推理接口。
// Source / 来源：airockchip/rknn_model_zoo, examples/yolo26/cpp/postprocess.cc.
// Revision / 版本：837db8e992815ff8647eadb709f07067f4bc3a71. See / 参见 licenses/SOURCES.md.
#include "internal/models/yolo26.h"

#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"
#include "internal/yolo26_decode.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct TensorLayout {
    int count = 0;
    int channels = 0;
    size_t anchor_stride = 0;
    size_t channel_stride = 0;
};

// Validate the logical tensor against the native buffer size. / 校验逻辑张量是否位于原生缓冲区内。
bool resolve_tensor_layout(const rknn_tensor_attr& attr, TensorLayout* layout) {
    if (attr.type != RKNN_TENSOR_INT8 || attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
        !std::isfinite(attr.scale) || attr.scale <= 0.0f || attr.dims[0] != 1) {
        return false;
    }
    if (attr.n_dims == 3) {
        // Rank-3 UNDEFINED keeps the exported [1,C,N] order on RV1106.
        // RV1106 的三维 UNDEFINED 张量保留导出的 [1,C,N] 顺序。
        if (attr.fmt == RKNN_TENSOR_NCHW || attr.fmt == RKNN_TENSOR_UNDEFINED) {
            layout->channels = attr.dims[1];
            layout->count = attr.dims[2];
            layout->anchor_stride = 1;
            layout->channel_stride = layout->count;
        } else if (attr.fmt == RKNN_TENSOR_NHWC) {
            layout->count = attr.dims[1];
            layout->channels = attr.dims[2];
            layout->anchor_stride = layout->channels;
            layout->channel_stride = 1;
        } else {
            return false;
        }
    } else if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
        // Flattened outputs may have an extra singleton H. / 展平的输出可能带有大小为 1 的高度维度。
        if (attr.dims[1] != 1 || (attr.w_stride != 0 && attr.w_stride < attr.dims[2])) {
            return false;
        }
        layout->count = attr.dims[2];
        layout->channels = attr.dims[3];
        layout->anchor_stride = layout->channels;
        layout->channel_stride = 1;
    } else {
        return false;
    }
    if (layout->count <= 0 || layout->channels <= 0) {
        return false;
    }
    const size_t required = static_cast<size_t>(layout->count - 1) * layout->anchor_stride +
                            static_cast<size_t>(layout->channels - 1) * layout->channel_stride + 1;
    return required <= attr.size_with_stride && required == attr.n_elems;
}

} // namespace

struct Yolo26PostProcessCtx {
    int scores_index = -1;
    int boxes_index = -1;
    TensorLayout scores_layout;
    TensorLayout boxes_layout;
    float box_threshold = 0.25f;
    char** labels = nullptr;
    std::vector<visiong::npu::yolo26::Candidate> candidates;
    std::vector<int8_t> maxima;

    ~Yolo26PostProcessCtx() { visiong::npu::yolo::free_c_labels(&labels, scores_layout.channels); }
};

int init_yolo26_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_yolo26_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

Yolo26PostProcessCtx* create_yolo26_post_process_ctx(rknn_app_context_t* app_ctx, const char* label_path,
                                                     float box_thresh) {
    if (app_ctx == nullptr || app_ctx->output_attrs == nullptr || app_ctx->io_num.n_output != 2 ||
        !std::isfinite(box_thresh) || box_thresh < 0.0f || box_thresh > 1.0f) {
        return nullptr;
    }
    std::unique_ptr<Yolo26PostProcessCtx> ctx(new Yolo26PostProcessCtx);
    TensorLayout layouts[2];
    for (int i = 0; i < 2; ++i) {
        if (!resolve_tensor_layout(app_ctx->output_attrs[i], &layouts[i])) {
            std::printf("YOLO26: unsupported native output[%d] layout/type/quantization\n", i);
            return nullptr;
        }
    }

    // Four-class heads need node names to distinguish boxes from scores.
    // 四分类模型的两路输出形状相同，需结合节点名称区分坐标与分数。
    for (int i = 0; i < 2; ++i) {
        const char* name = app_ctx->output_attrs[i].name;
        if (layouts[i].channels == 4 && (layouts[1 - i].channels != 4 || std::strstr(name, "Mul") != nullptr ||
                                         std::strstr(name, "boxes") != nullptr)) {
            ctx->boxes_index = i;
            ctx->scores_index = 1 - i;
            break;
        }
    }
    if (ctx->boxes_index < 0 || layouts[0].count != layouts[1].count) {
        std::printf("YOLO26: expected matching score and decoded xyxy output tensors\n");
        return nullptr;
    }
    ctx->scores_layout = layouts[ctx->scores_index];
    ctx->boxes_layout = layouts[ctx->boxes_index];
    ctx->box_threshold = box_thresh;

    if (label_path != nullptr && label_path[0] != '\0') {
        std::vector<std::string> labels;
        bool had_empty_lines = false;
        if (visiong::npu::yolo::load_non_empty_lines(label_path, &labels, &had_empty_lines) < 0 ||
            static_cast<int>(labels.size()) != ctx->scores_layout.channels ||
            visiong::npu::yolo::assign_c_labels(labels, &ctx->labels) < 0) {
            std::printf("YOLO26: failed to load labels matching the model class count\n");
            return nullptr;
        }
    }
    ctx->candidates.reserve(ctx->scores_layout.count);
    ctx->maxima.reserve(ctx->scores_layout.count);
    return ctx.release();
}

void destroy_yolo26_post_process_ctx(Yolo26PostProcessCtx* ctx) {
    delete ctx;
}

const char* coco_cls_to_name_yolo26(const Yolo26PostProcessCtx* ctx, int cls_id) {
    if (ctx == nullptr || ctx->labels == nullptr || cls_id < 0 || cls_id >= ctx->scores_layout.channels) {
        return "unknown";
    }
    return ctx->labels[cls_id];
}

int post_process_yolo26(rknn_app_context_t* app_ctx, Yolo26PostProcessCtx* ctx, float letterbox_scale,
                        int letterbox_pad_x, int letterbox_pad_y, object_detect_result_list* od_results) {
    if (app_ctx == nullptr || ctx == nullptr || od_results == nullptr || !std::isfinite(letterbox_scale) ||
        letterbox_scale <= 0.0f) {
        return -1;
    }
    std::memset(od_results, 0, sizeof(*od_results));
    const auto& score_attr = app_ctx->output_attrs[ctx->scores_index];
    const auto& box_attr = app_ctx->output_attrs[ctx->boxes_index];
    const auto& score_layout = ctx->scores_layout;
    const auto& box_layout = ctx->boxes_layout;
    if (app_ctx->output_mems[ctx->scores_index] == nullptr || app_ctx->output_mems[ctx->boxes_index] == nullptr) {
        return -1;
    }
    const auto* score_data = static_cast<const int8_t*>(app_ctx->output_mems[ctx->scores_index]->virt_addr);
    const auto* box_data = static_cast<const int8_t*>(app_ctx->output_mems[ctx->boxes_index]->virt_addr);
    if (score_data == nullptr || box_data == nullptr) {
        return -1;
    }

    const visiong::npu::yolo26::Scores scores{score_data,
                                              score_layout.count,
                                              score_layout.channels,
                                              score_layout.anchor_stride,
                                              score_layout.channel_stride,
                                              score_attr.zp,
                                              score_attr.scale};
    visiong::npu::yolo26::select(scores, ctx->box_threshold, OBJ_NUMB_MAX_SIZE, ctx->candidates, ctx->maxima);

    // Decode only retained candidates; one2one predictions require no NMS.
    // 仅解码保留的候选框；one2one 输出无需 NMS。
    const float inv_scale = 1.0f / letterbox_scale;
    for (const auto& candidate : ctx->candidates) {
        auto& detection = od_results->results[od_results->count++];
        float xyxy[4];
        for (int c = 0; c < 4; ++c) {
            const size_t offset = static_cast<size_t>(candidate.index) * box_layout.anchor_stride +
                                  static_cast<size_t>(c) * box_layout.channel_stride;
            xyxy[c] = (static_cast<int>(box_data[offset]) - box_attr.zp) * box_attr.scale;
        }
        detection.box.left = static_cast<int>((xyxy[0] - letterbox_pad_x) * inv_scale);
        detection.box.top = static_cast<int>((xyxy[1] - letterbox_pad_y) * inv_scale);
        detection.box.right = static_cast<int>((xyxy[2] - letterbox_pad_x) * inv_scale);
        detection.box.bottom = static_cast<int>((xyxy[3] - letterbox_pad_y) * inv_scale);
        detection.prop = (static_cast<int>(candidate.score) - score_attr.zp) * score_attr.scale;
        detection.cls_id = visiong::npu::yolo26::class_id(scores, candidate);
    }
    return 0;
}

int inference_yolo26_model(rknn_app_context_t* app_ctx, Yolo26PostProcessCtx* ctx, float letterbox_scale,
                           int letterbox_pad_x, int letterbox_pad_y, object_detect_result_list* od_results) {
    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "YOLO26") != 0) {
        return -1;
    }
    return post_process_yolo26(app_ctx, ctx, letterbox_scale, letterbox_pad_x, letterbox_pad_y, od_results);
}
