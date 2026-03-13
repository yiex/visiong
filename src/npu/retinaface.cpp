// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved. / Copyright (c) 2023 由 Rockchip Electronics Co., Ltd. 全部 Rights Reserved.
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

#include "internal/models/retinaface.h"

#include "common/internal/dma_alloc.h"
#include "internal/rknn_box_priors.h"
#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

namespace {

constexpr float kFaceScoreThreshold = 0.5f;
constexpr float kNmsThreshold = 0.2f;
constexpr float kVariance0 = 0.1f;
constexpr float kVariance1 = 0.2f;

float calculate_iou_xyxy(float x0_min, float y0_min, float x0_max, float y0_max, float x1_min, float y1_min,
                         float x1_max, float y1_max) {
    const float inter_w = std::max(0.0f, std::min(x0_max, x1_max) - std::max(x0_min, x1_min));
    const float inter_h = std::max(0.0f, std::min(y0_max, y1_max) - std::max(y0_min, y1_min));
    const float inter = inter_w * inter_h;
    const float area0 = std::max(0.0f, x0_max - x0_min) * std::max(0.0f, y0_max - y0_min);
    const float area1 = std::max(0.0f, x1_max - x1_min) * std::max(0.0f, y1_max - y1_min);
    const float union_area = area0 + area1 - inter;
    return (union_area <= 0.0f) ? 0.0f : (inter / union_area);
}

void nms_retinaface(const std::vector<float>& decoded_boxes, const std::vector<int>& valid_prior_indices,
                    std::vector<int>* order, float threshold) {
    for (size_t i = 0; i < order->size(); ++i) {
        const int det_n = (*order)[i];
        if (det_n < 0) {
            continue;
        }

        const int prior_n = valid_prior_indices[det_n];
        const float n_xmin = decoded_boxes[prior_n * 4 + 0];
        const float n_ymin = decoded_boxes[prior_n * 4 + 1];
        const float n_xmax = decoded_boxes[prior_n * 4 + 2];
        const float n_ymax = decoded_boxes[prior_n * 4 + 3];

        for (size_t j = i + 1; j < order->size(); ++j) {
            const int det_m = (*order)[j];
            if (det_m < 0) {
                continue;
            }

            const int prior_m = valid_prior_indices[det_m];
            const float m_xmin = decoded_boxes[prior_m * 4 + 0];
            const float m_ymin = decoded_boxes[prior_m * 4 + 1];
            const float m_xmax = decoded_boxes[prior_m * 4 + 2];
            const float m_ymax = decoded_boxes[prior_m * 4 + 3];

            const float iou = calculate_iou_xyxy(n_xmin, n_ymin, n_xmax, n_ymax, m_xmin, m_ymin, m_xmax, m_ymax);
            if (iou > threshold) {
                (*order)[j] = -1;
            }
        }
    }
}

const float (*select_priors(int model_width, int* num_priors))[4] {
    *num_priors = 0;
    if (model_width == 640) {
        *num_priors = 16800;
        return BOX_PRIORS_640;
    }
    if (model_width == 320) {
        *num_priors = 4200;
        return BOX_PRIORS_320;
    }
    return nullptr;
}

} // namespace

int init_retinaface_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_retinaface_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

int inference_retinaface_model(rknn_app_context_t* app_ctx, object_detect_result_list* od_results) {
    if (app_ctx == nullptr || od_results == nullptr || app_ctx->output_attrs == nullptr || app_ctx->io_num.n_output < 3) {
        return -1;
    }

    std::memset(od_results, 0, sizeof(object_detect_result_list));

    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "RetinaFace") != 0) {
        return -1;
    }

    const int model_width = app_ctx->model_width;
    const int model_height = app_ctx->model_height;

    int num_priors = 0;
    const float (*priors)[4] = select_priors(model_width, &num_priors);
    if (priors == nullptr || num_priors <= 0) {
        printf("Error: Unsupported RetinaFace model input width: %d\n", model_width);
        return -1;
    }

    const int8_t* location = reinterpret_cast<const int8_t*>(app_ctx->output_mems[0]->virt_addr);
    const int8_t* scores = reinterpret_cast<const int8_t*>(app_ctx->output_mems[1]->virt_addr);
    const int8_t* landms = reinterpret_cast<const int8_t*>(app_ctx->output_mems[2]->virt_addr);

    const int32_t loc_zp = app_ctx->output_attrs[0].zp;
    const float loc_scale = app_ctx->output_attrs[0].scale;
    const int32_t scores_zp = app_ctx->output_attrs[1].zp;
    const float scores_scale = app_ctx->output_attrs[1].scale;
    const int32_t landms_zp = app_ctx->output_attrs[2].zp;
    const float landms_scale = app_ctx->output_attrs[2].scale;

    std::vector<int> valid_prior_indices;
    std::vector<float> valid_scores;
    valid_prior_indices.reserve(num_priors);
    valid_scores.reserve(num_priors);

    std::vector<float> decoded_boxes(static_cast<size_t>(num_priors) * 4, 0.0f);
    std::vector<float> decoded_landmarks(static_cast<size_t>(num_priors) * 10, 0.0f);

    for (int i = 0; i < num_priors; ++i) {
        const float score = visiong::npu::yolo::dequantize_from_i8(scores[i * 2 + 1], scores_zp, scores_scale);
        if (score <= kFaceScoreThreshold) {
            continue;
        }

        valid_prior_indices.push_back(i);
        valid_scores.push_back(score);

        const int box_offset = i * 4;
        const float center_x =
            visiong::npu::yolo::dequantize_from_i8(location[box_offset + 0], loc_zp, loc_scale) * kVariance0 * priors[i][2] +
            priors[i][0];
        const float center_y =
            visiong::npu::yolo::dequantize_from_i8(location[box_offset + 1], loc_zp, loc_scale) * kVariance0 * priors[i][3] +
            priors[i][1];
        const float box_w =
            std::exp(visiong::npu::yolo::dequantize_from_i8(location[box_offset + 2], loc_zp, loc_scale) * kVariance1) *
            priors[i][2];
        const float box_h =
            std::exp(visiong::npu::yolo::dequantize_from_i8(location[box_offset + 3], loc_zp, loc_scale) * kVariance1) *
            priors[i][3];

        decoded_boxes[box_offset + 0] = center_x - box_w * 0.5f;
        decoded_boxes[box_offset + 1] = center_y - box_h * 0.5f;
        decoded_boxes[box_offset + 2] = center_x + box_w * 0.5f;
        decoded_boxes[box_offset + 3] = center_y + box_h * 0.5f;

        for (int k = 0; k < 5; ++k) {
            decoded_landmarks[i * 10 + 2 * k] =
                visiong::npu::yolo::dequantize_from_i8(landms[i * 10 + 2 * k], landms_zp, landms_scale) * kVariance0 *
                    priors[i][2] +
                priors[i][0];
            decoded_landmarks[i * 10 + 2 * k + 1] =
                visiong::npu::yolo::dequantize_from_i8(landms[i * 10 + 2 * k + 1], landms_zp, landms_scale) * kVariance0 *
                    priors[i][3] +
                priors[i][1];
        }
    }

    if (valid_prior_indices.empty()) {
        return 0;
    }

    std::vector<int> order;
    visiong::npu::yolo::sort_indices_desc(valid_scores, &order);
    nms_retinaface(decoded_boxes, valid_prior_indices, &order, kNmsThreshold);

    int face_count = 0;
    for (const int det_idx : order) {
        if (det_idx < 0) {
            continue;
        }
        if (face_count >= OBJ_NUMB_MAX_SIZE) {
            printf("Warning: detected more than %d faces, truncating.\n", OBJ_NUMB_MAX_SIZE);
            break;
        }

        const int prior_idx = valid_prior_indices[det_idx];
        const float x1 = decoded_boxes[prior_idx * 4 + 0] * model_width;
        const float y1 = decoded_boxes[prior_idx * 4 + 1] * model_height;
        const float x2 = decoded_boxes[prior_idx * 4 + 2] * model_width;
        const float y2 = decoded_boxes[prior_idx * 4 + 3] * model_height;

        od_results->results[face_count].box.left = visiong::npu::yolo::clamp_to_int(x1, 0, model_width - 1);
        od_results->results[face_count].box.top = visiong::npu::yolo::clamp_to_int(y1, 0, model_height - 1);
        od_results->results[face_count].box.right = visiong::npu::yolo::clamp_to_int(x2, 0, model_width - 1);
        od_results->results[face_count].box.bottom = visiong::npu::yolo::clamp_to_int(y2, 0, model_height - 1);
        od_results->results[face_count].prop = valid_scores[det_idx];
        od_results->results[face_count].cls_id = 0;

        for (int k = 0; k < 5; ++k) {
            const float point_x = decoded_landmarks[prior_idx * 10 + 2 * k] * model_width;
            const float point_y = decoded_landmarks[prior_idx * 10 + 2 * k + 1] * model_height;
            od_results->results[face_count].point[k].x = visiong::npu::yolo::clamp_to_int(point_x, 0, model_width - 1);
            od_results->results[face_count].point[k].y = visiong::npu::yolo::clamp_to_int(point_y, 0, model_height - 1);
        }

        ++face_count;
    }

    od_results->count = face_count;
    return 0;
}

