// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "npu/internal/npu_common.h"
#include "visiong/npu/NPU.h"

#include <vector>

int init_mlsd_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_mlsd_model(rknn_app_context_t* app_ctx);
int inference_mlsd_model(rknn_app_context_t* app_ctx,
                         int image_width,
                         int image_height,
                         float score_threshold,
                         float distance_threshold,
                         std::vector<LineSegment>* lines);
