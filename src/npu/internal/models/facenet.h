// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_MODELS_FACENET_H
#define VISIONG_NPU_INTERNAL_MODELS_FACENET_H

#include "npu/internal/npu_common.h"

#include <vector>

int init_facenet_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_facenet_model(rknn_app_context_t* app_ctx);
void output_normalization(rknn_app_context_t* app_ctx, uint8_t* output, std::vector<float>& out_fp32);

#endif  // VISIONG_NPU_INTERNAL_MODELS_FACENET_H
