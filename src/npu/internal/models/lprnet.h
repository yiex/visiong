// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_MODELS_LPRNET_H
#define VISIONG_NPU_INTERNAL_MODELS_LPRNET_H

#include "npu/internal/npu_common.h"

#include <string>

constexpr int LPRNET_MODEL_WIDTH = 94;
constexpr int LPRNET_MODEL_HEIGHT = 24;
constexpr int LPRNET_OUT_ROWS = 68;
constexpr int LPRNET_OUT_COLS = 18;

struct lprnet_result {
    std::string plate_name;
};

int init_lprnet_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_lprnet_model(rknn_app_context_t* app_ctx);
int inference_lprnet_model(rknn_app_context_t* app_ctx, lprnet_result* out_result);

#endif  // VISIONG_NPU_INTERNAL_MODELS_LPRNET_H
