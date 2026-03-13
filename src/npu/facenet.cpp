// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/facenet.h"

#include "internal/rknn_model_utils.h"
#include "internal/yolo_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr int kFaceFeatureDims = 128;
}

int init_facenet_model(const char* model_path, rknn_app_context_t* app_ctx) {
    const int ret = visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
    if (ret == 0) {
        printf("FaceNet model input height=%d, width=%d, channel=%d\n", app_ctx->model_height, app_ctx->model_width,
               app_ctx->model_channel);
    }
    return ret;
}

int release_facenet_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

void output_normalization(rknn_app_context_t* app_ctx, uint8_t* output, std::vector<float>& out_fp32) {
    out_fp32.assign(kFaceFeatureDims, 0.0f);

    if (app_ctx == nullptr || app_ctx->output_attrs == nullptr || output == nullptr || app_ctx->io_num.n_output < 1) {
        return;
    }

    const int feature_dims =
        std::min(kFaceFeatureDims, static_cast<int>(app_ctx->output_attrs[0].n_elems > 0 ? app_ctx->output_attrs[0].n_elems
                                                                                          : kFaceFeatureDims));
    const int32_t zp = app_ctx->output_attrs[0].zp;
    const float scale = app_ctx->output_attrs[0].scale;
    const int8_t* output_i8 = reinterpret_cast<const int8_t*>(output);

    float square_sum = 0.0f;
    for (int i = 0; i < feature_dims; ++i) {
        const float value = visiong::npu::yolo::dequantize_from_i8(output_i8[i], zp, scale);
        out_fp32[i] = value;
        square_sum += value * value;
    }

    const float norm = std::sqrt(square_sum);
    if (norm > 1e-6f) {
        for (int i = 0; i < feature_dims; ++i) {
            out_fp32[i] /= norm;
        }
    }
}

