// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/lprnet.h"

#include "common/internal/dma_alloc.h"
#include "internal/rknn_model_utils.h"

#include <cstdio>
#include <vector>

namespace {

const std::vector<std::string> kPlateCode = {
    u8"\u4EAC", u8"\u6CAA", u8"\u6D25", u8"\u6E1D", u8"\u5180", u8"\u664B", u8"\u8499", u8"\u8FBD",
    u8"\u5409", u8"\u9ED1", u8"\u82CF", u8"\u6D59", u8"\u7696", u8"\u95FD", u8"\u8D63", u8"\u9C81",
    u8"\u8C6B", u8"\u9102", u8"\u6E58", u8"\u7CA4", u8"\u6842", u8"\u743C", u8"\u5DDD", u8"\u8D35",
    u8"\u4E91", u8"\u85CF", u8"\u9655", u8"\u7518", u8"\u9752", u8"\u5B81", u8"\u65B0", "0",
    "1",        "2",        "3",        "4",        "5",        "6",        "7",        "8",
    "9",        "A",        "B",        "C",        "D",        "E",        "F",        "G",
    "H",        "J",        "K",        "L",        "M",        "N",        "P",        "Q",
    "R",        "S",        "T",        "U",        "V",        "W",        "X",        "Y",
    "Z",        "I",        "O",        "-",
};

}  // namespace

int init_lprnet_model(const char* model_path, rknn_app_context_t* app_ctx) {
    const int ret = visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
    if (ret == 0) {
        printf("LPRNet model loaded: H=%d, W=%d, C=%d\n", app_ctx->model_height, app_ctx->model_width,
               app_ctx->model_channel);
    }
    return ret;
}

int release_lprnet_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

int inference_lprnet_model(rknn_app_context_t* app_ctx, lprnet_result* out_result) {
    if (app_ctx == nullptr || out_result == nullptr) {
        return -1;
    }

    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "LPRNet") != 0) {
        return -1;
    }

    const int8_t* logits = reinterpret_cast<const int8_t*>(app_ctx->output_mems[0]->virt_addr);
    std::vector<int> best_path;
    best_path.reserve(LPRNET_OUT_COLS);

    for (int col = 0; col < LPRNET_OUT_COLS; ++col) {
        int best_idx = 0;
        int8_t best_val = logits[col];
        for (int row = 1; row < LPRNET_OUT_ROWS; ++row) {
            const int8_t value = logits[row * LPRNET_OUT_COLS + col];
            if (value > best_val) {
                best_val = value;
                best_idx = row;
            }
        }
        best_path.push_back(best_idx);
    }

    constexpr int blank_idx = LPRNET_OUT_ROWS - 1;
    out_result->plate_name.clear();

    int prev = blank_idx;
    for (const int idx : best_path) {
        if (idx != prev && idx != blank_idx && idx >= 0 && static_cast<size_t>(idx) < kPlateCode.size()) {
            out_result->plate_name += kPlateCode[idx];
        }
        prev = idx;
    }

    return 0;
}

