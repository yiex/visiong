// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "common/internal/dma_alloc.h"
#include "npu/internal/npu_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

namespace visiong::npu::rknn {

inline void dump_tensor_attr(const rknn_tensor_attr* attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

inline int init_zero_copy_model(const char* model_path, rknn_app_context_t* app_ctx) {
    if (model_path == nullptr || app_ctx == nullptr) {
        return -1;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, const_cast<char*>(model_path), 0, 0, nullptr);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query IN_OUT_NUM fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    const size_t input_slots = sizeof(app_ctx->input_mems) / sizeof(app_ctx->input_mems[0]);
    const size_t output_slots = sizeof(app_ctx->output_mems) / sizeof(app_ctx->output_mems[0]);
    if (io_num.n_input < 1 || io_num.n_input > input_slots) {
        printf("ERROR: unsupported input tensor count %u (capacity=%zu)\n", io_num.n_input, input_slots);
        rknn_destroy(ctx);
        return -1;
    }
    if (io_num.n_output < 1 || io_num.n_output > output_slots) {
        printf("ERROR: unsupported output tensor count %u (capacity=%zu)\n", io_num.n_output, output_slots);
        rknn_destroy(ctx);
        return -1;
    }

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query INPUT_ATTR fail! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query OUTPUT_ATTR fail! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;

    rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (input_mem == nullptr) {
        printf("rknn_create_mem for input failed\n");
        rknn_destroy(ctx);
        return -1;
    }
    ret = rknn_set_io_mem(ctx, input_mem, &input_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy(ctx);
        return -1;
    }

    std::vector<rknn_tensor_mem*> output_mems(io_num.n_output, nullptr);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (output_mems[i] == nullptr) {
            printf("rknn_create_mem for output[%u] failed\n", i);
            for (uint32_t j = 0; j < i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mem);
            rknn_destroy(ctx);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            for (uint32_t j = 0; j <= i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mem);
            rknn_destroy(ctx);
            return -1;
        }
    }

    rknn_tensor_attr* input_attrs_heap =
        static_cast<rknn_tensor_attr*>(std::malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    rknn_tensor_attr* output_attrs_heap =
        static_cast<rknn_tensor_attr*>(std::malloc(io_num.n_output * sizeof(rknn_tensor_attr)));
    if (input_attrs_heap == nullptr || output_attrs_heap == nullptr) {
        printf("malloc for tensor attrs failed\n");
        std::free(input_attrs_heap);
        std::free(output_attrs_heap);
        for (rknn_tensor_mem* output_mem : output_mems) {
            rknn_destroy_mem(ctx, output_mem);
        }
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy(ctx);
        return -1;
    }

    std::memcpy(input_attrs_heap, input_attrs.data(), io_num.n_input * sizeof(rknn_tensor_attr));
    std::memcpy(output_attrs_heap, output_attrs.data(), io_num.n_output * sizeof(rknn_tensor_attr));

    for (rknn_tensor_mem*& mem : app_ctx->input_mems) {
        mem = nullptr;
    }
    for (rknn_tensor_mem*& mem : app_ctx->output_mems) {
        mem = nullptr;
    }

    app_ctx->input_mems[0] = input_mem;
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = output_mems[i];
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = input_attrs_heap;
    app_ctx->output_attrs = output_attrs_heap;
    app_ctx->is_quant = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    app_ctx->net_mem = nullptr;
    app_ctx->max_mem = nullptr;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }

    return 0;
}

inline int release_zero_copy_model(rknn_app_context_t* app_ctx) {
    if (app_ctx == nullptr) {
        return -1;
    }

    rknn_context ctx = app_ctx->rknn_ctx;

    if (ctx != 0 && app_ctx->net_mem != nullptr) {
        rknn_destroy_mem(ctx, app_ctx->net_mem);
    }
    app_ctx->net_mem = nullptr;

    if (ctx != 0 && app_ctx->max_mem != nullptr) {
        rknn_destroy_mem(ctx, app_ctx->max_mem);
    }
    app_ctx->max_mem = nullptr;

    if (ctx != 0) {
        for (rknn_tensor_mem*& input_mem : app_ctx->input_mems) {
            if (input_mem != nullptr) {
                rknn_destroy_mem(ctx, input_mem);
                input_mem = nullptr;
            }
        }
        for (rknn_tensor_mem*& output_mem : app_ctx->output_mems) {
            if (output_mem != nullptr) {
                rknn_destroy_mem(ctx, output_mem);
                output_mem = nullptr;
            }
        }
    } else {
        for (rknn_tensor_mem*& input_mem : app_ctx->input_mems) {
            input_mem = nullptr;
        }
        for (rknn_tensor_mem*& output_mem : app_ctx->output_mems) {
            output_mem = nullptr;
        }
    }

    std::free(app_ctx->input_attrs);
    app_ctx->input_attrs = nullptr;
    std::free(app_ctx->output_attrs);
    app_ctx->output_attrs = nullptr;

    if (ctx != 0) {
        rknn_destroy(ctx);
        app_ctx->rknn_ctx = 0;
    }

    app_ctx->io_num.n_input = 0;
    app_ctx->io_num.n_output = 0;
    app_ctx->model_width = 0;
    app_ctx->model_height = 0;
    app_ctx->model_channel = 0;
    app_ctx->is_quant = false;
    return 0;
}

inline int run_model(rknn_app_context_t* app_ctx, const char* tag = "RKNN") {
    const char* log_tag = (tag != nullptr) ? tag : "RKNN";
    if (app_ctx == nullptr || app_ctx->rknn_ctx == 0) {
        printf("%s: invalid app context\n", log_tag);
        return -1;
    }

    constexpr int kMaxRunAttempts = 2;
    for (int attempt = 1; attempt <= kMaxRunAttempts; ++attempt) {
        const int ret = rknn_run(app_ctx->rknn_ctx, nullptr);
        if (ret >= 0) {
            return 0;
        }
        printf("%s: rknn_run failed, ret=%d (attempt %d/%d)\n", log_tag, ret, attempt, kMaxRunAttempts);
        if (attempt < kMaxRunAttempts) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    return -1;
}

inline int sync_output_tensors_to_cpu(rknn_app_context_t* app_ctx, const char* tag = "RKNN") {
    const char* log_tag = (tag != nullptr) ? tag : "RKNN";
    if (app_ctx == nullptr || app_ctx->io_num.n_output < 1) {
        printf("%s: invalid output tensor metadata\n", log_tag);
        return -1;
    }

    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        if (app_ctx->output_mems[i] == nullptr) {
            printf("%s: output_mems[%u] is null\n", log_tag, i);
            return -1;
        }
        dma_sync_device_to_cpu(app_ctx->output_mems[i]->fd);
    }
    return 0;
}

inline int run_and_sync_outputs(rknn_app_context_t* app_ctx, const char* tag = "RKNN") {
    if (run_model(app_ctx, tag) != 0) {
        return -1;
    }
    return sync_output_tensors_to_cpu(app_ctx, tag);
}

} // namespace visiong::npu::rknn

