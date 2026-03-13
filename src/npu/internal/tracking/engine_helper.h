// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_TRACKING_ENGINE_HELPER_H
#define VISIONG_NPU_INTERNAL_TRACKING_ENGINE_HELPER_H

#include "datatype.h"
#include "logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <rknn_api.h>

static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp = std::fopen(filename, "rb");
    if (fp == nullptr)
    {
        NN_LOG_ERROR("fopen %s fail!", filename);
        return nullptr;
    }
    std::fseek(fp, 0, SEEK_END);
    const long file_len = std::ftell(fp);
    if (file_len < 0)
    {
        NN_LOG_ERROR("ftell %s fail!", filename);
        std::fclose(fp);
        return nullptr;
    }

    const size_t model_len = static_cast<size_t>(file_len);
    unsigned char* model = static_cast<unsigned char*>(std::malloc(model_len));
    if (model == nullptr)
    {
        NN_LOG_ERROR("malloc %s fail!", filename);
        std::fclose(fp);
        return nullptr;
    }
    std::fseek(fp, 0, SEEK_SET);
    if (std::fread(model, 1, model_len, fp) != model_len)
    {
        NN_LOG_ERROR("fread %s fail!", filename);
        std::free(model);
        std::fclose(fp);
        return nullptr;
    }
    *model_size = static_cast<int>(model_len);
    std::fclose(fp);
    return model;
}

static void print_tensor_attr(rknn_tensor_attr* attr)
{
    NN_LOG_INFO("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f",
                attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
                attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
                get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static tensor_layout_e rknn_layout_convert(rknn_tensor_format fmt)
{
    switch (fmt)
    {
    case RKNN_TENSOR_NCHW:
        return NN_TENSOR_NCHW;
    case RKNN_TENSOR_NHWC:
        return NN_TENSOR_NHWC;
    default:
        return NN_TENSOR_OTHER;
    }
}

static rknn_tensor_format rknn_layout_convert(tensor_layout_e fmt)
{
    switch (fmt)
    {
    case NN_TENSOR_NCHW:
        return RKNN_TENSOR_NCHW;
    case NN_TENSOR_NHWC:
        return RKNN_TENSOR_NHWC;
    default:
        NN_LOG_ERROR("unsupported nn layout: %d\n", fmt);
        std::exit(1);
    }
}

static rknn_tensor_type rknn_type_convert(tensor_datatype_e type)
{
    switch (type)
    {
    case NN_TENSOR_INT8:
        return RKNN_TENSOR_INT8;
    case NN_TENSOR_UINT8:
        return RKNN_TENSOR_UINT8;
    case NN_TENSOR_FLOAT:
        return RKNN_TENSOR_FLOAT32;
    default:
        NN_LOG_ERROR("unsupported nn type: %d\n", type);
        std::exit(1);
    }
}

static tensor_datatype_e rknn_type_convert(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_UINT8:
        return NN_TENSOR_UINT8;
    case RKNN_TENSOR_FLOAT32:
        return NN_TENSOR_FLOAT;
    case RKNN_TENSOR_INT8:
        return NN_TENSOR_INT8;
    case RKNN_TENSOR_FLOAT16:
        return NN_TENSOR_FLOAT16;
    default:
        NN_LOG_ERROR("unsupported rknn type: %d\n", type);
        std::exit(1);
    }
}

static tensor_attr_s rknn_tensor_attr_convert(const rknn_tensor_attr& attr)
{
    tensor_attr_s shape{};
    shape.n_dims = attr.n_dims;
    shape.index = attr.index;
    for (uint32_t i = 0; i < attr.n_dims; ++i)
    {
        shape.dims[i] = attr.dims[i];
    }
    shape.size = (attr.size_with_stride > 0) ? attr.size_with_stride : attr.size;
    shape.n_elems = attr.n_elems;
    shape.layout = rknn_layout_convert(attr.fmt);
    shape.type = rknn_type_convert(attr.type);
    shape.zp = attr.zp;
    shape.scale = attr.scale;
    return shape;
}

[[maybe_unused]] static rknn_input tensor_data_to_rknn_input(const tensor_data_s& data)
{
    rknn_input input;
    std::memset(&input, 0, sizeof(input));
    input.index = data.attr.index;
    input.type = rknn_type_convert(data.attr.type);
    input.size = data.attr.size;
    input.fmt = rknn_layout_convert(data.attr.layout);
    input.pass_through = 0;
    input.buf = data.data;
    return input;
}

[[maybe_unused]] static void rknn_output_to_tensor_data(const rknn_output& output, tensor_data_s& data)
{
    data.attr.index = output.index;
    data.attr.size = output.size;
    NN_LOG_DEBUG("output size: %d", output.size);
    NN_LOG_DEBUG("output want_float: %d", output.want_float);
    std::memcpy(data.data, output.buf, output.size);
}

#endif  // VISIONG_NPU_INTERNAL_TRACKING_ENGINE_HELPER_H
