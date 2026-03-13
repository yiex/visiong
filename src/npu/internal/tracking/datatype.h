// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_TRACKING_DATATYPE_H
#define VISIONG_NPU_INTERNAL_TRACKING_DATATYPE_H

#include "error.h"
#include "logging.h"

#include <cstdint>
#include <cstdlib>

enum tensor_layout_e {
    NN_TENSORT_LAYOUT_UNKNOWN = 0,
    NN_TENSOR_NCHW = 1,
    NN_TENSOR_NHWC = 2,
    NN_TENSOR_OTHER = 3,
};

enum tensor_datatype_e {
    NN_TENSOR_INT8 = 1,
    NN_TENSOR_UINT8 = 2,
    NN_TENSOR_FLOAT = 3,
    NN_TENSOR_FLOAT16 = 4,
};

static const int g_max_num_dims = 4;

struct tensor_attr_s {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[g_max_num_dims];
    uint32_t n_elems;
    uint32_t size;
    tensor_datatype_e type;
    tensor_layout_e layout;
    int32_t zp;
    float scale;
};

struct tensor_data_s {
    tensor_attr_s attr;
    void* data;
};

static size_t nn_tensor_type_to_size(tensor_datatype_e type)
{
    switch (type)
    {
    case NN_TENSOR_INT8:
        return sizeof(int8_t);
    case NN_TENSOR_UINT8:
        return sizeof(uint8_t);
    case NN_TENSOR_FLOAT:
        return sizeof(float);
    case NN_TENSOR_FLOAT16:
        return sizeof(uint16_t);
    default:
        NN_LOG_ERROR("unsupported tensor type");
        std::exit(-1);
    }
}

[[maybe_unused]] static void nn_tensor_attr_to_cvimg_input_data(const tensor_attr_s& attr, tensor_data_s& data)
{
    if (attr.n_dims != 4)
    {
        NN_LOG_ERROR("unsupported input dims");
        std::exit(-1);
    }
    data.attr.n_dims = attr.n_dims;
    data.attr.index = 0;
    // Feed image tensors as UINT8 and let RKNN runtime handle input quantization. / Feed 图像 tensors 作为 UINT8 与 let RKNN 运行时 handle 输入 quantization.
    data.attr.type = (attr.type == NN_TENSOR_INT8) ? NN_TENSOR_INT8 : NN_TENSOR_UINT8;
    data.attr.layout = NN_TENSOR_NHWC;
    data.attr.zp = attr.zp;
    data.attr.scale = attr.scale;
    if (attr.layout == NN_TENSOR_NCHW)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[2];
        data.attr.dims[2] = attr.dims[3];
        data.attr.dims[3] = attr.dims[1];
    }
    else if (attr.layout == NN_TENSOR_NHWC)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[1];
        data.attr.dims[2] = attr.dims[2];
        data.attr.dims[3] = attr.dims[3];
    }
    else
    {
        NN_LOG_ERROR("unsupported input layout");
        std::exit(-1);
    }
    data.attr.n_elems = data.attr.dims[0] * data.attr.dims[1] * data.attr.dims[2] * data.attr.dims[3];
    const uint32_t compact_size = data.attr.n_elems * nn_tensor_type_to_size(data.attr.type);
    data.attr.size = (attr.size >= compact_size) ? attr.size : compact_size;
}

[[maybe_unused]] static void nn_tensor_attr_to_cvimg_input_data_float(const tensor_attr_s& attr, tensor_data_s& data)
{
    if (attr.n_dims != 4)
    {
        NN_LOG_ERROR("unsupported input dims");
        std::exit(-1);
    }
    data.attr.n_dims = attr.n_dims;
    data.attr.index = 0;
    data.attr.type = NN_TENSOR_FLOAT;
    data.attr.layout = NN_TENSOR_NHWC;
    data.attr.zp = attr.zp;
    data.attr.scale = attr.scale;
    if (attr.layout == NN_TENSOR_NCHW)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[2];
        data.attr.dims[2] = attr.dims[3];
        data.attr.dims[3] = attr.dims[1];
    }
    else if (attr.layout == NN_TENSOR_NHWC)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[1];
        data.attr.dims[2] = attr.dims[2];
        data.attr.dims[3] = attr.dims[3];
    }
    else
    {
        NN_LOG_ERROR("unsupported input layout");
        std::exit(-1);
    }
    data.attr.n_elems = data.attr.dims[0] * data.attr.dims[1] * data.attr.dims[2] * data.attr.dims[3];
    data.attr.size = data.attr.n_elems * sizeof(float);
}

[[maybe_unused]] static void nn_tensor_attr_to_tensor_input_data(const tensor_attr_s& attr, tensor_data_s& data)
{
    if (attr.n_dims != 4)
    {
        NN_LOG_ERROR("unsupported input dims");
        std::exit(-1);
    }
    data.attr.n_dims = attr.n_dims;
    data.attr.index = attr.index;
    data.attr.type = attr.type;
    data.attr.layout = attr.layout;
    data.attr.zp = attr.zp;
    data.attr.scale = attr.scale;
    for (uint32_t i = 0; i < attr.n_dims; ++i)
    {
        data.attr.dims[i] = attr.dims[i];
    }
    data.attr.n_elems = attr.n_elems;
    const uint32_t compact_size = data.attr.n_elems * nn_tensor_type_to_size(data.attr.type);
    data.attr.size = (attr.size >= compact_size) ? attr.size : compact_size;
}

#endif  // VISIONG_NPU_INTERNAL_TRACKING_DATATYPE_H
