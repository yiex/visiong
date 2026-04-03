// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/LowLevelNPU.h"
#include "visiong/core/ImageBuffer.h"

#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/core/BufferStateMachine.h"
#include "rknn_api.h"
#include "common/internal/dma_alloc.h"
#include "im2d.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"
#include <dlfcn.h>

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

struct LowLevelNPUImpl {
    rknn_context ctx = 0;
    bool initialized = false;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> input_attrs_initial;
    std::vector<rknn_tensor_attr> output_attrs;
    std::vector<rknn_tensor_mem*> input_mems;
    std::vector<rknn_tensor_mem*> output_mems;
    int64_t last_run_us = -1;
    mutable std::mutex mutex;
};

namespace {

struct SourceDmaContext {
    std::unique_ptr<RgaDmaBuffer> uploaded_dma;
    std::unique_ptr<RgaDmaBuffer> wrapped_dma;
    const RgaDmaBuffer* current = nullptr;
};

size_t tensor_type_size_bytes_impl(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32:
            return sizeof(float);
        case RKNN_TENSOR_FLOAT16:
            return sizeof(uint16_t);
        case RKNN_TENSOR_INT8:
            return sizeof(int8_t);
        case RKNN_TENSOR_UINT8:
            return sizeof(uint8_t);
        case RKNN_TENSOR_INT16:
            return sizeof(int16_t);
        case RKNN_TENSOR_UINT16:
            return sizeof(uint16_t);
        case RKNN_TENSOR_INT32:
            return sizeof(int32_t);
        case RKNN_TENSOR_UINT32:
            return sizeof(uint32_t);
        case RKNN_TENSOR_INT64:
            return sizeof(int64_t);
        case RKNN_TENSOR_BOOL:
            return sizeof(uint8_t);
        case RKNN_TENSOR_INT4:
            return 1;
        default:
            return 0;
    }
}

uint16_t float_to_half_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16U) & 0x8000U;
    uint32_t mantissa = bits & 0x007fffffU;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000U) >> static_cast<uint32_t>(1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x00001000U) >> 13U));
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) |
                                 ((mantissa + 0x00001000U) >> 13U));
}

float half_bits_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03ffU;
            const uint32_t exp32 = exponent + (127U - 15U);
            bits = sign | (exp32 << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        const uint32_t exp32 = exponent + (127U - 15U);
        bits = sign | (exp32 << 23U) | (mantissa << 13U);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

rknn_tensor_type parse_tensor_type(const std::string& tensor_type) {
    const std::string type = visiong::to_lower_copy(tensor_type);
    if (type == "float32" || type == "fp32") {
        return RKNN_TENSOR_FLOAT32;
    }
    if (type == "float16" || type == "fp16") {
        return RKNN_TENSOR_FLOAT16;
    }
    if (type == "int8") {
        return RKNN_TENSOR_INT8;
    }
    if (type == "uint8") {
        return RKNN_TENSOR_UINT8;
    }
    if (type == "int16") {
        return RKNN_TENSOR_INT16;
    }
    if (type == "uint16") {
        return RKNN_TENSOR_UINT16;
    }
    if (type == "int32") {
        return RKNN_TENSOR_INT32;
    }
    if (type == "uint32") {
        return RKNN_TENSOR_UINT32;
    }
    if (type == "int64") {
        return RKNN_TENSOR_INT64;
    }
    if (type == "bool") {
        return RKNN_TENSOR_BOOL;
    }
    throw std::invalid_argument("Unsupported tensor_type: '" + tensor_type + "'.");
}

rknn_tensor_format parse_tensor_format(const std::string& tensor_format) {
    const std::string format = visiong::to_lower_copy(tensor_format);
    if (format == "nhwc") {
        return RKNN_TENSOR_NHWC;
    }
    if (format == "nchw") {
        return RKNN_TENSOR_NCHW;
    }
    if (format == "nc1hwc2") {
        return RKNN_TENSOR_NC1HWC2;
    }
    throw std::invalid_argument("Unsupported tensor_format: '" + tensor_format + "'.");
}

rknn_core_mask parse_core_mask(const std::string& core_mask) {
    const std::string mask = visiong::to_lower_copy(core_mask);
    if (mask == "auto") {
        return RKNN_NPU_CORE_AUTO;
    }
    if (mask == "0" || mask == "core0") {
        return RKNN_NPU_CORE_0;
    }
    if (mask == "1" || mask == "core1") {
        return RKNN_NPU_CORE_1;
    }
    if (mask == "2" || mask == "core2") {
        return RKNN_NPU_CORE_2;
    }
    if (mask == "0_1" || mask == "core0_1") {
        return RKNN_NPU_CORE_0_1;
    }
    if (mask == "0_1_2" || mask == "core0_1_2") {
        return RKNN_NPU_CORE_0_1_2;
    }
    throw std::invalid_argument("Unsupported core mask: '" + core_mask + "'.");
}

bool parse_hwc_from_attr(const rknn_tensor_attr& attr, int* h, int* w, int* c) {
    if (h == nullptr || w == nullptr || c == nullptr) {
        return false;
    }
    if (attr.n_dims < 3) {
        return false;
    }

    if (attr.fmt == RKNN_TENSOR_NHWC) {
        const int32_t idx_c = static_cast<int32_t>(attr.n_dims) - 1;
        const int32_t idx_w = static_cast<int32_t>(attr.n_dims) - 2;
        const int32_t idx_h = static_cast<int32_t>(attr.n_dims) - 3;
        *c = static_cast<int>(attr.dims[idx_c]);
        *w = static_cast<int>(attr.dims[idx_w]);
        *h = static_cast<int>(attr.dims[idx_h]);
        return true;
    }

    if (attr.fmt == RKNN_TENSOR_NCHW) {
        if (attr.n_dims < 4) {
            return false;
        }
        const int32_t idx_w = static_cast<int32_t>(attr.n_dims) - 1;
        const int32_t idx_h = static_cast<int32_t>(attr.n_dims) - 2;
        const int32_t idx_c = static_cast<int32_t>(attr.n_dims) - 3;
        *w = static_cast<int>(attr.dims[idx_w]);
        *h = static_cast<int>(attr.dims[idx_h]);
        *c = static_cast<int>(attr.dims[idx_c]);
        return true;
    }

    return false;
}

bool build_strided_input_copy_plan(const rknn_tensor_attr& attr,
                                   size_t elem_size,
                                   int* dst_stride_bytes,
                                   int* src_stride_bytes,
                                   int* row_count,
                                   size_t* packed_bytes) {
    if (dst_stride_bytes == nullptr || src_stride_bytes == nullptr ||
        row_count == nullptr || packed_bytes == nullptr) {
        return false;
    }
    if (elem_size == 0 || attr.w_stride == 0 || attr.size_with_stride == 0 || attr.size_with_stride == attr.size) {
        return false;
    }

    int h = 0;
    int w = 0;
    int c = 0;
    if (!parse_hwc_from_attr(attr, &h, &w, &c)) {
        return false;
    }

    int batch = 1;
    if (attr.fmt == RKNN_TENSOR_NHWC) {
        for (uint32_t i = 0; i + 3 < attr.n_dims; ++i) {
            batch *= std::max(1, static_cast<int>(attr.dims[i]));
        }
        *src_stride_bytes = static_cast<int>(w * c * static_cast<int>(elem_size));
        *dst_stride_bytes = static_cast<int>(std::max(1u, attr.w_stride) * c * static_cast<uint32_t>(elem_size));
        *row_count = batch * h;
    } else if (attr.fmt == RKNN_TENSOR_NCHW) {
        for (uint32_t i = 0; i + 3 < attr.n_dims; ++i) {
            batch *= std::max(1, static_cast<int>(attr.dims[i]));
        }
        *src_stride_bytes = static_cast<int>(w * static_cast<int>(elem_size));
        *dst_stride_bytes = static_cast<int>(std::max(1u, attr.w_stride) * static_cast<uint32_t>(elem_size));
        *row_count = batch * c * h;
    } else {
        return false;
    }

    *packed_bytes = static_cast<size_t>(*src_stride_bytes) * static_cast<size_t>(std::max(0, *row_count));
    return (*dst_stride_bytes > *src_stride_bytes) && (*packed_bytes > 0);
}

int query_input_attr_with_fallback(rknn_context ctx, uint32_t index, rknn_tensor_attr* attr) {
    if (attr == nullptr) {
        return RKNN_ERR_PARAM_INVALID;
    }

    const rknn_query_cmd cmds[] = {
        RKNN_QUERY_INPUT_ATTR,
        RKNN_QUERY_NATIVE_INPUT_ATTR,
        RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,
        RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR,
    };

    int ret = RKNN_ERR_FAIL;
    for (const rknn_query_cmd cmd : cmds) {
        std::memset(attr, 0, sizeof(*attr));
        attr->index = index;
        ret = rknn_query(ctx, cmd, attr, sizeof(*attr));
        if (ret == RKNN_SUCC) {
            return ret;
        }
    }
    return ret;
}

int query_output_attr_with_fallback(rknn_context ctx, uint32_t index, rknn_tensor_attr* attr) {
    if (attr == nullptr) {
        return RKNN_ERR_PARAM_INVALID;
    }

    const rknn_query_cmd cmds[] = {
        RKNN_QUERY_OUTPUT_ATTR,
        RKNN_QUERY_NATIVE_OUTPUT_ATTR,
        RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
        RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR,
    };

    int ret = RKNN_ERR_FAIL;
    for (const rknn_query_cmd cmd : cmds) {
        std::memset(attr, 0, sizeof(*attr));
        attr->index = index;
        ret = rknn_query(ctx, cmd, attr, sizeof(*attr));
        if (ret == RKNN_SUCC) {
            return ret;
        }
    }
    return ret;
}

void upload_image_to_dma(const ImageBuffer& image, RgaDmaBuffer& dma) {
    const int bpp = get_bpp_for_format(static_cast<int>(image.format));
    copy_data_with_stride(
        dma.get_vir_addr(),
        dma.get_wstride() * bpp / 8,
        image.get_data(),
        image.w_stride * bpp / 8,
        image.height,
        image.width * bpp / 8);
    visiong::bufstate::mark_cpu_write(dma);
}

template <typename T>
T clamp_round_cast(double value) {
    const double low = static_cast<double>(std::numeric_limits<T>::lowest());
    const double high = static_cast<double>(std::numeric_limits<T>::max());
    if (value < low) {
        value = low;
    }
    if (value > high) {
        value = high;
    }
    return static_cast<T>(std::llround(value));
}

}  // namespace
LowLevelNPU::LowLevelNPU(const std::string& model_path, uint32_t init_flags) : m_impl(std::make_unique<LowLevelNPUImpl>()) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    int ret = rknn_init(&m_impl->ctx, const_cast<char*>(model_path.c_str()), 0, init_flags, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: rknn_init failed, ret=" + std::to_string(ret));
    }

    ret = rknn_query(m_impl->ctx, RKNN_QUERY_IN_OUT_NUM, &m_impl->io_num, sizeof(m_impl->io_num));
    if (ret != RKNN_SUCC) {
        rknn_destroy(m_impl->ctx);
        m_impl->ctx = 0;
        throw std::runtime_error("LowLevelNPU: RKNN_QUERY_IN_OUT_NUM failed, ret=" + std::to_string(ret));
    }

    m_impl->input_attrs.resize(m_impl->io_num.n_input);
    m_impl->input_attrs_initial.resize(m_impl->io_num.n_input);
    m_impl->output_attrs.resize(m_impl->io_num.n_output);
    m_impl->input_mems.assign(m_impl->io_num.n_input, nullptr);
    m_impl->output_mems.assign(m_impl->io_num.n_output, nullptr);

    for (uint32_t i = 0; i < m_impl->io_num.n_input; ++i) {
        ret = query_input_attr_with_fallback(m_impl->ctx, i, &m_impl->input_attrs[i]);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: query input attr failed for index " + std::to_string(i) +
                                     ", ret=" + std::to_string(ret));
        }

        rknn_tensor_attr bind_attr = m_impl->input_attrs[i];
        bind_attr.pass_through = 1;

        uint32_t bytes = bind_attr.size_with_stride;
        if (bytes == 0) {
            bytes = bind_attr.size;
        }
        if (bytes == 0) {
            const size_t elem_size = tensor_type_size_bytes_impl(bind_attr.type);
            if (elem_size == 0) {
                throw std::runtime_error("LowLevelNPU: unsupported input tensor type at index " +
                                         std::to_string(i));
            }
            bytes = static_cast<uint32_t>(bind_attr.n_elems * elem_size);
        }

        m_impl->input_mems[i] = rknn_create_mem(m_impl->ctx, bytes);
        if (m_impl->input_mems[i] == nullptr) {
            throw std::runtime_error("LowLevelNPU: rknn_create_mem failed for input " + std::to_string(i));
        }

        ret = rknn_set_io_mem(m_impl->ctx, m_impl->input_mems[i], &bind_attr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: rknn_set_io_mem failed for input " + std::to_string(i) +
                                     ", ret=" + std::to_string(ret));
        }

        m_impl->input_attrs[i] = bind_attr;
        m_impl->input_attrs_initial[i] = bind_attr;
    }

    for (uint32_t i = 0; i < m_impl->io_num.n_output; ++i) {
        ret = query_output_attr_with_fallback(m_impl->ctx, i, &m_impl->output_attrs[i]);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: query output attr failed for index " + std::to_string(i) +
                                     ", ret=" + std::to_string(ret));
        }

        rknn_tensor_attr bind_attr = m_impl->output_attrs[i];

        uint32_t bytes = bind_attr.size_with_stride;
        if (bytes == 0) {
            bytes = bind_attr.size;
        }
        if (bytes == 0) {
            const size_t elem_size = tensor_type_size_bytes_impl(bind_attr.type);
            if (elem_size == 0) {
                throw std::runtime_error("LowLevelNPU: unsupported output tensor type at index " +
                                         std::to_string(i));
            }
            bytes = static_cast<uint32_t>(bind_attr.n_elems * elem_size);
        }

        m_impl->output_mems[i] = rknn_create_mem(m_impl->ctx, bytes);
        if (m_impl->output_mems[i] == nullptr) {
            throw std::runtime_error("LowLevelNPU: rknn_create_mem failed for output " + std::to_string(i));
        }

        ret = rknn_set_io_mem(m_impl->ctx, m_impl->output_mems[i], &bind_attr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: rknn_set_io_mem failed for output " + std::to_string(i) +
                                     ", ret=" + std::to_string(ret));
        }

        m_impl->output_attrs[i] = bind_attr;
    }

    m_impl->initialized = true;
}

LowLevelNPU::~LowLevelNPU() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    if (m_impl->ctx != 0) {
        for (rknn_tensor_mem*& mem : m_impl->input_mems) {
            if (mem != nullptr) {
                rknn_destroy_mem(m_impl->ctx, mem);
                mem = nullptr;
            }
        }
        for (rknn_tensor_mem*& mem : m_impl->output_mems) {
            if (mem != nullptr) {
                rknn_destroy_mem(m_impl->ctx, mem);
                mem = nullptr;
            }
        }
        rknn_destroy(m_impl->ctx);
        m_impl->ctx = 0;
    }

    m_impl->initialized = false;
}

size_t tensor_type_size_bytes(rknn_tensor_type type) {
    return tensor_type_size_bytes_impl(type);
}

LowLevelTensorInfo to_tensor_info(const rknn_tensor_attr& attr) {
    LowLevelTensorInfo info;
    info.index = static_cast<int>(attr.index);
    info.name = attr.name;
    info.dims.reserve(attr.n_dims);
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        info.dims.push_back(static_cast<int64_t>(attr.dims[i]));
    }
    info.format = get_format_string(attr.fmt);
    info.type = get_type_string(attr.type);
    info.quant_type = get_qnt_type_string(attr.qnt_type);
    info.zero_point = attr.zp;
    info.scale = attr.scale;
    info.num_elements = attr.n_elems;
    info.size_bytes = attr.size;
    info.size_with_stride_bytes = attr.size_with_stride;
    info.w_stride = attr.w_stride;
    info.h_stride = attr.h_stride;
    info.pass_through = (attr.pass_through != 0);
    return info;
}

int LowLevelNPU::check_input_index_locked(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= m_impl->input_attrs.size()) {
        throw std::out_of_range("LowLevelNPU: input index out of range: " + std::to_string(index));
    }
    return index;
}

int LowLevelNPU::check_output_index_locked(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= m_impl->output_attrs.size()) {
        throw std::out_of_range("LowLevelNPU: output index out of range: " + std::to_string(index));
    }
    return index;
}

bool LowLevelNPU::is_initialized() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->initialized;
}

int LowLevelNPU::num_inputs() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<int>(m_impl->input_attrs.size());
}

int LowLevelNPU::num_outputs() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<int>(m_impl->output_attrs.size());
}

std::vector<LowLevelTensorInfo> LowLevelNPU::input_tensors() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    std::vector<LowLevelTensorInfo> out;
    out.reserve(m_impl->input_attrs.size());
    for (const rknn_tensor_attr& attr : m_impl->input_attrs) {
        out.push_back(to_tensor_info(attr));
    }
    return out;
}

std::vector<LowLevelTensorInfo> LowLevelNPU::output_tensors() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    std::vector<LowLevelTensorInfo> out;
    out.reserve(m_impl->output_attrs.size());
    for (const rknn_tensor_attr& attr : m_impl->output_attrs) {
        out.push_back(to_tensor_info(attr));
    }
    return out;
}

LowLevelTensorInfo LowLevelNPU::input_tensor(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);
    return to_tensor_info(m_impl->input_attrs[idx]);
}

LowLevelTensorInfo LowLevelNPU::output_tensor(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_output_index_locked(index);
    return to_tensor_info(m_impl->output_attrs[idx]);
}

std::vector<int64_t> LowLevelNPU::input_shape(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);
    std::vector<int64_t> shape;
    shape.reserve(m_impl->input_attrs[idx].n_dims);
    for (uint32_t i = 0; i < m_impl->input_attrs[idx].n_dims; ++i) {
        shape.push_back(static_cast<int64_t>(m_impl->input_attrs[idx].dims[i]));
    }
    return shape;
}

std::vector<int64_t> LowLevelNPU::output_shape(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_output_index_locked(index);
    std::vector<int64_t> shape;
    shape.reserve(m_impl->output_attrs[idx].n_dims);
    for (uint32_t i = 0; i < m_impl->output_attrs[idx].n_dims; ++i) {
        shape.push_back(static_cast<int64_t>(m_impl->output_attrs[idx].dims[i]));
    }
    return shape;
}

std::pair<std::string, std::string> LowLevelNPU::sdk_versions() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    rknn_sdk_version version{};
    const int ret = rknn_query(m_impl->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret != RKNN_SUCC) {
        return {"unknown", "unknown"};
    }
    return {version.api_version, version.drv_version};
}

int64_t LowLevelNPU::last_run_us() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->last_run_us;
}
void LowLevelNPU::set_core_mask(const std::string& core_mask) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const rknn_core_mask mask = parse_core_mask(core_mask);

    using SetCoreMaskFn = int (*)(rknn_context, rknn_core_mask);
    void* symbol = dlsym(RTLD_DEFAULT, "rknn_set_core_mask");
    if (symbol == nullptr) {
        throw std::runtime_error("LowLevelNPU: current librknnmrt does not export rknn_set_core_mask.");
    }

    SetCoreMaskFn set_core_mask_fn = reinterpret_cast<SetCoreMaskFn>(symbol);
    const int ret = set_core_mask_fn(m_impl->ctx, mask);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: set_core_mask failed, ret=" + std::to_string(ret));
    }
}

void LowLevelNPU::set_input_attr(int index,
                                 const std::string& tensor_type,
                                 const std::string& tensor_format,
                                 bool pass_through) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);

    rknn_tensor_attr attr = m_impl->input_attrs[idx];
    attr.type = parse_tensor_type(tensor_type);
    attr.fmt = parse_tensor_format(tensor_format);
    attr.pass_through = pass_through ? 1 : 0;

    const int ret = rknn_set_io_mem(m_impl->ctx, m_impl->input_mems[idx], &attr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: set_input_attr failed at index " + std::to_string(idx) +
                                 ", ret=" + std::to_string(ret));
    }

    m_impl->input_attrs[idx] = attr;
}

void LowLevelNPU::reset_input_attr(int index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);

    rknn_tensor_attr attr = m_impl->input_attrs_initial[idx];
    const int ret = rknn_set_io_mem(m_impl->ctx, m_impl->input_mems[idx], &attr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: reset_input_attr failed at index " + std::to_string(idx) +
                                 ", ret=" + std::to_string(ret));
    }

    m_impl->input_attrs[idx] = attr;
}

void LowLevelNPU::sync_input_to_device_locked(int index) {
    const int idx = check_input_index_locked(index);
    rknn_tensor_mem* mem = m_impl->input_mems[idx];
    if (mem == nullptr) {
        throw std::runtime_error("LowLevelNPU: input memory is null at index " + std::to_string(idx));
    }

    if (mem->fd >= 0) {
        const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
        visiong::bufstate::prepare_device_read(view, visiong::bufstate::BufferOwner::NPU);
        return;
    }

    const int ret = rknn_mem_sync(m_impl->ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: rknn_mem_sync TO_DEVICE failed, ret=" + std::to_string(ret));
    }
}

void LowLevelNPU::sync_output_from_device_locked(int index) const {
    const int idx = check_output_index_locked(index);
    rknn_tensor_mem* mem = m_impl->output_mems[idx];
    if (mem == nullptr) {
        throw std::runtime_error("LowLevelNPU: output memory is null at index " + std::to_string(idx));
    }

    if (mem->fd >= 0) {
        const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
        visiong::bufstate::prepare_cpu_read(view);
        return;
    }

    const int ret = rknn_mem_sync(m_impl->ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: rknn_mem_sync FROM_DEVICE failed, ret=" + std::to_string(ret));
    }
}

void LowLevelNPU::sync_input_to_device(int index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    sync_input_to_device_locked(index);
}

void LowLevelNPU::sync_output_from_device(int index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    sync_output_from_device_locked(index);
}

void LowLevelNPU::sync_all_outputs_from_device() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    for (size_t i = 0; i < m_impl->output_mems.size(); ++i) {
        sync_output_from_device_locked(static_cast<int>(i));
    }
}

void LowLevelNPU::set_input_buffer(int index,
                                   const void* data,
                                   size_t bytes,
                                   bool zero_pad,
                                   bool sync_to_device) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);

    if (bytes > 0 && data == nullptr) {
        throw std::invalid_argument("LowLevelNPU: set_input_buffer got null data pointer with non-zero bytes.");
    }

    rknn_tensor_mem* mem = m_impl->input_mems[idx];
    rknn_tensor_attr& attr = m_impl->input_attrs[idx];
    if (mem == nullptr || mem->virt_addr == nullptr) {
        throw std::runtime_error("LowLevelNPU: input memory is unavailable at index " + std::to_string(idx));
    }

    const size_t capacity = static_cast<size_t>(mem->size);
    if (bytes > capacity) {
        throw std::invalid_argument("LowLevelNPU: input bytes exceed capacity at index " + std::to_string(idx) +
                                    " (" + std::to_string(bytes) + " > " + std::to_string(capacity) + ").");
    }

    if (zero_pad) {
        std::memset(mem->virt_addr, 0, capacity);
    }

    const size_t elem_size = tensor_type_size_bytes_impl(attr.type);
    int dst_stride_bytes = 0;
    int src_stride_bytes = 0;
    int row_count = 0;
    size_t packed_bytes = 0;
    const bool use_strided_copy =
        build_strided_input_copy_plan(attr, elem_size, &dst_stride_bytes, &src_stride_bytes, &row_count, &packed_bytes) &&
        bytes <= packed_bytes && packed_bytes < capacity;

    if (bytes > 0) {
        if (use_strided_copy) {
            std::vector<uint8_t> packed(packed_bytes, 0);
            std::memcpy(packed.data(), data, bytes);
            copy_data_with_stride(mem->virt_addr,
                                  dst_stride_bytes,
                                  packed.data(),
                                  src_stride_bytes,
                                  row_count,
                                  src_stride_bytes);
        } else {
            std::memcpy(mem->virt_addr, data, bytes);
        }
    }
    if (mem->fd >= 0) {
        const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
        visiong::bufstate::mark_cpu_write(view);
    }

    if (sync_to_device) {
        sync_input_to_device_locked(idx);
    }
}

void LowLevelNPU::set_input_from_float(int index,
                                       const float* data,
                                       size_t count,
                                       bool quantize_if_needed,
                                       bool zero_pad,
                                       bool sync_to_device) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);

    if (count > 0 && data == nullptr) {
        throw std::invalid_argument("LowLevelNPU: set_input_from_float got null data pointer with non-zero count.");
    }

    rknn_tensor_mem* mem = m_impl->input_mems[idx];
    rknn_tensor_attr& attr = m_impl->input_attrs[idx];
    if (mem == nullptr || mem->virt_addr == nullptr) {
        throw std::runtime_error("LowLevelNPU: input memory is unavailable at index " + std::to_string(idx));
    }

    const size_t elem_size = tensor_type_size_bytes_impl(attr.type);
    if (elem_size == 0) {
        throw std::runtime_error("LowLevelNPU: unsupported input tensor type at index " + std::to_string(idx));
    }

    const size_t capacity_elems = static_cast<size_t>(mem->size) / elem_size;
    const size_t copy_elems = std::min(count, capacity_elems);

    if (zero_pad) {
        std::memset(mem->virt_addr, 0, mem->size);
    }

    int dst_stride_bytes = 0;
    int src_stride_bytes = 0;
    int row_count = 0;
    size_t packed_bytes = 0;
    const bool use_strided_copy =
        build_strided_input_copy_plan(attr, elem_size, &dst_stride_bytes, &src_stride_bytes, &row_count, &packed_bytes) &&
        packed_bytes < static_cast<size_t>(mem->size);
    std::vector<uint8_t> packed;
    void* write_ptr = mem->virt_addr;
    if (use_strided_copy) {
        packed.assign(packed_bytes, 0);
        write_ptr = packed.data();
    }

    const bool affine_qnt = quantize_if_needed && (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;

    switch (attr.type) {
        case RKNN_TENSOR_FLOAT32: {
            std::memcpy(write_ptr, data, copy_elems * sizeof(float));
            break;
        }
        case RKNN_TENSOR_FLOAT16: {
            uint16_t* dst = static_cast<uint16_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                dst[i] = float_to_half_bits(data[i]);
            }
            break;
        }
        case RKNN_TENSOR_INT8: {
            int8_t* dst = static_cast<int8_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<int8_t>(raw);
            }
            break;
        }
        case RKNN_TENSOR_UINT8: {
            uint8_t* dst = static_cast<uint8_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<uint8_t>(raw);
            }
            break;
        }
        case RKNN_TENSOR_INT16: {
            int16_t* dst = static_cast<int16_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<int16_t>(raw);
            }
            break;
        }
        case RKNN_TENSOR_UINT16: {
            uint16_t* dst = static_cast<uint16_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<uint16_t>(raw);
            }
            break;
        }
        case RKNN_TENSOR_INT32: {
            int32_t* dst = static_cast<int32_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<int32_t>(raw);
            }
            break;
        }
        case RKNN_TENSOR_UINT32: {
            uint32_t* dst = static_cast<uint32_t*>(write_ptr);
            for (size_t i = 0; i < copy_elems; ++i) {
                const double raw = affine_qnt
                                       ? static_cast<double>(data[i]) / static_cast<double>(scale) +
                                             static_cast<double>(attr.zp)
                                       : static_cast<double>(data[i]);
                dst[i] = clamp_round_cast<uint32_t>(raw);
            }
            break;
        }
        default:
            throw std::runtime_error("LowLevelNPU: set_input_from_float does not support target type " +
                                     std::string(get_type_string(attr.type)) + ".");
    }

    if (use_strided_copy) {
        copy_data_with_stride(mem->virt_addr,
                              dst_stride_bytes,
                              packed.data(),
                              src_stride_bytes,
                              row_count,
                              src_stride_bytes);
    }

    if (mem->fd >= 0) {
        const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
        visiong::bufstate::mark_cpu_write(view);
    }

    if (sync_to_device) {
        sync_input_to_device_locked(idx);
    }
}
SourceDmaContext prepare_source_dma(const ImageBuffer& image) {
    SourceDmaContext ctx;
    if (image.is_zero_copy() && image.get_dma_fd() >= 0) {
        ctx.wrapped_dma = std::make_unique<RgaDmaBuffer>(
            image.get_dma_fd(),
            const_cast<void*>(image.get_data()),
            image.get_size(),
            image.width,
            image.height,
            static_cast<int>(image.format),
            image.w_stride,
            image.h_stride);
        ctx.current = ctx.wrapped_dma.get();
        return ctx;
    }

    ctx.uploaded_dma = std::make_unique<RgaDmaBuffer>(
        image.width,
        image.height,
        static_cast<int>(image.format));
    upload_image_to_dma(image, *ctx.uploaded_dma);
    ctx.current = ctx.uploaded_dma.get();
    return ctx;
}

void LowLevelNPU::set_input_image(int index,
                                  const ImageBuffer& image,
                                  const std::string& color_order,
                                  bool keep_aspect,
                                  int pad_value,
                                  bool driver_convert) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_input_index_locked(index);

    if (!image.is_valid()) {
        throw std::invalid_argument("LowLevelNPU: input image is invalid.");
    }

    if (driver_convert) {
        rknn_tensor_attr converted = m_impl->input_attrs[idx];
        converted.type = RKNN_TENSOR_UINT8;
        converted.fmt = RKNN_TENSOR_NHWC;
        converted.pass_through = 0;
        const int ret = rknn_set_io_mem(m_impl->ctx, m_impl->input_mems[idx], &converted);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: set_input_image failed to enable driver convert, ret=" +
                                     std::to_string(ret));
        }
        m_impl->input_attrs[idx] = converted;
    }

    const rknn_tensor_attr& attr = m_impl->input_attrs[idx];
    int model_h = 0;
    int model_w = 0;
    int model_c = 0;
    if (!parse_hwc_from_attr(attr, &model_h, &model_w, &model_c)) {
        throw std::runtime_error("LowLevelNPU: set_input_image expects 3D/4D tensor layout.");
    }
    if (model_c != 3) {
        throw std::runtime_error("LowLevelNPU: set_input_image only supports 3-channel tensors.");
    }

    const std::string order = visiong::to_lower_copy(color_order);
    PIXEL_FORMAT_E target_format = RK_FMT_RGB888;
    if (order == "rgb") {
        target_format = RK_FMT_RGB888;
    } else if (order == "bgr") {
        target_format = RK_FMT_BGR888;
    } else {
        throw std::invalid_argument("LowLevelNPU: color_order must be 'rgb' or 'bgr'.");
    }

    ImageBuffer converted_image;
    const ImageBuffer* source = &image;
    if (image.format != target_format) {
        converted_image = image.to_format(target_format);
        source = &converted_image;
    }

    const uint8_t pad_u8 = static_cast<uint8_t>(std::max(0, std::min(255, pad_value)));

    const bool can_try_direct_rga =
        (m_impl->input_mems[idx] != nullptr && m_impl->input_mems[idx]->fd >= 0 && m_impl->input_mems[idx]->virt_addr != nullptr);

    if (can_try_direct_rga) {
        try {
            SourceDmaContext src_dma_ctx = prepare_source_dma(*source);
            const int w_stride = (attr.w_stride > 0) ? static_cast<int>(attr.w_stride) : model_w;
            const int h_stride = (attr.h_stride > 0) ? static_cast<int>(attr.h_stride) : model_h;
            RgaDmaBuffer dst_dma(
                m_impl->input_mems[idx]->fd,
                m_impl->input_mems[idx]->virt_addr,
                m_impl->input_mems[idx]->size,
                model_w,
                model_h,
                static_cast<int>(target_format),
                w_stride,
                h_stride);

            if (keep_aspect) {
                rga_letterbox_op(*src_dma_ctx.current,
                                 dst_dma,
                                 std::make_tuple(pad_u8, pad_u8, pad_u8),
                                 src_dma_ctx.uploaded_dma != nullptr,
                                 false);
            } else {
                visiong::bufstate::prepare_device_read(*src_dma_ctx.current, visiong::bufstate::BufferOwner::RGA);
                visiong::bufstate::prepare_device_write(dst_dma,
                                                        visiong::bufstate::BufferOwner::RGA,
                                                        visiong::bufstate::AccessIntent::Overwrite);
                if (imresize(src_dma_ctx.current->get_buffer(), dst_dma.get_buffer()) != IM_STATUS_SUCCESS) {
                    throw std::runtime_error("LowLevelNPU: RGA imresize failed.");
                }
                visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
            }
            visiong::bufstate::prepare_device_read(dst_dma, visiong::bufstate::BufferOwner::NPU);
            return;
        } catch (...) {
            // Fallback to CPU copy path below. / 回退 以 CPU 复制 路径 below.
        }
    }

    ImageBuffer prepared = keep_aspect ? source->letterbox(model_w, model_h, std::make_tuple(pad_u8, pad_u8, pad_u8))
                                       : source->resize(model_w, model_h);

    const int bpp = get_bpp_for_format(static_cast<int>(prepared.format));
    const int dst_w_stride = ((attr.w_stride > 0) ? static_cast<int>(attr.w_stride) : model_w) * bpp / 8;
    const int src_w_stride = prepared.w_stride * bpp / 8;
    const int row_bytes = model_w * bpp / 8;

    if (m_impl->input_mems[idx] == nullptr || m_impl->input_mems[idx]->virt_addr == nullptr) {
        throw std::runtime_error("LowLevelNPU: input memory is unavailable for set_input_image fallback.");
    }

    visiong::bufstate::prepare_cpu_read(prepared);
    copy_data_with_stride(m_impl->input_mems[idx]->virt_addr,
                          dst_w_stride,
                          prepared.get_data(),
                          src_w_stride,
                          model_h,
                          row_bytes);
    if (m_impl->input_mems[idx]->fd >= 0) {
        const auto view = visiong::bufstate::make_dma_view(m_impl->input_mems[idx]->fd,
                                                           m_impl->input_mems[idx]->virt_addr,
                                                           m_impl->input_mems[idx]->size);
        visiong::bufstate::mark_cpu_write(view);
    }
    sync_input_to_device_locked(idx);
}

void LowLevelNPU::query_perf_run_locked() {
    // Not all librknnmrt builds support RKNN_QUERY_PERF_RUN. / 不 全部 librknnmrt builds 支持 RKNN_QUERY_PERF_RUN.
    // We keep host-side timing as the portable fallback.
}
void LowLevelNPU::run(bool sync_outputs, bool non_block, int timeout_ms) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    if (!m_impl->initialized) {
        throw std::runtime_error("LowLevelNPU: runtime is not initialized.");
    }

    rknn_run_extend run_ext{};
    rknn_run_extend* run_ext_ptr = nullptr;
    if (non_block || timeout_ms > 0) {
        run_ext.non_block = non_block ? 1 : 0;
        run_ext.timeout_ms = timeout_ms;
        run_ext_ptr = &run_ext;
    }

    const auto run_start = std::chrono::steady_clock::now();
    int ret = rknn_run(m_impl->ctx, run_ext_ptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: rknn_run failed, ret=" + std::to_string(ret));
    }

    if (non_block) {
        ret = rknn_wait(m_impl->ctx, &run_ext);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("LowLevelNPU: rknn_wait failed, ret=" + std::to_string(ret));
        }
    }

    const auto run_end = std::chrono::steady_clock::now();
    m_impl->last_run_us = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start).count();

    for (size_t i = 0; i < m_impl->output_mems.size(); ++i) {
        rknn_tensor_mem* mem = m_impl->output_mems[i];
        if (mem != nullptr && mem->fd >= 0) {
            const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
            visiong::bufstate::mark_device_write(view, visiong::bufstate::BufferOwner::NPU);
        }
    }

    if (sync_outputs) {
        for (size_t i = 0; i < m_impl->output_mems.size(); ++i) {
            sync_output_from_device_locked(static_cast<int>(i));
        }
    }
}

void LowLevelNPU::wait(int timeout_ms) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    rknn_run_extend run_ext{};
    run_ext.timeout_ms = timeout_ms;

    const auto wait_start = std::chrono::steady_clock::now();
    const int ret = rknn_wait(m_impl->ctx, &run_ext);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("LowLevelNPU: rknn_wait failed, ret=" + std::to_string(ret));
    }
    const auto wait_end = std::chrono::steady_clock::now();
    m_impl->last_run_us = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

    for (size_t i = 0; i < m_impl->output_mems.size(); ++i) {
        rknn_tensor_mem* mem = m_impl->output_mems[i];
        if (mem != nullptr && mem->fd >= 0) {
            const auto view = visiong::bufstate::make_dma_view(mem->fd, mem->virt_addr, mem->size);
            visiong::bufstate::mark_device_write(view, visiong::bufstate::BufferOwner::NPU);
        }
    }
}

std::vector<uint8_t> LowLevelNPU::output_bytes(int index,
                                               bool with_stride,
                                               bool sync_from_device) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_output_index_locked(index);

    if (sync_from_device) {
        sync_output_from_device_locked(idx);
    }

    const rknn_tensor_attr& attr = m_impl->output_attrs[idx];
    const rknn_tensor_mem* mem = m_impl->output_mems[idx];
    if (mem == nullptr || mem->virt_addr == nullptr) {
        throw std::runtime_error("LowLevelNPU: output memory is unavailable at index " + std::to_string(idx));
    }

    uint32_t bytes = with_stride ? attr.size_with_stride : attr.size;
    if (bytes == 0) {
        const size_t elem_size = tensor_type_size_bytes_impl(attr.type);
        bytes = static_cast<uint32_t>(attr.n_elems * elem_size);
    }
    bytes = std::min(bytes, mem->size);

    std::vector<uint8_t> out(bytes);
    if (bytes > 0) {
        std::memcpy(out.data(), mem->virt_addr, bytes);
    }
    return out;
}

std::vector<float> LowLevelNPU::output_float(int index,
                                             bool dequantize_if_needed,
                                             bool sync_from_device) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const int idx = check_output_index_locked(index);

    if (sync_from_device) {
        sync_output_from_device_locked(idx);
    }

    const rknn_tensor_attr& attr = m_impl->output_attrs[idx];
    const rknn_tensor_mem* mem = m_impl->output_mems[idx];
    if (mem == nullptr || mem->virt_addr == nullptr) {
        throw std::runtime_error("LowLevelNPU: output memory is unavailable at index " + std::to_string(idx));
    }

    const size_t elem_size = tensor_type_size_bytes_impl(attr.type);
    if (elem_size == 0) {
        throw std::runtime_error("LowLevelNPU: unsupported output tensor type at index " + std::to_string(idx));
    }

    const size_t mem_elems = static_cast<size_t>(mem->size) / elem_size;
    const size_t count = std::min(static_cast<size_t>(attr.n_elems), mem_elems);

    std::vector<float> out(count, 0.0f);
    const bool affine_qnt = dequantize_if_needed && (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;

    switch (attr.type) {
        case RKNN_TENSOR_FLOAT32: {
            const float* src = static_cast<const float*>(mem->virt_addr);
            std::memcpy(out.data(), src, count * sizeof(float));
            break;
        }
        case RKNN_TENSOR_FLOAT16: {
            const uint16_t* src = static_cast<const uint16_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = half_bits_to_float(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_INT8: {
            const int8_t* src = static_cast<const int8_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_UINT8: {
            const uint8_t* src = static_cast<const uint8_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_INT16: {
            const int16_t* src = static_cast<const int16_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_UINT16: {
            const uint16_t* src = static_cast<const uint16_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_INT32: {
            const int32_t* src = static_cast<const int32_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_UINT32: {
            const uint32_t* src = static_cast<const uint32_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = affine_qnt ? (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * scale
                                    : static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_INT64: {
            const int64_t* src = static_cast<const int64_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = static_cast<float>(src[i]);
            }
            break;
        }
        case RKNN_TENSOR_BOOL: {
            const uint8_t* src = static_cast<const uint8_t*>(mem->virt_addr);
            for (size_t i = 0; i < count; ++i) {
                out[i] = src[i] ? 1.0f : 0.0f;
            }
            break;
        }
        default:
            throw std::runtime_error("LowLevelNPU: output_float does not support tensor type " +
                                     std::string(get_type_string(attr.type)) + ".");
    }

    return out;
}

int LowLevelNPU::input_dma_fd(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const size_t idx = static_cast<size_t>(check_input_index_locked(index));
    const auto& mems = m_impl->input_mems;
    return (mems[idx] != nullptr) ? mems[idx]->fd : -1;
}

int LowLevelNPU::output_dma_fd(int index) const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const size_t idx = static_cast<size_t>(check_output_index_locked(index));
    const auto& mems = m_impl->output_mems;
    return (mems[idx] != nullptr) ? mems[idx]->fd : -1;
}
