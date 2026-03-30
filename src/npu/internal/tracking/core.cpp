// SPDX-License-Identifier: LGPL-3.0-or-later
#include "core.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include "logging.h"

static inline float softmax_pos_2class(float neg_logit, float pos_logit)
{
    const float m = std::max(neg_logit, pos_logit);
    const float e_neg = std::exp(neg_logit - m);
    const float e_pos = std::exp(pos_logit - m);
    const float denom = e_neg + e_pos;
    if (denom <= 0.0f)
    {
        return 0.5f;
    }
    return e_pos / denom;
}

namespace {

using SteadyClock = std::chrono::steady_clock;

static inline double elapsed_ms(const SteadyClock::time_point& start, const SteadyClock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

static bool nanotrack_profile_enabled()
{
    static const bool enabled = []() {
        const char* env = std::getenv("VISIONG_PROFILE_NANOTRACK");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

struct NanoTrackUpdateTiming
{
    double copy_ms = 0.0;
    double x_run_ms = 0.0;
    double pack_ms = 0.0;
    double h_run_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
};

static NanoTrackUpdateTiming g_last_update_timing;

struct NanoTrackProfileStats
{
    uint64_t track_calls = 0;
    double track_crop_ms = 0.0;
    double track_update_ms = 0.0;
    double track_tail_ms = 0.0;
    double track_total_ms = 0.0;
    double update_copy_ms = 0.0;
    double update_x_run_ms = 0.0;
    double update_pack_ms = 0.0;
    double update_h_run_ms = 0.0;
    double update_post_ms = 0.0;
    double update_total_ms = 0.0;
};

static NanoTrackProfileStats& nanotrack_profile_stats()
{
    static NanoTrackProfileStats stats;
    return stats;
}

static void nanotrack_profile_record(double crop_ms, double update_ms, double tail_ms, double total_ms)
{
    if (!nanotrack_profile_enabled())
    {
        return;
    }

    auto& stats = nanotrack_profile_stats();
    stats.track_calls += 1;
    stats.track_crop_ms += crop_ms;
    stats.track_update_ms += update_ms;
    stats.track_tail_ms += tail_ms;
    stats.track_total_ms += total_ms;
    stats.update_copy_ms += g_last_update_timing.copy_ms;
    stats.update_x_run_ms += g_last_update_timing.x_run_ms;
    stats.update_pack_ms += g_last_update_timing.pack_ms;
    stats.update_h_run_ms += g_last_update_timing.h_run_ms;
    stats.update_post_ms += g_last_update_timing.post_ms;
    stats.update_total_ms += g_last_update_timing.total_ms;

    if ((stats.track_calls % 120) == 0)
    {
        const double inv = 1.0 / static_cast<double>(stats.track_calls);
        NN_LOG_INFO("[NanoTrackProfile] calls=%llu track(avg_ms): total=%.4f crop=%.4f update=%.4f tail=%.4f",
                    static_cast<unsigned long long>(stats.track_calls),
                    stats.track_total_ms * inv,
                    stats.track_crop_ms * inv,
                    stats.track_update_ms * inv,
                    stats.track_tail_ms * inv);
        NN_LOG_INFO("[NanoTrackProfile] calls=%llu update(avg_ms): total=%.4f copy=%.4f x_run=%.4f pack=%.4f h_run=%.4f post=%.4f",
                    static_cast<unsigned long long>(stats.track_calls),
                    stats.update_total_ms * inv,
                    stats.update_copy_ms * inv,
                    stats.update_x_run_ms * inv,
                    stats.update_pack_ms * inv,
                    stats.update_h_run_ms * inv,
                    stats.update_post_ms * inv);
    }
}

} // namespace

static inline float softmax_pos_2class_int8_lut(int8_t neg_q, int8_t pos_q, float scale)
{
    constexpr int kDiffRange = 255;
    constexpr int kLutSize = kDiffRange * 2 + 1;
    thread_local float cached_scale = std::numeric_limits<float>::quiet_NaN();
    thread_local std::array<float, kLutSize> lut{};

    if (!(scale > 0.0f))
    {
        return 0.5f;
    }

    if (!std::isfinite(cached_scale) || std::fabs(cached_scale - scale) > 1e-8f)
    {
        for (int diff = -kDiffRange; diff <= kDiffRange; ++diff)
        {
            const float x = static_cast<float>(diff) * scale;
            lut[static_cast<size_t>(diff + kDiffRange)] = 1.0f / (1.0f + std::exp(x));
        }
        cached_scale = scale;
    }

    int diff = static_cast<int>(neg_q) - static_cast<int>(pos_q);
    diff = std::max(-kDiffRange, std::min(kDiffRange, diff));
    return lut[static_cast<size_t>(diff + kDiffRange)];
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;
            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}
void transpose_nchw_to_nhwc(const tensor_data_s& input, tensor_data_s& output, float scale, int32_t zp) {
    const int N = input.attr.dims[0];
    const int C = input.attr.dims[1];
    const int H = input.attr.dims[2];
    const int W = input.attr.dims[3];

    if (input.attr.type == NN_TENSOR_FLOAT && output.attr.type == NN_TENSOR_FLOAT)
    {
        const float* src = static_cast<const float*>(input.data);
        float* dst = static_cast<float*>(output.data);
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        const int src_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                        const int dst_idx = n * (H * W * C) + h * (W * C) + w * C + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        return;
    }

    if (input.attr.type == NN_TENSOR_INT8 && output.attr.type == NN_TENSOR_INT8)
    {
        const int8_t* src = static_cast<const int8_t*>(input.data);
        int8_t* dst = static_cast<int8_t*>(output.data);
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        const int src_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                        const int dst_idx = n * (H * W * C) + h * (W * C) + w * C + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        return;
    }

    if (input.attr.type == NN_TENSOR_INT8 && output.attr.type == NN_TENSOR_FLOAT)
    {
        const int8_t* src = static_cast<const int8_t*>(input.data);
        float* dst = static_cast<float*>(output.data);
        const float deq_scale = (scale == 0.0f) ? 1.0f : scale;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        const int src_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                        const int dst_idx = n * (H * W * C) + h * (W * C) + w * C + c;
                        dst[dst_idx] = (static_cast<float>(src[src_idx]) - static_cast<float>(zp)) * deq_scale;
                    }
                }
            }
        }
        return;
    }

    throw std::runtime_error("transpose_nchw_to_nhwc: unsupported tensor type combination");
}

static inline int8_t requant_int8_value(int8_t src_q, float src_scale, int32_t src_zp, float dst_scale, int32_t dst_zp)
{
    const float src_s = (src_scale == 0.0f) ? 1.0f : src_scale;
    const float dst_s = (dst_scale == 0.0f) ? 1.0f : dst_scale;
    const float real_val = (static_cast<float>(src_q) - static_cast<float>(src_zp)) * src_s;
    int32_t q = static_cast<int32_t>(std::nearbyint(real_val / dst_s + static_cast<float>(dst_zp)));
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

static void pack_nchw_int8_to_nc1hwc2_requant(const tensor_data_s& src_nchw,
                                               tensor_data_s& dst_nc1hwc2,
                                               float src_scale,
                                               int32_t src_zp,
                                               float dst_scale,
                                               int32_t dst_zp)
{
    if (src_nchw.attr.type != NN_TENSOR_INT8 || dst_nc1hwc2.attr.type != NN_TENSOR_INT8)
    {
        throw std::runtime_error("pack_nchw_int8_to_nc1hwc2_requant: only INT8 tensors are supported");
    }

    const int N = static_cast<int>(src_nchw.attr.dims[0]);
    const int C = static_cast<int>(src_nchw.attr.dims[1]);
    const int H = static_cast<int>(src_nchw.attr.dims[2]);
    const int W = static_cast<int>(src_nchw.attr.dims[3]);
    constexpr int C2 = 16;
    const int C1 = (C + C2 - 1) / C2;

    const size_t needed = static_cast<size_t>(N) * C1 * H * W * C2;
    if (dst_nc1hwc2.attr.size < needed)
    {
        throw std::runtime_error("pack_nchw_int8_to_nc1hwc2_requant: destination buffer is too small");
    }

    const int8_t* src = static_cast<const int8_t*>(src_nchw.data);
    int8_t* dst = static_cast<int8_t*>(dst_nc1hwc2.data);
    const int8_t dst_fill = static_cast<int8_t>(std::max(-128, std::min(127, dst_zp)));
    std::fill(dst, dst + dst_nc1hwc2.attr.size, dst_fill);

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            const int c1 = c / C2;
            const int c2 = c % C2;
            for (int h = 0; h < H; ++h)
            {
                for (int w = 0; w < W; ++w)
                {
                    const size_t src_idx = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
                    const size_t dst_idx = ((((static_cast<size_t>(n) * C1 + c1) * H + h) * W + w) * C2) + c2;
                    dst[dst_idx] = requant_int8_value(src[src_idx], src_scale, src_zp, dst_scale, dst_zp);
                }
            }
        }
    }
}

static void copy_bgr_to_input_tensor(const cv::Mat& src_bgr, tensor_data_s& tensor)
{
    if (src_bgr.empty())
    {
        throw std::runtime_error("copy_bgr_to_input_tensor: empty input image");
    }

    const int n = static_cast<int>(tensor.attr.dims[0]);
    const int h = static_cast<int>(tensor.attr.dims[1]);
    const int w = static_cast<int>(tensor.attr.dims[2]);
    const int c = static_cast<int>(tensor.attr.dims[3]);
    if (n != 1 || c != src_bgr.channels() || h != src_bgr.rows || w != src_bgr.cols)
    {
        throw std::runtime_error("copy_bgr_to_input_tensor: tensor/image shape mismatch");
    }

    const size_t src_row_bytes = static_cast<size_t>(src_bgr.cols * src_bgr.channels());
    const uint8_t* src_base = static_cast<const uint8_t*>(src_bgr.data);

    const size_t default_row_bytes = static_cast<size_t>(w * c);
    size_t dst_row_bytes = default_row_bytes;
    const size_t denom = static_cast<size_t>(n) * static_cast<size_t>(h) * static_cast<size_t>(c);
    if (denom > 0 && tensor.attr.size % denom == 0)
    {
        const size_t inferred_w_stride = tensor.attr.size / denom;
        if (inferred_w_stride >= static_cast<size_t>(w))
        {
            dst_row_bytes = inferred_w_stride * static_cast<size_t>(c);
        }
    }

    if (tensor.attr.type == NN_TENSOR_UINT8)
    {
        uint8_t* dst = static_cast<uint8_t*>(tensor.data);
        memset(dst, 0, tensor.attr.size);
        for (int row = 0; row < h; ++row)
        {
            const uint8_t* src_row = src_base + row * src_bgr.step;
            uint8_t* dst_row = dst + static_cast<size_t>(row) * dst_row_bytes;
            memcpy(dst_row, src_row, src_row_bytes);
        }
        return;
    }

    if (tensor.attr.type == NN_TENSOR_INT8)
    {
        const float scale = (tensor.attr.scale == 0.0f) ? 1.0f : tensor.attr.scale;
        const int32_t zp = tensor.attr.zp;
        int8_t* dst = static_cast<int8_t*>(tensor.data);
        memset(dst, 0, tensor.attr.size);
        for (int row = 0; row < h; ++row)
        {
            const uint8_t* src_row = src_base + row * src_bgr.step;
            int8_t* dst_row = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(dst) + static_cast<size_t>(row) * dst_row_bytes);
            for (int col = 0; col < w * c; ++col)
            {
                int q = static_cast<int>(std::round(static_cast<float>(src_row[col]) / scale + static_cast<float>(zp)));
                q = std::max(-128, std::min(127, q));
                dst_row[col] = static_cast<int8_t>(q);
            }
        }
        return;
    }

    if (tensor.attr.type == NN_TENSOR_FLOAT)
    {
        float* dst = static_cast<float*>(tensor.data);
        const int elems = static_cast<int>(src_bgr.total() * src_bgr.channels());
        for (int i = 0; i < elems; ++i)
        {
            dst[i] = static_cast<float>(src_base[i]);
        }
        return;
    }

    throw std::runtime_error("copy_bgr_to_input_tensor: unsupported input tensor type");
}

static inline float tensor_value_to_f32(const tensor_data_s& tensor, int idx, float scale, int32_t zp)
{
    switch (tensor.attr.type)
    {
    case NN_TENSOR_FLOAT:
        return static_cast<const float*>(tensor.data)[idx];
    case NN_TENSOR_INT8:
        return (static_cast<float>(static_cast<const int8_t*>(tensor.data)[idx]) - static_cast<float>(zp)) * scale;
    case NN_TENSOR_UINT8:
        return (static_cast<float>(static_cast<const uint8_t*>(tensor.data)[idx]) - static_cast<float>(zp)) * scale;
    default:
        throw std::runtime_error("tensor_value_to_f32: unsupported tensor type");
    }
}

NanoTrackCore::NanoTrackCore()
{
    t_engine_ = CreateRKNNEngine();
    x_engine_ = CreateRKNNEngine();
    h_engine_ = CreateRKNNEngine();
    t_input_tensor_.data = nullptr;
    x_input_tensor_.data = nullptr;
    h_input_tensor_1.data = nullptr;
    h_input_tensor_2.data = nullptr;
}
// Destructor / 详见英文原注释。
NanoTrackCore::~NanoTrackCore()
{
    if (t_input_tensor_.data != nullptr)
    {
        free(t_input_tensor_.data);
        t_input_tensor_.data = nullptr;
    }
    if (x_input_tensor_.data != nullptr)
    {
        free(x_input_tensor_.data);
        x_input_tensor_.data = nullptr;
    }
    if (h_input_tensor_1.data != nullptr)
    {
        free(h_input_tensor_1.data);
        h_input_tensor_1.data = nullptr;
    }
    if (h_input_tensor_2.data != nullptr)
    {
        free(h_input_tensor_2.data);
        h_input_tensor_2.data = nullptr;
    }

    for (auto &tensor : t_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
    for (auto &tensor : x_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
    for (auto &tensor : h_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
}

nn_error_e NanoTrackCore::LoadModel(const char *modelTName,const char *modelXName, const char *modelHName)
{
    // Load template model / Load template 模式l
    auto ret = t_engine_->LoadModelFile(modelTName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelTName load model file failed");
        return ret;
    }
    // get input tensor / get 输入 tensor
    auto t_input_shapes = t_engine_->GetInputShapes();

    nn_tensor_attr_to_cvimg_input_data(t_input_shapes[0], t_input_tensor_);
    t_input_tensor_.data = malloc(t_input_tensor_.attr.size);

    auto output_shapes = t_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16 || output_shapes[0].type == NN_TENSOR_FLOAT)
    {
        t_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (size_t i = 0; i < output_shapes.size(); ++i)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (uint32_t j = 0; j < output_shapes[i].n_dims; ++j)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type; / tensor.attr.type = 输出_形状[i].type;
        tensor.attr.type = t_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.layout = output_shapes[i].layout;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        t_output_tensors_.push_back(tensor);
        t_out_zps_.push_back(output_shapes[i].zp);
        t_out_scales_.push_back(output_shapes[i].scale);
    }

    // Load search model / Load search 模式l
    ret = x_engine_->LoadModelFile(modelXName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelXName load model file failed");
        return ret;
    }
    // get input tensor / get 输入 tensor
    auto x_input_shapes = x_engine_->GetInputShapes();

    nn_tensor_attr_to_cvimg_input_data(x_input_shapes[0], x_input_tensor_);
    x_input_tensor_.data = malloc(x_input_tensor_.attr.size);

    output_shapes = x_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16 || output_shapes[0].type == NN_TENSOR_FLOAT)
    {
        x_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (size_t i = 0; i < output_shapes.size(); ++i)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (uint32_t j = 0; j < output_shapes[i].n_dims; ++j)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type; / tensor.attr.type = 输出_形状[i].type;
        tensor.attr.type = x_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.layout = output_shapes[i].layout;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        x_output_tensors_.push_back(tensor);
        x_out_zps_.push_back(output_shapes[i].zp);
        x_out_scales_.push_back(output_shapes[i].scale);
    }

    // Load head model / Load head 模式l
    ret = h_engine_->LoadModelFile(modelHName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelHName load model file failed");
        return ret;
    }

    // get input tensor / get 输入 tensor
    auto h_input_shapes = h_engine_->GetInputShapes();

    nn_tensor_attr_to_tensor_input_data(h_input_shapes[0], h_input_tensor_1);
    h_input_tensor_1.data = malloc(h_input_tensor_1.attr.size);

    nn_tensor_attr_to_tensor_input_data(h_input_shapes[1], h_input_tensor_2);
    h_input_tensor_2.data = malloc(h_input_tensor_2.attr.size);

    output_shapes = h_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16 || output_shapes[0].type == NN_TENSOR_FLOAT)
    {
        h_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (size_t i = 0; i < output_shapes.size(); ++i)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (uint32_t j = 0; j < output_shapes[i].n_dims; ++j)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type; / tensor.attr.type = 输出_形状[i].type;
        tensor.attr.type = h_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.layout = output_shapes[i].layout;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        h_output_tensors_.push_back(tensor);
        h_out_zps_.push_back(output_shapes[i].zp);
        h_out_scales_.push_back(output_shapes[i].scale);
    }

    if (!t_out_scales_.empty() && !x_out_scales_.empty())
    {
        NN_LOG_INFO("[NanoTrack] Requant map: T(out zp=%d scale=%.6f) -> H1(in zp=%d scale=%.6f), X(out zp=%d scale=%.6f) -> H2(in zp=%d scale=%.6f)",
                    t_out_zps_[0], t_out_scales_[0],
                    h_input_tensor_1.attr.zp, h_input_tensor_1.attr.scale,
                    x_out_zps_[0], x_out_scales_[0],
                    h_input_tensor_2.attr.zp, h_input_tensor_2.attr.scale);
    }
    return NN_SUCCESS;
}


// Build the cosine window / 构建 cosine window
void NanoTrackCore::create_window()
{
    int score_size= cfg.score_size;
    std::vector<float> hanning(score_size,0);
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    } 
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j]; 
        }
    }    
}

// Build the cosine window / 构建 cosine window
void NanoTrackCore::create_grids()
{
    /*
    each element of feature map on input search image / 输入搜索图像上特征图的每个元素位置。
    :return: H*W*2 (position for each element) / 返回 H*W*2，表示每个元素的位置。
    */
    int sz = cfg.score_size;   //16x16

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    const float ori = -static_cast<float>(sz / 2) * static_cast<float>(cfg.total_stride);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            // Centered point grid, consistent with official NanoTrack Python implementation. / Centered point grid, consistent 与 official Nano跟踪 Python 实现.
            this->grid_to_search_x[i * sz + j] = ori + static_cast<float>(j * cfg.total_stride);
            this->grid_to_search_y[i * sz + j] = ori + static_cast<float>(i * cfg.total_stride);
        }
    }
}

void NanoTrackCore::init(const cv::Mat &img, cv::Rect bbox)
{

    create_window(); 
    create_grids(); 

    cv::Point2f target_pos ={0.f, 0.f}; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h

    target_pos.x = bbox.x + (bbox.width - 1) * 0.5f; 
    target_pos.y = bbox.y + (bbox.height - 1) * 0.5f;
    target_sz.x=bbox.width;
    target_sz.y=bbox.height;
    
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));  

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    
    z_crop = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z), avg_chans);

    copy_bgr_to_input_tensor(z_crop, t_input_tensor_);
    std::vector<tensor_data_s> t_inputs;
    t_inputs.push_back(t_input_tensor_);
    // Run the template backbone.
    const nn_error_e t_ret = t_engine_->Run(t_inputs, t_output_tensors_, t_want_float_);
    if (t_ret != NN_SUCCESS)
    {
        throw std::runtime_error("NanoTrackCore::init template backbone run failed, ret=" + std::to_string(static_cast<int>(t_ret)));
    }

    this->state.channel_ave=avg_chans;
    this->state.im_h=img.rows;
    this->state.im_w=img.cols;
    this->state.target_pos=target_pos;
    this->state.target_sz= target_sz;
}


float NanoTrackCore::track(const cv::Mat &img)
{
    const bool do_profile = nanotrack_profile_enabled();
    SteadyClock::time_point track_start;
    SteadyClock::time_point crop_start;
    SteadyClock::time_point update_start;
    if (do_profile)
    {
        track_start = SteadyClock::now();
        crop_start = track_start;
    }

    cv::Point2f target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;
    
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  
    float scale_z = cfg.exemplar_size / s_z;  

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2; 
    float pad = d_search / scale_z; 
    float s_x = s_z + 2*pad;

    cv::Mat x_crop;  
    x_crop  = get_subwindow_tracking(img, target_pos, cfg.instance_size, int(s_x),state.channel_ave);
    SteadyClock::time_point crop_end;
    if (do_profile)
    {
        crop_end = SteadyClock::now();
        update_start = crop_end;
    }

    // update / 详见英文原注释。
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;
    
    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);
    SteadyClock::time_point update_end;
    if (do_profile)
    {
        update_end = SteadyClock::now();
    }

    target_pos.x = std::max(0.0f, std::min(static_cast<float>(state.im_w), target_pos.x));
    target_pos.y = std::max(0.0f, std::min(static_cast<float>(state.im_h), target_pos.y));
    target_sz.x = std::max(10.0f, std::min(static_cast<float>(state.im_w), target_sz.x));
    target_sz.y = std::max(10.0f, std::min(static_cast<float>(state.im_h), target_sz.y));

    // get max confidence and update target state / get max confidence 与 update target 状态
    state.cls_score_max = cls_score_max;
    state.target_pos = target_pos;
    state.target_sz = target_sz;
    
    float cx = target_pos.x;
    float cy = target_pos.y;
    float w = target_sz.x;
    float h = target_sz.y;

    state.bbox = cv::Rect (
        static_cast<int>(cx - w / 2),
        static_cast<int>(cy - h / 2),
        static_cast<int>(w),
        static_cast<int>(h)
    );

    if (do_profile)
    {
        const SteadyClock::time_point track_end = SteadyClock::now();
        nanotrack_profile_record(
            elapsed_ms(crop_start, crop_end),
            elapsed_ms(update_start, update_end),
            elapsed_ms(update_end, track_end),
            elapsed_ms(track_start, track_end));
    }
   
    return cls_score_max;
}


cv::Mat NanoTrackCore::get_subwindow_tracking(const cv::Mat &im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = pos.x - c + 0.5;
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = pos.y - c + 0.5;
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
       
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));
    return im_path; 
}



void NanoTrackCore::update(const cv::Mat &x_crops, cv::Point2f &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max)
{
    const bool do_profile = nanotrack_profile_enabled();
    NanoTrackUpdateTiming timing;
    SteadyClock::time_point update_start;
    if (do_profile)
    {
        update_start = SteadyClock::now();
    }

    SteadyClock::time_point stage_start;
    if (do_profile)
    {
        stage_start = SteadyClock::now();
    }
    copy_bgr_to_input_tensor(x_crops, x_input_tensor_);
    if (do_profile)
    {
        timing.copy_ms = elapsed_ms(stage_start, SteadyClock::now());
    }
    std::vector<tensor_data_s> x_inputs;
    x_inputs.push_back(x_input_tensor_);
    if (do_profile)
    {
        stage_start = SteadyClock::now();
    }
    if (x_engine_->Run(x_inputs, x_output_tensors_, x_want_float_) != NN_SUCCESS)
    {
        throw std::runtime_error("NanoTrackCore::update search backbone run failed");
    }
    if (do_profile)
    {
        timing.x_run_ms = elapsed_ms(stage_start, SteadyClock::now());
        stage_start = SteadyClock::now();
    }

    pack_nchw_int8_to_nc1hwc2_requant(
        t_output_tensors_[0], h_input_tensor_1,
        t_out_scales_[0], t_out_zps_[0],
        h_input_tensor_1.attr.scale, h_input_tensor_1.attr.zp);
    pack_nchw_int8_to_nc1hwc2_requant(
        x_output_tensors_[0], h_input_tensor_2,
        x_out_scales_[0], x_out_zps_[0],
        h_input_tensor_2.attr.scale, h_input_tensor_2.attr.zp);
    if (do_profile)
    {
        timing.pack_ms = elapsed_ms(stage_start, SteadyClock::now());
    }

    std::vector<tensor_data_s> h_inputs;
    h_inputs.push_back(h_input_tensor_1);
    h_inputs.push_back(h_input_tensor_2);

    if (do_profile)
    {
        stage_start = SteadyClock::now();
    }
    if (h_engine_->Run(h_inputs, h_output_tensors_, h_want_float_) != NN_SUCCESS)
    {
        throw std::runtime_error("NanoTrackCore::update head run failed");
    }
    if (do_profile)
    {
        timing.h_run_ms = elapsed_ms(stage_start, SteadyClock::now());
        stage_start = SteadyClock::now();
    }

    int channels_cls = h_output_tensors_[0].attr.dims[1];
    int rows = h_output_tensors_[0].attr.dims[2];
    int cols = h_output_tensors_[0].attr.dims[3];
    int channel_size = rows * cols;

    if (channels_cls < 2)
    {
        throw std::runtime_error("NanoTrackCore::update cls output channel must be >= 2");
    }

    std::vector<float> cls_scores;
    cls_scores.reserve(channel_size);
    for (int i = 0; i < channel_size; i++)
    {
        const float cls_neg = tensor_value_to_f32(h_output_tensors_[0], i, h_out_scales_[0], h_out_zps_[0]);
        const float cls_pos = tensor_value_to_f32(h_output_tensors_[0], channel_size + i, h_out_scales_[0], h_out_zps_[0]);
        cls_scores.push_back(softmax_pos_2class(cls_neg, cls_pos));
    }

    std::vector<float> pred_x1(channel_size), pred_y1(channel_size), pred_x2(channel_size), pred_y2(channel_size);
    std::vector<float> w(cols * rows, 0), h(cols * rows, 0);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            const int base_idx = i * cols + j;
            const float d0 = tensor_value_to_f32(h_output_tensors_[1], base_idx + channel_size * 0, h_out_scales_[1], h_out_zps_[1]);
            const float d1 = tensor_value_to_f32(h_output_tensors_[1], base_idx + channel_size * 1, h_out_scales_[1], h_out_zps_[1]);
            const float d2 = tensor_value_to_f32(h_output_tensors_[1], base_idx + channel_size * 2, h_out_scales_[1], h_out_zps_[1]);
            const float d3 = tensor_value_to_f32(h_output_tensors_[1], base_idx + channel_size * 3, h_out_scales_[1], h_out_zps_[1]);

            pred_x1[base_idx] = this->grid_to_search_x[base_idx] - d0;
            pred_y1[base_idx] = this->grid_to_search_y[base_idx] - d1;
            pred_x2[base_idx] = this->grid_to_search_x[base_idx] + d2;
            pred_y2[base_idx] = this->grid_to_search_y[base_idx] + d3;

            w[base_idx] = pred_x2[base_idx] - pred_x1[base_idx];
            h[base_idx] = pred_y2[base_idx] - pred_y1[base_idx];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows * cols, 0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i] - 1) * cfg.penalty_k);
    }

    std::vector<float> pscore(rows * cols, 0);
    float maxScore = 0;
    int max_idx = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_scores[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence;
        if (pscore[i] > maxScore)
        {
            maxScore = pscore[i];
            max_idx = i;
        }
    }

    float pred_x1_real = pred_x1[max_idx];
    float pred_y1_real = pred_y1[max_idx];
    float pred_x2_real = pred_x2[max_idx];
    float pred_y2_real = pred_y2[max_idx];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs;
    float diff_ys = pred_ys;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    float lr = penalty[max_idx] * cls_scores[max_idx] * cfg.lr;

    auto res_xs = float(target_pos.x + diff_xs);
    auto res_ys = float(target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = res_xs;
    target_pos.y = res_ys;
    target_sz.x = res_w;
    target_sz.y = res_h;
    cls_score_max = cls_scores[max_idx];

    if (do_profile)
    {
        const SteadyClock::time_point update_end = SteadyClock::now();
        timing.post_ms = elapsed_ms(stage_start, update_end);
        timing.total_ms = elapsed_ms(update_start, update_end);
        g_last_update_timing = timing;
    }
}

