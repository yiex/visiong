// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/models/mlsd.h"

#include "internal/rknn_model_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

enum class SpatialOrder {
    NHWC,
    NHCW,
    NCHW,
};

struct SpatialTensorLayout {
    int map_h = 0;
    int map_w = 0;
    int channels = 0;
    SpatialOrder order = SpatialOrder::NHWC;

    int offset(int y, int x, int c) const {
        switch (order) {
            case SpatialOrder::NHWC:
                return ((y * map_w) + x) * channels + c;
            case SpatialOrder::NHCW:
                return ((y * channels) + c) * map_w + x;
            case SpatialOrder::NCHW:
                return ((c * map_h) + y) * map_w + x;
        }
        return 0;
    }
};

float fp16_to_f32(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    uint32_t exp = (static_cast<uint32_t>(value) >> 10U) & 0x1FU;
    uint32_t mant = static_cast<uint32_t>(value) & 0x03FFU;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1U;
                --exp;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + (127U - 15U)) << 23U) | (mant << 13U);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13U);
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23U) | (mant << 13U);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float tensor_value_as_f32(const rknn_tensor_attr& attr, const void* data, int index) {
    if (data == nullptr || index < 0 || static_cast<uint32_t>(index) >= attr.n_elems) {
        return 0.0f;
    }
    switch (attr.type) {
        case RKNN_TENSOR_FLOAT32:
            return static_cast<const float*>(data)[index];
        case RKNN_TENSOR_FLOAT16:
            return fp16_to_f32(static_cast<const uint16_t*>(data)[index]);
        case RKNN_TENSOR_INT8:
            if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
                return (static_cast<int>(static_cast<const int8_t*>(data)[index]) - attr.zp) * attr.scale;
            }
            return static_cast<float>(static_cast<const int8_t*>(data)[index]);
        case RKNN_TENSOR_UINT8:
            if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
                return (static_cast<int>(static_cast<const uint8_t*>(data)[index]) - attr.zp) * attr.scale;
            }
            return static_cast<float>(static_cast<const uint8_t*>(data)[index]);
        case RKNN_TENSOR_INT32:
            if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
                return (static_cast<int>(static_cast<const int32_t*>(data)[index]) - attr.zp) * attr.scale;
            }
            return static_cast<float>(static_cast<const int32_t*>(data)[index]);
        default:
            return 0.0f;
    }
}

int tensor_value_as_i32(const rknn_tensor_attr& attr, const void* data, int index) {
    return static_cast<int>(std::lround(tensor_value_as_f32(attr, data, index)));
}

bool attr_has_last_dim(const rknn_tensor_attr& attr, int value) {
    return attr.n_dims > 0 && static_cast<int>(attr.dims[attr.n_dims - 1]) == value;
}

bool resolve_spatial_layout(const rknn_tensor_attr& attr,
                            int min_channels,
                            int preferred_channels,
                            SpatialTensorLayout* layout) {
    if (layout == nullptr || attr.n_dims < 4) {
        return false;
    }
    const int d1 = static_cast<int>(attr.dims[1]);
    const int d2 = static_cast<int>(attr.dims[2]);
    const int d3 = static_cast<int>(attr.dims[3]);
    if (d1 <= 0 || d2 <= 0 || d3 <= 0) {
        return false;
    }

    if (preferred_channels > 0) {
        if (d3 == preferred_channels) {
            *layout = {d1, d2, d3, SpatialOrder::NHWC};
            return true;
        }
        if (d2 == preferred_channels) {
            *layout = {d1, d3, d2, SpatialOrder::NHCW};
            return true;
        }
        if (d1 == preferred_channels) {
            *layout = {d2, d3, d1, SpatialOrder::NCHW};
            return true;
        }
    }

    if (d3 >= min_channels && d1 == d2) {
        *layout = {d1, d2, d3, SpatialOrder::NHWC};
        return true;
    }
    if (d2 >= min_channels && d1 == d3) {
        *layout = {d1, d3, d2, SpatialOrder::NHCW};
        return true;
    }
    if (d1 >= min_channels && d2 == d3) {
        *layout = {d2, d3, d1, SpatialOrder::NCHW};
        return true;
    }
    return false;
}

struct MlsdOutputLayout {
    int pts = -1;
    int scores = -1;
    int vmap = -1;
    int topk = 0;
    SpatialTensorLayout vmap_layout;
};

bool resolve_mlsd_outputs(const rknn_app_context_t* app_ctx, MlsdOutputLayout* layout) {
    if (app_ctx == nullptr || layout == nullptr || app_ctx->io_num.n_output < 3 || app_ctx->output_attrs == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        SpatialTensorLayout vmap_layout;
        if (attr.n_elems > 0 && resolve_spatial_layout(attr, 4, 4, &vmap_layout)) {
            layout->vmap = static_cast<int>(i);
            layout->vmap_layout = vmap_layout;
            continue;
        }
        if (attr.n_elems > 0 && attr_has_last_dim(attr, 2)) {
            layout->pts = static_cast<int>(i);
            layout->topk = attr.n_elems / 2;
            continue;
        }
    }
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        if (static_cast<int>(i) == layout->pts || static_cast<int>(i) == layout->vmap) {
            continue;
        }
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        if (layout->topk <= 0 || attr.n_elems == static_cast<uint32_t>(layout->topk)) {
            layout->scores = static_cast<int>(i);
            if (layout->topk <= 0) {
                layout->topk = attr.n_elems;
            }
            break;
        }
    }
    return layout->pts >= 0 && layout->scores >= 0 && layout->vmap >= 0 && layout->topk > 0 &&
           layout->vmap_layout.map_h > 0 && layout->vmap_layout.map_w > 0;
}

struct MlsdCandidate {
    int y = 0;
    int x = 0;
    float score = 0.0f;
};

struct MlsdUpsampledMaps {
    int map_h = 0;
    int map_w = 0;
    std::vector<float> center;
    std::vector<float> disp;

    float disp_at(int y, int x, int c) const {
        return disp[((y * map_w) + x) * 4 + c];
    }
};

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

float logit_threshold(float threshold) {
    const float clamped = std::max(1.0e-6f, std::min(threshold, 1.0f - 1.0e-6f));
    return std::log(clamped / (1.0f - clamped));
}

int quant_threshold_gt(const rknn_tensor_attr& attr, float threshold) {
    const float raw_threshold = logit_threshold(threshold);
    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && attr.scale > 0.0f) {
        return static_cast<int>(std::floor(raw_threshold / attr.scale + static_cast<float>(attr.zp)));
    }
    return static_cast<int>(std::floor(raw_threshold));
}

float dequant_i8_score(const rknn_tensor_attr& attr, int8_t value) {
    float raw = static_cast<float>(value);
    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
        raw = (static_cast<int>(value) - attr.zp) * attr.scale;
    }
    return sigmoid(raw);
}

float dequant_u8_score(const rknn_tensor_attr& attr, uint8_t value) {
    float raw = static_cast<float>(value);
    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
        raw = (static_cast<int>(value) - attr.zp) * attr.scale;
    }
    return sigmoid(raw);
}

using QuantScoreLut = std::array<float, 256>;

QuantScoreLut make_int8_score_lut(const rknn_tensor_attr& attr) {
    QuantScoreLut lut{};
    for (int i = -128; i <= 127; ++i) {
        lut[static_cast<size_t>(i + 128)] = dequant_i8_score(attr, static_cast<int8_t>(i));
    }
    return lut;
}

QuantScoreLut make_uint8_score_lut(const rknn_tensor_attr& attr) {
    QuantScoreLut lut{};
    for (int i = 0; i <= 255; ++i) {
        lut[static_cast<size_t>(i)] = dequant_u8_score(attr, static_cast<uint8_t>(i));
    }
    return lut;
}

template <typename ValueT>
size_t quant_lut_index(ValueT value) {
    if constexpr (std::is_same<ValueT, int8_t>::value) {
        return static_cast<size_t>(static_cast<int>(value) + 128);
    } else {
        return static_cast<size_t>(value);
    }
}

template <typename ValueT>
bool is_quant_peak(const ValueT* data, const SpatialTensorLayout& layout, int y, int x, int channel) {
    const ValueT center = data[layout.offset(y, x, channel)];
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= layout.map_h) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if ((dx == 0 && dy == 0) || xx < 0 || xx >= layout.map_w) {
                continue;
            }
            if (data[layout.offset(yy, xx, channel)] > center) {
                return false;
            }
        }
    }
    return true;
}

template <typename ValueT>
void collect_quant_candidates_scalar(const ValueT* data,
                                     const SpatialTensorLayout& layout,
                                     int center_channel,
                                     int threshold_quant,
                                     float score_threshold,
                                     const QuantScoreLut& score_lut,
                                     std::vector<MlsdCandidate>* candidates) {
    for (int y = 0; y < layout.map_h; ++y) {
        for (int x = 0; x < layout.map_w; ++x) {
            const ValueT center = data[layout.offset(y, x, center_channel)];
            if (static_cast<int>(center) <= threshold_quant || !is_quant_peak(data, layout, y, x, center_channel)) {
                continue;
            }
            const float score = score_lut[quant_lut_index(center)];
            if (score > score_threshold) {
                candidates->push_back({y, x, score});
            }
        }
    }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void collect_nhcw_int8_candidates_neon(const int8_t* data,
                                       const SpatialTensorLayout& layout,
                                       int center_channel,
                                       int threshold_quant,
                                       float score_threshold,
                                       const QuantScoreLut& score_lut,
                                       std::vector<MlsdCandidate>* candidates) {
    if (threshold_quant >= 127) {
        return;
    }
    const bool skip_threshold_compare = threshold_quant < -128;
    const int8x16_t threshold = vdupq_n_s8(static_cast<int8_t>(std::max(-128, threshold_quant)));

    auto try_push_scalar = [&](int y, int x) {
        const int idx = layout.offset(y, x, center_channel);
        const int8_t center = data[idx];
        if (static_cast<int>(center) <= threshold_quant ||
            !is_quant_peak(data, layout, y, x, center_channel)) {
            return;
        }
        const float score = score_lut[static_cast<size_t>(static_cast<int>(center) + 128)];
        if (score > score_threshold) {
            candidates->push_back({y, x, score});
        }
    };

    if (layout.map_w < 3 || layout.map_h < 3) {
        for (int y = 0; y < layout.map_h; ++y) {
            for (int x = 0; x < layout.map_w; ++x) {
                try_push_scalar(y, x);
            }
        }
        return;
    }

    for (int y = 0; y < layout.map_h; ++y) {
        if (y == 0 || y == layout.map_h - 1) {
            for (int x = 0; x < layout.map_w; ++x) {
                try_push_scalar(y, x);
            }
            continue;
        }

        try_push_scalar(y, 0);

        const int prev_base = layout.offset(y - 1, 0, center_channel);
        const int row_base = layout.offset(y, 0, center_channel);
        const int next_base = layout.offset(y + 1, 0, center_channel);
        int x = 1;
        for (; x + 15 < layout.map_w - 1; x += 16) {
            const int8x16_t center = vld1q_s8(data + row_base + x);
            uint8x16_t mask = skip_threshold_compare ? vdupq_n_u8(0xFF) : vcgtq_s8(center, threshold);
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + row_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + row_base + x + 1)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + prev_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + prev_base + x)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + prev_base + x + 1)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + next_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + next_base + x)));
            mask = vandq_u8(mask, vcgeq_s8(center, vld1q_s8(data + next_base + x + 1)));
            const uint64x2_t mask64 = vreinterpretq_u64_u8(mask);
            if ((vgetq_lane_u64(mask64, 0) | vgetq_lane_u64(mask64, 1)) == 0) {
                continue;
            }
            uint8_t mask_bytes[16];
            vst1q_u8(mask_bytes, mask);
            for (int lane = 0; lane < 16; ++lane) {
                if (mask_bytes[lane] == 0) {
                    continue;
                }
                const int xx = x + lane;
                const int8_t q = data[row_base + xx];
                const float score = score_lut[static_cast<size_t>(static_cast<int>(q) + 128)];
                if (score > score_threshold) {
                    candidates->push_back({y, xx, score});
                }
            }
        }
        for (; x < layout.map_w; ++x) {
            try_push_scalar(y, x);
        }
    }
}

void collect_nhcw_uint8_candidates_neon(const uint8_t* data,
                                        const SpatialTensorLayout& layout,
                                        int center_channel,
                                        int threshold_quant,
                                        float score_threshold,
                                        const QuantScoreLut& score_lut,
                                        std::vector<MlsdCandidate>* candidates) {
    if (threshold_quant >= 255) {
        return;
    }
    const bool skip_threshold_compare = threshold_quant < 0;
    const uint8x16_t threshold = vdupq_n_u8(static_cast<uint8_t>(std::max(0, threshold_quant)));

    auto try_push_scalar = [&](int y, int x) {
        const int idx = layout.offset(y, x, center_channel);
        const uint8_t center = data[idx];
        if (static_cast<int>(center) <= threshold_quant ||
            !is_quant_peak(data, layout, y, x, center_channel)) {
            return;
        }
        const float score = score_lut[static_cast<size_t>(center)];
        if (score > score_threshold) {
            candidates->push_back({y, x, score});
        }
    };

    if (layout.map_w < 3 || layout.map_h < 3) {
        for (int y = 0; y < layout.map_h; ++y) {
            for (int x = 0; x < layout.map_w; ++x) {
                try_push_scalar(y, x);
            }
        }
        return;
    }

    for (int y = 0; y < layout.map_h; ++y) {
        if (y == 0 || y == layout.map_h - 1) {
            for (int x = 0; x < layout.map_w; ++x) {
                try_push_scalar(y, x);
            }
            continue;
        }

        try_push_scalar(y, 0);

        const int prev_base = layout.offset(y - 1, 0, center_channel);
        const int row_base = layout.offset(y, 0, center_channel);
        const int next_base = layout.offset(y + 1, 0, center_channel);
        int x = 1;
        for (; x + 15 < layout.map_w - 1; x += 16) {
            const uint8x16_t center = vld1q_u8(data + row_base + x);
            uint8x16_t mask = skip_threshold_compare ? vdupq_n_u8(0xFF) : vcgtq_u8(center, threshold);
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + row_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + row_base + x + 1)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + prev_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + prev_base + x)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + prev_base + x + 1)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + next_base + x - 1)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + next_base + x)));
            mask = vandq_u8(mask, vcgeq_u8(center, vld1q_u8(data + next_base + x + 1)));
            const uint64x2_t mask64 = vreinterpretq_u64_u8(mask);
            if ((vgetq_lane_u64(mask64, 0) | vgetq_lane_u64(mask64, 1)) == 0) {
                continue;
            }
            uint8_t mask_bytes[16];
            vst1q_u8(mask_bytes, mask);
            for (int lane = 0; lane < 16; ++lane) {
                if (mask_bytes[lane] == 0) {
                    continue;
                }
                const int xx = x + lane;
                const uint8_t q = data[row_base + xx];
                const float score = score_lut[static_cast<size_t>(q)];
                if (score > score_threshold) {
                    candidates->push_back({y, xx, score});
                }
            }
        }
        for (; x < layout.map_w; ++x) {
            try_push_scalar(y, x);
        }
    }
}
#endif

bool collect_quant_candidates_fast(const rknn_tensor_attr& raw_attr,
                                   const void* raw_data,
                                   const SpatialTensorLayout& layout,
                                   int center_channel,
                                   float score_threshold,
                                   std::vector<MlsdCandidate>* candidates) {
    if (raw_data == nullptr || candidates == nullptr) {
        return false;
    }
    const int threshold_quant = quant_threshold_gt(raw_attr, score_threshold);
    if (raw_attr.type == RKNN_TENSOR_INT8) {
        const auto* data = static_cast<const int8_t*>(raw_data);
        const QuantScoreLut score_lut = make_int8_score_lut(raw_attr);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (layout.order == SpatialOrder::NHCW) {
            collect_nhcw_int8_candidates_neon(data, layout, center_channel,
                                              threshold_quant, score_threshold, score_lut, candidates);
            return true;
        }
#endif
        collect_quant_candidates_scalar(data, layout, center_channel,
                                        threshold_quant, score_threshold, score_lut, candidates);
        return true;
    }
    if (raw_attr.type == RKNN_TENSOR_UINT8) {
        const auto* data = static_cast<const uint8_t*>(raw_data);
        const QuantScoreLut score_lut = make_uint8_score_lut(raw_attr);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (layout.order == SpatialOrder::NHCW) {
            collect_nhcw_uint8_candidates_neon(data, layout, center_channel,
                                               threshold_quant, score_threshold, score_lut, candidates);
            return true;
        }
#endif
        collect_quant_candidates_scalar(data, layout, center_channel,
                                        threshold_quant, score_threshold, score_lut, candidates);
        return true;
    }
    return false;
}

inline float dist2(float x1, float y1, float x2, float y2) {
    const float dx = x1 - x2;
    const float dy = y1 - y2;
    return dx * dx + dy * dy;
}

bool should_suppress_line(const LineSegment& candidate, const std::vector<LineSegment>& lines) {
    const float dx = candidate.x2 - candidate.x1;
    const float dy = candidate.y2 - candidate.y1;
    const float len = candidate.length > 0.0f ? candidate.length : std::sqrt(dx * dx + dy * dy);
    if (len < 1.0f) {
        return true;
    }
    const float len2 = len * len;
    const float cx = (candidate.x1 + candidate.x2) * 0.5f;
    const float cy = (candidate.y1 + candidate.y2) * 0.5f;
    constexpr float kCosAngleThreshold = 0.990268f;  // cos(8 deg)
    constexpr float kCosAngleThreshold2 = kCosAngleThreshold * kCosAngleThreshold;
    constexpr float kCenterDistThreshold = 14.0f;
    constexpr float kCenterDistThreshold2 = kCenterDistThreshold * kCenterDistThreshold;
    constexpr float kEndpointDistThreshold = 24.0f;

    for (const LineSegment& existing : lines) {
        const float ex = existing.x2 - existing.x1;
        const float ey = existing.y2 - existing.y1;
        const float elen = std::max(existing.length, 1.0f);
        const float dot = dx * ex + dy * ey;
        if (dot * dot < kCosAngleThreshold2 * len2 * elen * elen) {
            continue;
        }

        const float ecx = (existing.x1 + existing.x2) * 0.5f;
        const float ecy = (existing.y1 + existing.y2) * 0.5f;
        if (dist2(cx, cy, ecx, ecy) > kCenterDistThreshold2) {
            continue;
        }

        const float direct = std::sqrt(dist2(candidate.x1, candidate.y1, existing.x1, existing.y1)) +
                             std::sqrt(dist2(candidate.x2, candidate.y2, existing.x2, existing.y2));
        const float reverse = std::sqrt(dist2(candidate.x1, candidate.y1, existing.x2, existing.y2)) +
                              std::sqrt(dist2(candidate.x2, candidate.y2, existing.x1, existing.y1));
        if (std::min(direct, reverse) * 0.5f <= kEndpointDistThreshold) {
            return true;
        }
    }
    return false;
}

void push_line_if_distinct(LineSegment line, std::vector<LineSegment>* lines) {
    if (lines == nullptr || should_suppress_line(line, *lines)) {
        return;
    }
    lines->push_back(line);
}

bool build_upsampled_maps(const rknn_tensor_attr& raw_attr,
                          const void* raw_data,
                          const SpatialTensorLayout& src_layout,
                          int dst_w,
                          int dst_h,
                          int center_channel,
                          int disp_channel,
                          MlsdUpsampledMaps* out) {
    if (raw_data == nullptr || out == nullptr || dst_w <= 0 || dst_h <= 0) {
        return false;
    }

    constexpr int kChannels = 5;
    const size_t src_pixels = static_cast<size_t>(src_layout.map_h) * static_cast<size_t>(src_layout.map_w);
    std::vector<float> src(src_pixels * kChannels);
    for (int y = 0; y < src_layout.map_h; ++y) {
        for (int x = 0; x < src_layout.map_w; ++x) {
            const size_t base = (static_cast<size_t>(y) * src_layout.map_w + x) * kChannels;
            src[base + 0] = tensor_value_as_f32(raw_attr, raw_data, src_layout.offset(y, x, center_channel));
            src[base + 1] = tensor_value_as_f32(raw_attr, raw_data, src_layout.offset(y, x, disp_channel + 0));
            src[base + 2] = tensor_value_as_f32(raw_attr, raw_data, src_layout.offset(y, x, disp_channel + 1));
            src[base + 3] = tensor_value_as_f32(raw_attr, raw_data, src_layout.offset(y, x, disp_channel + 2));
            src[base + 4] = tensor_value_as_f32(raw_attr, raw_data, src_layout.offset(y, x, disp_channel + 3));
        }
    }

    out->map_h = dst_h;
    out->map_w = dst_w;
    out->center.assign(static_cast<size_t>(dst_h) * dst_w, 0.0f);
    out->disp.assign(static_cast<size_t>(dst_h) * dst_w * 4U, 0.0f);

    const float y_scale = static_cast<float>(src_layout.map_h) / static_cast<float>(dst_h);
    const float x_scale = static_cast<float>(src_layout.map_w) / static_cast<float>(dst_w);
    for (int y = 0; y < dst_h; ++y) {
        const float src_y_raw = (static_cast<float>(y) + 0.5f) * y_scale - 0.5f;
        const float src_y = std::max(0.0f, src_y_raw);
        const int y0 = std::min(static_cast<int>(std::floor(src_y)), src_layout.map_h - 1);
        const int y1 = std::min(y0 + 1, src_layout.map_h - 1);
        const float wy = src_y - static_cast<float>(y0);
        for (int x = 0; x < dst_w; ++x) {
            const float src_x_raw = (static_cast<float>(x) + 0.5f) * x_scale - 0.5f;
            const float src_x = std::max(0.0f, src_x_raw);
            const int x0 = std::min(static_cast<int>(std::floor(src_x)), src_layout.map_w - 1);
            const int x1 = std::min(x0 + 1, src_layout.map_w - 1);
            const float wx = src_x - static_cast<float>(x0);

            const size_t p00 = (static_cast<size_t>(y0) * src_layout.map_w + x0) * kChannels;
            const size_t p01 = (static_cast<size_t>(y0) * src_layout.map_w + x1) * kChannels;
            const size_t p10 = (static_cast<size_t>(y1) * src_layout.map_w + x0) * kChannels;
            const size_t p11 = (static_cast<size_t>(y1) * src_layout.map_w + x1) * kChannels;
            const size_t dst_idx = static_cast<size_t>(y) * dst_w + x;
            for (int c = 0; c < kChannels; ++c) {
                const float top = src[p00 + c] + (src[p01 + c] - src[p00 + c]) * wx;
                const float bottom = src[p10 + c] + (src[p11 + c] - src[p10 + c]) * wx;
                const float value = top + (bottom - top) * wy;
                if (c == 0) {
                    out->center[dst_idx] = value;
                } else {
                    out->disp[dst_idx * 4U + static_cast<size_t>(c - 1)] = value;
                }
            }
        }
    }
    return true;
}

void collect_float_candidates(const MlsdUpsampledMaps& maps,
                              float score_threshold,
                              std::vector<MlsdCandidate>* candidates) {
    if (candidates == nullptr) {
        return;
    }
    const float raw_threshold = logit_threshold(score_threshold);
    for (int y = 0; y < maps.map_h; ++y) {
        for (int x = 0; x < maps.map_w; ++x) {
            const size_t idx = static_cast<size_t>(y) * maps.map_w + x;
            const float center = maps.center[idx];
            if (center <= raw_threshold) {
                continue;
            }
            bool is_peak = true;
            for (int dy = -1; dy <= 1 && is_peak; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= maps.map_h) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = x + dx;
                    if ((dx == 0 && dy == 0) || xx < 0 || xx >= maps.map_w) {
                        continue;
                    }
                    if (maps.center[static_cast<size_t>(yy) * maps.map_w + xx] > center) {
                        is_peak = false;
                        break;
                    }
                }
            }
            if (is_peak) {
                candidates->push_back({y, x, sigmoid(center)});
            }
        }
    }
}

void decode_upsampled_line(int y,
                           int x,
                           float score,
                           const MlsdUpsampledMaps& maps,
                           int image_width,
                           int image_height,
                           float distance_threshold,
                           std::vector<LineSegment>* lines) {
    const float dx1 = maps.disp_at(y, x, 0);
    const float dy1 = maps.disp_at(y, x, 1);
    const float dx2 = maps.disp_at(y, x, 2);
    const float dy2 = maps.disp_at(y, x, 3);
    const float map_len = std::sqrt((dx1 - dx2) * (dx1 - dx2) + (dy1 - dy2) * (dy1 - dy2));
    if (map_len <= distance_threshold) {
        return;
    }

    const float x_scale = static_cast<float>(image_width) / static_cast<float>(maps.map_w);
    const float y_scale = static_cast<float>(image_height) / static_cast<float>(maps.map_h);
    LineSegment line;
    line.x1 = (static_cast<float>(x) + dx1) * x_scale;
    line.y1 = (static_cast<float>(y) + dy1) * y_scale;
    line.x2 = (static_cast<float>(x) + dx2) * x_scale;
    line.y2 = (static_cast<float>(y) + dy2) * y_scale;
    line.x1 = std::max(0.0f, std::min(line.x1, static_cast<float>(image_width - 1)));
    line.y1 = std::max(0.0f, std::min(line.y1, static_cast<float>(image_height - 1)));
    line.x2 = std::max(0.0f, std::min(line.x2, static_cast<float>(image_width - 1)));
    line.y2 = std::max(0.0f, std::min(line.y2, static_cast<float>(image_height - 1)));
    line.score = std::max(0.0f, std::min(score, 1.0f));
    line.length = std::sqrt((line.x1 - line.x2) * (line.x1 - line.x2) + (line.y1 - line.y2) * (line.y1 - line.y2));
    push_line_if_distinct(line, lines);
}

void decode_mlsd_line(int y,
                      int x,
                      float score,
                      const rknn_tensor_attr& vmap_attr,
                      const void* vmap_data,
                      const SpatialTensorLayout& layout,
                      int image_width,
                      int image_height,
                      int coord_map_w,
                      int coord_map_h,
                      float center_step_x,
                      float center_step_y,
                      int channel_base,
                      float distance_threshold,
                      std::vector<LineSegment>* lines) {
    const float dx1 = tensor_value_as_f32(vmap_attr, vmap_data, layout.offset(y, x, channel_base + 0));
    const float dy1 = tensor_value_as_f32(vmap_attr, vmap_data, layout.offset(y, x, channel_base + 1));
    const float dx2 = tensor_value_as_f32(vmap_attr, vmap_data, layout.offset(y, x, channel_base + 2));
    const float dy2 = tensor_value_as_f32(vmap_attr, vmap_data, layout.offset(y, x, channel_base + 3));
    const float map_len = std::sqrt((dx1 - dx2) * (dx1 - dx2) + (dy1 - dy2) * (dy1 - dy2));
    if (map_len <= distance_threshold) {
        return;
    }

    const float x_scale = static_cast<float>(image_width) / static_cast<float>(coord_map_w);
    const float y_scale = static_cast<float>(image_height) / static_cast<float>(coord_map_h);
    const float center_x = static_cast<float>(x) * center_step_x;
    const float center_y = static_cast<float>(y) * center_step_y;
    LineSegment line;
    line.x1 = (center_x + dx1) * x_scale;
    line.y1 = (center_y + dy1) * y_scale;
    line.x2 = (center_x + dx2) * x_scale;
    line.y2 = (center_y + dy2) * y_scale;
    line.x1 = std::max(0.0f, std::min(line.x1, static_cast<float>(image_width - 1)));
    line.y1 = std::max(0.0f, std::min(line.y1, static_cast<float>(image_height - 1)));
    line.x2 = std::max(0.0f, std::min(line.x2, static_cast<float>(image_width - 1)));
    line.y2 = std::max(0.0f, std::min(line.y2, static_cast<float>(image_height - 1)));
    line.score = std::max(0.0f, std::min(score, 1.0f));
    line.length = std::sqrt((line.x1 - line.x2) * (line.x1 - line.x2) + (line.y1 - line.y2) * (line.y1 - line.y2));
    push_line_if_distinct(line, lines);
}

int inference_mlsd_raw_map(rknn_app_context_t* app_ctx,
                           int image_width,
                           int image_height,
                           float score_threshold,
                           float distance_threshold,
                           std::vector<LineSegment>* lines) {
    const rknn_tensor_attr& raw_attr = app_ctx->output_attrs[0];
    const void* raw_data = app_ctx->output_mems[0]->virt_addr;
    SpatialTensorLayout raw_layout;
    if (!resolve_spatial_layout(raw_attr, 12, 16, &raw_layout)) {
        std::printf("ERROR: unexpected MLSD raw output layout.\n");
        return -1;
    }
    constexpr int kTopK = 200;
    constexpr int kOrgCenterChannel = 7;
    constexpr int kOrgDispChannel = 8;
    const int coord_map_w = std::max(raw_layout.map_w, app_ctx->model_width / 2);
    const int coord_map_h = std::max(raw_layout.map_h, app_ctx->model_height / 2);
    const float center_step_x = static_cast<float>(coord_map_w) / static_cast<float>(raw_layout.map_w);
    const float center_step_y = static_cast<float>(coord_map_h) / static_cast<float>(raw_layout.map_h);

    std::vector<MlsdCandidate> candidates;
    candidates.reserve(static_cast<size_t>(coord_map_h) * static_cast<size_t>(coord_map_w) / 16U);

    if (coord_map_w != raw_layout.map_w || coord_map_h != raw_layout.map_h) {
        MlsdUpsampledMaps maps;
        if (!build_upsampled_maps(raw_attr, raw_data, raw_layout, coord_map_w, coord_map_h,
                                  kOrgCenterChannel, kOrgDispChannel, &maps)) {
            return -1;
        }
        collect_float_candidates(maps, score_threshold, &candidates);

        auto by_score_desc = [](const MlsdCandidate& a, const MlsdCandidate& b) {
            return a.score > b.score;
        };
        if (candidates.size() > kTopK) {
            std::partial_sort(candidates.begin(), candidates.begin() + kTopK, candidates.end(), by_score_desc);
            candidates.resize(kTopK);
        } else {
            std::sort(candidates.begin(), candidates.end(), by_score_desc);
        }

        lines->reserve(candidates.size());
        for (const MlsdCandidate& candidate : candidates) {
            decode_upsampled_line(candidate.y,
                                  candidate.x,
                                  candidate.score,
                                  maps,
                                  image_width,
                                  image_height,
                                  distance_threshold,
                                  lines);
        }
        return 0;
    }

    if (!collect_quant_candidates_fast(raw_attr, raw_data, raw_layout, kOrgCenterChannel,
                                       score_threshold, &candidates)) {
        auto raw_center = [&](int yy, int xx) {
            return tensor_value_as_f32(raw_attr, raw_data, raw_layout.offset(yy, xx, kOrgCenterChannel));
        };

        for (int y = 0; y < raw_layout.map_h; ++y) {
            for (int x = 0; x < raw_layout.map_w; ++x) {
                const float center_raw = raw_center(y, x);
                bool is_peak = true;
                for (int dy = -1; dy <= 1 && is_peak; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= raw_layout.map_h) {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = x + dx;
                        if ((dx == 0 && dy == 0) || xx < 0 || xx >= raw_layout.map_w) {
                            continue;
                        }
                        if (raw_center(yy, xx) > center_raw) {
                            is_peak = false;
                            break;
                        }
                    }
                }
                if (!is_peak) {
                    continue;
                }
                const float score = sigmoid(center_raw);
                if (score > score_threshold) {
                    candidates.push_back({y, x, score});
                }
            }
        }
    }
    auto by_score_desc = [](const MlsdCandidate& a, const MlsdCandidate& b) {
        return a.score > b.score;
    };
    if (candidates.size() > kTopK) {
        std::partial_sort(candidates.begin(), candidates.begin() + kTopK, candidates.end(), by_score_desc);
        candidates.resize(kTopK);
    } else {
        std::sort(candidates.begin(), candidates.end(), by_score_desc);
    }

    lines->reserve(candidates.size());
    for (const MlsdCandidate& candidate : candidates) {
        decode_mlsd_line(candidate.y,
                         candidate.x,
                         candidate.score,
                         raw_attr,
                         raw_data,
                         raw_layout,
                         image_width,
                         image_height,
                         coord_map_w,
                         coord_map_h,
                         center_step_x,
                         center_step_y,
                         kOrgDispChannel,
                         distance_threshold,
                         lines);
    }
    return 0;
}

}  // namespace

int init_mlsd_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::init_zero_copy_model(model_path, app_ctx);
}

int release_mlsd_model(rknn_app_context_t* app_ctx) {
    return visiong::npu::rknn::release_zero_copy_model(app_ctx);
}

int inference_mlsd_model(rknn_app_context_t* app_ctx,
                         int image_width,
                         int image_height,
                         float score_threshold,
                         float distance_threshold,
                         std::vector<LineSegment>* lines) {
    if (app_ctx == nullptr || lines == nullptr || image_width <= 0 || image_height <= 0) {
        return -1;
    }
    lines->clear();
    if (visiong::npu::rknn::run_and_sync_outputs(app_ctx, "MLSD") != 0) {
        return -1;
    }

    if (app_ctx->io_num.n_output == 1) {
        return inference_mlsd_raw_map(app_ctx,
                                      image_width,
                                      image_height,
                                      score_threshold,
                                      distance_threshold,
                                      lines);
    }
    MlsdOutputLayout layout;
    if (!resolve_mlsd_outputs(app_ctx, &layout)) {
        std::printf("ERROR: unexpected MLSD output layout.\n");
        return -1;
    }

    const rknn_tensor_attr& pts_attr = app_ctx->output_attrs[layout.pts];
    const rknn_tensor_attr& score_attr = app_ctx->output_attrs[layout.scores];
    const rknn_tensor_attr& vmap_attr = app_ctx->output_attrs[layout.vmap];
    const void* pts_data = app_ctx->output_mems[layout.pts]->virt_addr;
    const void* score_data = app_ctx->output_mems[layout.scores]->virt_addr;
    const void* vmap_data = app_ctx->output_mems[layout.vmap]->virt_addr;

    const int max_x_index = std::max(0, layout.vmap_layout.map_w - 1);
    const int max_y_index = std::max(0, layout.vmap_layout.map_h - 1);

    lines->reserve(static_cast<size_t>(layout.topk));
    for (int i = 0; i < layout.topk; ++i) {
        const int y = std::max(0, std::min(tensor_value_as_i32(pts_attr, pts_data, i * 2 + 0), max_y_index));
        const int x = std::max(0, std::min(tensor_value_as_i32(pts_attr, pts_data, i * 2 + 1), max_x_index));
        const float score = tensor_value_as_f32(score_attr, score_data, i);
        if (score <= score_threshold) {
            continue;
        }
        decode_mlsd_line(y, x, score, vmap_attr, vmap_data, layout.vmap_layout,
                         image_width, image_height,
                         layout.vmap_layout.map_w, layout.vmap_layout.map_h,
                         1.0f, 1.0f, 0, distance_threshold, lines);
    }
    return 0;
}
