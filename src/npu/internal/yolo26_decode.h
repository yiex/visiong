// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Modified for VisionG: INT8/NEON reduction, deferred decoding and deterministic TopK.
// VisionG 修改：INT8/NEON 归约、延迟解码、稳定 TopK 排序和工作内存复用。
// Source / 来源：airockchip/rknn_model_zoo, examples/yolo26/cpp/postprocess.cc.
// Revision / 版本：837db8e992815ff8647eadb709f07067f4bc3a71. See / 参见 licenses/SOURCES.md.
#ifndef VISIONG_NPU_INTERNAL_YOLO26_DECODE_H
#define VISIONG_NPU_INTERNAL_YOLO26_DECODE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace visiong::npu::yolo26 {

// INT8 scores in [C,N] or [N,C], with strides measured in elements.
// INT8 分数支持 [C,N] 或 [N,C] 布局，步长以元素为单位。
struct Scores {
    const int8_t* data = nullptr;
    int count = 0;
    int classes = 0;
    size_t anchor_stride = 0;
    size_t class_stride = 0;
    int zero_point = 0;
    float scale = 1.0f;
};
struct Candidate {
    int index;
    int8_t score;
};

inline bool better(const Candidate& a, const Candidate& b) {
    return a.score != b.score ? a.score > b.score : a.index < b.index;
}

inline void select(const Scores& scores, float threshold, size_t limit, std::vector<Candidate>& candidates,
                   std::vector<int8_t>& maxima, bool use_neon = true) {
    candidates.clear();
    if (scores.data == nullptr || scores.count <= 0 || scores.classes <= 0 || !std::isfinite(scores.scale) ||
        scores.scale <= 0.0f || !std::isfinite(threshold) || limit == 0) {
        return;
    }
    maxima.assign(static_cast<size_t>(scores.count), static_cast<int8_t>(-128));
    if (scores.anchor_stride == 1) {
        // Reduce 16 anchors at once for planar scores. / 平面布局每次归约 16 个候选点的类别分数。
        for (int c = 0; c < scores.classes; ++c) {
            const int8_t* row = scores.data + static_cast<size_t>(c) * scores.class_stride;
            int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if (use_neon) {
                for (; i + 16 <= scores.count; i += 16) {
                    vst1q_s8(maxima.data() + i, vmaxq_s8(vld1q_s8(maxima.data() + i), vld1q_s8(row + i)));
                }
            }
#else
            (void)use_neon;
#endif
            for (; i < scores.count; ++i) {
                maxima[i] = std::max(maxima[i], row[i]);
            }
        }
    } else {
        // Interleaved scores reduce contiguous channels in INT8. / 交错布局直接归约连续的 INT8 通道。
        for (int i = 0; i < scores.count; ++i) {
            const int8_t* row = scores.data + static_cast<size_t>(i) * scores.anchor_stride;
            int8_t best = -128;
            int c = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if (use_neon && scores.class_stride == 1) {
                int8x16_t v = vdupq_n_s8(-128);
                for (; c + 16 <= scores.classes; c += 16) {
                    v = vmaxq_s8(v, vld1q_s8(row + c));
                }
                int8x8_t m = vmax_s8(vget_low_s8(v), vget_high_s8(v));
                m = vmax_s8(m, vext_s8(m, m, 4));
                m = vmax_s8(m, vext_s8(m, m, 2));
                m = vmax_s8(m, vext_s8(m, m, 1));
                best = vget_lane_s8(m, 0);
            }
#endif
            for (; c < scores.classes; ++c) {
                best = std::max(best, row[static_cast<size_t>(c) * scores.class_stride]);
            }
            maxima[i] = best;
        }
    }
    // Find the rejected INT8 boundary once, matching float threshold comparisons.
    // 一次性查找被拒绝的 INT8 边界，保持与浮点阈值比较一致，避免逐类别反量化。
    int low = -128;
    int high = 128;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        const float value = (mid - scores.zero_point) * scores.scale;
        if (value <= threshold) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    const int q = low - 1;
    for (int i = 0; i < scores.count; ++i) {
        if (maxima[i] > q) {
            candidates.push_back({i, maxima[i]});
        }
    }
    if (candidates.size() > limit) {
        std::nth_element(candidates.begin(), candidates.begin() + limit, candidates.end(), better);
        candidates.resize(limit);
    }
    std::sort(candidates.begin(), candidates.end(), better);
}

inline int class_id(const Scores& scores, const Candidate& candidate) {
    const int8_t* row = scores.data + static_cast<size_t>(candidate.index) * scores.anchor_stride;
    for (int c = 0; c < scores.classes; ++c) {
        if (row[static_cast<size_t>(c) * scores.class_stride] == candidate.score) {
            return c;
        }
    }
    return -1;
}
} // namespace visiong::npu::yolo26

#endif // VISIONG_NPU_INTERNAL_YOLO26_DECODE_H
