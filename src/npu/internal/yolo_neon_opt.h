// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// Ultra-aggressive NEON post-process helpers for ARMv7-A (Cortex-A7) single core. / Ultra-aggressive NEON post-process 辅助函数 用于 ARMv7-A (Cortex-A7) single core.
// Not designed for readability/maintainability.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace visiong { namespace npu { namespace yolo { namespace neonopt {

#ifndef YOLO_NEON_ALWAYS_INLINE
#define YOLO_NEON_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

YOLO_NEON_ALWAYS_INLINE void prefetch_l1(const void* p) {
#if defined(__GNUC__)
    __builtin_prefetch(p, 0, 3);
#else
    (void)p;
#endif
}

// Horizontal max for int8x16 -> int8 (ARMv7 compatible) / Horizontal max 用于 int8x16 -> int8 (ARMv7 compatible)
YOLO_NEON_ALWAYS_INLINE int8_t hmax_s8x16(int8x16_t v) {
    int8x8_t lo = vget_low_s8(v);
    int8x8_t hi = vget_high_s8(v);
    int8x8_t m = vmax_s8(lo, hi);
    // reduce 8 -> 1 / 详见英文原注释。
    int8x8_t t;
    t = vext_s8(m, m, 4);
    m = vmax_s8(m, t);
    t = vext_s8(m, m, 2);
    m = vmax_s8(m, t);
    t = vext_s8(m, m, 1);
    m = vmax_s8(m, t);
    return vget_lane_s8(m, 0);
}

// Argmax over int8 array. Returns max value, writes first index of max. / Argmax over int8 array. 返回 max value, 写入 首个 index 的 max.
YOLO_NEON_ALWAYS_INLINE int8_t argmax_i8(const int8_t* p, int n, int* out_idx) {
    int i = 0;
    int8x16_t vmax = vdupq_n_s8((int8_t)-128);
    for (; i + 16 <= n; i += 16) {
        prefetch_l1(p + i + 64);
        int8x16_t v = vld1q_s8(p + i);
        vmax = vmaxq_s8(vmax, v);
    }
    int8_t maxv = hmax_s8x16(vmax);
    for (; i < n; ++i) {
        const int8_t v = p[i];
        if (v > maxv) maxv = v;
    }
    int idx = 0;
    for (int j = 0; j < n; ++j) {
        if (p[j] == maxv) {
            idx = j;
            break;
        }
    }
    *out_idx = idx;
    return maxv;
}

// Argmax over float array. Returns max value, writes first index of max. / Argmax over float array. 返回 max value, 写入 首个 index 的 max.
YOLO_NEON_ALWAYS_INLINE float argmax_f32(const float* p, int n, int* out_idx) {
    int i = 0;
    float32x4_t vmax = vdupq_n_f32(-3.402823466e+38f);
    for (; i + 4 <= n; i += 4) {
        prefetch_l1(p + i + 32);
        float32x4_t v = vld1q_f32(p + i);
        vmax = vmaxq_f32(vmax, v);
    }
    float32x2_t m2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    m2 = vpmax_f32(m2, m2);
    float maxv = vget_lane_f32(m2, 0);
    for (; i < n; ++i) {
        const float v = p[i];
        if (v > maxv) maxv = v;
    }
    int idx = 0;
    for (int j = 0; j < n; ++j) {
        if (p[j] == maxv) {
            idx = j;
            break;
        }
    }
    *out_idx = idx;
    return maxv;
}

// Build exp table for int8 values using bit-pattern indexing (uint8_t cast). Table size must be 256. / 构建 exp table 用于 int8 values using bit-pattern indexing (uint8_t cast). Table 尺寸 必须 被 256.
YOLO_NEON_ALWAYS_INLINE void build_exp_lut_i8(float* table256, int32_t zp, float scale) {
    for (int i = 0; i < 256; ++i) {
        const int8_t q = (int8_t)(uint8_t)i;
        const float x = (static_cast<float>((int)q - (int)zp)) * scale;
        table256[i] = std::exp(x);
    }
}

// DFL expectation directly from quantized logits (int8) using exp LUT. / DFL expectation 直接 from quantized logits (int8) using exp LUT.
// logits layout: [4][dfl_len]
YOLO_NEON_ALWAYS_INLINE void dfl_expect_i8_lut(const int8_t* logits, int dfl_len, const float* exp_lut256, float out4[4]) {
    // Very hot path: keep it brutally simple. / Very hot 路径: 保持 it brutally simple.
    for (int b = 0; b < 4; ++b) {
        const int8_t* p = logits + b * dfl_len;
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int i = 0; i < dfl_len; ++i) {
            const float e = exp_lut256[(uint8_t)p[i]];
            sum += e;
            wsum += e * static_cast<float>(i);
        }
        out4[b] = (sum > 0.0f) ? (wsum / sum) : 0.0f;
    }
}

// Build top-k indices by descending score (nth_element + sort). topk<=0 -> full sort. / 构建 top-k indices 由 descending 得分 (nth_element + sort). topk<=0 -> full sort.
YOLO_NEON_ALWAYS_INLINE void topk_indices_desc(const std::vector<float>& scores, std::vector<int>* order, int topk) {
    order->resize(scores.size());
    std::iota(order->begin(), order->end(), 0);
    auto cmp = [&scores](int a, int b) { return scores[(size_t)a] > scores[(size_t)b]; };
    if (topk > 0 && (int)order->size() > topk) {
        std::nth_element(order->begin(), order->begin() + topk, order->end(), cmp);
        order->resize((size_t)topk);
    }
    std::sort(order->begin(), order->end(), cmp);
}

// Class-wise NMS over XYWH boxes. Returns kept indices into original candidate arrays, in global score order. / Class-wise NMS over XYWH 框. 返回 kept indices into original candidate arrays, 在 global 得分 order.
// - boxes_xywh: size = n*4, layout [x,y,w,h]
// - scores: size=n / - scores: 尺寸=n
// - class_ids: size=n
// - class_count: number of classes / - class_count: number 的 classes
// - topk: maximum candidates entering NMS (e.g., 1024)
// - out_indices cap: out_cap / 详见英文原注释。
// IoU uses the same +1.0 convention as the original calculate_iou_xywh.
YOLO_NEON_ALWAYS_INLINE int nms_topk_classwise_xywh_neon(
    const std::vector<float>& boxes_xywh,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    int class_count,
    float nms_threshold,
    int topk,
    int* out_indices,
    int out_cap) {

    const int n = (int)scores.size();
    if (n <= 0) return 0;

    static thread_local std::vector<int> order;
    topk_indices_desc(scores, &order, topk);
    const int k = (int)order.size();
    if (k <= 0) return 0;

    // Packed by global score-desc order. / Packed 由 global score-desc order.
    static thread_local std::vector<int> orig;
    static thread_local std::vector<int> cls;
    static thread_local std::vector<float> x1;
    static thread_local std::vector<float> y1;
    static thread_local std::vector<float> x2;
    static thread_local std::vector<float> y2;
    static thread_local std::vector<float> area;

    orig.resize((size_t)k);
    cls.resize((size_t)k);
    x1.resize((size_t)k);
    y1.resize((size_t)k);
    x2.resize((size_t)k);
    y2.resize((size_t)k);
    area.resize((size_t)k);

    for (int p = 0; p < k; ++p) {
        const int idx = order[(size_t)p];
        orig[(size_t)p] = idx;
        const size_t b = (size_t)idx * 4U;
        const float bx = boxes_xywh[b + 0];
        const float by = boxes_xywh[b + 1];
        const float bw = boxes_xywh[b + 2];
        const float bh = boxes_xywh[b + 3];
        x1[(size_t)p] = bx;
        y1[(size_t)p] = by;
        x2[(size_t)p] = bx + bw;
        y2[(size_t)p] = by + bh;
        // match original: (x2-x1+1)*(y2-y1+1) == (w+1)*(h+1) / 匹配 original: (x2-x1+1)*(y2-y1+1) == (w+1)*(h+1)
        area[(size_t)p] = (bw + 1.0f) * (bh + 1.0f);
        cls[(size_t)p] = class_ids[(size_t)idx];
    }

    // Keep flags for packed positions. / 保持 flags 用于 packed positions.
    static thread_local std::vector<uint8_t> keep;
    keep.assign((size_t)k, 0);

    // Group packed positions by class (prefix-sum). / Group packed positions 由 类别 (prefix-sum).
    if (class_count <= 0) {
        // fallback: treat as single class / 回退: treat 作为 single 类别
        class_count = 1;
    }

    static thread_local std::vector<int> counts;
    static thread_local std::vector<int> offsets;
    static thread_local std::vector<int> cls_pos;

    counts.assign((size_t)class_count, 0);
    for (int p = 0; p < k; ++p) {
        const int c = cls[(size_t)p];
        if ((unsigned)c < (unsigned)class_count) {
            counts[(size_t)c]++;
        }
    }
    offsets.resize((size_t)class_count + 1U);
    offsets[0] = 0;
    for (int c = 0; c < class_count; ++c) {
        offsets[(size_t)c + 1U] = offsets[(size_t)c] + counts[(size_t)c];
    }
    cls_pos.resize((size_t)k);
    // cursor = offsets (reuse counts as cursor) / cursor = offsets (复用 counts 作为 cursor)
    for (int c = 0; c < class_count; ++c) {
        counts[(size_t)c] = offsets[(size_t)c];
    }
    for (int p = 0; p < k; ++p) {
        const int c = cls[(size_t)p];
        if ((unsigned)c < (unsigned)class_count) {
            cls_pos[(size_t)counts[(size_t)c]++] = p;
        }
    }

    // Scratch arrays for one class. / Scratch arrays 用于 一个 类别.
    static thread_local std::vector<float> sx1;
    static thread_local std::vector<float> sy1;
    static thread_local std::vector<float> sx2;
    static thread_local std::vector<float> sy2;
    static thread_local std::vector<float> sarea;
    static thread_local std::vector<int> spos;
    static thread_local std::vector<uint8_t> suppressed;

    // NMS threshold vector. / 详见英文原注释。
    const float32x4_t vth = vdupq_n_f32(nms_threshold);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    const float32x4_t vzero = vdupq_n_f32(0.0f);

    for (int c = 0; c < class_count; ++c) {
        const int start = offsets[(size_t)c];
        const int end = offsets[(size_t)c + 1U];
        const int len = end - start;
        if (len <= 0) continue;

        sx1.resize((size_t)len);
        sy1.resize((size_t)len);
        sx2.resize((size_t)len);
        sy2.resize((size_t)len);
        sarea.resize((size_t)len);
        spos.resize((size_t)len);
        suppressed.assign((size_t)len, 0);

        for (int i = 0; i < len; ++i) {
            const int p = cls_pos[(size_t)start + (size_t)i];
            spos[(size_t)i] = p;
            sx1[(size_t)i] = x1[(size_t)p];
            sy1[(size_t)i] = y1[(size_t)p];
            sx2[(size_t)i] = x2[(size_t)p];
            sy2[(size_t)i] = y2[(size_t)p];
            sarea[(size_t)i] = area[(size_t)p];
        }

        for (int i = 0; i < len; ++i) {
            if (suppressed[(size_t)i]) continue;
            keep[(size_t)spos[(size_t)i]] = 1;

            const float xi1 = sx1[(size_t)i];
            const float yi1 = sy1[(size_t)i];
            const float xi2 = sx2[(size_t)i];
            const float yi2 = sy2[(size_t)i];
            const float ai = sarea[(size_t)i];

            const float32x4_t vxi1 = vdupq_n_f32(xi1);
            const float32x4_t vyi1 = vdupq_n_f32(yi1);
            const float32x4_t vxi2 = vdupq_n_f32(xi2);
            const float32x4_t vyi2 = vdupq_n_f32(yi2);
            const float32x4_t vai = vdupq_n_f32(ai);

            int j = i + 1;
            for (; j + 4 <= len; j += 4) {
                // NOTE: we intentionally do NOT skip already-suppressed j-lanes; extra math is cheaper than branches. / NOTE: we intentionally do 不 跳过 already-suppressed j-lanes; 额外 math 为 cheaper than branches.
                float32x4_t vx1 = vld1q_f32(&sx1[(size_t)j]);
                float32x4_t vy1 = vld1q_f32(&sy1[(size_t)j]);
                float32x4_t vx2 = vld1q_f32(&sx2[(size_t)j]);
                float32x4_t vy2 = vld1q_f32(&sy2[(size_t)j]);
                float32x4_t va2 = vld1q_f32(&sarea[(size_t)j]);

                float32x4_t vxx1 = vmaxq_f32(vxi1, vx1);
                float32x4_t vyy1 = vmaxq_f32(vyi1, vy1);
                float32x4_t vxx2 = vminq_f32(vxi2, vx2);
                float32x4_t vyy2 = vminq_f32(vyi2, vy2);

                float32x4_t vw = vsubq_f32(vaddq_f32(vxx2, vone), vxx1);
                float32x4_t vh = vsubq_f32(vaddq_f32(vyy2, vone), vyy1);
                vw = vmaxq_f32(vw, vzero);
                vh = vmaxq_f32(vh, vzero);

                float32x4_t vinter = vmulq_f32(vw, vh);
                float32x4_t vunion = vsubq_f32(vaddq_f32(vai, va2), vinter);

                // iou = inter / union / 详见英文原注释。
                float32x4_t vrec = vrecpeq_f32(vunion);
                vrec = vmulq_f32(vrecpsq_f32(vunion, vrec), vrec);
                vrec = vmulq_f32(vrecpsq_f32(vunion, vrec), vrec);
                float32x4_t viou = vmulq_f32(vinter, vrec);
                uint32x4_t m = vcgtq_f32(viou, vth);

                uint32_t mask[4];
                vst1q_u32(mask, m);
                if (mask[0]) suppressed[(size_t)j + 0U] = 1;
                if (mask[1]) suppressed[(size_t)j + 1U] = 1;
                if (mask[2]) suppressed[(size_t)j + 2U] = 1;
                if (mask[3]) suppressed[(size_t)j + 3U] = 1;
            }

            for (; j < len; ++j) {
                if (suppressed[(size_t)j]) continue;
                const float xx1 = std::max(xi1, sx1[(size_t)j]);
                const float yy1 = std::max(yi1, sy1[(size_t)j]);
                const float xx2 = std::min(xi2, sx2[(size_t)j]);
                const float yy2 = std::min(yi2, sy2[(size_t)j]);
                const float w = std::max(0.0f, (xx2 - xx1 + 1.0f));
                const float h = std::max(0.0f, (yy2 - yy1 + 1.0f));
                const float inter = w * h;
                const float uni = ai + sarea[(size_t)j] - inter;
                const float iou = (uni > 0.0f) ? (inter / uni) : 0.0f;
                if (iou > nms_threshold) suppressed[(size_t)j] = 1;
            }
        }
    }

    int out_n = 0;
    for (int p = 0; p < k && out_n < out_cap; ++p) {
        if (keep[(size_t)p]) {
            out_indices[out_n++] = orig[(size_t)p];
        }
    }
    return out_n;
}

}}}} // namespace visiong::npu::yolo::neonopt

