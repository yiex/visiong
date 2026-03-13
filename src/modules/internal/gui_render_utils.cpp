// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/gui_render_utils.h"
#include "nuklear.h"

#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void cpu_fill_rect_bgr888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int x, int y, int w, int h, nk_color color) {
    int x0 = std::max(0, x), y0 = std::max(0, y);
    int x1 = std::min(dst_w, x + w), y1 = std::min(dst_h, y + h);
    if (x1 <= x0 || y1 <= y0) return;

    const uint32_t fg_r = color.r, fg_g = color.g, fg_b = color.b, fg_a = color.a;
    const int dst_stride_bytes = dst_stride_pixels * 3;
    if (fg_a == 255) {
        for (int i = y0; i < y1; ++i) {
            uint8_t* pixel_row = dst + i * dst_stride_bytes + x0 * 3;
            for (int j = x0; j < x1; ++j) {
                *pixel_row++ = static_cast<uint8_t>(fg_b);
                *pixel_row++ = static_cast<uint8_t>(fg_g);
                *pixel_row++ = static_cast<uint8_t>(fg_r);
            }
        }
    } else if (fg_a > 0) {
        const uint32_t inv_alpha = 255 - fg_a;
        for (int i = y0; i < y1; ++i) {
            uint8_t* pixel = dst + i * dst_stride_bytes + x0 * 3;
            for (int j = x0; j < x1; ++j) {
                pixel[0] = static_cast<uint8_t>((fg_b * fg_a + pixel[0] * inv_alpha) >> 8);
                pixel[1] = static_cast<uint8_t>((fg_g * fg_a + pixel[1] * inv_alpha) >> 8);
                pixel[2] = static_cast<uint8_t>((fg_r * fg_a + pixel[2] * inv_alpha) >> 8);
                pixel += 3;
            }
        }
    }
}

void cpu_render_text_bgr888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    const uint8_t* atlas, int atlas_w, int dst_x, int dst_y, int w, int h,
    int src_x, int src_y, nk_color color) {
    const uint32_t fg_r = color.r, fg_g = color.g, fg_b = color.b, fg_a = color.a;
    const int dst_stride_bytes = dst_stride_pixels * 3;
    for (int i = 0; i < h; ++i) {
        const int y = dst_y + i;
        if (y < 0 || y >= dst_h) continue;
        const uint8_t* atlas_row = atlas + (src_y + i) * atlas_w;
        uint8_t* pixel_row = dst + y * dst_stride_bytes;
        for (int j = 0; j < w; ++j) {
            const int x = dst_x + j;
            if (x < 0 || x >= dst_w) continue;
            const uint32_t glyph_alpha = atlas_row[src_x + j];
            if (glyph_alpha == 0) continue;
            const uint32_t final_alpha = (glyph_alpha * fg_a) >> 8;
            if (final_alpha == 0) continue;
            if (final_alpha == 255) {
                pixel_row[x * 3 + 0] = static_cast<uint8_t>(fg_b);
                pixel_row[x * 3 + 1] = static_cast<uint8_t>(fg_g);
                pixel_row[x * 3 + 2] = static_cast<uint8_t>(fg_r);
            } else {
                const uint32_t inv_alpha = 255 - final_alpha;
                uint8_t* pixel = pixel_row + x * 3;
                pixel[0] = static_cast<uint8_t>((fg_b * final_alpha + pixel[0] * inv_alpha) >> 8);
                pixel[1] = static_cast<uint8_t>((fg_g * final_alpha + pixel[1] * inv_alpha) >> 8);
                pixel[2] = static_cast<uint8_t>((fg_r * final_alpha + pixel[2] * inv_alpha) >> 8);
            }
        }
    }
}

void cpu_fill_rect_rgba8888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int x, int y, int w, int h, nk_color color) {
    int x0 = std::max(0, x), y0 = std::max(0, y);
    int x1 = std::min(dst_w, x + w), y1 = std::min(dst_h, y + h);
    if (x1 <= x0 || y1 <= y0) return;

    const uint32_t fg_r = color.r, fg_g = color.g, fg_b = color.b, fg_a = color.a;
    const int dst_stride_bytes = dst_stride_pixels * 4;

    for (int i = y0; i < y1; ++i) {
        uint8_t* p = dst + i * dst_stride_bytes + x0 * 4;
        for (int j = x0; j < x1; ++j) {
            if (fg_a == 255) {
                p[0] = static_cast<uint8_t>(fg_r);
                p[1] = static_cast<uint8_t>(fg_g);
                p[2] = static_cast<uint8_t>(fg_b);
                p[3] = 255;
            } else if (fg_a > 0) {
                p[0] = static_cast<uint8_t>(fg_r);
                p[1] = static_cast<uint8_t>(fg_g);
                p[2] = static_cast<uint8_t>(fg_b);
                p[3] = static_cast<uint8_t>(fg_a);
            }
            p += 4;
        }
    }
}

void cpu_render_text_rgba8888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    const uint8_t* atlas, int atlas_w, int dst_x, int dst_y, int w, int h,
    int src_x, int src_y, nk_color color) {
    const uint32_t fg_r = color.r, fg_g = color.g, fg_b = color.b, fg_a = color.a;
    const int dst_stride_bytes = dst_stride_pixels * 4;

    for (int i = 0; i < h; ++i) {
        const int y = dst_y + i;
        if (y < 0 || y >= dst_h) continue;
        const uint8_t* atlas_row = atlas + (src_y + i) * atlas_w;
        uint8_t* pixel_row = dst + y * dst_stride_bytes;
        for (int j = 0; j < w; ++j) {
            const int x = dst_x + j;
            if (x < 0 || x >= dst_w) continue;
            const uint32_t glyph_a = atlas_row[src_x + j];
            if (glyph_a == 0) continue;
            const uint32_t final_fg_a = (glyph_a * fg_a) >> 8;
            if (final_fg_a == 0) continue;
            uint8_t* p = pixel_row + x * 4;
            p[0] = static_cast<uint8_t>(fg_r);
            p[1] = static_cast<uint8_t>(fg_g);
            p[2] = static_cast<uint8_t>(fg_b);
            p[3] = static_cast<uint8_t>(final_fg_a);
        }
    }
}

#if defined(__ARM_NEON)
static void cpu_resize_bgr888_bilinear_neon(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h) {
    (void)dst_w;
    (void)dst_h;

    constexpr int kFracBits = 8;
    constexpr int kFracVal = 1 << kFracBits;
    const uint64_t x_ratio = ((uint64_t)(src_w - 1) * kFracVal) / roi_w;
    const uint64_t y_ratio = ((uint64_t)(src_h - 1) * kFracVal) / roi_h;
    uint64_t y_src_fixed = 0;

    for (int y = 0; y < roi_h; ++y) {
        int y_src_i = static_cast<int>(y_src_fixed >> kFracBits);
        y_src_i = std::min(y_src_i, src_h - 2);

        const uint16_t y_weight = static_cast<uint16_t>(y_src_fixed & (kFracVal - 1));
        const uint16x8_t y_w_u16 = vdupq_n_u16(y_weight);
        const uint16x8_t y_inv_w_u16 = vdupq_n_u16(kFracVal - y_weight);

        const uint8_t* row1 = src + y_src_i * src_stride_pixels * 3;
        const uint8_t* row2 = src + (y_src_i + 1) * src_stride_pixels * 3;
        uint8_t* dst_row = dst + (roi_y + y) * dst_stride_pixels * 3 + roi_x * 3;

        uint64_t x_src_fixed = 0;
        int x = 0;
        for (; x <= roi_w - 8; x += 8) {
            uint32_t x_coords_fixed[8];
            for (int i = 0; i < 8; ++i) {
                x_coords_fixed[i] = static_cast<uint32_t>(x_src_fixed + i * x_ratio);
            }

            const uint32x4_t x_src_fixed_vec1 = vld1q_u32(x_coords_fixed);
            const uint32x4_t x_src_fixed_vec2 = vld1q_u32(x_coords_fixed + 4);

            const uint32x4_t x_src_i_vec1 = vshrq_n_u32(x_src_fixed_vec1, kFracBits);
            const uint32x4_t x_src_i_vec2 = vshrq_n_u32(x_src_fixed_vec2, kFracBits);

            uint16x8_t x_w_u16 = vcombine_u16(vmovn_u32(x_src_fixed_vec1), vmovn_u32(x_src_fixed_vec2));
            x_w_u16 = vandq_u16(x_w_u16, vdupq_n_u16(kFracVal - 1));
            const uint16x8_t x_inv_w_u16 = vsubq_u16(vdupq_n_u16(kFracVal), x_w_u16);

            uint32_t x_indices[8];
            vst1q_u32(x_indices, x_src_i_vec1);
            vst1q_u32(x_indices + 4, x_src_i_vec2);

            uint8x8x3_t p1, p2, p3, p4;
            for (int i = 0; i < 8; ++i) {
                const int clamped_idx = std::min(static_cast<int>(x_indices[i]), src_w - 2);
                for (int c = 0; c < 3; ++c) {
                    p1.val[c][i] = row1[clamped_idx * 3 + c];
                    p2.val[c][i] = row1[(clamped_idx + 1) * 3 + c];
                    p3.val[c][i] = row2[clamped_idx * 3 + c];
                    p4.val[c][i] = row2[(clamped_idx + 1) * 3 + c];
                }
            }

            uint8x8x3_t result_pixels;
            for (int c = 0; c < 3; ++c) {
                uint32x4_t top_lo = vmull_u16(vget_low_u16(vmovl_u8(p1.val[c])), vget_low_u16(x_inv_w_u16));
                top_lo = vmlal_u16(top_lo, vget_low_u16(vmovl_u8(p2.val[c])), vget_low_u16(x_w_u16));
                uint32x4_t top_hi = vmull_u16(vget_high_u16(vmovl_u8(p1.val[c])), vget_high_u16(x_inv_w_u16));
                top_hi = vmlal_u16(top_hi, vget_high_u16(vmovl_u8(p2.val[c])), vget_high_u16(x_w_u16));

                uint32x4_t bot_lo = vmull_u16(vget_low_u16(vmovl_u8(p3.val[c])), vget_low_u16(x_inv_w_u16));
                bot_lo = vmlal_u16(bot_lo, vget_low_u16(vmovl_u8(p4.val[c])), vget_low_u16(x_w_u16));
                uint32x4_t bot_hi = vmull_u16(vget_high_u16(vmovl_u8(p3.val[c])), vget_high_u16(x_inv_w_u16));
                bot_hi = vmlal_u16(bot_hi, vget_high_u16(vmovl_u8(p4.val[c])), vget_high_u16(x_w_u16));

                uint32x4_t blend_lo = vmulq_u32(top_lo, vmovl_u16(vget_low_u16(y_inv_w_u16)));
                blend_lo = vmlaq_u32(blend_lo, bot_lo, vmovl_u16(vget_low_u16(y_w_u16)));
                uint32x4_t blend_hi = vmulq_u32(top_hi, vmovl_u16(vget_high_u16(y_inv_w_u16)));
                blend_hi = vmlaq_u32(blend_hi, bot_hi, vmovl_u16(vget_high_u16(y_w_u16)));

                const uint16x8_t final_u16 =
                    vcombine_u16(vshrn_n_u32(blend_lo, kFracBits * 2), vshrn_n_u32(blend_hi, kFracBits * 2));
                result_pixels.val[c] = vqmovn_u16(final_u16);
            }

            vst3_u8(dst_row + x * 3, result_pixels);
            x_src_fixed += static_cast<uint64_t>(x_ratio) * 8;
        }

        for (; x < roi_w; ++x) {
            int x_src_i = static_cast<int>(x_src_fixed >> kFracBits);
            x_src_i = std::min(x_src_i, src_w - 2);
            const int x_weight = static_cast<int>(x_src_fixed & (kFracVal - 1));

            for (int c = 0; c < 3; ++c) {
                const int p1_c = row1[x_src_i * 3 + c];
                const int p2_c = row1[(x_src_i + 1) * 3 + c];
                const int p3_c = row2[x_src_i * 3 + c];
                const int p4_c = row2[(x_src_i + 1) * 3 + c];
                const int top = (p1_c * (kFracVal - x_weight) + p2_c * x_weight);
                const int bottom = (p3_c * (kFracVal - x_weight) + p4_c * x_weight);
                dst_row[x * 3 + c] = static_cast<uint8_t>(((uint64_t)top * (kFracVal - y_weight) +
                                                           (uint64_t)bottom * y_weight) >>
                                                          (kFracBits * 2));
            }
            x_src_fixed += x_ratio;
        }
        y_src_fixed += y_ratio;
    }
}
#else
static void cpu_resize_bgr888_bilinear_c(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h) {
    (void)dst_w;
    (void)dst_h;
    const float x_ratio = static_cast<float>(src_w - 1) / roi_w;
    const float y_ratio = static_cast<float>(src_h - 1) / roi_h;

    for (int y_dst = 0; y_dst < roi_h; ++y_dst) {
        const float y_src_f = y_dst * y_ratio;
        const int y_src_i = static_cast<int>(y_src_f);
        const float y_diff = y_src_f - y_src_i;

        const uint8_t* row1 = src + std::min(y_src_i, src_h - 2) * src_stride_pixels * 3;
        const uint8_t* row2 = src + std::min(y_src_i + 1, src_h - 1) * src_stride_pixels * 3;
        uint8_t* dst_row = dst + (roi_y + y_dst) * dst_stride_pixels * 3 + roi_x * 3;

        for (int x_dst = 0; x_dst < roi_w; ++x_dst) {
            const float x_src_f = x_dst * x_ratio;
            const int x_src_i = static_cast<int>(x_src_f);
            const float x_diff = x_src_f - x_src_i;

            const int x_src_i_clamped = std::min(x_src_i, src_w - 2);
            const uint8_t* p1 = row1 + x_src_i_clamped * 3;
            const uint8_t* p2 = row1 + (x_src_i_clamped + 1) * 3;
            const uint8_t* p3 = row2 + x_src_i_clamped * 3;
            const uint8_t* p4 = row2 + (x_src_i_clamped + 1) * 3;

            for (int c = 0; c < 3; ++c) {
                const float top = p1[c] * (1.0f - x_diff) + p2[c] * x_diff;
                const float bottom = p3[c] * (1.0f - x_diff) + p4[c] * x_diff;
                dst_row[x_dst * 3 + c] = static_cast<uint8_t>(top * (1.0f - y_diff) + bottom * y_diff);
            }
        }
    }
}
#endif

void cpu_resize_bgr888_bilinear(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || roi_w <= 0 || roi_h <= 0) {
        return;
    }

    if (src_w == 1 || src_h == 1) {
        const float x_ratio = roi_w > 1 ? static_cast<float>(src_w - 1) / static_cast<float>(roi_w - 1) : 0.0f;
        const float y_ratio = roi_h > 1 ? static_cast<float>(src_h - 1) / static_cast<float>(roi_h - 1) : 0.0f;
        for (int y_dst = 0; y_dst < roi_h; ++y_dst) {
            const int current_y = roi_y + y_dst;
            if (current_y < 0 || current_y >= dst_h) {
                continue;
            }
            const int src_y = std::clamp(static_cast<int>(y_dst * y_ratio), 0, src_h - 1);
            const uint8_t* src_row = src + src_y * src_stride_pixels * 3;
            uint8_t* dst_row = dst + current_y * dst_stride_pixels * 3 + roi_x * 3;
            for (int x_dst = 0; x_dst < roi_w; ++x_dst) {
                const int current_x = roi_x + x_dst;
                if (current_x < 0 || current_x >= dst_w) {
                    continue;
                }
                const int src_x = std::clamp(static_cast<int>(x_dst * x_ratio), 0, src_w - 1);
                const uint8_t* src_pixel = src_row + src_x * 3;
                uint8_t* dst_pixel = dst_row + x_dst * 3;
                dst_pixel[0] = src_pixel[0];
                dst_pixel[1] = src_pixel[1];
                dst_pixel[2] = src_pixel[2];
            }
        }
        return;
    }

#if defined(__ARM_NEON)
    cpu_resize_bgr888_bilinear_neon(src, src_w, src_h, src_stride_pixels,
                                    dst, dst_w, dst_h, dst_stride_pixels,
                                    roi_x, roi_y, roi_w, roi_h);
#else
    cpu_resize_bgr888_bilinear_c(src, src_w, src_h, src_stride_pixels,
                                 dst, dst_w, dst_h, dst_stride_pixels,
                                 roi_x, roi_y, roi_w, roi_h);
#endif
}

