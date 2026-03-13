// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_GUI_RENDER_UTILS_H
#define VISIONG_MODULES_GUI_RENDER_UTILS_H

#include <cstdint>

struct nk_color;

void cpu_fill_rect_bgr888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int x, int y, int w, int h, nk_color color);

void cpu_render_text_bgr888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    const uint8_t* atlas, int atlas_w, int dst_x, int dst_y, int w, int h,
    int src_x, int src_y, nk_color color);

void cpu_fill_rect_rgba8888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int x, int y, int w, int h, nk_color color);

void cpu_render_text_rgba8888(
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    const uint8_t* atlas, int atlas_w, int dst_x, int dst_y, int w, int h,
    int src_x, int src_y, nk_color color);

void cpu_resize_bgr888_bilinear(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int dst_w, int dst_h, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h);

#endif // VISIONG_MODULES_GUI_RENDER_UTILS_H