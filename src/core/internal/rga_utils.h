// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <tuple>
#include <vector>

#include "visiong/core/RgaHelper.h"

struct ImageBuffer;

int get_bpp_for_format(int format);
const char* PixelFormatToString(int format);
int convert_mpi_to_rga_format(int mpi_format);
void copy_data_with_stride(void* dst,
                           int dst_stride_bytes,
                           const void* src,
                           int src_stride_bytes,
                           int height,
                           int copy_width_bytes);
void copy_data_from_stride(void* dst,
                           const void* src,
                           int dst_width_bytes,
                           int height,
                           int src_stride_bytes);
void rga_letterbox_op(const RgaDmaBuffer& src_dma,
                      const RgaDmaBuffer& dst_dma,
                      std::tuple<uint8_t, uint8_t, uint8_t> color,
                      bool sync_src_cpu_to_device = true,
                      bool sync_dst_device_to_cpu = true);
void rga_letterbox_yuv_op(const rga_buffer_t& src_buf,
                          const RgaDmaBuffer& dst_dma_buf,
                          int scaled_w,
                          int scaled_h,
                          int pad_x,
                          int pad_y);
void gray_to_yuv420sp_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data);
void convert_bgr_to_compact_rgb(const ImageBuffer& src_bgr, std::vector<unsigned char>& dst_rgb_data);

