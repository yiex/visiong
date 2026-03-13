// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/dma_alloc.h"
#include "im2d.hpp"
#include "rga.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

//------------------------------------------------------------------------------
// Software color conversion helpers / 软件 color conversion 辅助函数
//------------------------------------------------------------------------------




namespace {

int align_up(int value, int base) {
    return (value + base - 1) & (~(base - 1));
}

void copy_row_bytes_simd(char* dst_row, const char* src_row, int copy_width_bytes) {
#if defined(__ARM_NEON)
    int offset = 0;
    for (; offset + 64 <= copy_width_bytes; offset += 64) {
        uint8x16_t v0 = vld1q_u8(reinterpret_cast<const uint8_t*>(src_row + offset + 0));
        uint8x16_t v1 = vld1q_u8(reinterpret_cast<const uint8_t*>(src_row + offset + 16));
        uint8x16_t v2 = vld1q_u8(reinterpret_cast<const uint8_t*>(src_row + offset + 32));
        uint8x16_t v3 = vld1q_u8(reinterpret_cast<const uint8_t*>(src_row + offset + 48));
        vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + offset + 0), v0);
        vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + offset + 16), v1);
        vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + offset + 32), v2);
        vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + offset + 48), v3);
    }
    for (; offset + 16 <= copy_width_bytes; offset += 16) {
        uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(src_row + offset));
        vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + offset), v);
    }
    if (offset < copy_width_bytes) {
        std::memcpy(dst_row + offset, src_row + offset, static_cast<size_t>(copy_width_bytes - offset));
    }
#else
    std::memcpy(dst_row, src_row, static_cast<size_t>(copy_width_bytes));
#endif
}

}  // namespace

int get_bpp_for_format(int format) {
    switch (format) {
        case RK_FMT_RGB565:
        case RK_FMT_BGR565:
            return 16;
        case RK_FMT_RGB888:
        case RK_FMT_BGR888:
            return 24;
        case RK_FMT_RGBA8888:
        case RK_FMT_BGRA8888:
            return 32;
        case RK_FMT_YUV420SP:
        case RK_FMT_YUV420SP_VU:
            return 12;
        case static_cast<int>(visiong::kGray8Format):
            return 8;
        default:
            throw std::invalid_argument("Unsupported pixel format in get_bpp_for_format: " + std::to_string(format));
    }
}

const char* PixelFormatToString(int format) {
    switch (static_cast<int>(format)) {
        case RK_FMT_RGB888:
            return "RGB888";
        case RK_FMT_BGR888:
            return "BGR888";
        case RK_FMT_YUV420SP:
            return "YUV420SP";
        case RK_FMT_YUV420SP_VU:
            return "YUV420SP_VU";
        case RK_FMT_RGBA8888:
            return "RGBA8888";
        case RK_FMT_RGB565:
            return "BGR565";
        case RK_FMT_BGR565:
            return "RGB565";
        case static_cast<int>(visiong::kGray8Format):
            return "GRAY8";
        default:
            return "Unknown";
    }
}

int convert_mpi_to_rga_format(int mpi_format) {
    switch (mpi_format) {
        case RK_FMT_RGB565:
            return RK_FORMAT_BGR_565;
        case RK_FMT_BGR565:
            return RK_FORMAT_RGB_565;
        case RK_FMT_RGB888:
            return RK_FORMAT_RGB_888;
        case RK_FMT_RGBA8888:
            return RK_FORMAT_RGBA_8888;
        case RK_FMT_BGR888:
            return RK_FORMAT_BGR_888;
        case RK_FMT_BGRA8888:
            return RK_FORMAT_BGRA_8888;
        case RK_FMT_YUV420SP:
            return RK_FORMAT_YCbCr_420_SP;
        case RK_FMT_YUV420SP_VU:
            return RK_FORMAT_YCrCb_420_SP;
        case static_cast<int>(visiong::kGray8Format):
            return RK_FORMAT_YCbCr_400;
        default:
            throw std::invalid_argument("Unsupported MPI format for RGA conversion: " + std::to_string(mpi_format));
    }
}

void copy_data_with_stride(void* dst,
                           int dst_stride_bytes,
                           const void* src,
                           int src_stride_bytes,
                           int height,
                           int copy_width_bytes) {
    if (dst_stride_bytes == src_stride_bytes && dst_stride_bytes == copy_width_bytes) {
        std::memcpy(dst, src, static_cast<size_t>(src_stride_bytes) * height);
        return;
    }

    char* dst_row = static_cast<char*>(dst);
    const char* src_row = static_cast<const char*>(src);
    for (int i = 0; i < height; ++i) {
        copy_row_bytes_simd(dst_row, src_row, copy_width_bytes);
        dst_row += dst_stride_bytes;
        src_row += src_stride_bytes;
    }
}

void copy_data_from_stride(void* dst,
                           const void* src,
                           int dst_width_bytes,
                           int height,
                           int src_stride_bytes) {
    if (dst_width_bytes == src_stride_bytes) {
        std::memcpy(dst, src, static_cast<size_t>(src_stride_bytes) * height);
        return;
    }

    char* dst_row = static_cast<char*>(dst);
    const char* src_row = static_cast<const char*>(src);
    for (int i = 0; i < height; ++i) {
        copy_row_bytes_simd(dst_row, src_row, dst_width_bytes);
        dst_row += dst_width_bytes;
        src_row += src_stride_bytes;
    }
}

RgaDmaBuffer::RgaDmaBuffer(int width, int height, int format, int wstride, int hstride, size_t size)
    : m_width(width),
      m_height(height),
      m_wstride((wstride > 0) ? wstride : align_up(width, 16)),
      m_hstride((hstride > 0) ? hstride : height),
      m_mpi_format(format),
      m_size(0),
      m_fd(-1),
      m_handle(0),
      m_vir_addr(nullptr),
      m_is_owner(true) {
    if (size > 0) {
        m_size = size;
    } else {
        const int bpp = get_bpp_for_format(format);
        m_size = static_cast<size_t>(m_wstride) * m_hstride * bpp / 8;
    }

    if (dma_buf_alloc(RV1106_CMA_HEAP_PATH, m_size, &m_fd, &m_vir_addr) < 0) {
        throw std::runtime_error("RgaDmaBuffer: DMA alloc failed.");
    }

    m_handle = importbuffer_fd(m_fd, m_size);
    if (m_handle == 0) {
        dma_buf_free(m_size, &m_fd, m_vir_addr);
        m_fd = -1;
        m_vir_addr = nullptr;
        throw std::runtime_error("RgaDmaBuffer: import DMA failed.");
    }
}

RgaDmaBuffer::RgaDmaBuffer(int fd,
                           void* vir_addr,
                           size_t size,
                           int width,
                           int height,
                           int format,
                           int wstride,
                           int hstride)
    : m_width(width),
      m_height(height),
      m_wstride(wstride),
      m_hstride(hstride),
      m_mpi_format(format),
      m_size(size),
      m_fd(fd),
      m_handle(0),
      m_vir_addr(vir_addr),
      m_is_owner(false) {
    m_handle = importbuffer_fd(m_fd, m_size);
    if (m_handle == 0) {
        throw std::runtime_error("RgaDmaBuffer: import existing DMA fd failed.");
    }
}

RgaDmaBuffer::~RgaDmaBuffer() {
    if (m_handle != 0) {
        releasebuffer_handle(m_handle);
    }
    if (m_is_owner && m_fd != -1) {
        dma_buf_free(m_size, &m_fd, m_vir_addr);
    }
}

rga_buffer_t RgaDmaBuffer::get_buffer() const {
    return wrapbuffer_handle(m_handle,
                             m_width,
                             m_height,
                             convert_mpi_to_rga_format(m_mpi_format),
                             m_wstride,
                             m_hstride);
}

#if defined(__ARM_NEON)
// Use NEON to swap R/B channels for BGR<->RGB conversion. / 使用 NEON 以 swap R/B channels 用于 BGR<->RGB conversion.
static void bgr888_to_rgb888_neon(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    dst_data.resize(src.width * src.height * 3);
    const uint8_t* src_base = (const uint8_t*)src.get_data();
    uint8_t* dst_base = dst_data.data();

    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride * 3;
        uint8_t* dst_row = dst_base + y * src.width * 3;
        int x = 0;
        // Process 16 pixels per iteration (48 bytes). / 详见英文原注释。
        for (; x <= src.width - 16; x += 16) {
            uint8x16x3_t bgr_pixels = vld3q_u8(src_row + x * 3);
            uint8x16x3_t rgb_pixels;
            rgb_pixels.val[0] = bgr_pixels.val[2]; // R
            rgb_pixels.val[1] = bgr_pixels.val[1]; // G
            rgb_pixels.val[2] = bgr_pixels.val[0]; // B
            vst3q_u8(dst_row + x * 3, rgb_pixels);
        }
        for (; x < src.width; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 3 + 2]; // R
            dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // G
            dst_row[x * 3 + 2] = src_row[x * 3 + 0]; // B
        }
    }
}
#else
// Portable C++ fallback for non-NEON targets. / Portable C++ 回退 用于 non-NEON targets.
static void bgr888_to_rgb888_c(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    dst_data.resize(src.width * src.height * 3);
    const uint8_t* src_base = (const uint8_t*)src.get_data();
    uint8_t* dst_base = dst_data.data();
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride * 3;
        uint8_t* dst_row = dst_base + y * src.width * 3;
        for (int x = 0; x < src.width; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 3 + 2]; // R
            dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // G
            dst_row[x * 3 + 2] = src_row[x * 3 + 0]; // B
        }
    }
}
#endif
void gray_to_rgba8888_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    dst_data.resize(src.width * src.height * 4);
    const uint8_t* src_base = (const uint8_t*)src.get_data();
    uint8_t* dst_base = dst_data.data();

#if defined(__ARM_NEON)
    // NEON optimized grayscale->RGBA conversion / NEON optimized 灰度图->RGBA conversion
    uint8x16x4_t rgba_pixels;
    rgba_pixels.val[3] = vdupq_n_u8(255); // Alpha channel is fully opaque.

    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint8_t* dst_row = dst_base + y * src.width * 4;
        int x = 0;
        for (; x <= src.width - 16; x += 16) {
            uint8x16_t gray = vld1q_u8(src_row + x);
            rgba_pixels.val[0] = gray; // R
            rgba_pixels.val[1] = gray; // G
            rgba_pixels.val[2] = gray; // B
            vst4q_u8(dst_row + x * 4, rgba_pixels);
        }
        // Remaining pixels that don't fit one NEON vector. / Remaining pixels don't fit 一个 NEON vector.
        for (; x < src.width; ++x) {
            uint8_t val = src_row[x];
            dst_row[x * 4 + 0] = val; // R
            dst_row[x * 4 + 1] = val; // G
            dst_row[x * 4 + 2] = val; // B
            dst_row[x * 4 + 3] = 255; // A
        }
    }
#else
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint8_t* dst_row = dst_base + y * src.width * 4;
        for (int x = 0; x < src.width; ++x) {
            uint8_t val = src_row[x];
            dst_row[x * 4 + 0] = val; // R
            dst_row[x * 4 + 1] = val; // G
            dst_row[x * 4 + 2] = val; // B
            dst_row[x * 4 + 3] = 255; // A
        }
    }
#endif
}
// Grayscale -> YUV420SP / 灰度图 -> YUV420SP
void gray_to_yuv420sp_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    int width = src.width;
    int height = src.height;
    size_t y_plane_size = static_cast<size_t>(width) * height;
    size_t uv_plane_size = y_plane_size / 2;
    dst_data.resize(y_plane_size + uv_plane_size);

    const uint8_t* src_base = static_cast<const uint8_t*>(src.get_data());
    uint8_t* dst_y_plane = dst_data.data();
    uint8_t* dst_uv_plane = dst_data.data() + y_plane_size;

    copy_data_with_stride(dst_y_plane, width, src_base, src.w_stride, height, width);

#if defined(__ARM_NEON)
    int uv_pixels = width * height / 2;
    int i = 0;
    uint8x16_t val_128 = vdupq_n_u8(128);
    for (; i <= uv_pixels - 16; i += 16) {
        vst1q_u8(dst_uv_plane + i, val_128);
    }
    for (; i < uv_pixels; ++i) {
        dst_uv_plane[i] = 128;
    }
#else
    memset(dst_uv_plane, 128, uv_plane_size);
#endif
}

void gray_to_bgr888_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    dst_data.resize(src.width * src.height * 3);
    const uint8_t* src_base = (const uint8_t*)src.get_data();
    uint8_t* dst_base = dst_data.data();
#if defined(__ARM_NEON)
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint8_t* dst_row = dst_base + y * src.width * 3;
        for (int x = 0; x <= src.width - 16; x += 16) {
            uint8x16_t gray = vld1q_u8(src_row + x);
            uint8x16x3_t bgr = {{gray, gray, gray}};
            vst3q_u8(dst_row + x * 3, bgr);
        }
        for (int x = (src.width / 16) * 16; x < src.width; ++x) {
            dst_row[x * 3 + 0] = dst_row[x * 3 + 1] = dst_row[x * 3 + 2] = src_row[x];
        }
    }
#else
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint8_t* dst_row = dst_base + y * src.width * 3;
        for (int x = 0; x < src.width; ++x) {
            dst_row[x * 3] = dst_row[x * 3 + 1] = dst_row[x * 3 + 2] = src_row[x];
        }
    }
#endif
}

void gray_to_rgb888_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    gray_to_bgr888_sw(src, dst_data);
}

void gray_to_rgb565_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    visiong::bufstate::prepare_cpu_read(src);
    dst_data.resize(src.width * src.height * 2);
    const uint8_t* src_base = (const uint8_t*)src.get_data();
    uint16_t* dst_base = (uint16_t*)dst_data.data();
#if defined(__ARM_NEON)
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint16_t* dst_row = dst_base + y * src.width;
        int x = 0;
        for (; x <= src.width - 8; x += 8) {
            uint8x8_t gray_u8 = vld1_u8(src_row + x);
            uint16x8_t gray_u16 = vmovl_u8(gray_u8);
            uint16x8_t r = vshlq_n_u16(vshrq_n_u16(gray_u16, 3), 11);
            uint16x8_t g = vshlq_n_u16(vshrq_n_u16(gray_u16, 2), 5);
            uint16x8_t b = vshrq_n_u16(gray_u16, 3);
            vst1q_u16(dst_row + x, vorrq_u16(vorrq_u16(r, g), b));
        }
        for (; x < src.width; ++x) {
            uint8_t val = src_row[x];
            dst_row[x] = ((val & 0xF8) << 8) | ((val & 0xFC) << 3) | (val >> 3);
        }
    }
#else
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* src_row = src_base + y * src.w_stride;
        uint16_t* dst_row = dst_base + y * src.width;
        for (int x = 0; x < src.width; ++x) {
            uint8_t val = src_row[x];
            dst_row[x] = ((val & 0xF8) << 8) | ((val & 0xFC) << 3) | (val >> 3);
        }
    }
#endif
}

void gray_to_bgr565_sw(const ImageBuffer& src, std::vector<unsigned char>& dst_data) {
    gray_to_rgb565_sw(src, dst_data);
}

//################################################################################# / 详见英文原注释。
//##
// RGA helper operations / RGA 辅助函数 operations
//##
//################################################################################# / 详见英文原注释。

void rga_letterbox_op(const RgaDmaBuffer& src_dma,
                      const RgaDmaBuffer& dst_dma,
                      std::tuple<uint8_t, uint8_t, uint8_t> color,
                      bool sync_src_cpu_to_device,
                      bool sync_dst_device_to_cpu) {
    (void)sync_src_cpu_to_device;
    (void)sync_dst_device_to_cpu;
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);

    rga_buffer_t src_buf = src_dma.get_buffer();
    rga_buffer_t dst_buf = dst_dma.get_buffer();

    int target_width = dst_buf.width;
    int target_height = dst_buf.height;
    int src_width = src_buf.width;
    int src_height = src_buf.height;
    float scale = std::min(static_cast<float>(target_width) / src_width, static_cast<float>(target_height) / src_height);
    int new_w = static_cast<int>(src_width * scale) & ~1;
    int new_h = static_cast<int>(src_height * scale) & ~1;
    int pad_x = (target_width - new_w) / 2;
    int pad_y = (target_height - new_h) / 2;
    im_rect full_rect = {0, 0, target_width, target_height};
    uint32_t r = std::get<0>(color);
    uint32_t g = std::get<1>(color);
    uint32_t b = std::get<2>(color);
    // RGA fill color layout is BGRA. / RGA fill color layout 为 BGRA.
    uint32_t fill_color = (0xFF << 24) | (b << 16) | (g << 8) | r;
    if (imfill(dst_buf, full_rect, fill_color) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA letterbox step (imfill) failed");
    }
    im_rect src_rect = {0, 0, src_width, src_height};
    im_rect dst_paste_rect = {pad_x, pad_y, new_w, new_h};
    if (improcess(src_buf, dst_buf, {}, src_rect, dst_paste_rect, {}, IM_SYNC) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA letterbox step (improcess) failed");
    }

    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
}


void rga_letterbox_yuv_op(const rga_buffer_t& src_buf, const RgaDmaBuffer& dst_dma_buf, int scaled_w, int scaled_h, int pad_x, int pad_y) {
    void* dst_vir_addr = dst_dma_buf.get_vir_addr();
    if (!dst_vir_addr) {
        throw std::runtime_error("rga_letterbox_yuv_op: Destination buffer has no virtual address for manual fill.");
    }
    visiong::bufstate::prepare_cpu_write(dst_dma_buf, visiong::bufstate::AccessIntent::Overwrite);
    int y_plane_size = dst_dma_buf.get_wstride() * dst_dma_buf.get_hstride();
    memset(dst_vir_addr, 0, y_plane_size);
    void* uv_plane_start = static_cast<char*>(dst_vir_addr) + y_plane_size;
    size_t uv_plane_size = dst_dma_buf.get_size() - y_plane_size;
    memset(uv_plane_start, 128, uv_plane_size);
    visiong::bufstate::mark_cpu_write(dst_dma_buf);
    visiong::bufstate::prepare_device_write(dst_dma_buf,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::ReadModifyWrite);

    im_rect src_rect = {0, 0, src_buf.width, src_buf.height};
    im_rect dst_paste_rect = {pad_x, pad_y, scaled_w, scaled_h};
    IM_STATUS ret = improcess(src_buf, dst_dma_buf.get_buffer(), {}, src_rect, dst_paste_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        throw std::runtime_error("rga_letterbox_yuv_op step (improcess) failed");
    }

    visiong::bufstate::mark_device_write(dst_dma_buf, visiong::bufstate::BufferOwner::RGA);
}

//################################################################################# / 详见英文原注释。
//##
// ImageBuffer member implementations / 图像缓冲区 member implementations
//#################################################################################
// For BGR<->RGB888 channel swap, NEON is usually faster than RGA. / 用于 BGR<->RGB888 channel swap, NEON 为 usually faster than RGA.
// Portable C++ fallback implementation.
[[maybe_unused]] static void cpu_resize_bgr888_bilinear_c(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int /*dst_w*/, int /*dst_h*/, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h) 
{
    const float x_ratio = (float)(src_w - 1) / roi_w;
    const float y_ratio = (float)(src_h - 1) / roi_h;
    
    for (int y_dst = 0; y_dst < roi_h; ++y_dst) {
        float y_src_f = y_dst * y_ratio;
        int y_src_i = (int)y_src_f;
        float y_diff = y_src_f - y_src_i;

        const uint8_t* row1 = src + std::min(y_src_i, src_h - 2) * src_stride_pixels * 3;
        const uint8_t* row2 = src + std::min(y_src_i + 1, src_h - 1) * src_stride_pixels * 3;
        uint8_t* dst_row = dst + (roi_y + y_dst) * dst_stride_pixels * 3 + roi_x * 3;

        for (int x_dst = 0; x_dst < roi_w; ++x_dst) {
            float x_src_f = x_dst * x_ratio;
            int x_src_i = (int)x_src_f;
            float x_diff = x_src_f - x_src_i;

            int x_src_i_clamped = std::min(x_src_i, src_w - 2);
            const uint8_t* p1 = row1 + x_src_i_clamped * 3;
            const uint8_t* p2 = row1 + (x_src_i_clamped + 1) * 3;
            const uint8_t* p3 = row2 + x_src_i_clamped * 3;
            const uint8_t* p4 = row2 + (x_src_i_clamped + 1) * 3;

            for (int c = 0; c < 3; ++c) {
                float top = p1[c] * (1.0f - x_diff) + p2[c] * x_diff;
                float bottom = p3[c] * (1.0f - x_diff) + p4[c] * x_diff;
                dst_row[x_dst * 3 + c] = (uint8_t)(top * (1.0f - y_diff) + bottom * y_diff);
            }
        }
    }
}

#if defined(__ARM_NEON)
// NEON version. / 详见英文原注释。
[[maybe_unused]] static void cpu_resize_bgr888_bilinear_neon(
    const uint8_t* src, int src_w, int src_h, int src_stride_pixels,
    uint8_t* dst, int /*dst_w*/, int /*dst_h*/, int dst_stride_pixels,
    int roi_x, int roi_y, int roi_w, int roi_h)
{
    const int FRAC_BITS = 8;
    const int FRAC_VAL = 1 << FRAC_BITS;
    
    uint64_t x_ratio = ((uint64_t)(src_w - 1) * FRAC_VAL) / roi_w;
    uint64_t y_ratio = ((uint64_t)(src_h - 1) * FRAC_VAL) / roi_h;

    uint64_t y_src_fixed = 0;

    for (int y = 0; y < roi_h; ++y) {
        int y_src_i = y_src_fixed >> FRAC_BITS;
        y_src_i = std::min(y_src_i, src_h - 2);
        
        uint16_t y_weight = y_src_fixed & (FRAC_VAL - 1);
        uint16x8_t y_w_u16 = vdupq_n_u16(y_weight);
        uint16x8_t y_inv_w_u16 = vdupq_n_u16(FRAC_VAL - y_weight);
        
        const uint8_t* row1 = src + y_src_i * src_stride_pixels * 3;
        const uint8_t* row2 = src + (y_src_i + 1) * src_stride_pixels * 3;
        uint8_t* dst_row = dst + (roi_y + y) * dst_stride_pixels * 3 + roi_x * 3;
        
        uint64_t x_src_fixed = 0;
        int x = 0;

        for (; x <= roi_w - 8; x += 8) {
            uint32_t x_coords_fixed[8];
            for(int i=0; i<8; ++i) x_coords_fixed[i] = x_src_fixed + i * x_ratio;

            uint32x4_t x_src_fixed_vec1 = vld1q_u32(x_coords_fixed);
            uint32x4_t x_src_fixed_vec2 = vld1q_u32(x_coords_fixed + 4);

            uint32x4_t x_src_i_vec1 = vshrq_n_u32(x_src_fixed_vec1, FRAC_BITS);
            uint32x4_t x_src_i_vec2 = vshrq_n_u32(x_src_fixed_vec2, FRAC_BITS);
            
            uint16x8_t x_w_u16 = vcombine_u16(vmovn_u32(x_src_fixed_vec1), vmovn_u32(x_src_fixed_vec2));
            x_w_u16 = vandq_u16(x_w_u16, vdupq_n_u16(FRAC_VAL - 1));
            uint16x8_t x_inv_w_u16 = vsubq_u16(vdupq_n_u16(FRAC_VAL), x_w_u16);
            
            uint32_t x_indices[8];
            vst1q_u32(x_indices, x_src_i_vec1);
            vst1q_u32(x_indices + 4, x_src_i_vec2);
            
            uint8x8x3_t p1, p2, p3, p4;
            for(int i=0; i<8; ++i) {
                int clamped_idx = std::min((int)x_indices[i], src_w - 2);
                for(int c=0; c<3; ++c) {
                    p1.val[c][i] = row1[clamped_idx * 3 + c];
                    p2.val[c][i] = row1[(clamped_idx + 1) * 3 + c];
                    p3.val[c][i] = row2[clamped_idx * 3 + c];
                    p4.val[c][i] = row2[(clamped_idx + 1) * 3 + c];
                }
            }
            
            uint8x8x3_t result_pixels;
            for (int c = 0; c < 3; ++c) {
                // Horizontal interpolation with 32-bit accumulators avoids overflow. / Horizontal interpolation 与 32-bit accumulators avoids overflow.
                uint32x4_t top_lo = vmull_u16(vget_low_u16(vmovl_u8(p1.val[c])), vget_low_u16(x_inv_w_u16));
                top_lo = vmlal_u16(top_lo, vget_low_u16(vmovl_u8(p2.val[c])), vget_low_u16(x_w_u16));
                uint32x4_t top_hi = vmull_u16(vget_high_u16(vmovl_u8(p1.val[c])), vget_high_u16(x_inv_w_u16));
                top_hi = vmlal_u16(top_hi, vget_high_u16(vmovl_u8(p2.val[c])), vget_high_u16(x_w_u16));

                uint32x4_t bot_lo = vmull_u16(vget_low_u16(vmovl_u8(p3.val[c])), vget_low_u16(x_inv_w_u16));
                bot_lo = vmlal_u16(bot_lo, vget_low_u16(vmovl_u8(p4.val[c])), vget_low_u16(x_w_u16));
                uint32x4_t bot_hi = vmull_u16(vget_high_u16(vmovl_u8(p3.val[c])), vget_high_u16(x_inv_w_u16));
                bot_hi = vmlal_u16(bot_hi, vget_high_u16(vmovl_u8(p4.val[c])), vget_high_u16(x_w_u16));

                // Vertical blend step. / 详见英文原注释。
                uint32x4_t blend_lo = vmulq_u32(top_lo, vmovl_u16(vget_low_u16(y_inv_w_u16)));
                blend_lo = vmlaq_u32(blend_lo, bot_lo, vmovl_u16(vget_low_u16(y_w_u16)));
                uint32x4_t blend_hi = vmulq_u32(top_hi, vmovl_u16(vget_high_u16(y_inv_w_u16)));
                blend_hi = vmlaq_u32(blend_hi, bot_hi, vmovl_u16(vget_high_u16(y_w_u16)));

                // Normalize fixed-point accumulators back to 8-bit lanes. / Normalize fixed-point accumulators back 以 8-bit lanes.
                uint16x8_t final_u16 = vcombine_u16(vshrn_n_u32(blend_lo, FRAC_BITS * 2), vshrn_n_u32(blend_hi, FRAC_BITS * 2));
                result_pixels.val[c] = vqmovn_u16(final_u16);
            }

            vst3_u8(dst_row + x * 3, result_pixels);
            x_src_fixed += (uint64_t)x_ratio * 8;
        }
        
        // C++ for remaining pixels / C++ 用于 remaining pixels
        for (; x < roi_w; ++x) {
            int x_src_i = x_src_fixed >> FRAC_BITS;
            x_src_i = std::min(x_src_i, src_w - 2);
            int x_weight = x_src_fixed & (FRAC_VAL - 1);

            for (int c = 0; c < 3; ++c) {
                int p1_c = row1[x_src_i * 3 + c];
                int p2_c = row1[(x_src_i + 1) * 3 + c];
                int p3_c = row2[x_src_i * 3 + c];
                int p4_c = row2[(x_src_i + 1) * 3 + c];
                int top = (p1_c * (FRAC_VAL - x_weight) + p2_c * x_weight);
                int bottom = (p3_c * (FRAC_VAL - x_weight) + p4_c * x_weight);
                dst_row[x * 3 + c] = (uint8_t)(((uint64_t)top * (FRAC_VAL - y_weight) + (uint64_t)bottom * y_weight) >> (FRAC_BITS * 2));
            }
            x_src_fixed += x_ratio;
        }
        y_src_fixed += y_ratio;
    }
}
#endif
ImageBuffer ImageBuffer::to_format(PIXEL_FORMAT_E new_format) const {
    if (!is_valid()) throw std::runtime_error("to_format: invalid source image");
    if (this->format == new_format) return this->copy();

    // NOTE: BGR<->RGB is intentionally routed through the RGA conversion path below / NOTE: BGR<->RGB 为 intentionally routed through RGA conversion 路径 below
    // so callers can rely on a single hardware conversion pipeline.

    if (this->format == visiong::kGray8Format) {
        std::vector<unsigned char> converted_data;
        switch(new_format) {
            case RK_FMT_BGR888: gray_to_bgr888_sw(*this, converted_data); break;
            case RK_FMT_RGB888: gray_to_rgb888_sw(*this, converted_data); break;
            case RK_FMT_BGR565: gray_to_bgr565_sw(*this, converted_data); break;
            case RK_FMT_RGB565: gray_to_rgb565_sw(*this, converted_data); break;
            case RK_FMT_YUV420SP: case RK_FMT_YUV420SP_VU: gray_to_yuv420sp_sw(*this, converted_data); break;
            case RK_FMT_RGBA8888:
            case RK_FMT_BGRA8888:
                // Grayscale->BGRA and Grayscale->RGBA share the same channel values. / 灰度图->BGRA 与 灰度图->RGBA share same channel values.
                gray_to_rgba8888_sw(*this, converted_data);
                break;
            default:
                if (new_format == visiong::kGray8Format) { return this->copy(); }
                throw std::runtime_error("Unsupported software conversion from Grayscale to target format " + std::to_string(new_format));
        }
        return ImageBuffer(this->width, this->height, new_format, std::move(converted_data));
    }

    // Zero-copy path: wrap existing DMA source and convert directly into DMA dst. / 零拷贝路径: wrap 现有 DMA 源 与 convert 直接 into DMA dst.
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
        visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
        auto dst_sptr = std::make_shared<RgaDmaBuffer>(this->width, this->height, static_cast<int>(new_format));
        visiong::bufstate::prepare_device_write(*dst_sptr,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (imcvtcolor(src_wrapper.get_buffer(), dst_sptr->get_buffer(), convert_mpi_to_rga_format(this->format), convert_mpi_to_rga_format(new_format)) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA imcvtcolor failed");
        }
        ImageBuffer out(this->width, this->height, new_format, std::move(dst_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    RgaDmaBuffer src_dma(this->width, this->height, static_cast<int>(this->format));
    int bpp = get_bpp_for_format(this->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, this->get_data(), this->w_stride * bpp / 8, this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

    RgaDmaBuffer dst_dma(this->width, this->height, new_format);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);
    if (imcvtcolor(src_dma.get_buffer(), dst_dma.get_buffer(), convert_mpi_to_rga_format(this->format), convert_mpi_to_rga_format(new_format)) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcvtcolor failed");
    }
    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);

    ImageBuffer dst_img;
    dst_img.copy_from_dma(dst_dma);
    return dst_img;
}

ImageBuffer ImageBuffer::to_grayscale() const {
    return this->to_format(visiong::kGray8Format);
}

ImageBuffer ImageBuffer::resize(int new_width, int new_height) const {
    if (!is_valid()) throw std::runtime_error("resize: invalid source image");

    if (new_width < 41 || new_height < 41 || this->width % 2 != 0 || this->height % 2 != 0 || new_width % 2 != 0 || new_height % 2 != 0) {
        const ImageBuffer& bgr_src = this->get_bgr_version();
        visiong::bufstate::prepare_cpu_read(bgr_src);
        std::vector<unsigned char> dst_data(new_width * new_height * 3);

        #if defined(__ARM_NEON)
        cpu_resize_bgr888_bilinear_neon((const uint8_t*)bgr_src.get_data(), bgr_src.width, bgr_src.height, bgr_src.w_stride, dst_data.data(), new_width, new_height, new_width, 0, 0, new_width, new_height);
        #else
        cpu_resize_bgr888_bilinear_c((const uint8_t*)bgr_src.get_data(), bgr_src.width, bgr_src.height, bgr_src.w_stride, dst_data.data(), new_width, new_height, new_width, 0, 0, new_width, new_height);
        #endif

        return ImageBuffer(new_width, new_height, RK_FMT_BGR888, std::move(dst_data));
    }

    // Zero-copy path for resize. / 零拷贝路径 用于 缩放.
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
        visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
        auto dst_sptr = std::make_shared<RgaDmaBuffer>(new_width, new_height, static_cast<int>(this->format));
        visiong::bufstate::prepare_device_write(*dst_sptr,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (imresize(src_wrapper.get_buffer(), dst_sptr->get_buffer()) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA imresize failed");
        }
        ImageBuffer out(new_width, new_height, this->format, std::move(dst_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    RgaDmaBuffer src_dma(this->width, this->height, static_cast<int>(this->format));
    int bpp = get_bpp_for_format(this->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, this->get_data(), this->w_stride * bpp / 8, this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

    RgaDmaBuffer dst_dma(new_width, new_height, this->format);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);
    if (imresize(src_dma.get_buffer(), dst_dma.get_buffer()) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed");
    }
    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);

    ImageBuffer dst_img;
    dst_img.copy_from_dma(dst_dma);
    return dst_img;
}

ImageBuffer ImageBuffer::crop(const std::tuple<int, int, int, int>& rect_tuple) const {
    if (!is_valid()) {
        throw std::runtime_error("crop: invalid source image");
    }

    int x = std::get<0>(rect_tuple);
    int y = std::get<1>(rect_tuple);
    int w = std::get<2>(rect_tuple);
    int h = std::get<3>(rect_tuple);

    // Zero-copy path for letterbox. / 零拷贝路径 用于 letterbox.
    int adj_x = x & ~1;
    int adj_y = y & ~1;
    int adj_w = w & ~1;
    int adj_h = h & ~1;

    if (adj_w <= 0 || adj_h <= 0) {
        throw std::runtime_error("crop: Crop dimensions must be positive after alignment to even numbers.");
    }
    if ((adj_x + adj_w) > this->width || (adj_y + adj_h) > this->height) {
        std::string err_msg = "crop: Adjusted crop rectangle [" + std::to_string(adj_x) + ", " +
                              std::to_string(adj_y) + ", " + std::to_string(adj_w) + ", " + std::to_string(adj_h) +
                              "] is out of source image bounds (" + std::to_string(this->width) + "x" + std::to_string(this->height) + ").";
        throw std::runtime_error(err_msg);
    }

    im_rect crop_rect = {adj_x, adj_y, adj_w, adj_h};

    // Zero-copy path for crop. / 零拷贝路径 用于 裁切.
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
        visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
        auto dst_sptr = std::make_shared<RgaDmaBuffer>(adj_w, adj_h, static_cast<int>(this->format));
        visiong::bufstate::prepare_device_write(*dst_sptr,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (imcrop(src_wrapper.get_buffer(), dst_sptr->get_buffer(), crop_rect) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA imcrop failed.");
        }
        ImageBuffer out(adj_w, adj_h, this->format, std::move(dst_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    // CPU fallback: upload to temporary DMA, crop via RGA, then copy back. / CPU 回退: upload 以 临时 DMA, 裁切 via RGA, 然后 复制 back.
    RgaDmaBuffer src_dma(this->width, this->height, static_cast<int>(this->format));
    int bpp = get_bpp_for_format(this->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8,
                          this->get_data(), this->w_stride * bpp / 8,
                          this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

    RgaDmaBuffer dst_dma(adj_w, adj_h, this->format);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);
    if (imcrop(src_dma.get_buffer(), dst_dma.get_buffer(), crop_rect) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcrop failed.");
    }
    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);

    ImageBuffer dst_img;
    dst_img.copy_from_dma(dst_dma);
    return dst_img;
}

ImageBuffer ImageBuffer::letterbox(int target_width, int target_height, std::tuple<unsigned char, unsigned char, unsigned char> color) const {
    if (!is_valid()) throw std::runtime_error("letterbox: invalid source image");

    // Zero-copy path for letterbox. / 零拷贝路径 用于 letterbox.
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
        auto dst_sptr = std::make_shared<RgaDmaBuffer>(target_width, target_height, static_cast<int>(this->format));
        rga_letterbox_op(src_wrapper, *dst_sptr, color);
        ImageBuffer out(target_width, target_height, this->format, std::move(dst_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    RgaDmaBuffer src_dma(this->width, this->height, static_cast<int>(this->format));
    int bpp = get_bpp_for_format(this->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, this->get_data(), this->w_stride * bpp / 8, this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);

    RgaDmaBuffer dst_dma(target_width, target_height, this->format);
    rga_letterbox_op(src_dma, dst_dma, color);

    ImageBuffer dst_img;
    dst_img.copy_from_dma(dst_dma);
    return dst_img;
}

ImageBuffer ImageBuffer::create(int w, int h, PIXEL_FORMAT_E fmt, std::tuple<unsigned char, unsigned char, unsigned char> color_rgb) {
    auto [r, g, b] = color_rgb;
    return create(w, h, fmt, std::make_tuple(r, g, b, 255));
}

ImageBuffer ImageBuffer::create(int width, int height, PIXEL_FORMAT_E format, std::tuple<unsigned char, unsigned char, unsigned char, unsigned char> color_rgba) {
    if (width < 2 || height < 2) throw std::runtime_error("create: dimensions must be >= 2");
    auto dst_sptr = std::make_shared<RgaDmaBuffer>(width, height, static_cast<int>(format));
    im_rect fill_rect = {0, 0, width, height};
    uint32_t r = std::get<0>(color_rgba), g = std::get<1>(color_rgba), b = std::get<2>(color_rgba), a = std::get<3>(color_rgba);
    uint32_t fill_color = (a << 24) | (b << 16) | (g << 8) | r;

    visiong::bufstate::prepare_device_write(*dst_sptr,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);
    if (imfill(dst_sptr->get_buffer(), fill_rect, fill_color) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imfill failed in create()");
    }
    ImageBuffer out(width, height, format, std::move(dst_sptr));
    visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
    return out;
}

ImageBuffer& ImageBuffer::paste(const ImageBuffer& img_to_paste, int x, int y) {
    if (!this->is_valid() || !img_to_paste.is_valid()) {
        throw std::runtime_error("paste: Both source and destination images must be valid.");
    }
    
    if (this->is_zero_copy()) {
        *this = this->copy();
    }
    
    ImageBuffer bgr_img_to_paste_owner;
    const ImageBuffer* bgr_img_to_paste = &img_to_paste;
    if (img_to_paste.format != RK_FMT_BGR888) {
        bgr_img_to_paste_owner = img_to_paste.to_format(RK_FMT_BGR888);
        bgr_img_to_paste = &bgr_img_to_paste_owner;
    }
    
    if (this->format != RK_FMT_BGR888) {
        *this = this->to_format(RK_FMT_BGR888);
    }
    
    // Allow partial paste by clipping to destination bounds / 允许 partial paste 由 clipping 以 目标 bounds
    int src_x = 0, src_y = 0;
    int dst_x = x, dst_y = y;
    int paste_w = bgr_img_to_paste->width;
    int paste_h = bgr_img_to_paste->height;

    if (dst_x < 0) { src_x = -dst_x; paste_w -= src_x; dst_x = 0; }
    if (dst_y < 0) { src_y = -dst_y; paste_h -= src_y; dst_y = 0; }
    if (dst_x + paste_w > this->width) { paste_w = this->width - dst_x; }
    if (dst_y + paste_h > this->height) { paste_h = this->height - dst_y; }

    if (paste_w <= 0 || paste_h <= 0) {
        throw std::runtime_error("paste: No overlap with destination image bounds.");
    }

    const bool dst_dma_backed = this->is_zero_copy() && this->get_dma_fd() >= 0;
    const bool src_dma_backed = bgr_img_to_paste->is_zero_copy() && bgr_img_to_paste->get_dma_fd() >= 0;
    const bool both_cpu_backed = !dst_dma_backed && !src_dma_backed;
    constexpr int kCpuPasteMaxPixels = 320 * 320;
    if (both_cpu_backed || paste_w * paste_h <= kCpuPasteMaxPixels) {
        unsigned char* dst_base = static_cast<unsigned char*>(this->get_data());
        const unsigned char* src_base = static_cast<const unsigned char*>(bgr_img_to_paste->get_data());
        const size_t row_bytes = static_cast<size_t>(paste_w) * 3;
        for (int row = 0; row < paste_h; ++row) {
            unsigned char* dst_row = dst_base +
                                     (static_cast<size_t>(dst_y + row) * this->w_stride + dst_x) * 3;
            const unsigned char* src_row = src_base +
                                           (static_cast<size_t>(src_y + row) * bgr_img_to_paste->w_stride + src_x) * 3;
            std::memmove(dst_row, src_row, row_bytes);
        }
        return *this;
    }

    // Let RgaDmaBuffer compute aligned strides for both source and destination. / Let RgaDma缓冲区 计算 对齐后的 strides 用于 both 源 与 目标.
    int bpp = get_bpp_for_format(bgr_img_to_paste->format);

    // Create source DMA buffer (small image). / Create 源 DMA 缓冲区 (小 图像).
    RgaDmaBuffer src_dma(bgr_img_to_paste->width, bgr_img_to_paste->height, bgr_img_to_paste->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, bgr_img_to_paste->get_data(), bgr_img_to_paste->w_stride * bpp / 8, bgr_img_to_paste->height, bgr_img_to_paste->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

    // Create destination DMA buffer (background image). / Create 目标 DMA 缓冲区 (background 图像).
    RgaDmaBuffer dst_dma(this->width, this->height, this->format);
    copy_data_with_stride(dst_dma.get_vir_addr(), dst_dma.get_wstride() * bpp / 8, this->get_data(), this->w_stride * bpp / 8, this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(dst_dma);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::ReadModifyWrite);
    
    im_rect src_rect = {src_x, src_y, paste_w, paste_h};
    im_rect paste_rect = {dst_x, dst_y, paste_w, paste_h};
    
    IM_STATUS ret = improcess(src_dma.get_buffer(), dst_dma.get_buffer(), {}, src_rect, paste_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA improcess for paste operation failed.");
    }

    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
    visiong::bufstate::prepare_cpu_read(dst_dma);
    copy_data_with_stride(this->get_data(), this->w_stride * bpp / 8, dst_dma.get_vir_addr(), dst_dma.get_wstride() * bpp / 8, this->height, this->width * bpp / 8);
    
    return *this;
}
void convert_bgr_to_compact_rgb(const ImageBuffer& src_bgr, std::vector<unsigned char>& dst_rgb_data) {
    visiong::bufstate::prepare_cpu_read(src_bgr);
#if defined(__ARM_NEON)
    bgr888_to_rgb888_neon(src_bgr, dst_rgb_data);
#else
    bgr888_to_rgb888_c(src_bgr, dst_rgb_data);
#endif
}

