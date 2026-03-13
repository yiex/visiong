// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayFB.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/dma_alloc.h"
#include "im2d.hpp"
#include "core/internal/logger.h"
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <chrono>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
// Keep NEON includes guarded for non-ARM builds. / 在非 ARM 构建下对 NEON 头文件进行条件保护。
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

struct RectROI {
    int x;
    int y;
    int w;
    int h;
};

struct DisplayTask {
    std::unique_ptr<ImageBuffer> image;
    RectROI roi;
};

constexpr char kFbDevice[] = "/dev/fb0";

struct DisplayFB::Impl {
    explicit Impl(Mode display_mode) : mode(display_mode) {}

    Mode mode;
    std::atomic<bool> initialized{false};
    std::atomic<bool> is_running{false};
    int fb_fd = -1;
    void* fb_mem = nullptr;
    long screen_size = 0;
    int screen_width = 0;
    int screen_height = 0;
    int bits_per_pixel = 0;
    int line_length = 0;
    int screen_format_mpi = RK_FMT_BUTT;
    std::thread display_thread;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    std::unique_ptr<DisplayTask> latest_task;
};
// Custom pseudo pixel formats used only inside DisplayFB. / 仅在 DisplayFB 内部使用的自定义伪像素格式。
constexpr PIXEL_FORMAT_E CUSTOM_FMT_BGR666_PACKED24 = static_cast<PIXEL_FORMAT_E>(0x2001);
constexpr PIXEL_FORMAT_E CUSTOM_FMT_RGB666_PACKED24 = static_cast<PIXEL_FORMAT_E>(0x2002);

static bool is_yuv420sp_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU;
}

static RectROI clamp_and_align_roi(const RectROI& requested_roi, const ImageBuffer& image) {
    RectROI roi = requested_roi;
    if (roi.w <= 0 || roi.h <= 0) {
        roi = {0, 0, image.width, image.height};
    }

    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    if (roi.x >= image.width || roi.y >= image.height) {
        throw std::runtime_error("DisplayFB ROI origin is outside source image bounds.");
    }

    roi.w = std::min(roi.w, image.width - roi.x);
    roi.h = std::min(roi.h, image.height - roi.y);

    if (is_yuv420sp_format(image.format)) {
        roi.x &= ~1;
        roi.y &= ~1;
        roi.w &= ~1;
        roi.h &= ~1;
        if (roi.x >= image.width || roi.y >= image.height) {
            throw std::runtime_error("DisplayFB ROI became invalid after YUV alignment.");
        }
        if (roi.x + roi.w > image.width) {
            roi.w = (image.width - roi.x) & ~1;
        }
        if (roi.y + roi.h > image.height) {
            roi.h = (image.height - roi.y) & ~1;
        }
    }

    if (roi.w <= 0 || roi.h <= 0) {
        throw std::runtime_error("DisplayFB ROI is empty after clamping/alignment.");
    }
    return roi;
}

static void copy_roi_to_dma_buffer(const ImageBuffer& src, const RectROI& roi, RgaDmaBuffer& dst_dma) {
    if (dst_dma.get_width() != roi.w || dst_dma.get_height() != roi.h) {
        throw std::runtime_error("DisplayFB ROI DMA size mismatch.");
    }
    if (dst_dma.get_mpi_format() != static_cast<int>(src.format)) {
        throw std::runtime_error("DisplayFB ROI DMA format mismatch.");
    }

    if (is_yuv420sp_format(src.format)) {
        const uint8_t* src_base = static_cast<const uint8_t*>(src.get_data());
        uint8_t* dst_base = static_cast<uint8_t*>(dst_dma.get_vir_addr());
        const int src_stride = src.w_stride;
        const int dst_stride = dst_dma.get_wstride();

        const uint8_t* src_y = src_base + static_cast<size_t>(roi.y) * src_stride + roi.x;
        copy_data_with_stride(dst_base, dst_stride, src_y, src_stride, roi.h, roi.w);

        const size_t src_y_plane_size = static_cast<size_t>(src.w_stride) * src.h_stride;
        const size_t dst_y_plane_size = static_cast<size_t>(dst_dma.get_wstride()) * dst_dma.get_hstride();
        const uint8_t* src_uv_base = src_base + src_y_plane_size;
        uint8_t* dst_uv_base = dst_base + dst_y_plane_size;
        const uint8_t* src_uv = src_uv_base + static_cast<size_t>(roi.y / 2) * src_stride + roi.x;
        copy_data_with_stride(dst_uv_base, dst_stride, src_uv, src_stride, roi.h / 2, roi.w);
    } else {
        const int bytes_per_pixel = get_bpp_for_format(src.format) / 8;
        const char* roi_src_ptr = static_cast<const char*>(src.get_data()) +
                                  static_cast<size_t>(roi.y) * src.w_stride * bytes_per_pixel +
                                  static_cast<size_t>(roi.x) * bytes_per_pixel;
        copy_data_with_stride(dst_dma.get_vir_addr(), dst_dma.get_wstride() * bytes_per_pixel, roi_src_ptr,
                              src.w_stride * bytes_per_pixel, roi.h, roi.w * bytes_per_pixel);
    }

    dma_sync_cpu_to_device(dst_dma.get_fd());
}


#if defined(__ARM_NEON)
// NEON: Gray8 -> BGR565 (BBBBBGGGGGGRRRRR) / NEON：Gray8 转 BGR565（BBBBBGGGGGGRRRRR）。
static void gray8_to_bgr565_neon(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    uint8_t* dst_rows = static_cast<uint8_t*>(dst);
    const uint8_t* src_rows = src;
    for (int y = 0; y < height; ++y) {
        uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(dst_rows);
        const uint8_t* src_ptr = src_rows;
        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16_t gray_u8 = vld1q_u8(src_ptr);
            uint16x8_t gray_u16_low = vmovl_u8(vget_low_u8(gray_u8));
            uint16x8_t gray_u16_high = vmovl_u8(vget_high_u8(gray_u8));
            uint16x8_t b_low = vshlq_n_u16(vshrq_n_u16(gray_u16_low, 3), 11);
            uint16x8_t g_low = vshlq_n_u16(vshrq_n_u16(gray_u16_low, 2), 5);
            uint16x8_t r_low = vshrq_n_u16(gray_u16_low, 3);
            uint16x8_t b_high = vshlq_n_u16(vshrq_n_u16(gray_u16_high, 3), 11);
            uint16x8_t g_high = vshlq_n_u16(vshrq_n_u16(gray_u16_high, 2), 5);
            uint16x8_t r_high = vshrq_n_u16(gray_u16_high, 3);
            vst1q_u16(dst_ptr, vorrq_u16(vorrq_u16(b_low, g_low), r_low));
            vst1q_u16(dst_ptr + 8, vorrq_u16(vorrq_u16(b_high, g_high), r_high));
            src_ptr += 16;
            dst_ptr += 16;
        }
        for (; x < width; ++x) {
            uint8_t y_val = *src_ptr++;
            uint16_t b = (y_val & 0xF8) >> 3, g = (y_val & 0xFC) >> 2, r = (y_val & 0xF8) >> 3;
            *dst_ptr++ = (b << 11) | (g << 5) | r;
        }
        dst_rows += dst_stride;
        src_rows += src_stride;
    }
}

// NEON: Gray8 -> RGB565 (RRRRRGGGGGGBBBBB) / NEON：Gray8 转 RGB565（RRRRRGGGGGGBBBBB）。
static void gray8_to_rgb565_neon(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    uint8_t* dst_rows = static_cast<uint8_t*>(dst);
    const uint8_t* src_rows = src;
    for (int y = 0; y < height; ++y) {
        uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(dst_rows);
        const uint8_t* src_ptr = src_rows;
        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16_t gray_u8 = vld1q_u8(src_ptr);
            uint16x8_t gray_u16_low = vmovl_u8(vget_low_u8(gray_u8));
            uint16x8_t gray_u16_high = vmovl_u8(vget_high_u8(gray_u8));
            uint16x8_t r_low = vshlq_n_u16(vshrq_n_u16(gray_u16_low, 3), 11);
            uint16x8_t g_low = vshlq_n_u16(vshrq_n_u16(gray_u16_low, 2), 5);
            uint16x8_t b_low = vshrq_n_u16(gray_u16_low, 3);
            uint16x8_t r_high = vshlq_n_u16(vshrq_n_u16(gray_u16_high, 3), 11);
            uint16x8_t g_high = vshlq_n_u16(vshrq_n_u16(gray_u16_high, 2), 5);
            uint16x8_t b_high = vshrq_n_u16(gray_u16_high, 3);
            vst1q_u16(dst_ptr, vorrq_u16(vorrq_u16(r_low, g_low), b_low));
            vst1q_u16(dst_ptr + 8, vorrq_u16(vorrq_u16(r_high, g_high), b_high));
            src_ptr += 16;
            dst_ptr += 16;
        }
        for (; x < width; ++x) {
            uint8_t y_val = *src_ptr++;
            uint16_t r = (y_val & 0xF8) >> 3, g = (y_val & 0xFC) >> 2, b = (y_val & 0xF8) >> 3;
            *dst_ptr++ = (r << 11) | (g << 5) | b;
        }
        dst_rows += dst_stride;
        src_rows += src_stride;
    }
}

// NEON: Gray8 -> BGR666 packed in 24-bit (B: xxBBBBBB, G: xxGGGGGG, R: xxRRRRRR) / NEON：Gray8 -> 24 位打包 BGR666（B: xxBBBBBB，G: xxGGGGGG，R: xxRRRRRR）。
static void gray8_to_bgr666_packed24_neon(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    uint8_t* dst_rows = static_cast<uint8_t*>(dst);
    const uint8_t* src_rows = src;
    for (int y = 0; y < height; ++y) {
        uint8_t* dst_ptr = dst_rows;
        const uint8_t* src_ptr = src_rows;
        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16_t gray_u8 = vld1q_u8(src_ptr);
            uint8x16_t gray_6bit = vshrq_n_u8(gray_u8, 2); // 8-bit to 6-bit
            uint8x16x3_t bgr_6bit = {{ gray_6bit, gray_6bit, gray_6bit }};
            vst3q_u8(dst_ptr, bgr_6bit); // Interleaved store
            src_ptr += 16;
            dst_ptr += 16 * 3;
        }
        for (; x < width; ++x) {
            uint8_t val_6bit = (*src_ptr++) >> 2;
            *dst_ptr++ = val_6bit; // B
            *dst_ptr++ = val_6bit; // G
            *dst_ptr++ = val_6bit; // R
        }
        dst_rows += dst_stride;
        src_rows += src_stride;
    }
}

static void bgr888_to_bgr666_packed24_neon(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    uint8_t* dst_rows = static_cast<uint8_t*>(dst);
    const uint8_t* src_rows = src;
    for (int y = 0; y < height; ++y) {
        uint8_t* dst_ptr = dst_rows;
        const uint8_t* src_ptr = src_rows;
        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16x3_t bgr888 = vld3q_u8(src_ptr);
            uint8x16x3_t bgr666;
            bgr666.val[0] = vshrq_n_u8(bgr888.val[0], 2); // B
            bgr666.val[1] = vshrq_n_u8(bgr888.val[1], 2); // G
            bgr666.val[2] = vshrq_n_u8(bgr888.val[2], 2); // R
            vst3q_u8(dst_ptr, bgr666);
            src_ptr += 16 * 3;
            dst_ptr += 16 * 3;
        }
        for (; x < width; ++x) {
            *dst_ptr++ = (*src_ptr++) >> 2; // B
            *dst_ptr++ = (*src_ptr++) >> 2; // G
            *dst_ptr++ = (*src_ptr++) >> 2; // R
        }
        dst_rows += dst_stride;
        src_rows += src_stride;
    }
}

static void bgr888_to_rgb666_packed24_neon(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    uint8_t* dst_rows = static_cast<uint8_t*>(dst);
    const uint8_t* src_rows = src;
    for (int y = 0; y < height; ++y) {
        uint8_t* dst_ptr = dst_rows;
        const uint8_t* src_ptr = src_rows;
        int x = 0;
        for (; x <= width - 16; x += 16) {
            uint8x16x3_t bgr888 = vld3q_u8(src_ptr); // Loads B,G,R into val[0], val[1], val[2]
            uint8x16x3_t rgb666;
            rgb666.val[0] = vshrq_n_u8(bgr888.val[2], 2); // Output R = Input R >> 2
            rgb666.val[1] = vshrq_n_u8(bgr888.val[1], 2); // Output G = Input G >> 2
            rgb666.val[2] = vshrq_n_u8(bgr888.val[0], 2); // Output B = Input B >> 2
            vst3q_u8(dst_ptr, rgb666);
            src_ptr += 16 * 3;
            dst_ptr += 16 * 3;
        }
        for (; x < width; ++x) {
            uint8_t b = *src_ptr++;
            uint8_t g = *src_ptr++;
            uint8_t r = *src_ptr++;
            *dst_ptr++ = r >> 2; // R
            *dst_ptr++ = g >> 2; // G
            *dst_ptr++ = b >> 2; // B
        }
        dst_rows += dst_stride;
        src_rows += src_stride;
    }
}

#else
// C: Gray8 -> BGR565 / C 路径：Gray8 -> BGR565。
static void gray8_to_bgr565_c(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
     for(int y=0; y < height; ++y) {
        uint16_t* dst_row = (uint16_t*)((char*)dst + y * dst_stride);
        const uint8_t* src_row = src + y * src_stride;
        for(int x=0; x < width; ++x) {
           uint8_t y_val = src_row[x];
           uint16_t b = (y_val & 0xF8) >> 3, g = (y_val & 0xFC) >> 2, r = (y_val & 0xF8) >> 3;
           dst_row[x] = (b << 11) | (g << 5) | r;
        }
    }
}

// C: Gray8 -> RGB565 / C 路径：Gray8 -> RGB565。
static void gray8_to_rgb565_c(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
     for(int y=0; y < height; ++y) {
        uint16_t* dst_row = (uint16_t*)((char*)dst + y * dst_stride);
        const uint8_t* src_row = src + y * src_stride;
        for(int x=0; x < width; ++x) {
           uint8_t y_val = src_row[x];
           uint16_t r = (y_val & 0xF8) >> 3, g = (y_val & 0xFC) >> 2, b = (y_val & 0xF8) >> 3;
           dst_row[x] = (r << 11) | (g << 5) | b;
        }
    }
}

// C: Gray8 -> BGR666 packed in 24-bit / C 路径：Gray8 -> 24 位打包 BGR666。
static void gray8_to_bgr666_packed24_c(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    for(int y=0; y < height; ++y) {
        uint8_t* dst_row = (uint8_t*)dst + y * dst_stride;
        const uint8_t* src_row = src + y * src_stride;
        for(int x=0; x < width; ++x) {
           uint8_t val_6bit = src_row[x] >> 2;
           dst_row[x*3 + 0] = val_6bit; // B
           dst_row[x*3 + 1] = val_6bit; // G
           dst_row[x*3 + 2] = val_6bit; // R
        }
    }
}

// C: BGR888 -> BGR666 packed in 24-bit / C 路径：BGR888 -> 24 位打包 BGR666。
static void bgr888_to_bgr666_packed24_c(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    for(int y=0; y < height; ++y) {
        uint8_t* dst_row = (uint8_t*)dst + y * dst_stride;
        const uint8_t* src_row = src + y * src_stride;
        for(int x=0; x < width; ++x) {
           dst_row[x*3 + 0] = src_row[x*3 + 0] >> 2; // B
           dst_row[x*3 + 1] = src_row[x*3 + 1] >> 2; // G
           dst_row[x*3 + 2] = src_row[x*3 + 2] >> 2; // R
        }
    }
}

static void bgr888_to_rgb666_packed24_c(const uint8_t* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    for(int y=0; y < height; ++y) {
        uint8_t* dst_row = (uint8_t*)dst + y * dst_stride;
        const uint8_t* src_row = src + y * src_stride;
        for(int x=0; x < width; ++x) {
           dst_row[x*3 + 0] = src_row[x*3 + 2] >> 2; // R
           dst_row[x*3 + 1] = src_row[x*3 + 1] >> 2; // G
           dst_row[x*3 + 2] = src_row[x*3 + 0] >> 2; // B
        }
    }
}

#endif

DisplayFB::DisplayFB(Mode mode)
    : m_impl(std::make_unique<Impl>(mode)) {
    m_impl->fb_fd = open(kFbDevice, O_RDWR);
    if (m_impl->fb_fd < 0) throw std::runtime_error("DisplayFB Error: Cannot open framebuffer device.");
    
    fb_var_screeninfo vinfo;
    if (ioctl(m_impl->fb_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) { 
        close(m_impl->fb_fd);
        throw std::runtime_error("DisplayFB Error: Failed to get variable screen info.");
    }
    
    // Detect framebuffer pixel format at runtime (including RGB666 variants). / 在运行时检测 framebuffer 像素格式（包括 RGB666 变体）。
    switch (vinfo.bits_per_pixel) {
        case 16:
            if (vinfo.red.offset == 11 && vinfo.green.offset == 5 && vinfo.blue.offset == 0) {
                m_impl->screen_format_mpi = RK_FMT_RGB565;
            } else if (vinfo.red.offset == 0 && vinfo.green.offset == 5 && vinfo.blue.offset == 11) {
                m_impl->screen_format_mpi = RK_FMT_BGR565;
            } else {
                close(m_impl->fb_fd);
                throw std::runtime_error("DisplayFB Error: Unsupported 16bpp pixel format configuration.");
            }
            break;
        case 24:
            // Check RGB666 (channel width is 6 bits). / 检查 RGB666（通道位宽为 6 位）。
            if (vinfo.red.length == 6 && vinfo.green.length == 6 && vinfo.blue.length == 6) {
                if (vinfo.red.offset == 16 && vinfo.green.offset == 8 && vinfo.blue.offset == 0) {
                    m_impl->screen_format_mpi = CUSTOM_FMT_BGR666_PACKED24; // FB is BGR666
                } else if (vinfo.red.offset == 0 && vinfo.green.offset == 8 && vinfo.blue.offset == 16) {
                    m_impl->screen_format_mpi = CUSTOM_FMT_RGB666_PACKED24; // FB is RGB666
                } else {
                    close(m_impl->fb_fd);
                    throw std::runtime_error("DisplayFB Error: Unsupported RGB666 packing order.");
                }
            } 
            // Otherwise check standard RGB888 layouts. / 否则检查标准 RGB888 布局。
            else if (vinfo.red.length == 8 && vinfo.green.length == 8 && vinfo.blue.length == 8) {
                if (vinfo.red.offset == 0 && vinfo.green.offset == 8 && vinfo.blue.offset == 16) {
                    m_impl->screen_format_mpi = RK_FMT_RGB888;
                } else if (vinfo.red.offset == 16 && vinfo.green.offset == 8 && vinfo.blue.offset == 0) {
                    m_impl->screen_format_mpi = RK_FMT_BGR888;
                } else {
                    close(m_impl->fb_fd);
                    throw std::runtime_error("DisplayFB Error: Unsupported 24bpp pixel format configuration.");
                }
            } else {
                close(m_impl->fb_fd);
                throw std::runtime_error("DisplayFB Error: Ambiguous 24bpp format detected.");
            }
            break;
        case 32:
            if (vinfo.red.offset == 16 && vinfo.green.offset == 8 && vinfo.blue.offset == 0) {
                 m_impl->screen_format_mpi = RK_FMT_BGRA8888;
            } else if (vinfo.red.offset == 0 && vinfo.green.offset == 8 && vinfo.blue.offset == 16) {
                 m_impl->screen_format_mpi = RK_FMT_RGBA8888;
            } else {
                close(m_impl->fb_fd);
                throw std::runtime_error("DisplayFB Error: Unsupported 32bpp pixel format configuration.");
            }
            break;
        default:
            close(m_impl->fb_fd);
            throw std::runtime_error("DisplayFB Error: Unsupported bpp: " + std::to_string(vinfo.bits_per_pixel));
    }
    // Pixel format detection done. / 像素格式检测完成。
    
    fb_fix_screeninfo finfo;
    if (ioctl(m_impl->fb_fd, FBIOGET_FSCREENINFO, &finfo) < 0) { 
        close(m_impl->fb_fd);
        throw std::runtime_error("DisplayFB Error: Failed to get fixed screen info.");
    }
    
    m_impl->screen_width = vinfo.xres; 
    m_impl->screen_height = vinfo.yres; 
    m_impl->bits_per_pixel = vinfo.bits_per_pixel; 
    m_impl->line_length = finfo.line_length;
    m_impl->screen_size = finfo.smem_len;

    m_impl->fb_mem = mmap(0, m_impl->screen_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_impl->fb_fd, 0);
    if (m_impl->fb_mem == MAP_FAILED) { 
        close(m_impl->fb_fd);
        throw std::runtime_error("DisplayFB Error: Failed to map framebuffer device.");
    }

    memset(m_impl->fb_mem, 0, m_impl->screen_size);

    m_impl->is_running = true;
    try {
        m_impl->display_thread = std::thread(&DisplayFB::display_thread_func, this);
    } catch (...) {
        m_impl->is_running = false;
        munmap(m_impl->fb_mem, m_impl->screen_size);
        m_impl->fb_mem = nullptr;
        close(m_impl->fb_fd);
        m_impl->fb_fd = -1;
        throw;
    }
    m_impl->initialized = true;
    
    VISIONG_LOG_INFO("DisplayFB",
                     "Initialized successfully. Mode: "
                         << (m_impl->mode == Mode::LOW_REFRESH ? "LOW_REFRESH" : "HIGH_REFRESH")
                         << ", Target: " << m_impl->screen_width << "x" << m_impl->screen_height
                         << ", BPP: " << m_impl->bits_per_pixel
                         << ", Detected Format ID: " << static_cast<int>(m_impl->screen_format_mpi));
}

DisplayFB::~DisplayFB() { release(); }

void DisplayFB::release() {
    if (m_impl->is_running) {
        VISIONG_LOG_INFO("DisplayFB", "Releasing...");
        m_impl->is_running = false;
        m_impl->task_cv.notify_all();
        if (m_impl->display_thread.joinable()) { m_impl->display_thread.join(); }
        VISIONG_LOG_INFO("DisplayFB", "Thread joined.");
    }
    {
        std::lock_guard<std::mutex> lock(m_impl->task_mutex);
        m_impl->latest_task.reset();
    }
    if (m_impl->fb_mem && m_impl->fb_mem != MAP_FAILED) { munmap(m_impl->fb_mem, m_impl->screen_size); m_impl->fb_mem = nullptr; }
    if (m_impl->fb_fd >= 0) { close(m_impl->fb_fd); m_impl->fb_fd = -1; }
    m_impl->initialized = false;
    VISIONG_LOG_INFO("DisplayFB", "Released.");
}

bool DisplayFB::is_initialized() const { return m_impl->initialized.load(); }

bool DisplayFB::display(const ImageBuffer& img_buf) { return display(img_buf, std::make_tuple(0, 0, 0, 0)); }

bool DisplayFB::display(ImageBuffer&& img_buf) { return display(std::move(img_buf), std::make_tuple(0, 0, 0, 0)); }

bool DisplayFB::display(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi) {
    if (m_impl->mode == Mode::LOW_REFRESH) {
        // In low-refresh mode, avoid expensive deep copies when a pending frame / 在低刷新模式下，如果已有待显示帧，尽量避免高开销深拷贝。
        // is already queued for the display thread.
        std::lock_guard<std::mutex> lock(m_impl->task_mutex);
        if (m_impl->latest_task != nullptr) {
            return true;
        }
    }
    ImageBuffer owned_copy(img_buf);
    return display(std::move(owned_copy), roi);
}

bool DisplayFB::display(ImageBuffer&& img_buf, const std::tuple<int, int, int, int>& roi) {
    if (!m_impl->is_running || !img_buf.is_valid()) return false;

    auto [roi_x, roi_y, roi_w, roi_h] = roi;
    RectROI normalized_roi;
    if (roi_w <= 0 || roi_h <= 0) normalized_roi = {0, 0, img_buf.width, img_buf.height};
    else normalized_roi = {roi_x, roi_y, roi_w, roi_h};

    auto task = std::make_unique<DisplayTask>();
    task->image = std::make_unique<ImageBuffer>(std::move(img_buf));
    task->roi = normalized_roi;
    
    {
        std::lock_guard<std::mutex> lock(m_impl->task_mutex);
        m_impl->latest_task = std::move(task);
    }

    m_impl->task_cv.notify_one();
    
    return true;
}

void DisplayFB::display_thread_func() {
    VISIONG_LOG_INFO("DisplayFB",
                     "Display thread started in "
                         << (m_impl->mode == Mode::LOW_REFRESH ? "LOW_REFRESH" : "HIGH_REFRESH")
                         << " mode.");
    
    std::unique_ptr<RgaDmaBuffer> screen_dma_buf_color;
    std::vector<uint8_t> screen_cpu_buf;
    // Cache source DMA buffers and reuse across frames. / 缓存源 DMA 缓冲区，并在帧之间复用。
    std::unique_ptr<RgaDmaBuffer> cached_src_dma;
    int cached_src_w = 0, cached_src_h = 0;
    PIXEL_FORMAT_E cached_src_fmt = RK_FMT_BUTT;
    std::unique_ptr<RgaDmaBuffer> cached_gray_src_dma;
    int cached_gray_src_w = 0, cached_gray_src_h = 0;
    std::unique_ptr<RgaDmaBuffer> cached_gray_scaled_dma;
    int cached_gray_scaled_w = 0, cached_gray_scaled_h = 0;
    try {
        // RGB666 targets are converted via BGR888 intermediate. / RGB666 目标通过 BGR888 中间格式进行转换。
        PIXEL_FORMAT_E rga_target_format = static_cast<PIXEL_FORMAT_E>(m_impl->screen_format_mpi);
        if (m_impl->screen_format_mpi == CUSTOM_FMT_BGR666_PACKED24 || m_impl->screen_format_mpi == CUSTOM_FMT_RGB666_PACKED24) {
            rga_target_format = RK_FMT_BGR888;
        }
        screen_dma_buf_color = std::make_unique<RgaDmaBuffer>(m_impl->screen_width, m_impl->screen_height, rga_target_format);
        screen_cpu_buf.resize(m_impl->screen_size);
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("DisplayFB", "Display thread fatal error: " << e.what());
        m_impl->is_running = false;
        return;
    }

    while (m_impl->is_running) {
        std::unique_ptr<DisplayTask> task_to_process;

        if (m_impl->mode == Mode::LOW_REFRESH) {
            std::unique_lock<std::mutex> lock(m_impl->task_mutex);
            m_impl->task_cv.wait_for(lock, std::chrono::milliseconds(200),
                               [this] { return m_impl->latest_task != nullptr || !m_impl->is_running; });
            if (!m_impl->is_running) break;
            if (m_impl->latest_task) task_to_process = std::move(m_impl->latest_task);
        } else {
            std::unique_lock<std::mutex> lock(m_impl->task_mutex);
            m_impl->task_cv.wait(lock, [this]{ return m_impl->latest_task != nullptr || !m_impl->is_running; });
            if (!m_impl->is_running) break;
            if (m_impl->latest_task) task_to_process = std::move(m_impl->latest_task);
        }

        if (!task_to_process) continue;
        
        const ImageBuffer& current_image = *task_to_process->image;
        bool is_grayscale = (current_image.format == visiong::kGray8Format);

        try {
            const RectROI roi = clamp_and_align_roi(task_to_process->roi, current_image);
            float scale = std::min(static_cast<float>(m_impl->screen_width) / roi.w, static_cast<float>(m_impl->screen_height) / roi.h);
            int new_w = static_cast<int>(roi.w * scale) & ~1, new_h = static_cast<int>(roi.h * scale) & ~1;
            if (new_w <= 0 || new_h <= 0) {
                throw std::runtime_error("DisplayFB computed invalid scaled size.");
            }
            int pad_x = (m_impl->screen_width - new_w) / 2, pad_y = (m_impl->screen_height - new_h) / 2;

            if (!is_grayscale) {
                const bool can_use_src_zero_copy = current_image.is_zero_copy() && current_image.get_dma_fd() >= 0;
                const im_rect full_rect = {0, 0, m_impl->screen_width, m_impl->screen_height};
                const im_rect dst_rect = {pad_x, pad_y, new_w, new_h};
                constexpr uint32_t kBlackFillColor = 0xFF000000;

                if (can_use_src_zero_copy) {
                    RgaDmaBuffer src_wrapper(current_image.get_dma_fd(), const_cast<void*>(current_image.get_data()),
                                             current_image.get_size(), current_image.width, current_image.height,
                                             static_cast<int>(current_image.format), current_image.w_stride,
                                             current_image.h_stride);
                    dma_sync_cpu_to_device(src_wrapper.get_fd());
                    im_rect src_rect = {roi.x, roi.y, roi.w, roi.h};
                    if (imfill(screen_dma_buf_color->get_buffer(), full_rect, kBlackFillColor) != IM_STATUS_SUCCESS) {
                        throw std::runtime_error("DisplayFB RGA imfill failed.");
                    }
                    if (improcess(src_wrapper.get_buffer(), screen_dma_buf_color->get_buffer(), {}, src_rect, dst_rect,
                                  {}, IM_SYNC) != IM_STATUS_SUCCESS) {
                        throw std::runtime_error("DisplayFB RGA improcess (zero-copy ROI) failed.");
                    }
                } else {
                    // Reuse DMA buffers and only reallocate when size/format changes. / 复用 DMA 缓冲区，仅在尺寸或格式变化时重新分配。
                    if (!cached_src_dma || cached_src_w != roi.w || cached_src_h != roi.h ||
                        cached_src_fmt != current_image.format) {
                        cached_src_dma = std::make_unique<RgaDmaBuffer>(roi.w, roi.h, current_image.format);
                        cached_src_w = roi.w;
                        cached_src_h = roi.h;
                        cached_src_fmt = current_image.format;
                    }
                    copy_roi_to_dma_buffer(current_image, roi, *cached_src_dma);
                    im_rect src_rect = {0, 0, roi.w, roi.h};
                    if (imfill(screen_dma_buf_color->get_buffer(), full_rect, kBlackFillColor) != IM_STATUS_SUCCESS) {
                        throw std::runtime_error("DisplayFB RGA imfill failed.");
                    }
                    if (improcess(cached_src_dma->get_buffer(), screen_dma_buf_color->get_buffer(), {}, src_rect,
                                  dst_rect, {}, IM_SYNC) != IM_STATUS_SUCCESS) {
                        throw std::runtime_error("DisplayFB RGA improcess (CPU ROI upload) failed.");
                    }
                }

                dma_sync_device_to_cpu(screen_dma_buf_color->get_fd());

                if (m_impl->screen_format_mpi == CUSTOM_FMT_BGR666_PACKED24) {
                    #if defined(__ARM_NEON)
                    bgr888_to_bgr666_packed24_neon(static_cast<const uint8_t*>(screen_dma_buf_color->get_vir_addr()), screen_cpu_buf.data(), m_impl->screen_width, m_impl->screen_height, screen_dma_buf_color->get_wstride() * 3, m_impl->line_length);
                    #else
                    bgr888_to_bgr666_packed24_c(static_cast<const uint8_t*>(screen_dma_buf_color->get_vir_addr()), screen_cpu_buf.data(), m_impl->screen_width, m_impl->screen_height, screen_dma_buf_color->get_wstride() * 3, m_impl->line_length);
                    #endif
                    memcpy(m_impl->fb_mem, screen_cpu_buf.data(), m_impl->screen_size);
                } else if (m_impl->screen_format_mpi == CUSTOM_FMT_RGB666_PACKED24) {
                     #if defined(__ARM_NEON)
                    bgr888_to_rgb666_packed24_neon(static_cast<const uint8_t*>(screen_dma_buf_color->get_vir_addr()), screen_cpu_buf.data(), m_impl->screen_width, m_impl->screen_height, screen_dma_buf_color->get_wstride() * 3, m_impl->line_length);
                    #else
                    bgr888_to_rgb666_packed24_c(static_cast<const uint8_t*>(screen_dma_buf_color->get_vir_addr()), screen_cpu_buf.data(), m_impl->screen_width, m_impl->screen_height, screen_dma_buf_color->get_wstride() * 3, m_impl->line_length);
                    #endif
                    memcpy(m_impl->fb_mem, screen_cpu_buf.data(), m_impl->screen_size);
                }
                else {
                    copy_data_with_stride(m_impl->fb_mem, m_impl->line_length, screen_dma_buf_color->get_vir_addr(), screen_dma_buf_color->get_wstride() * m_impl->bits_per_pixel / 8, m_impl->screen_height, m_impl->screen_width * m_impl->bits_per_pixel / 8);
                }

            } else {
                // Reuse grayscale DMA buffers to avoid per-frame allocations. / 复用灰度 DMA 缓冲区，避免逐帧分配。
                if (!cached_gray_src_dma || cached_gray_src_w != roi.w || cached_gray_src_h != roi.h) {
                    cached_gray_src_dma = std::make_unique<RgaDmaBuffer>(roi.w, roi.h, current_image.format);
                    cached_gray_src_w = roi.w;
                    cached_gray_src_h = roi.h;
                }
                const void* roi_src_ptr = static_cast<const char*>(current_image.get_data()) + (size_t)roi.y * current_image.w_stride + roi.x;
                copy_data_with_stride(cached_gray_src_dma->get_vir_addr(), cached_gray_src_dma->get_wstride(), roi_src_ptr, current_image.w_stride, roi.h, roi.w);
                
                dma_sync_cpu_to_device(cached_gray_src_dma->get_fd());
                
                if (!cached_gray_scaled_dma || cached_gray_scaled_w != new_w || cached_gray_scaled_h != new_h) {
                    cached_gray_scaled_dma = std::make_unique<RgaDmaBuffer>(new_w, new_h, current_image.format);
                    cached_gray_scaled_w = new_w;
                    cached_gray_scaled_h = new_h;
                }
                if (imresize(cached_gray_src_dma->get_buffer(), cached_gray_scaled_dma->get_buffer()) != IM_STATUS_SUCCESS) throw std::runtime_error("RGA imresize for grayscale failed");
                
                dma_sync_device_to_cpu(cached_gray_scaled_dma->get_fd());
                
                memset(screen_cpu_buf.data(), 0, screen_cpu_buf.size());
                void* paste_pos_in_sw = screen_cpu_buf.data() + (size_t)pad_y * m_impl->line_length + (size_t)pad_x * (m_impl->bits_per_pixel / 8);
                
                // Choose the proper NEON/C conversion path based on detected FB format. / 根据检测到的 FB 格式选择合适的 NEON/C 转换路径。
                if (m_impl->screen_format_mpi == RK_FMT_BGR565) {
                    #if defined(__ARM_NEON)
                    gray8_to_bgr565_neon(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #else
                    gray8_to_bgr565_c(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #endif
                } else if (m_impl->screen_format_mpi == RK_FMT_RGB565) {
                    #if defined(__ARM_NEON)
                    gray8_to_rgb565_neon(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #else
                    gray8_to_rgb565_c(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #endif
                } else if (m_impl->screen_format_mpi == CUSTOM_FMT_BGR666_PACKED24) {
                    #if defined(__ARM_NEON)
                    gray8_to_bgr666_packed24_neon(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #else
                    gray8_to_bgr666_packed24_c(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #endif
                } else if (m_impl->screen_format_mpi == CUSTOM_FMT_RGB666_PACKED24) {
                    // For grayscale, reusing BGR666 conversion is valid because R=G=B. / 对于灰度图，复用 BGR666 转换是成立的，因为 R=G=B。
                     #if defined(__ARM_NEON)
                    gray8_to_bgr666_packed24_neon(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #else
                    gray8_to_bgr666_packed24_c(static_cast<uint8_t*>(cached_gray_scaled_dma->get_vir_addr()), paste_pos_in_sw, new_w, new_h, cached_gray_scaled_dma->get_wstride(), m_impl->line_length);
                    #endif
                }
                else {
                    throw std::runtime_error("Grayscale to " + std::to_string(m_impl->bits_per_pixel) + "bpp conversion via CPU is not implemented for this specific format.");
                }
                
                memcpy(m_impl->fb_mem, screen_cpu_buf.data(), m_impl->screen_size);
            }
        } catch (const std::exception& e) {
            VISIONG_LOG_ERROR("DisplayFB", "Display thread processing error: " << e.what());
        }
    }
    VISIONG_LOG_INFO("DisplayFB", "Display thread exited.");
}

int DisplayFB::get_screen_width() const { return m_impl->screen_width; }

int DisplayFB::get_screen_height() const { return m_impl->screen_height; }

