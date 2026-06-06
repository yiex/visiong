// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplaySPI.h"

#include "common/internal/dma_alloc.h"
#include "common/internal/string_utils.h"
#include "visiong/modules/DisplayFB.h"
#include "core/internal/logger.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/RgaHelper.h"
#include "visiong/core/pinmux.h"
#include "visiong/uapi/visiong_hw.h"
#include "im2d.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <fcntl.h>
#include <functional>
#include <fstream>
#include <iostream>
#include <linux/spi/spidev.h>
#include <limits.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr uint8_t kCmdSwReset = 0x01;
constexpr uint8_t kCmdSleepOut = 0x11;
constexpr uint8_t kCmdNormalOn = 0x13;
constexpr uint8_t kCmdInvertOff = 0x20;
constexpr uint8_t kCmdInvertOn = 0x21;
constexpr uint8_t kCmdDisplayOn = 0x29;
constexpr uint8_t kCmdColumnAddr = 0x2A;
constexpr uint8_t kCmdRowAddr = 0x2B;
constexpr uint8_t kCmdMemoryWrite = 0x2C;
constexpr uint8_t kCmdMadctl = 0x36;
constexpr uint8_t kCmdPixelFormat = 0x3A;

constexpr uint8_t kMadctlMy = 0x80;
constexpr uint8_t kMadctlMx = 0x40;
constexpr uint8_t kMadctlMv = 0x20;
constexpr uint8_t kMadctlBgr = 0x08;

constexpr char kDefaultSpiBusPath[] = "/dev/spidev0.0";
constexpr char kDefaultDcPin[] = "GPIO1_C3";
constexpr char kDefaultResetPin[] = "GPIO1_C2";
constexpr char kDefaultBackend[] = "auto";
constexpr uint32_t kDefaultSpeedHz = 50000000;
constexpr uint32_t kDefaultSourceClockHz = 200000000;
constexpr size_t kDefaultTransferChunkSize = 4096;
constexpr size_t kDefaultHwRegTransferChunkSize = 32768;
constexpr size_t kMinHwRegTransferChunkSize = 4096;

constexpr uintptr_t kRv1106Spi0Base = 0xff500000;
constexpr uintptr_t kRv1106Spi1Base = 0xff510000;
constexpr uintptr_t kRv1106CruBase = 0xff3a0000;
constexpr size_t kRegisterMapSize = 0x1000;
constexpr size_t kCruMapSize = 0x20000;

constexpr uint32_t kSpiCtrlr0 = 0x0000;
constexpr uint32_t kSpiCtrlr1 = 0x0004;
constexpr uint32_t kSpiSsienr = 0x0008;
constexpr uint32_t kSpiSer = 0x000c;
constexpr uint32_t kSpiBaudr = 0x0010;
constexpr uint32_t kSpiTxftlr = 0x0014;
constexpr uint32_t kSpiRxftlr = 0x0018;
constexpr uint32_t kSpiTxflr = 0x001c;
constexpr uint32_t kSpiRxflr = 0x0020;
constexpr uint32_t kSpiSr = 0x0024;
constexpr uint32_t kSpiImr = 0x002c;
constexpr uint32_t kSpiIcr = 0x0038;
constexpr uint32_t kSpiDmacr = 0x003c;
constexpr uint32_t kSpiVersion = 0x0048;
constexpr uint32_t kSpiTxdr = 0x0400;

constexpr uint32_t kSrBusy = 1u << 0;
constexpr uint32_t kSrTxFifoFull = 1u << 1;
constexpr uint32_t kSrRxFifoEmpty = 1u << 3;
constexpr uint32_t kCr0Dfs8Bit = 0x1u;
constexpr uint32_t kCr0ScphOffset = 6;
constexpr uint32_t kCr0SsdOne = 1u << 10;
constexpr uint32_t kCr0EmBig = 1u << 11;
constexpr uint32_t kCr0Bht8Bit = 1u << 13;
constexpr uint32_t kCr0FrfSpi = 0u << 16;
constexpr uint32_t kCr0XfmTo = 1u << 18;
constexpr uint32_t kSpiMaxTransferLen = 0xffff;
constexpr uint32_t kSpiVer2Type1 = 0x05ec0002;
constexpr uint32_t kSpiVer2Type2 = 0x00110002;

struct RectROI {
    int x;
    int y;
    int w;
    int h;
};

struct ScaledRect {
    RectROI roi;
    int x;
    int y;
    int w;
    int h;
};

int normalize_rotation(int rotation_degrees) {
    int rotation = rotation_degrees % 360;
    if (rotation < 0) {
        rotation += 360;
    }
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
        throw std::invalid_argument("DisplaySPI rotation must be 0, 90, 180, or 270 degrees.");
    }
    return rotation;
}

int logical_width_for(const DisplaySPIConfig& config) {
    const int rotation = normalize_rotation(config.rotation_degrees);
    return (rotation == 90 || rotation == 270) ? config.height : config.width;
}

int logical_height_for(const DisplaySPIConfig& config) {
    const int rotation = normalize_rotation(config.rotation_degrees);
    return (rotation == 90 || rotation == 270) ? config.width : config.height;
}

uint8_t madctl_for_rotation(const DisplaySPIConfig& config) {
    uint8_t madctl = 0;
    switch (normalize_rotation(config.rotation_degrees)) {
        case 0:
            madctl = kMadctlMx | kMadctlMy;
            break;
        case 90:
            madctl = kMadctlMy | kMadctlMv;
            break;
        case 180:
            madctl = 0;
            break;
        case 270:
            madctl = kMadctlMx | kMadctlMv;
            break;
        default:
            break;
    }
    if (config.bgr) {
        madctl |= kMadctlBgr;
    }
    return madctl;
}

RectROI clamp_and_align_roi(const RectROI& requested_roi, const ImageBuffer& image) {
    RectROI roi = requested_roi;
    if (roi.w <= 0 || roi.h <= 0) {
        roi = {0, 0, image.width, image.height};
    }

    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    if (roi.x >= image.width || roi.y >= image.height) {
        throw std::runtime_error("DisplaySPI ROI origin is outside source image bounds.");
    }

    roi.w = std::min(roi.w, image.width - roi.x);
    roi.h = std::min(roi.h, image.height - roi.y);

    if (visiong::is_yuv420sp_format(image.format)) {
        roi.x &= ~1;
        roi.y &= ~1;
        roi.w &= ~1;
        roi.h &= ~1;
        if (roi.x >= image.width || roi.y >= image.height) {
            throw std::runtime_error("DisplaySPI ROI became invalid after YUV alignment.");
        }
        if (roi.x + roi.w > image.width) {
            roi.w = (image.width - roi.x) & ~1;
        }
        if (roi.y + roi.h > image.height) {
            roi.h = (image.height - roi.y) & ~1;
        }
    }

    if (roi.w <= 0 || roi.h <= 0) {
        throw std::runtime_error("DisplaySPI ROI is empty after clamping/alignment.");
    }
    return roi;
}

ScaledRect compute_scaled_rect(const RectROI& requested_roi,
                                const ImageBuffer& image,
                                int screen_width,
                                int screen_height) {
    const RectROI roi = clamp_and_align_roi(requested_roi, image);
    const float scale = std::min(static_cast<float>(screen_width) / roi.w,
                                  static_cast<float>(screen_height) / roi.h);
    int scaled_w = static_cast<int>(roi.w * scale) & ~1;
    int scaled_h = static_cast<int>(roi.h * scale) & ~1;
    if (scaled_w <= 0 || scaled_h <= 0) {
        throw std::runtime_error("DisplaySPI computed invalid scaled size.");
    }

    scaled_w = std::min(scaled_w, screen_width);
    scaled_h = std::min(scaled_h, screen_height);
    return ScaledRect{roi, (screen_width - scaled_w) / 2, (screen_height - scaled_h) / 2, scaled_w, scaled_h};
}

void copy_roi_to_dma_buffer(const ImageBuffer& src, const RectROI& roi, RgaDmaBuffer& dst_dma) {
    if (dst_dma.get_width() != roi.w || dst_dma.get_height() != roi.h) {
        throw std::runtime_error("DisplaySPI ROI DMA size mismatch.");
    }
    if (dst_dma.get_mpi_format() != static_cast<int>(src.format)) {
        throw std::runtime_error("DisplaySPI ROI DMA format mismatch.");
    }

    if (visiong::is_yuv420sp_format(src.format)) {
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

#if !defined(__ARM_NEON)
void rgb565_native_to_be_c(const uint8_t* src,
                           uint8_t* dst,
                           int width,
                           int height,
                           int src_stride_bytes,
                           int dst_stride_bytes) {
    const int row_bytes = width * 2;
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + static_cast<size_t>(y) * src_stride_bytes;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride_bytes;
        for (int x = 0; x < row_bytes; x += 2) {
            dst_row[x] = src_row[x + 1];
            dst_row[x + 1] = src_row[x];
        }
    }
}

void gray8_to_rgb565_be_c(const uint8_t* src,
                          uint8_t* dst,
                          int width,
                          int height,
                          int src_stride_bytes,
                          int dst_stride_bytes) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + static_cast<size_t>(y) * src_stride_bytes;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride_bytes;
        for (int x = 0; x < width; ++x) {
            const uint8_t gray = src_row[x];
            const uint16_t rgb565 = static_cast<uint16_t>(((gray & 0xF8) << 8) |
                                                           ((gray & 0xFC) << 3) |
                                                           (gray >> 3));
            dst_row[x * 2] = static_cast<uint8_t>(rgb565 >> 8);
            dst_row[x * 2 + 1] = static_cast<uint8_t>(rgb565 & 0xFF);
        }
    }
}
#endif

#if defined(__ARM_NEON)
void rgb565_native_to_be_neon(const uint8_t* src,
                              uint8_t* dst,
                              int width,
                              int height,
                              int src_stride_bytes,
                              int dst_stride_bytes) {
    const int row_bytes = width * 2;
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + static_cast<size_t>(y) * src_stride_bytes;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride_bytes;
        int x = 0;
        for (; x <= row_bytes - 64; x += 64) {
            const uint8x16_t v0 = vrev16q_u8(vld1q_u8(src_row + x + 0));
            const uint8x16_t v1 = vrev16q_u8(vld1q_u8(src_row + x + 16));
            const uint8x16_t v2 = vrev16q_u8(vld1q_u8(src_row + x + 32));
            const uint8x16_t v3 = vrev16q_u8(vld1q_u8(src_row + x + 48));
            vst1q_u8(dst_row + x + 0, v0);
            vst1q_u8(dst_row + x + 16, v1);
            vst1q_u8(dst_row + x + 32, v2);
            vst1q_u8(dst_row + x + 48, v3);
        }
        for (; x <= row_bytes - 16; x += 16) {
            vst1q_u8(dst_row + x, vrev16q_u8(vld1q_u8(src_row + x)));
        }
        for (; x < row_bytes; x += 2) {
            dst_row[x] = src_row[x + 1];
            dst_row[x + 1] = src_row[x];
        }
    }
}

void gray8_to_rgb565_be_neon(const uint8_t* src,
                             uint8_t* dst,
                             int width,
                             int height,
                             int src_stride_bytes,
                             int dst_stride_bytes) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + static_cast<size_t>(y) * src_stride_bytes;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride_bytes;
        int x = 0;
        for (; x <= width - 8; x += 8) {
            const uint16x8_t gray = vmovl_u8(vld1_u8(src_row + x));
            const uint16x8_t r = vshlq_n_u16(vshrq_n_u16(gray, 3), 11);
            const uint16x8_t g = vshlq_n_u16(vshrq_n_u16(gray, 2), 5);
            const uint16x8_t b = vshrq_n_u16(gray, 3);
            const uint16x8_t rgb565 = vorrq_u16(vorrq_u16(r, g), b);
            vst1q_u8(dst_row + x * 2, vrev16q_u8(vreinterpretq_u8_u16(rgb565)));
        }
        for (; x < width; ++x) {
            const uint8_t gray = src_row[x];
            const uint16_t rgb565 = static_cast<uint16_t>(((gray & 0xF8) << 8) |
                                                           ((gray & 0xFC) << 3) |
                                                           (gray >> 3));
            dst_row[x * 2] = static_cast<uint8_t>(rgb565 >> 8);
            dst_row[x * 2 + 1] = static_cast<uint8_t>(rgb565 & 0xFF);
        }
    }
}
#endif

void rgb565_native_to_be(const uint8_t* src,
                         uint8_t* dst,
                         int width,
                         int height,
                         int src_stride_bytes,
                         int dst_stride_bytes) {
#if defined(__ARM_NEON)
    rgb565_native_to_be_neon(src, dst, width, height, src_stride_bytes, dst_stride_bytes);
#else
    rgb565_native_to_be_c(src, dst, width, height, src_stride_bytes, dst_stride_bytes);
#endif
}

void gray8_to_rgb565_be(const uint8_t* src,
                        uint8_t* dst,
                        int width,
                        int height,
                        int src_stride_bytes,
                        int dst_stride_bytes) {
#if defined(__ARM_NEON)
    gray8_to_rgb565_be_neon(src, dst, width, height, src_stride_bytes, dst_stride_bytes);
#else
    gray8_to_rgb565_be_c(src, dst, width, height, src_stride_bytes, dst_stride_bytes);
#endif
}

void fill_rgb565_be(std::vector<uint8_t>& buffer, uint16_t color_rgb565) {
    const uint8_t hi = static_cast<uint8_t>(color_rgb565 >> 8);
    const uint8_t lo = static_cast<uint8_t>(color_rgb565 & 0xFF);
    if (hi == lo) {
        std::memset(buffer.data(), hi, buffer.size());
        return;
    }
#if defined(__ARM_NEON)
    const uint8x16_t pattern = vreinterpretq_u8_u16(
        vdupq_n_u16(static_cast<uint16_t>((static_cast<uint16_t>(lo) << 8) | hi)));
    size_t i = 0;
    for (; i + 32 <= buffer.size(); i += 32) {
        vst1q_u8(buffer.data() + i, pattern);
        vst1q_u8(buffer.data() + i + 16, pattern);
    }
    for (; i + 1 < buffer.size(); i += 2) {
        buffer[i] = hi;
        buffer[i + 1] = lo;
    }
#else
    for (size_t i = 0; i + 1 < buffer.size(); i += 2) {
        buffer[i] = hi;
        buffer[i + 1] = lo;
    }
#endif
}

}  // namespace

struct DisplaySPI::Impl {
    std::string spi_bus_path;
    std::string chip_model;
    DisplaySPIConfig config;
    std::string backend = kDefaultBackend;
    std::string active_backend;
    int spi_fd = -1;
    int hw_fd = -1;
    bool hw_reg_accel_unavailable = false;
    int mem_fd = -1;
    volatile uint8_t* spi_regs = nullptr;
    volatile uint8_t* cru_regs = nullptr;
    uintptr_t spi_phys_base = 0;
    int spi_index = 0;
    int chip_select = 0;
    uint32_t register_source_clock_hz = kDefaultSourceClockHz;
    uint32_t register_fifo_len = 32;
    std::string power_control_path;
    std::string previous_power_control;
    std::string released_spi_child;
    std::string released_spi_driver_path;
    std::string released_spi_driver_name;
    std::string released_platform_device;
    std::string released_platform_driver_path;
    std::string released_platform_driver_name;
    std::atomic<bool> initialized{false};
    int screen_width = 0;
    int screen_height = 0;
    std::mutex lock;
    std::mutex spi_lock;
    std::unique_ptr<visiong::pinmux::Controller> gpio;
    std::vector<uint8_t> transfer_buffer;
    std::vector<uint8_t> region_transfer_buffer;

    std::thread transfer_thread;
    std::mutex transfer_lock;
    std::condition_variable transfer_cv;
    bool transfer_running = false;
    bool transfer_stop = false;
    bool transfer_pending = false;
    int pending_buffer_index = -1;
    int active_buffer_index = -1;
    std::vector<std::vector<uint8_t>> frame_buffers;

    std::unique_ptr<RgaDmaBuffer> screen_dma;
    std::unique_ptr<RgaDmaBuffer> cached_src_dma;
    int cached_src_w = 0;
    int cached_src_h = 0;
    PIXEL_FORMAT_E cached_src_fmt = RK_FMT_BUTT;

    std::unique_ptr<RgaDmaBuffer> cached_gray_src_dma;
    int cached_gray_src_w = 0;
    int cached_gray_src_h = 0;
    std::unique_ptr<RgaDmaBuffer> cached_gray_scaled_dma;
    int cached_gray_scaled_w = 0;
    int cached_gray_scaled_h = 0;

    std::function<void(Impl&)> init_sequence;
};

namespace {

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string trim_copy(const std::string& value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }
    return value.substr(first, last - first);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool parse_uint_at(const std::string& value, size_t pos, int* out, size_t* end_pos) {
    if (pos >= value.size() || !std::isdigit(static_cast<unsigned char>(value[pos]))) {
        return false;
    }
    int parsed = 0;
    while (pos < value.size() && std::isdigit(static_cast<unsigned char>(value[pos]))) {
        parsed = parsed * 10 + (value[pos] - '0');
        ++pos;
    }
    if (out) {
        *out = parsed;
    }
    if (end_pos) {
        *end_pos = pos;
    }
    return true;
}

bool parse_spi_bus_name(const std::string& spi_bus, int* spi_index, int* chip_select) {
    std::string token = visiong::to_lower_copy(trim_copy(spi_bus));
    if (token.empty()) {
        token = kDefaultSpiBusPath;
    }

    if (token == "ff500000.spi" || token == "/spi@ff500000" || token == "spi@ff500000") {
        *spi_index = 0;
        *chip_select = 0;
        return true;
    }
    if (token == "ff510000.spi" || token == "/spi@ff510000" || token == "spi@ff510000") {
        *spi_index = 1;
        *chip_select = 0;
        return true;
    }

    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }

    size_t pos = std::string::npos;
    if (starts_with(token, "spidev")) {
        pos = 6;
    } else if (starts_with(token, "spi")) {
        pos = 3;
    } else {
        const size_t embedded = token.find("spidev");
        if (embedded != std::string::npos) {
            pos = embedded + 6;
        }
    }
    if (pos == std::string::npos) {
        return false;
    }

    int bus = 0;
    size_t end = 0;
    if (!parse_uint_at(token, pos, &bus, &end)) {
        return false;
    }

    int cs = 0;
    if (end < token.size() && token[end] == '.') {
        size_t cs_end = 0;
        if (!parse_uint_at(token, end + 1, &cs, &cs_end)) {
            return false;
        }
        end = cs_end;
    }

    if (bus < 0 || bus > 1 || cs < 0 || cs > 1) {
        return false;
    }
    *spi_index = bus;
    *chip_select = cs;
    return true;
}

uintptr_t register_spi_base_for_index(int spi_index) {
    switch (spi_index) {
        case 0:
            return kRv1106Spi0Base;
        case 1:
            return kRv1106Spi1Base;
        default:
            throw std::invalid_argument("[DisplaySPI] Register backend currently supports spi0 and spi1 on RV1103/RV1106.");
    }
}

std::string platform_device_for_spi(int spi_index) {
    switch (spi_index) {
        case 0:
            return "ff500000.spi";
        case 1:
            return "ff510000.spi";
        default:
            return "";
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return "";
    }
    std::string value;
    std::getline(in, value, '\0');
    return trim_copy(value);
}

bool write_text_file(const std::string& path, const std::string& value) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << value;
    return static_cast<bool>(out);
}

std::string read_symlink_target(const std::string& path) {
    char buffer[PATH_MAX] = {};
    const ssize_t len = ::readlink(path.c_str(), buffer, sizeof(buffer) - 1);
    if (len <= 0) {
        return "";
    }
    buffer[len] = '\0';
    return std::string(buffer);
}

std::string path_basename(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

template <typename ImplT>
std::string spi_child_name(const ImplT& impl) {
    return "spi" + std::to_string(impl.spi_index) + "." + std::to_string(impl.chip_select);
}

std::string current_spi_child_driver_name(const std::string& child) {
    const std::string driver_link = "/sys/bus/spi/devices/" + child + "/driver";
    const std::string driver_path = read_symlink_target(driver_link);
    return driver_path.empty() ? "" : path_basename(driver_path);
}

bool unbind_spi_child_driver(const std::string& child, const std::string& driver_name) {
    if (child.empty() || driver_name.empty()) {
        return true;
    }
    return write_text_file("/sys/bus/spi/drivers/" + driver_name + "/unbind", child);
}

template <typename ImplT>
void release_spi_child_driver(ImplT& impl) {
    const std::string child = spi_child_name(impl);
    const std::string driver_name = current_spi_child_driver_name(child);
    if (driver_name.empty()) {
        return;
    }
    if (impl.released_spi_child == child && !impl.released_spi_driver_name.empty()) {
        (void)unbind_spi_child_driver(child, driver_name);
        return;
    }

    const std::string driver_path = "/sys/bus/spi/drivers/" + driver_name;
    if (!unbind_spi_child_driver(child, driver_name)) {
        std::cerr << "[DisplaySPI] Warning: failed to release " << child << " from SPI driver "
                  << driver_name << "; userspace SPI backend will continue anyway." << std::endl;
        return;
    }

    impl.released_spi_child = child;
    impl.released_spi_driver_path = driver_path;
    impl.released_spi_driver_name = driver_name;
    std::cerr << "[DisplaySPI] Warning: released " << child << " from SPI driver "
              << driver_name << " for userspace DisplaySPI backend." << std::endl;
}

template <typename ImplT>
void restore_spi_child_driver(ImplT& impl) {
    if (impl.released_spi_child.empty() || impl.released_spi_driver_path.empty()) {
        return;
    }
    const std::string current_driver = current_spi_child_driver_name(impl.released_spi_child);
    if (current_driver == impl.released_spi_driver_name) {
        const std::string override_path = "/sys/bus/spi/devices/" + impl.released_spi_child + "/driver_override";
        (void)write_text_file(override_path, "\n");
        impl.released_spi_child.clear();
        impl.released_spi_driver_path.clear();
        impl.released_spi_driver_name.clear();
        return;
    }
    if (!current_driver.empty() && current_driver != impl.released_spi_driver_name) {
        (void)unbind_spi_child_driver(impl.released_spi_child, current_driver);
    }
    const std::string override_path = "/sys/bus/spi/devices/" + impl.released_spi_child + "/driver_override";
    (void)write_text_file(override_path, impl.released_spi_driver_name + "\n");

    const std::string bind_path = impl.released_spi_driver_path + "/bind";
    if (!write_text_file(bind_path, impl.released_spi_child)) {
        std::cerr << "[DisplaySPI] Warning: failed to restore " << impl.released_spi_child
                  << " to SPI driver " << impl.released_spi_driver_name << "." << std::endl;
    } else {
        (void)write_text_file(override_path, "\n");
    }
    impl.released_spi_child.clear();
    impl.released_spi_driver_path.clear();
    impl.released_spi_driver_name.clear();
}

bool wait_for_path(const std::string& path, int attempts, int sleep_ms_value) {
    for (int i = 0; i < attempts; ++i) {
        if (::access(path.c_str(), F_OK) == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_value));
    }
    return ::access(path.c_str(), F_OK) == 0;
}

std::string current_platform_driver_name(const std::string& device) {
    const std::string driver_link = "/sys/bus/platform/devices/" + device + "/driver";
    const std::string driver_path = read_symlink_target(driver_link);
    return driver_path.empty() ? "" : path_basename(driver_path);
}

bool unbind_platform_device(const std::string& device, const std::string& driver_name) {
    if (device.empty() || driver_name.empty()) {
        return true;
    }
    return write_text_file("/sys/bus/platform/drivers/" + driver_name + "/unbind", device);
}

template <typename ImplT>
void release_platform_spi_driver(ImplT& impl) {
    const std::string device = platform_device_for_spi(impl.spi_index);
    if (device.empty()) {
        return;
    }
    const std::string driver_name = current_platform_driver_name(device);
    if (driver_name.empty()) {
        return;
    }
    if (impl.released_platform_device == device && !impl.released_platform_driver_name.empty()) {
        return;
    }

    const std::string driver_path = "/sys/bus/platform/drivers/" + driver_name;
    if (!unbind_platform_device(device, driver_name)) {
        std::cerr << "[DisplaySPI] Warning: failed to release platform SPI device " << device
                  << " from driver " << driver_name
                  << "; register backend will continue with PIO fallback if DMA is unavailable." << std::endl;
        return;
    }

    impl.released_platform_device = device;
    impl.released_platform_driver_path = driver_path;
    impl.released_platform_driver_name = driver_name;
    std::cerr << "[DisplaySPI] Warning: released platform SPI device " << device
              << " from driver " << driver_name << " so DisplaySPI register DMA can own the bus." << std::endl;
}

template <typename ImplT>
void restore_platform_spi_driver(ImplT& impl) {
    if (impl.released_platform_device.empty() || impl.released_platform_driver_path.empty()) {
        return;
    }

    const std::string current_driver = current_platform_driver_name(impl.released_platform_device);
    if (current_driver == impl.released_platform_driver_name) {
        impl.released_platform_device.clear();
        impl.released_platform_driver_path.clear();
        impl.released_platform_driver_name.clear();
        return;
    }
    if (!current_driver.empty() && current_driver != impl.released_platform_driver_name) {
        (void)unbind_platform_device(impl.released_platform_device, current_driver);
    }

    const std::string bind_path = impl.released_platform_driver_path + "/bind";
    if (!write_text_file(bind_path, impl.released_platform_device)) {
        std::cerr << "[DisplaySPI] Warning: failed to restore platform SPI device "
                  << impl.released_platform_device << " to driver "
                  << impl.released_platform_driver_name << "." << std::endl;
    } else {
        std::cerr << "[DisplaySPI] Restored platform SPI device " << impl.released_platform_device
                  << " to driver " << impl.released_platform_driver_name << "." << std::endl;
        (void)wait_for_path("/sys/bus/spi/devices/" + spi_child_name(impl), 20, 10);
    }

    impl.released_platform_device.clear();
    impl.released_platform_driver_path.clear();
    impl.released_platform_driver_name.clear();
}

uint32_t reg_read(volatile uint8_t* base, uint32_t offset) {
    return *reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(base) + offset);
}

void reg_write(volatile uint8_t* base, uint32_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(base) + offset) = value;
    __sync_synchronize();
}

void reg_write_relaxed(volatile uint8_t* base, uint32_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(base) + offset) = value;
}

template <typename ImplT>
uint32_t spi_reg_read(const ImplT& impl, uint32_t offset) {
    return reg_read(impl.spi_regs, offset);
}

template <typename ImplT>
void spi_reg_write(ImplT& impl, uint32_t offset, uint32_t value) {
    reg_write(impl.spi_regs, offset, value);
}

template <typename ImplT>
void spi_reg_write_relaxed(ImplT& impl, uint32_t offset, uint32_t value) {
    reg_write_relaxed(impl.spi_regs, offset, value);
}

template <typename ImplT>
void cru_hiword_update(ImplT& impl, uint32_t offset, uint32_t shift, uint32_t width, uint32_t value) {
    const uint32_t mask = ((1u << width) - 1u) << shift;
    reg_write(impl.cru_regs, offset, (mask << 16) | ((value << shift) & mask));
}

template <typename ImplT>
void cru_enable_gate_bits(ImplT& impl, uint32_t offset, uint32_t bits) {
    reg_write(impl.cru_regs, offset, bits << 16);
}

template <typename ImplT>
void enable_register_spi_clocks(ImplT& impl) {
    if (!impl.cru_regs) {
        return;
    }

    if (impl.spi_index == 0) {
        constexpr uint32_t kVepuClkSel0 = 0x1a000 + 0x300;
        constexpr uint32_t kVepuClkGate1 = 0x1a000 + 0x800 + 0x4;
        cru_hiword_update(impl, kVepuClkSel0, 12, 2, 0);  // clk_spi0 source: 200 MHz
        cru_enable_gate_bits(impl, kVepuClkGate1, (1u << 2) | (1u << 3) | (1u << 4));
    } else if (impl.spi_index == 1) {
        constexpr uint32_t kPeriClkSel6 = 0x12000 + 0x300 + 0x18;
        constexpr uint32_t kPeriClkGate3 = 0x12000 + 0x800 + 0x0c;
        cru_hiword_update(impl, kPeriClkSel6, 3, 2, 0);  // clk_spi1 source: 200 MHz
        cru_enable_gate_bits(impl, kPeriClkGate3, (1u << 6) | (1u << 7));
    }
}

template <typename ImplT>
void hold_kernel_runtime_power(ImplT& impl) {
    const std::string platform = platform_device_for_spi(impl.spi_index);
    if (platform.empty()) {
        return;
    }
    const std::string control = "/sys/bus/platform/devices/" + platform + "/power/control";
    if (::access(control.c_str(), F_OK) != 0) {
        return;
    }

    impl.power_control_path = control;
    impl.previous_power_control = read_text_file(control);
    if (impl.previous_power_control != "on") {
        write_text_file(control, "on");
    }
}

template <typename ImplT>
void restore_kernel_runtime_power(ImplT& impl) {
    if (!impl.power_control_path.empty() && !impl.previous_power_control.empty()) {
        write_text_file(impl.power_control_path, impl.previous_power_control);
    }
    impl.power_control_path.clear();
    impl.previous_power_control.clear();
}

uint32_t even_spi_divisor(uint32_t source_hz, uint32_t speed_hz) {
    if (source_hz == 0) {
        source_hz = kDefaultSourceClockHz;
    }
    if (speed_hz == 0) {
        speed_hz = kDefaultSpeedHz;
    }
    uint32_t div = (source_hz + speed_hz - 1) / speed_hz;
    if (div < 2) {
        div = 2;
    }
    if (div & 1u) {
        ++div;
    }
    return std::min<uint32_t>(div, 65534);
}

template <typename ImplT>
void drain_rx_fifo(ImplT& impl) {
    int guard = 256;
    while (guard-- > 0 && !(spi_reg_read(impl, kSpiSr) & kSrRxFifoEmpty)) {
        (void)spi_reg_read(impl, kSpiRxflr);
    }
}

template <typename ImplT>
void configure_register_spi_transfer(ImplT& impl, size_t len) {
    const uint32_t mode = impl.config.spi_mode & 0x3u;
    const uint32_t cr0 = kCr0FrfSpi |
                         kCr0Bht8Bit |
                         kCr0SsdOne |
                         kCr0EmBig |
                         (mode << kCr0ScphOffset) |
                         kCr0XfmTo |
                         kCr0Dfs8Bit;

    spi_reg_write(impl, kSpiSsienr, 0);
    spi_reg_write(impl, kSpiImr, 0);
    spi_reg_write(impl, kSpiIcr, 0xffffffff);
    spi_reg_write(impl, kSpiDmacr, 0);
    spi_reg_write(impl, kSpiCtrlr0, cr0);
    spi_reg_write(impl, kSpiCtrlr1, static_cast<uint32_t>(len - 1));
    spi_reg_write(impl, kSpiTxftlr, impl.register_fifo_len / 2);
    spi_reg_write(impl, kSpiRxftlr, 0);
    spi_reg_write(impl, kSpiBaudr, even_spi_divisor(impl.register_source_clock_hz, impl.config.speed_hz));
}

template <typename ImplT>
void wait_register_spi_idle(ImplT& impl, size_t len) {
    const auto bit_time_us = (static_cast<uint64_t>(len) * 8u * 1000000u) /
                             std::max<uint32_t>(1, impl.config.speed_hz);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(static_cast<int64_t>(bit_time_us * 32 + 500000));
    while (spi_reg_read(impl, kSpiSr) & kSrBusy) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error("[DisplaySPI] Register SPI transfer timed out waiting for idle.");
        }
        std::this_thread::yield();
    }
}

template <typename ImplT>
void set_gpio_value(ImplT& impl, const std::string& pin_name, int value) {
    if (!pin_name.empty() && impl.gpio) {
        impl.gpio->gpio_set_value(pin_name, value);
    }
}

template <typename ImplT>
void request_gpio_output(ImplT& impl,
                         const std::string& pin_name,
                         int default_value,
                         const char* label) {
    if (pin_name.empty()) {
        return;
    }
    if (!impl.gpio) {
        impl.gpio = std::make_unique<visiong::pinmux::Controller>();
    }
    visiong::pinmux::GpioLineConfig cfg;
    cfg.direction = "output";
    cfg.default_value = default_value ? 1 : 0;
    cfg.consumer = std::string("visiong-displayspi-") + label;
    if (!impl.gpio->gpio_request_line(pin_name, cfg)) {
        const auto conflict = impl.gpio->check_conflict(pin_name, "gpio");
        if (conflict.conflict) {
            std::cerr << "[DisplaySPI] Warning: " << pin_name << " is busy before GPIO request: "
                      << conflict.reason << " mux_owner=" << conflict.runtime.mux_owner
                      << " gpio_owner=" << conflict.runtime.gpio_owner << std::endl;
        }

        (void)impl.gpio->release_conflict(pin_name);
        if (impl.gpio->gpio_request_line(pin_name, cfg)) {
            std::cerr << "[DisplaySPI] Warning: released previous owner and requested GPIO pin "
                      << pin_name << " for " << label << "." << std::endl;
            return;
        }

        const auto after = impl.gpio->check_conflict(pin_name, "gpio");
        std::string detail = after.reason;
        if (detail.empty()) {
            detail = conflict.reason;
        }
        if (detail.empty()) {
            detail = "line may be held by a kernel GPIO consumer or another process.";
        }
        throw std::runtime_error("[DisplaySPI] Failed to request GPIO pin: " + pin_name + ". " + detail);
    }
}

template <typename ImplT>
void request_control_gpios(ImplT& impl) {
    if (impl.config.dc_pin.empty()) {
        throw std::invalid_argument("[DisplaySPI] dc_pin is required, for example 'GPIO1_C3'.");
    }
    request_gpio_output(impl, impl.config.dc_pin, 1, "dc");
    request_gpio_output(impl, impl.config.reset_pin, 1, "reset");
    request_gpio_output(impl, impl.config.backlight_pin, 0, "backlight");
}

template <typename ImplT>
void open_spi_device(ImplT& impl) {
    impl.spi_fd = ::open(impl.spi_bus_path.c_str(), O_RDWR | O_CLOEXEC);
    if (impl.spi_fd < 0) {
        throw std::runtime_error("[DisplaySPI] Failed to open SPI bus " + impl.spi_bus_path + ": " +
                                  std::strerror(errno));
    }

    uint8_t mode = impl.config.spi_mode;
    uint8_t bits = impl.config.bits_per_word;
    uint32_t speed = impl.config.speed_hz;
    if (::ioctl(impl.spi_fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ::ioctl(impl.spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ::ioctl(impl.spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        const std::string err = std::strerror(errno);
        ::close(impl.spi_fd);
        impl.spi_fd = -1;
        throw std::runtime_error("[DisplaySPI] Failed to configure SPI bus: " + err);
    }
}

template <typename ImplT>
void bind_spidev_and_open_spi_device(ImplT& impl) {
    if (DisplayFB::is_any_active()) {
        throw std::runtime_error(
            "[DisplaySPI] Refusing to rebind SPI to spidev while DisplayFB is active. "
            "Call DisplayFB.release() first, or pass backend='spidev' for an already exposed free SPI device.");
    }

    if (!parse_spi_bus_name(impl.spi_bus_path, &impl.spi_index, &impl.chip_select)) {
        throw std::invalid_argument("[DisplaySPI] Cannot resolve SPI controller for spidev backend from '" +
                                    impl.spi_bus_path + "'.");
    }

    const std::string child = spi_child_name(impl);
    const std::string device_path = "/sys/bus/spi/devices/" + child;
    if (::access(device_path.c_str(), F_OK) != 0) {
        throw std::runtime_error("[DisplaySPI] SPI child device is not present: " + child);
    }

    if (current_spi_child_driver_name(child) != "spidev") {
        release_spi_child_driver(impl);
        const std::string override_path = device_path + "/driver_override";
        (void)write_text_file(override_path, "spidev\n");
        if (!write_text_file("/sys/bus/spi/drivers/spidev/bind", child)) {
            throw std::runtime_error("[DisplaySPI] Failed to bind " + child + " to spidev.");
        }
    }

    for (int i = 0; i < 20 && ::access(impl.spi_bus_path.c_str(), F_OK) != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    open_spi_device(impl);
}

template <typename ImplT>
void ensure_register_spi_legacy_maps(ImplT& impl) {
    if (impl.spi_regs) {
        return;
    }
    if (impl.mem_fd >= 0 || impl.cru_regs) {
        throw std::runtime_error("[DisplaySPI] Register SPI legacy backend is partially initialized.");
    }

    impl.mem_fd = ::open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (impl.mem_fd < 0) {
        throw std::runtime_error("[DisplaySPI] Failed to open /dev/mem for register SPI backend: " +
                                  std::string(std::strerror(errno)));
    }

    void* cru_map = ::mmap(nullptr, kCruMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, impl.mem_fd, kRv1106CruBase);
    if (cru_map == MAP_FAILED) {
        const std::string err = std::strerror(errno);
        ::close(impl.mem_fd);
        impl.mem_fd = -1;
        restore_kernel_runtime_power(impl);
        throw std::runtime_error("[DisplaySPI] Failed to mmap RV1106 CRU registers: " + err);
    }
    impl.cru_regs = static_cast<volatile uint8_t*>(cru_map);
    enable_register_spi_clocks(impl);

    void* spi_map = ::mmap(nullptr, kRegisterMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, impl.mem_fd,
                           impl.spi_phys_base);
    if (spi_map == MAP_FAILED) {
        const std::string err = std::strerror(errno);
        ::munmap(const_cast<uint8_t*>(impl.cru_regs), kCruMapSize);
        impl.cru_regs = nullptr;
        ::close(impl.mem_fd);
        impl.mem_fd = -1;
        restore_kernel_runtime_power(impl);
        throw std::runtime_error("[DisplaySPI] Failed to mmap SPI registers at 0x" +
                                  std::to_string(static_cast<unsigned long long>(impl.spi_phys_base)) + ": " + err);
    }
    impl.spi_regs = static_cast<volatile uint8_t*>(spi_map);

    const uint32_t version = spi_reg_read(impl, kSpiVersion);
    impl.register_fifo_len = (version == kSpiVer2Type1 || version == kSpiVer2Type2) ? 64 : 32;
    release_spi_child_driver(impl);

    spi_reg_write(impl, kSpiSsienr, 0);
    spi_reg_write(impl, kSpiImr, 0);
    spi_reg_write(impl, kSpiIcr, 0xffffffff);
    spi_reg_write(impl, kSpiDmacr, 0);
}

template <typename ImplT>
void open_register_spi_device(ImplT& impl) {
    if (DisplayFB::is_any_active()) {
        throw std::runtime_error(
            "[DisplaySPI] Cannot use the register SPI backend while DisplayFB is active. "
            "Call DisplayFB.release() first, or use backend='spidev' with a free SPI device.");
    }
    if (impl.config.bits_per_word != 8) {
        throw std::invalid_argument("[DisplaySPI] Register SPI backend currently supports 8 bits per word only.");
    }
    if (!parse_spi_bus_name(impl.spi_bus_path, &impl.spi_index, &impl.chip_select)) {
        throw std::invalid_argument("[DisplaySPI] Cannot resolve SPI controller for register backend from '" +
                                    impl.spi_bus_path + "'. Use spi0, spi1, spi0.0, /dev/spidev0.0, or ff500000.spi.");
    }

    impl.spi_phys_base = register_spi_base_for_index(impl.spi_index);
    impl.register_source_clock_hz = impl.config.source_clock_hz ? impl.config.source_clock_hz : kDefaultSourceClockHz;
    impl.hw_fd = ::open("/dev/visiong-hw", O_RDWR | O_CLOEXEC);
    if (impl.hw_fd < 0) {
        impl.hw_reg_accel_unavailable = true;
    }
    hold_kernel_runtime_power(impl);
    release_spi_child_driver(impl);
    if (!impl.hw_reg_accel_unavailable) {
        release_platform_spi_driver(impl);
    }

    if (impl.hw_reg_accel_unavailable) {
        ensure_register_spi_legacy_maps(impl);
    }
}

template <typename ImplT>
void close_register_spi_device(ImplT& impl) {
    if (impl.hw_fd >= 0) {
        visiong_hw_spi_reg_release request{};
        request.size = sizeof(request);
        request.bus = static_cast<uint32_t>(impl.spi_index);
        (void)::ioctl(impl.hw_fd, VISIONG_HW_SPI_REG_RELEASE, &request);
        ::close(impl.hw_fd);
        impl.hw_fd = -1;
    }
    if (impl.spi_regs) {
        spi_reg_write(impl, kSpiSsienr, 0);
        spi_reg_write(impl, kSpiSer, 0);
        ::munmap(const_cast<uint8_t*>(impl.spi_regs), kRegisterMapSize);
        impl.spi_regs = nullptr;
    }
    if (impl.cru_regs) {
        ::munmap(const_cast<uint8_t*>(impl.cru_regs), kCruMapSize);
        impl.cru_regs = nullptr;
    }
    if (impl.mem_fd >= 0) {
        ::close(impl.mem_fd);
        impl.mem_fd = -1;
    }
    restore_platform_spi_driver(impl);
    restore_spi_child_driver(impl);
    restore_kernel_runtime_power(impl);
}

template <typename ImplT>
bool spi_transport_is_open(const ImplT& impl) {
    if (impl.active_backend == "spidev") {
        return impl.spi_fd >= 0;
    }
    if (impl.active_backend == "reg") {
        return (impl.hw_fd >= 0 && !impl.hw_reg_accel_unavailable) || impl.spi_regs != nullptr;
    }
    return false;
}

template <typename ImplT>
void open_display_transport(ImplT& impl) {
    const std::string backend = visiong::to_lower_copy(trim_copy(impl.backend.empty() ? kDefaultBackend : impl.backend));
    if (backend == "spidev" || backend == "linux") {
        try {
            open_spi_device(impl);
        } catch (const std::exception& first_error) {
            try {
                bind_spidev_and_open_spi_device(impl);
            } catch (const std::exception& second_error) {
                throw std::runtime_error(std::string(first_error.what()) +
                                         "; spidev bind/open error: " + second_error.what());
            }
        }
        impl.active_backend = "spidev";
        return;
    }
    if (backend == "reg" || backend == "register" || backend == "direct") {
        open_register_spi_device(impl);
        impl.active_backend = "reg";
        return;
    }
    if (backend != "auto") {
        throw std::invalid_argument("[DisplaySPI] Unsupported SPI backend '" + impl.backend +
                                    "'. Supported: auto, spidev, reg.");
    }

    std::string spidev_error;
    try {
        open_spi_device(impl);
        impl.active_backend = "spidev";
        return;
    } catch (const std::exception& e) {
        spidev_error = e.what();
    }

    try {
        bind_spidev_and_open_spi_device(impl);
        impl.active_backend = "spidev";
        VISIONG_LOG_INFO("DisplaySPI", "bound SPI child to spidev for kernel SPI transfer. Previous error: "
                                           << spidev_error);
        return;
    } catch (const std::exception& e) {
        spidev_error += "; spidev bind/open error: ";
        spidev_error += e.what();
    }

    try {
        open_register_spi_device(impl);
        impl.active_backend = "reg";
        VISIONG_LOG_INFO("DisplaySPI", "spidev backend unavailable, using register backend. Reason: " << spidev_error);
        return;
    } catch (const std::exception& e) {
        throw std::runtime_error("[DisplaySPI] Failed to open SPI using backend='auto'. spidev error: " +
                                  spidev_error + "; register error: " + e.what());
    }
}

template <typename ImplT>
void spidev_transfer(ImplT& impl, const void* data, size_t len) {
    if (impl.spi_fd < 0 || data == nullptr || len == 0) {
        return;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    const size_t chunk_size =
        impl.config.transfer_chunk_size > 0 ? impl.config.transfer_chunk_size : kDefaultTransferChunkSize;
    size_t offset = 0;
    while (offset < len) {
        const size_t chunk = std::min(chunk_size, len - offset);
        spi_ioc_transfer transfer{};
        transfer.tx_buf = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr + offset));
        transfer.len = static_cast<uint32_t>(chunk);
        transfer.speed_hz = impl.config.speed_hz;
        transfer.bits_per_word = impl.config.bits_per_word;

        if (::ioctl(impl.spi_fd, SPI_IOC_MESSAGE(1), &transfer) < 1) {
            throw std::runtime_error("[DisplaySPI] SPI transfer failed: " + std::string(std::strerror(errno)));
        }
        offset += chunk;
    }
}

template <typename ImplT>
void register_spi_transfer(ImplT& impl, const void* data, size_t len) {
    if (data == nullptr || len == 0) {
        return;
    }
    if (impl.hw_fd < 0 && !impl.spi_regs) {
        ensure_register_spi_legacy_maps(impl);
    }

    if (impl.hw_fd >= 0 && !impl.hw_reg_accel_unavailable) {
        const uint8_t* ptr = static_cast<const uint8_t*>(data);
        size_t offset = 0;
        size_t hw_chunk_limit = impl.config.transfer_chunk_size;
        if (hw_chunk_limit == 0 || hw_chunk_limit == kDefaultTransferChunkSize) {
            hw_chunk_limit = kDefaultHwRegTransferChunkSize;
        }
        hw_chunk_limit = std::max<size_t>(kMinHwRegTransferChunkSize,
                                          std::min<size_t>(hw_chunk_limit, kSpiMaxTransferLen));
        bool completed_by_hw = true;
        while (offset < len) {
            size_t chunk = std::min<size_t>(len - offset, hw_chunk_limit);
            visiong_hw_spi_reg_transfer request{};
            request.size = sizeof(request);
            request.bus = static_cast<uint32_t>(impl.spi_index);
            request.chip_select = static_cast<uint32_t>(impl.chip_select);
            request.speed_hz = impl.config.speed_hz;
            request.source_clock_hz = impl.register_source_clock_hz;
            request.mode = impl.config.spi_mode & 0x3u;
            request.bits_per_word = impl.config.bits_per_word;
            request.flags = VISIONG_HW_SPI_REG_TX_ONLY;
            request.tx_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr + offset));
            request.tx_len = static_cast<uint32_t>(chunk);
            request.dummy = 0xff;

            if (::ioctl(impl.hw_fd, VISIONG_HW_SPI_REG_TRANSFER, &request) != 0) {
                if (errno == ENOTTY || errno == ENODEV || errno == EOPNOTSUPP || errno == ENOSYS) {
                    impl.hw_reg_accel_unavailable = true;
                    completed_by_hw = false;
                    break;
                }
                if ((errno == ENOMEM || errno == EAGAIN) && hw_chunk_limit > kMinHwRegTransferChunkSize) {
                    hw_chunk_limit = std::max<size_t>(kMinHwRegTransferChunkSize, hw_chunk_limit / 2);
                    continue;
                }
                throw std::runtime_error("[DisplaySPI] HW register SPI transfer failed: " +
                                         std::string(std::strerror(errno)));
            }
            if (request.status != VISIONG_HW_SPI_STATUS_DONE || request.transferred != chunk) {
                throw std::runtime_error("[DisplaySPI] HW register SPI transfer returned incomplete status.");
            }
            offset += chunk;
        }
        if (completed_by_hw) {
            return;
        }
    }

    ensure_register_spi_legacy_maps(impl);
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    size_t configured_chunk = impl.config.transfer_chunk_size;
    if (configured_chunk == 0 || configured_chunk == kDefaultTransferChunkSize) {
        configured_chunk = kSpiMaxTransferLen;
    }
    const size_t chunk_size = std::min<size_t>(configured_chunk, kSpiMaxTransferLen);
    size_t offset = 0;
    while (offset < len) {
        const size_t chunk = std::min(chunk_size, len - offset);
        configure_register_spi_transfer(impl, chunk);
        spi_reg_write(impl, kSpiSer, 1u << impl.chip_select);
        spi_reg_write(impl, kSpiSsienr, 1);

        size_t written = 0;
        auto progress_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (written < chunk) {
            const uint32_t tx_level = spi_reg_read(impl, kSpiTxflr);
            if (tx_level < impl.register_fifo_len) {
                const size_t writable = std::min<size_t>(impl.register_fifo_len - tx_level, chunk - written);
                for (size_t i = 0; i < writable; ++i) {
                    spi_reg_write_relaxed(impl, kSpiTxdr, ptr[offset + written + i]);
                }
                __sync_synchronize();
                written += writable;
                progress_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            } else {
                std::this_thread::yield();
            }
            if (std::chrono::steady_clock::now() > progress_deadline) {
                spi_reg_write(impl, kSpiSsienr, 0);
                spi_reg_write(impl, kSpiSer, 0);
                throw std::runtime_error("[DisplaySPI] Register SPI transfer timed out while filling TX FIFO.");
            }
        }

        wait_register_spi_idle(impl, chunk);
        spi_reg_write(impl, kSpiSsienr, 0);
        spi_reg_write(impl, kSpiSer, 0);
        offset += chunk;
    }
}

template <typename ImplT>
void spi_transfer(ImplT& impl, const void* data, size_t len) {
    if (impl.active_backend == "reg") {
        register_spi_transfer(impl, data, len);
    } else {
        spidev_transfer(impl, data, len);
    }
}

template <typename ImplT>
void write_command(ImplT& impl, uint8_t command) {
    set_gpio_value(impl, impl.config.dc_pin, 0);
    spi_transfer(impl, &command, 1);
}

template <typename ImplT>
void write_data(ImplT& impl, const void* data, size_t len) {
    set_gpio_value(impl, impl.config.dc_pin, 1);
    spi_transfer(impl, data, len);
}

template <typename ImplT>
void write_data_u8(ImplT& impl, uint8_t value) {
    write_data(impl, &value, 1);
}

template <typename ImplT>
void hardware_reset(ImplT& impl) {
    if (impl.config.reset_pin.empty()) {
        write_command(impl, kCmdSwReset);
        sleep_ms(150);
        return;
    }

    set_gpio_value(impl, impl.config.reset_pin, 1);
    sleep_ms(10);
    set_gpio_value(impl, impl.config.reset_pin, 0);
    sleep_ms(20);
    set_gpio_value(impl, impl.config.reset_pin, 1);
    sleep_ms(120);
}

template <typename ImplT>
void set_window(ImplT& impl, int x, int y, int w, int h) {
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > impl.screen_width || y + h > impl.screen_height) {
        throw std::out_of_range("[DisplaySPI] Window is outside the configured screen.");
    }

    const uint16_t x0 = static_cast<uint16_t>(impl.config.x_offset + x);
    const uint16_t x1 = static_cast<uint16_t>(impl.config.x_offset + x + w - 1);
    const uint16_t y0 = static_cast<uint16_t>(impl.config.y_offset + y);
    const uint16_t y1 = static_cast<uint16_t>(impl.config.y_offset + y + h - 1);

    const uint8_t col_data[4] = {
        static_cast<uint8_t>(x0 >> 8),
        static_cast<uint8_t>(x0 & 0xFF),
        static_cast<uint8_t>(x1 >> 8),
        static_cast<uint8_t>(x1 & 0xFF),
    };
    const uint8_t row_data[4] = {
        static_cast<uint8_t>(y0 >> 8),
        static_cast<uint8_t>(y0 & 0xFF),
        static_cast<uint8_t>(y1 >> 8),
        static_cast<uint8_t>(y1 & 0xFF),
    };

    write_command(impl, kCmdColumnAddr);
    write_data(impl, col_data, sizeof(col_data));
    write_command(impl, kCmdRowAddr);
    write_data(impl, row_data, sizeof(row_data));
    write_command(impl, kCmdMemoryWrite);
}

template <typename ImplT>
void ensure_transfer_buffer(ImplT& impl) {
    const size_t required = static_cast<size_t>(impl.screen_width) * impl.screen_height * 2;
    if (impl.transfer_buffer.size() != required) {
        impl.transfer_buffer.assign(required, 0);
    }
}

template <typename ImplT>
void ensure_frame_buffers(ImplT& impl) {
    const size_t required = static_cast<size_t>(impl.screen_width) * impl.screen_height * 2;
    const size_t count = std::max<size_t>(2, impl.config.buffer_count);
    if (impl.frame_buffers.size() != count ||
        (!impl.frame_buffers.empty() && impl.frame_buffers.front().size() != required)) {
        impl.frame_buffers.assign(count, std::vector<uint8_t>(required));
        impl.transfer_pending = false;
        impl.pending_buffer_index = -1;
        impl.active_buffer_index = -1;
    }
}

template <typename ImplT>
void ensure_screen_dma(ImplT& impl) {
    if (!impl.screen_dma ||
        impl.screen_dma->get_width() != impl.screen_width ||
        impl.screen_dma->get_height() != impl.screen_height ||
        impl.screen_dma->get_mpi_format() != RK_FMT_RGB565) {
        impl.screen_dma = std::make_unique<RgaDmaBuffer>(impl.screen_width, impl.screen_height, RK_FMT_RGB565);
    }
}

template <typename ImplT>
void transfer_full_frame(ImplT& impl, const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> spi_guard(impl.spi_lock);
    if (!impl.initialized || !spi_transport_is_open(impl)) {
        return;
    }
    set_window(impl, 0, 0, impl.screen_width, impl.screen_height);
    write_data(impl, data, size);
}

template <typename ImplT>
void submit_frame_async(ImplT& impl) {
    std::lock_guard<std::mutex> transfer_guard(impl.transfer_lock);
    ensure_frame_buffers(impl);

    int target_index = -1;
    for (size_t i = 0; i < impl.frame_buffers.size(); ++i) {
        const int index = static_cast<int>(i);
        if (index != impl.active_buffer_index && index != impl.pending_buffer_index) {
            target_index = index;
            break;
        }
    }
    if (target_index < 0) {
        target_index = impl.pending_buffer_index >= 0 ? impl.pending_buffer_index : 0;
    }

    std::memcpy(impl.frame_buffers[static_cast<size_t>(target_index)].data(),
                impl.transfer_buffer.data(),
                impl.transfer_buffer.size());
    impl.pending_buffer_index = target_index;
    impl.transfer_pending = true;
    impl.transfer_cv.notify_one();
}

template <typename ImplT>
void start_transfer_worker(ImplT& impl) {
    if (!impl.config.multi_buffering || impl.transfer_running) {
        return;
    }

    ensure_frame_buffers(impl);
    impl.transfer_stop = false;
    impl.transfer_pending = false;
    impl.pending_buffer_index = -1;
    impl.active_buffer_index = -1;
    impl.transfer_running = true;
    impl.transfer_thread = std::thread([&impl]() {
        while (true) {
            int buffer_index = -1;
            {
                std::unique_lock<std::mutex> transfer_guard(impl.transfer_lock);
                impl.transfer_cv.wait(transfer_guard, [&impl]() {
                    return impl.transfer_stop || impl.transfer_pending;
                });
                if (impl.transfer_stop) {
                    break;
                }
                buffer_index = impl.pending_buffer_index;
                impl.pending_buffer_index = -1;
                impl.transfer_pending = false;
                impl.active_buffer_index = buffer_index;
            }

            try {
                const auto& frame = impl.frame_buffers[static_cast<size_t>(buffer_index)];
                transfer_full_frame(impl, frame.data(), frame.size());
            } catch (const std::exception& e) {
                VISIONG_LOG_ERROR("DisplaySPI", "Async transfer failed: " << e.what());
            }

            {
                std::lock_guard<std::mutex> transfer_guard(impl.transfer_lock);
                if (impl.active_buffer_index == buffer_index) {
                    impl.active_buffer_index = -1;
                }
                impl.transfer_cv.notify_all();
            }
        }
    });
}

template <typename ImplT>
void stop_transfer_worker(ImplT& impl) {
    {
        std::lock_guard<std::mutex> transfer_guard(impl.transfer_lock);
        if (!impl.transfer_running) {
            return;
        }
        impl.transfer_stop = true;
        impl.transfer_cv.notify_one();
    }
    if (impl.transfer_thread.joinable()) {
        impl.transfer_thread.join();
    }
    {
        std::lock_guard<std::mutex> transfer_guard(impl.transfer_lock);
        impl.transfer_running = false;
        impl.transfer_stop = false;
        impl.transfer_pending = false;
        impl.pending_buffer_index = -1;
        impl.active_buffer_index = -1;
        impl.transfer_cv.notify_all();
    }
}

template <typename ImplT>
void wait_transfer_idle(ImplT& impl) {
    if (!impl.config.multi_buffering) {
        return;
    }

    std::unique_lock<std::mutex> transfer_guard(impl.transfer_lock);
    impl.transfer_cv.wait(transfer_guard, [&impl]() {
        return !impl.transfer_running || (!impl.transfer_pending && impl.active_buffer_index < 0);
    });
}

template <typename ImplT>
bool clip_rect_to_screen(const ImplT& impl, int* x, int* y, int* w, int* h) {
    if (!x || !y || !w || !h || *w <= 0 || *h <= 0) {
        return false;
    }
    const int x0 = std::max(0, *x);
    const int y0 = std::max(0, *y);
    const int x1 = std::min(impl.screen_width, *x + *w);
    const int y1 = std::min(impl.screen_height, *y + *h);
    if (x0 >= x1 || y0 >= y1) {
        return false;
    }
    *x = x0;
    *y = y0;
    *w = x1 - x0;
    *h = y1 - y0;
    return true;
}

template <typename ImplT>
void transfer_shadow_region(ImplT& impl, int x, int y, int w, int h) {
    if (!clip_rect_to_screen(impl, &x, &y, &w, &h)) {
        return;
    }

    std::lock_guard<std::mutex> spi_guard(impl.spi_lock);
    if (!impl.initialized || !spi_transport_is_open(impl)) {
        return;
    }
    set_window(impl, x, y, w, h);
    const size_t screen_stride = static_cast<size_t>(impl.screen_width) * 2;
    const size_t row_bytes = static_cast<size_t>(w) * 2;

    const uint8_t* first_row = impl.transfer_buffer.data() +
                               static_cast<size_t>(y) * screen_stride +
                               static_cast<size_t>(x) * 2;
    if (w == impl.screen_width) {
        write_data(impl, first_row, row_bytes * static_cast<size_t>(h));
        return;
    }
    if (h == 1) {
        write_data(impl, first_row, row_bytes);
        return;
    }

    const size_t total_bytes = row_bytes * static_cast<size_t>(h);
    if (impl.region_transfer_buffer.size() < total_bytes) {
        impl.region_transfer_buffer.resize(total_bytes);
    }
    for (int row = 0; row < h; ++row) {
        const uint8_t* src = impl.transfer_buffer.data() +
                             static_cast<size_t>(y + row) * screen_stride +
                             static_cast<size_t>(x) * 2;
        std::memcpy(impl.region_transfer_buffer.data() + static_cast<size_t>(row) * row_bytes,
                    src,
                    row_bytes);
    }
    write_data(impl, impl.region_transfer_buffer.data(), total_bytes);
}

template <typename ImplT>
void fill_shadow_rect(ImplT& impl, int x, int y, int w, int h, uint16_t color_rgb565) {
    if (!clip_rect_to_screen(impl, &x, &y, &w, &h)) {
        return;
    }
    ensure_transfer_buffer(impl);

    const uint8_t hi = static_cast<uint8_t>(color_rgb565 >> 8);
    const uint8_t lo = static_cast<uint8_t>(color_rgb565 & 0xFF);
    const size_t stride = static_cast<size_t>(impl.screen_width) * 2;
    uint8_t* first_row = impl.transfer_buffer.data() +
                         static_cast<size_t>(y) * stride +
                         static_cast<size_t>(x) * 2;
    const size_t row_bytes = static_cast<size_t>(w) * 2;
    if (hi == lo) {
        std::memset(first_row, hi, row_bytes);
    } else {
        for (int col = 0; col < w; ++col) {
            first_row[col * 2] = hi;
            first_row[col * 2 + 1] = lo;
        }
    }
    for (int row = 1; row < h; ++row) {
        uint8_t* dst = impl.transfer_buffer.data() +
                       static_cast<size_t>(y + row) * stride +
                       static_cast<size_t>(x) * 2;
        std::memcpy(dst, first_row, row_bytes);
    }
}

template <typename ImplT>
void put_shadow_pixel(ImplT& impl, int x, int y, uint16_t color_rgb565) {
    if (x < 0 || y < 0 || x >= impl.screen_width || y >= impl.screen_height) {
        return;
    }
    ensure_transfer_buffer(impl);
    uint8_t* dst = impl.transfer_buffer.data() +
                   (static_cast<size_t>(y) * impl.screen_width + static_cast<size_t>(x)) * 2;
    dst[0] = static_cast<uint8_t>(color_rgb565 >> 8);
    dst[1] = static_cast<uint8_t>(color_rgb565 & 0xFF);
}

template <typename ImplT>
void put_shadow_thick_pixel(ImplT& impl, int x, int y, uint16_t color_rgb565, int thickness) {
    thickness = std::max(1, thickness);
    const int half = thickness / 2;
    fill_shadow_rect(impl, x - half, y - half, thickness, thickness, color_rgb565);
}

template <typename ImplT>
bool copy_rgb565_be_to_shadow(ImplT& impl,
                              int x,
                              int y,
                              int w,
                              int h,
                              const uint8_t* src,
                              size_t src_size,
                              size_t src_stride,
                              bool src_native_endian) {
    if (src == nullptr || w <= 0 || h <= 0) {
        return false;
    }
    if (src_stride == 0) {
        src_stride = static_cast<size_t>(w) * 2;
    }
    if (src_stride < static_cast<size_t>(w) * 2) {
        throw std::invalid_argument("[DisplaySPI] RGB565 source stride is smaller than row width.");
    }
    const size_t needed = src_stride * static_cast<size_t>(h - 1) + static_cast<size_t>(w) * 2;
    if (src_size < needed) {
        throw std::invalid_argument("[DisplaySPI] RGB565 source buffer is smaller than requested area.");
    }

    int src_x = 0;
    int src_y = 0;
    if (x < 0) {
        src_x = -x;
        w -= src_x;
        x = 0;
    }
    if (y < 0) {
        src_y = -y;
        h -= src_y;
        y = 0;
    }
    if (w <= 0 || h <= 0 || x >= impl.screen_width || y >= impl.screen_height) {
        return false;
    }
    w = std::min(w, impl.screen_width - x);
    h = std::min(h, impl.screen_height - y);
    if (w <= 0 || h <= 0) {
        return false;
    }

    ensure_transfer_buffer(impl);
    const size_t dst_stride = static_cast<size_t>(impl.screen_width) * 2;
    const uint8_t* src_base = src + static_cast<size_t>(src_y) * src_stride + static_cast<size_t>(src_x) * 2;
    uint8_t* dst_base = impl.transfer_buffer.data() +
                        static_cast<size_t>(y) * dst_stride +
                        static_cast<size_t>(x) * 2;
    if (src_native_endian) {
        rgb565_native_to_be(src_base, dst_base, w, h, static_cast<int>(src_stride), static_cast<int>(dst_stride));
    } else {
        for (int row = 0; row < h; ++row) {
            const uint8_t* src_row = src_base + static_cast<size_t>(row) * src_stride;
            uint8_t* dst_row = dst_base + static_cast<size_t>(row) * dst_stride;
            std::memcpy(dst_row, src_row, static_cast<size_t>(w) * 2);
        }
    }
    return true;
}

template <typename ImplT>
void render_gray_to_transfer(ImplT& impl,
                             const ImageBuffer& image,
                             const ScaledRect& target) {
    ensure_transfer_buffer(impl);
    const bool covers_full_screen = target.x == 0 &&
                                    target.y == 0 &&
                                    target.w == impl.screen_width &&
                                    target.h == impl.screen_height;
    if (!covers_full_screen) {
        std::fill(impl.transfer_buffer.begin(), impl.transfer_buffer.end(), 0);
    }

    if (!impl.cached_gray_src_dma ||
        impl.cached_gray_src_w != target.roi.w ||
        impl.cached_gray_src_h != target.roi.h) {
        impl.cached_gray_src_dma = std::make_unique<RgaDmaBuffer>(target.roi.w, target.roi.h, image.format);
        impl.cached_gray_src_w = target.roi.w;
        impl.cached_gray_src_h = target.roi.h;
    }

    const void* roi_src = static_cast<const uint8_t*>(image.get_data()) +
                           static_cast<size_t>(target.roi.y) * image.w_stride + target.roi.x;
    copy_data_with_stride(impl.cached_gray_src_dma->get_vir_addr(),
                           impl.cached_gray_src_dma->get_wstride(),
                           roi_src,
                           image.w_stride,
                           target.roi.h,
                           target.roi.w);
    dma_sync_cpu_to_device(impl.cached_gray_src_dma->get_fd());

    if (!impl.cached_gray_scaled_dma ||
        impl.cached_gray_scaled_w != target.w ||
        impl.cached_gray_scaled_h != target.h) {
        impl.cached_gray_scaled_dma = std::make_unique<RgaDmaBuffer>(target.w, target.h, image.format);
        impl.cached_gray_scaled_w = target.w;
        impl.cached_gray_scaled_h = target.h;
    }

    if (imresize(impl.cached_gray_src_dma->get_buffer(), impl.cached_gray_scaled_dma->get_buffer()) !=
        IM_STATUS_SUCCESS) {
        throw std::runtime_error("[DisplaySPI] RGA grayscale resize failed.");
    }
    dma_sync_device_to_cpu(impl.cached_gray_scaled_dma->get_fd());

    uint8_t* dst = impl.transfer_buffer.data() +
                    static_cast<size_t>(target.y) * impl.screen_width * 2 +
                    static_cast<size_t>(target.x) * 2;
    gray8_to_rgb565_be(static_cast<const uint8_t*>(impl.cached_gray_scaled_dma->get_vir_addr()),
                        dst,
                        target.w,
                        target.h,
                        impl.cached_gray_scaled_dma->get_wstride(),
                        impl.screen_width * 2);
}

template <typename ImplT>
void render_color_to_transfer(ImplT& impl,
                              const ImageBuffer& image,
                              const ScaledRect& target) {
    ensure_transfer_buffer(impl);
    ensure_screen_dma(impl);

    const im_rect full_rect = {0, 0, impl.screen_width, impl.screen_height};
    const im_rect dst_rect = {target.x, target.y, target.w, target.h};
    constexpr uint32_t kBlackFillColor = 0x00000000;

    const bool covers_full_screen = target.x == 0 &&
                                    target.y == 0 &&
                                    target.w == impl.screen_width &&
                                    target.h == impl.screen_height;
    if (!covers_full_screen) {
        if (imfill(impl.screen_dma->get_buffer(), full_rect, kBlackFillColor) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("[DisplaySPI] RGA screen clear failed.");
        }
    }

    const bool can_use_src_zero_copy = image.is_zero_copy() && image.get_dma_fd() >= 0;
    if (can_use_src_zero_copy) {
        RgaDmaBuffer src_wrapper(image.get_dma_fd(),
                                  const_cast<void*>(image.get_data()),
                                  image.get_size(),
                                  image.width,
                                  image.height,
                                  static_cast<int>(image.format),
                                  image.w_stride,
                                  image.h_stride);
        dma_sync_cpu_to_device(src_wrapper.get_fd());
        const im_rect src_rect = {target.roi.x, target.roi.y, target.roi.w, target.roi.h};
        if (improcess(src_wrapper.get_buffer(), impl.screen_dma->get_buffer(), {}, src_rect, dst_rect, {}, IM_SYNC) !=
            IM_STATUS_SUCCESS) {
            throw std::runtime_error("[DisplaySPI] RGA zero-copy render failed.");
        }
    } else {
        if (!impl.cached_src_dma ||
            impl.cached_src_w != target.roi.w ||
            impl.cached_src_h != target.roi.h ||
            impl.cached_src_fmt != image.format) {
            impl.cached_src_dma = std::make_unique<RgaDmaBuffer>(target.roi.w, target.roi.h, image.format);
            impl.cached_src_w = target.roi.w;
            impl.cached_src_h = target.roi.h;
            impl.cached_src_fmt = image.format;
        }
        copy_roi_to_dma_buffer(image, target.roi, *impl.cached_src_dma);
        const im_rect src_rect = {0, 0, target.roi.w, target.roi.h};
        if (improcess(impl.cached_src_dma->get_buffer(), impl.screen_dma->get_buffer(), {}, src_rect, dst_rect, {},
                      IM_SYNC) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("[DisplaySPI] RGA render failed.");
        }
    }

    dma_sync_device_to_cpu(impl.screen_dma->get_fd());
    rgb565_native_to_be(static_cast<const uint8_t*>(impl.screen_dma->get_vir_addr()),
                         impl.transfer_buffer.data(),
                         impl.screen_width,
                         impl.screen_height,
                         impl.screen_dma->get_wstride() * 2,
                         impl.screen_width * 2);
}

std::vector<uint8_t> render_roi_to_rgb565_be(const ImageBuffer& image, const RectROI& roi) {
    if (roi.w <= 0 || roi.h <= 0) {
        return {};
    }

    std::vector<uint8_t> out(static_cast<size_t>(roi.w) * roi.h * 2);
    if (image.format == visiong::kGray8Format) {
        const uint8_t* src = static_cast<const uint8_t*>(image.get_data()) +
                             static_cast<size_t>(roi.y) * image.w_stride + roi.x;
        gray8_to_rgb565_be(src, out.data(), roi.w, roi.h, image.w_stride, roi.w * 2);
        return out;
    }

    if (image.format == RK_FMT_RGB565) {
        const int bytes_per_pixel = 2;
        const uint8_t* src = static_cast<const uint8_t*>(image.get_data()) +
                             static_cast<size_t>(roi.y) * image.w_stride * bytes_per_pixel +
                             static_cast<size_t>(roi.x) * bytes_per_pixel;
        rgb565_native_to_be(src, out.data(), roi.w, roi.h, image.w_stride * bytes_per_pixel, roi.w * 2);
        return out;
    }

    RgaDmaBuffer dst_dma(roi.w, roi.h, RK_FMT_RGB565);
    const im_rect dst_rect = {0, 0, roi.w, roi.h};
    const bool can_use_src_zero_copy = image.is_zero_copy() && image.get_dma_fd() >= 0;
    if (can_use_src_zero_copy) {
        RgaDmaBuffer src_wrapper(image.get_dma_fd(),
                                  const_cast<void*>(image.get_data()),
                                  image.get_size(),
                                  image.width,
                                  image.height,
                                  static_cast<int>(image.format),
                                  image.w_stride,
                                  image.h_stride);
        dma_sync_cpu_to_device(src_wrapper.get_fd());
        const im_rect src_rect = {roi.x, roi.y, roi.w, roi.h};
        if (improcess(src_wrapper.get_buffer(), dst_dma.get_buffer(), {}, src_rect, dst_rect, {}, IM_SYNC) !=
            IM_STATUS_SUCCESS) {
            throw std::runtime_error("[DisplaySPI] RGA area zero-copy render failed.");
        }
    } else {
        RgaDmaBuffer src_dma(roi.w, roi.h, image.format);
        copy_roi_to_dma_buffer(image, roi, src_dma);
        const im_rect src_rect = {0, 0, roi.w, roi.h};
        if (improcess(src_dma.get_buffer(), dst_dma.get_buffer(), {}, src_rect, dst_rect, {}, IM_SYNC) !=
            IM_STATUS_SUCCESS) {
            throw std::runtime_error("[DisplaySPI] RGA area render failed.");
        }
    }

    dma_sync_device_to_cpu(dst_dma.get_fd());
    rgb565_native_to_be(static_cast<const uint8_t*>(dst_dma.get_vir_addr()),
                         out.data(),
                         roi.w,
                         roi.h,
                         dst_dma.get_wstride() * 2,
                         roi.w * 2);
    return out;
}

template <typename ImplT>
void init_st7735(ImplT& impl) {
    hardware_reset(impl);

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    const uint8_t fr1[] = {0x01, 0x2C, 0x2D};
    write_command(impl, 0xB1);
    write_data(impl, fr1, sizeof(fr1));
    const uint8_t fr2[] = {0x01, 0x2C, 0x2D};
    write_command(impl, 0xB2);
    write_data(impl, fr2, sizeof(fr2));
    const uint8_t fr3[] = {0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D};
    write_command(impl, 0xB3);
    write_data(impl, fr3, sizeof(fr3));

    write_command(impl, 0xB4);
    write_data_u8(impl, 0x07);

    const uint8_t pw1[] = {0xA2, 0x02, 0x84};
    write_command(impl, 0xC0);
    write_data(impl, pw1, sizeof(pw1));
    write_command(impl, 0xC1);
    write_data_u8(impl, 0xC5);
    const uint8_t pw3[] = {0x0A, 0x00};
    write_command(impl, 0xC2);
    write_data(impl, pw3, sizeof(pw3));
    const uint8_t pw4[] = {0x8A, 0x2A};
    write_command(impl, 0xC3);
    write_data(impl, pw4, sizeof(pw4));
    const uint8_t pw5[] = {0x8A, 0xEE};
    write_command(impl, 0xC4);
    write_data(impl, pw5, sizeof(pw5));

    write_command(impl, 0xC5);
    write_data_u8(impl, 0x0E);

    write_command(impl, kCmdInvertOff);

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x05);

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    write_command(impl, 0x26);
    write_data_u8(impl, 0x01);

    const uint8_t pg[] = {0x02, 0x1C, 0x07, 0x12, 0x37, 0x32, 0x29, 0x2D,
                          0x29, 0x25, 0x2B, 0x39, 0x00, 0x01, 0x03, 0x10};
    write_command(impl, 0xE0);
    write_data(impl, pg, sizeof(pg));
    const uint8_t ng[] = {0x03, 0x1D, 0x07, 0x06, 0x2E, 0x2C, 0x29, 0x2D,
                          0x2E, 0x2E, 0x37, 0x3F, 0x00, 0x00, 0x02, 0x10};
    write_command(impl, 0xE1);
    write_data(impl, ng, sizeof(ng));

    write_command(impl, kCmdNormalOn);
    sleep_ms(10);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

template <typename ImplT>
void init_st7789(ImplT& impl) {
    hardware_reset(impl);

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x55);

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    write_command(impl, impl.config.invert ? kCmdInvertOn : kCmdInvertOff);

    write_command(impl, kCmdNormalOn);
    sleep_ms(10);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

template <typename ImplT>
void init_st7796(ImplT& impl) {
    hardware_reset(impl);

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    const uint8_t dfc[] = {0x02, 0x02, 0x0B};
    write_command(impl, 0xB6);
    write_data(impl, dfc, sizeof(dfc));

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x55);

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    write_command(impl, impl.config.invert ? kCmdInvertOn : kCmdInvertOff);

    write_command(impl, kCmdNormalOn);
    sleep_ms(10);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

template <typename ImplT>
void init_ili9341(ImplT& impl) {
    hardware_reset(impl);

    const uint8_t pca[] = {0x39, 0x2C, 0x00, 0x34, 0x02};
    write_command(impl, 0xCB);
    write_data(impl, pca, sizeof(pca));

    const uint8_t pcb[] = {0x00, 0xC1, 0x30};
    write_command(impl, 0xCF);
    write_data(impl, pcb, sizeof(pcb));

    const uint8_t dtca[] = {0x85, 0x00, 0x78};
    write_command(impl, 0xE8);
    write_data(impl, dtca, sizeof(dtca));

    const uint8_t dtcb[] = {0x00, 0x00};
    write_command(impl, 0xEA);
    write_data(impl, dtcb, sizeof(dtcb));

    const uint8_t pon[] = {0x64, 0x03, 0x12, 0x81};
    write_command(impl, 0xED);
    write_data(impl, pon, sizeof(pon));

    write_command(impl, 0xC0);
    write_data_u8(impl, 0x23);

    write_command(impl, 0xC1);
    write_data_u8(impl, 0x10);

    const uint8_t vm1[] = {0x3E, 0x28};
    write_command(impl, 0xC5);
    write_data(impl, vm1, sizeof(vm1));

    write_command(impl, 0xC7);
    write_data_u8(impl, 0x86);

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x55);

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    const uint8_t fr[] = {0x00, 0x18};
    write_command(impl, 0xB1);
    write_data(impl, fr, sizeof(fr));

    const uint8_t div[] = {0x08, 0x82, 0x27};
    write_command(impl, 0xB6);
    write_data(impl, div, sizeof(div));

    write_command(impl, 0xF2);
    write_data_u8(impl, 0x00);

    write_command(impl, 0x26);
    write_data_u8(impl, 0x01);

    const uint8_t pg[] = {0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1,
                          0x37, 0x07, 0x10, 0x03, 0x0E, 0x09, 0x00};
    write_command(impl, 0xE0);
    write_data(impl, pg, sizeof(pg));
    const uint8_t ng[] = {0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1,
                          0x48, 0x08, 0x0F, 0x0C, 0x31, 0x36, 0x0F};
    write_command(impl, 0xE1);
    write_data(impl, ng, sizeof(ng));

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    write_command(impl, impl.config.invert ? kCmdInvertOn : kCmdInvertOff);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

template <typename ImplT>
void init_ili9163(ImplT& impl) {
    hardware_reset(impl);

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    const uint8_t fr[] = {0x08, 0x08};
    write_command(impl, 0xB1);
    write_data(impl, fr, sizeof(fr));

    const uint8_t pw1[] = {0x0F, 0x06};
    write_command(impl, 0xC0);
    write_data(impl, pw1, sizeof(pw1));

    write_command(impl, 0xC1);
    write_data_u8(impl, 0x04);

    write_command(impl, 0xC2);
    write_data_u8(impl, 0x03);

    write_command(impl, 0xC5);
    write_data_u8(impl, 0x48);

    write_command(impl, 0x26);
    write_data_u8(impl, 0x01);

    const uint8_t pg[] = {0x36, 0x29, 0x12, 0x22, 0x1C, 0x15, 0x42, 0xB7,
                          0x2F, 0x13, 0x12, 0x0A, 0x11, 0x0B, 0x06};
    write_command(impl, 0xE0);
    write_data(impl, pg, sizeof(pg));
    const uint8_t ng[] = {0x09, 0x16, 0x1D, 0x0D, 0x13, 0x2A, 0x3D, 0x48,
                          0x10, 0x0C, 0x0D, 0x05, 0x0E, 0x14, 0x19};
    write_command(impl, 0xE1);
    write_data(impl, ng, sizeof(ng));

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x05);

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    write_command(impl, impl.config.invert ? kCmdInvertOn : kCmdInvertOff);

    write_command(impl, kCmdNormalOn);
    sleep_ms(10);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

template <typename ImplT>
void init_ili9488(ImplT& impl) {
    hardware_reset(impl);

    write_command(impl, kCmdSleepOut);
    sleep_ms(120);

    const uint8_t pw1[] = {0x17, 0x15};
    write_command(impl, 0xC0);
    write_data(impl, pw1, sizeof(pw1));

    write_command(impl, 0xC1);
    write_data_u8(impl, 0x41);

    const uint8_t vm[] = {0x00, 0x12, 0x80};
    write_command(impl, 0xC5);
    write_data(impl, vm, sizeof(vm));

    write_command(impl, kCmdPixelFormat);
    write_data_u8(impl, 0x55);

    write_command(impl, 0xB0);
    write_data_u8(impl, 0x00);

    write_command(impl, 0xB1);
    write_data_u8(impl, 0xA0);

    write_command(impl, 0xB6);
    write_data_u8(impl, 0x02);

    write_command(impl, 0x26);
    write_data_u8(impl, 0x01);

    const uint8_t pg[] = {0x00, 0x07, 0x10, 0x09, 0x17, 0x0B, 0x41, 0x89,
                          0x4B, 0x0A, 0x0C, 0x0E, 0x18, 0x1B, 0x0F};
    write_command(impl, 0xE0);
    write_data(impl, pg, sizeof(pg));
    const uint8_t ng[] = {0x00, 0x20, 0x21, 0x02, 0x08, 0x09, 0x04, 0x4B,
                          0x24, 0x0B, 0x13, 0x15, 0x27, 0x2A, 0x0F};
    write_command(impl, 0xE1);
    write_data(impl, ng, sizeof(ng));

    write_command(impl, kCmdMadctl);
    write_data_u8(impl, madctl_for_rotation(impl.config));

    write_command(impl, impl.config.invert ? kCmdInvertOn : kCmdInvertOff);

    write_command(impl, kCmdDisplayOn);
    sleep_ms(120);

    set_gpio_value(impl, impl.config.backlight_pin, 1);
}

}  // namespace

std::unique_ptr<DisplaySPIDevice> create_display_spi_device(const std::string& chip_model,
                                                             const std::string& spi_bus_path,
                                                             const DisplaySPIConfig& config) {
    return std::make_unique<DisplaySPI>(chip_model, spi_bus_path, config);
}

DisplaySPI::DisplaySPI(const std::string& chip_model,
                       const std::string& spi_bus_path,
                       const DisplaySPIConfig& config)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->spi_bus_path = spi_bus_path.empty() ? kDefaultSpiBusPath : spi_bus_path;
    m_impl->config = config;
    m_impl->backend = m_impl->config.backend.empty() ? kDefaultBackend : m_impl->config.backend;
    if (m_impl->config.dc_pin.empty()) {
        m_impl->config.dc_pin = kDefaultDcPin;
    }
    if (m_impl->config.reset_pin.empty()) {
        m_impl->config.reset_pin = kDefaultResetPin;
    }
    if (m_impl->config.speed_hz == 0) {
        m_impl->config.speed_hz = kDefaultSpeedHz;
    }
    if (m_impl->config.source_clock_hz == 0) {
        m_impl->config.source_clock_hz = kDefaultSourceClockHz;
    }
    if (m_impl->config.transfer_chunk_size == 0) {
        m_impl->config.transfer_chunk_size = kDefaultTransferChunkSize;
    }
    m_impl->config.buffer_count = std::max<size_t>(2, m_impl->config.buffer_count);

    const std::string model = visiong::to_lower_copy(chip_model);
    m_impl->chip_model = model;

    if (model == "st7735" || model == "st7735s" || model == "st7735r") {
        m_impl->init_sequence = [](Impl& impl) { init_st7735(impl); };
    } else if (model == "st7789" || model == "st7789v" || model == "st7789vw" || model == "st7789vi") {
        m_impl->init_sequence = [](Impl& impl) { init_st7789(impl); };
    } else if (model == "st7796" || model == "st7796s") {
        m_impl->init_sequence = [](Impl& impl) { init_st7796(impl); };
    } else if (model == "ili9341" || model == "ili9341v") {
        m_impl->init_sequence = [](Impl& impl) { init_ili9341(impl); };
    } else if (model == "ili9163" || model == "ili9163c" || model == "ili9163v") {
        m_impl->init_sequence = [](Impl& impl) { init_ili9163(impl); };
    } else if (model == "ili9488" || model == "ili9488l") {
        m_impl->init_sequence = [](Impl& impl) { init_ili9488(impl); };
    } else {
        throw std::invalid_argument("[DisplaySPI] Unsupported chip model: " + chip_model +
                                    ". Supported: ST7735, ST7789, ST7796, ILI9341, ILI9163, ILI9488.");
    }

    try {
        configure_geometry(config.width, config.height, config.rotation_degrees);
        request_control_gpios(*m_impl);
        open_display_transport(*m_impl);
        m_impl->init_sequence(*m_impl);
        m_impl->initialized = true;
        start_transfer_worker(*m_impl);
        clear(0);
        VISIONG_LOG_INFO("DisplaySPI",
                         m_impl->chip_model << " initialized on "
                                            << m_impl->spi_bus_path << " via " << m_impl->active_backend
                                            << ", screen " << m_impl->screen_width
                                            << "x" << m_impl->screen_height << ", speed "
                                            << m_impl->config.speed_hz << " Hz");
    } catch (...) {
        release();
        throw;
    }
}

DisplaySPI::~DisplaySPI() {
    release();
}

void DisplaySPI::release() {
    stop_transfer_worker(*m_impl);

    std::lock_guard<std::mutex> guard(m_impl->lock);

    if (m_impl->spi_fd >= 0) {
        std::lock_guard<std::mutex> spi_guard(m_impl->spi_lock);
        ::close(m_impl->spi_fd);
        m_impl->spi_fd = -1;
    }
    if (m_impl->hw_fd >= 0 || m_impl->mem_fd >= 0 || m_impl->spi_regs || m_impl->cru_regs ||
        !m_impl->released_spi_child.empty() || !m_impl->released_platform_device.empty() ||
        !m_impl->power_control_path.empty()) {
        std::lock_guard<std::mutex> spi_guard(m_impl->spi_lock);
        close_register_spi_device(*m_impl);
    }
    m_impl->active_backend.clear();
    m_impl->initialized = false;
    m_impl->screen_dma.reset();
    m_impl->cached_src_dma.reset();
    m_impl->cached_gray_src_dma.reset();
    m_impl->cached_gray_scaled_dma.reset();
    m_impl->transfer_buffer.clear();
    m_impl->transfer_buffer.shrink_to_fit();
    m_impl->region_transfer_buffer.clear();
    m_impl->region_transfer_buffer.shrink_to_fit();
    m_impl->frame_buffers.clear();
    m_impl->frame_buffers.shrink_to_fit();
    m_impl->gpio.reset();
}

bool DisplaySPI::is_initialized() const {
    return m_impl->initialized;
}

bool DisplaySPI::display(const ImageBuffer& img_buf) {
    return display(img_buf, std::make_tuple(0, 0, 0, 0));
}

bool DisplaySPI::display(ImageBuffer&& img_buf) {
    return display(static_cast<const ImageBuffer&>(img_buf), std::make_tuple(0, 0, 0, 0));
}

bool DisplaySPI::display(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi) {
    if (!img_buf.is_valid()) {
        return false;
    }

    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }

    try {
        const auto [roi_x, roi_y, roi_w, roi_h] = roi;
        const RectROI requested_roi = (roi_w <= 0 || roi_h <= 0) ? RectROI{0, 0, img_buf.width, img_buf.height}
                                                                  : RectROI{roi_x, roi_y, roi_w, roi_h};
        const ScaledRect target = compute_scaled_rect(requested_roi, img_buf,
                                                       m_impl->screen_width,
                                                       m_impl->screen_height);
        if (img_buf.format == visiong::kGray8Format) {
            render_gray_to_transfer(*m_impl, img_buf, target);
        } else {
            render_color_to_transfer(*m_impl, img_buf, target);
        }

        if (m_impl->config.multi_buffering) {
            submit_frame_async(*m_impl);
        } else {
            transfer_full_frame(*m_impl, m_impl->transfer_buffer.data(), m_impl->transfer_buffer.size());
        }
        return true;
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("DisplaySPI", "Display failed: " << e.what());
        return false;
    }
}

bool DisplaySPI::display(ImageBuffer&& img_buf, const std::tuple<int, int, int, int>& roi) {
    return display(static_cast<const ImageBuffer&>(img_buf), roi);
}

bool DisplaySPI::display_area(const ImageBuffer& img_buf, int x, int y) {
    return display_area(img_buf, x, y, std::make_tuple(0, 0, 0, 0));
}

bool DisplaySPI::display_area(ImageBuffer&& img_buf, int x, int y) {
    return display_area(static_cast<const ImageBuffer&>(img_buf), x, y, std::make_tuple(0, 0, 0, 0));
}

bool DisplaySPI::display_area(const ImageBuffer& img_buf,
                              int x,
                              int y,
                              const std::tuple<int, int, int, int>& roi) {
    if (!img_buf.is_valid()) {
        return false;
    }

    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }

    try {
        wait_transfer_idle(*m_impl);
        const auto [roi_x, roi_y, roi_w, roi_h] = roi;
        RectROI src_roi = (roi_w <= 0 || roi_h <= 0) ? RectROI{0, 0, img_buf.width, img_buf.height}
                                                      : RectROI{roi_x, roi_y, roi_w, roi_h};
        src_roi = clamp_and_align_roi(src_roi, img_buf);

        if (x < 0) {
            const int crop = std::min(src_roi.w, -x);
            src_roi.x += crop;
            src_roi.w -= crop;
            x = 0;
        }
        if (y < 0) {
            const int crop = std::min(src_roi.h, -y);
            src_roi.y += crop;
            src_roi.h -= crop;
            y = 0;
        }
        if (src_roi.w <= 0 || src_roi.h <= 0 || x >= m_impl->screen_width || y >= m_impl->screen_height) {
            return true;
        }
        src_roi.w = std::min(src_roi.w, m_impl->screen_width - x);
        src_roi.h = std::min(src_roi.h, m_impl->screen_height - y);
        if (src_roi.w <= 0 || src_roi.h <= 0) {
            return true;
        }

        const std::vector<uint8_t> pixels = render_roi_to_rgb565_be(img_buf, src_roi);
        if (!copy_rgb565_be_to_shadow(*m_impl,
                                      x,
                                      y,
                                      src_roi.w,
                                      src_roi.h,
                                      pixels.data(),
                                      pixels.size(),
                                      static_cast<size_t>(src_roi.w) * 2,
                                      false)) {
            return true;
        }
        transfer_shadow_region(*m_impl, x, y, src_roi.w, src_roi.h);
        return true;
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("DisplaySPI", "Display area failed: " << e.what());
        return false;
    }
}

bool DisplaySPI::display_area(ImageBuffer&& img_buf,
                              int x,
                              int y,
                              const std::tuple<int, int, int, int>& roi) {
    return display_area(static_cast<const ImageBuffer&>(img_buf), x, y, roi);
}

bool DisplaySPI::draw_rgb565(int x,
                             int y,
                             int w,
                             int h,
                             const void* data,
                             size_t size_bytes,
                             size_t stride_bytes,
                             bool source_is_native_endian) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }

    try {
        wait_transfer_idle(*m_impl);
        const int original_x = x;
        const int original_y = y;
        const bool copied = copy_rgb565_be_to_shadow(*m_impl,
                                                     x,
                                                     y,
                                                     w,
                                                     h,
                                                     static_cast<const uint8_t*>(data),
                                                     size_bytes,
                                                     stride_bytes,
                                                     source_is_native_endian);
        if (!copied) {
            return true;
        }

        if (original_x < 0) {
            w += original_x;
            x = 0;
        }
        if (original_y < 0) {
            h += original_y;
            y = 0;
        }
        w = std::min(w, m_impl->screen_width - x);
        h = std::min(h, m_impl->screen_height - y);
        transfer_shadow_region(*m_impl, x, y, w, h);
        return true;
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("DisplaySPI", "draw_rgb565 failed: " << e.what());
        return false;
    }
}

bool DisplaySPI::draw_pixel(int x, int y, uint16_t color_rgb565) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }
    wait_transfer_idle(*m_impl);
    put_shadow_pixel(*m_impl, x, y, color_rgb565);
    transfer_shadow_region(*m_impl, x, y, 1, 1);
    return true;
}

bool DisplaySPI::draw_line(int x0, int y0, int x1, int y1, uint16_t color_rgb565, int thickness) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }
    wait_transfer_idle(*m_impl);
    ensure_transfer_buffer(*m_impl);

    thickness = std::max(1, thickness);
    const int dirty_x0 = std::min(x0, x1) - thickness;
    const int dirty_y0 = std::min(y0, y1) - thickness;
    const int dirty_x1 = std::max(x0, x1) + thickness;
    const int dirty_y1 = std::max(y0, y1) + thickness;

    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        put_shadow_thick_pixel(*m_impl, x0, y0, color_rgb565, thickness);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    transfer_shadow_region(*m_impl, dirty_x0, dirty_y0, dirty_x1 - dirty_x0 + 1, dirty_y1 - dirty_y0 + 1);
    return true;
}

bool DisplaySPI::draw_rectangle(int x,
                                int y,
                                int w,
                                int h,
                                uint16_t color_rgb565,
                                int thickness,
                                bool fill) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }
    wait_transfer_idle(*m_impl);
    ensure_transfer_buffer(*m_impl);
    if (w <= 0 || h <= 0) {
        return true;
    }
    thickness = std::max(1, thickness);

    if (fill) {
        fill_shadow_rect(*m_impl, x, y, w, h, color_rgb565);
    } else {
        const int t = std::min(thickness, std::min(w, h));
        fill_shadow_rect(*m_impl, x, y, w, t, color_rgb565);
        fill_shadow_rect(*m_impl, x, y + h - t, w, t, color_rgb565);
        fill_shadow_rect(*m_impl, x, y, t, h, color_rgb565);
        fill_shadow_rect(*m_impl, x + w - t, y, t, h, color_rgb565);
    }
    transfer_shadow_region(*m_impl, x, y, w, h);
    return true;
}

bool DisplaySPI::draw_circle(int cx, int cy, int radius, uint16_t color_rgb565, int thickness, bool fill) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!m_impl->initialized || !spi_transport_is_open(*m_impl)) {
        return false;
    }
    wait_transfer_idle(*m_impl);
    ensure_transfer_buffer(*m_impl);
    if (radius <= 0) {
        return true;
    }
    thickness = std::max(1, thickness);

    if (fill) {
        for (int dy = -radius; dy <= radius; ++dy) {
            const int row_half = static_cast<int>(std::sqrt(static_cast<double>(radius * radius - dy * dy)));
            fill_shadow_rect(*m_impl, cx - row_half, cy + dy, row_half * 2 + 1, 1, color_rgb565);
        }
    } else {
        const int inner = std::max(0, radius - thickness + 1);
        for (int dy = -radius; dy <= radius; ++dy) {
            const int outer_half = static_cast<int>(std::sqrt(static_cast<double>(radius * radius - dy * dy)));
            int inner_half = -1;
            if (inner > 0 && std::abs(dy) <= inner) {
                inner_half = static_cast<int>(std::sqrt(static_cast<double>(inner * inner - dy * dy)));
            }
            if (inner_half < 0) {
                fill_shadow_rect(*m_impl, cx - outer_half, cy + dy, outer_half * 2 + 1, 1, color_rgb565);
            } else {
                fill_shadow_rect(*m_impl, cx - outer_half, cy + dy, outer_half - inner_half, 1, color_rgb565);
                fill_shadow_rect(*m_impl, cx + inner_half + 1, cy + dy, outer_half - inner_half, 1, color_rgb565);
            }
        }
    }
    transfer_shadow_region(*m_impl,
                           cx - radius - thickness,
                           cy - radius - thickness,
                           radius * 2 + thickness * 2 + 1,
                           radius * 2 + thickness * 2 + 1);
    return true;
}

bool DisplaySPI::draw_cross(int cx, int cy, uint16_t color_rgb565, int size, int thickness) {
    size = std::max(1, size);
    const int half = size / 2;
    const bool h_ok = draw_line(cx - half, cy, cx + half, cy, color_rgb565, thickness);
    const bool v_ok = draw_line(cx, cy - half, cx, cy + half, color_rgb565, thickness);
    return h_ok && v_ok;
}

void DisplaySPI::clear(uint16_t color_rgb565) {
    std::lock_guard<std::mutex> guard(m_impl->lock);
    if (!spi_transport_is_open(*m_impl)) {
        return;
    }

    ensure_transfer_buffer(*m_impl);
    fill_rgb565_be(m_impl->transfer_buffer, color_rgb565);
    if (m_impl->config.multi_buffering) {
        submit_frame_async(*m_impl);
    } else {
        transfer_full_frame(*m_impl, m_impl->transfer_buffer.data(), m_impl->transfer_buffer.size());
    }
}

void DisplaySPI::configure_geometry(int width, int height, int rotation_degrees) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("DisplaySPI width and height must be positive.");
    }

    const bool restart_worker = m_impl->initialized && m_impl->config.multi_buffering;
    stop_transfer_worker(*m_impl);

    {
        std::lock_guard<std::mutex> guard(m_impl->lock);
        m_impl->config.width = width;
        m_impl->config.height = height;
        m_impl->config.rotation_degrees = normalize_rotation(rotation_degrees);
        m_impl->screen_width = logical_width_for(m_impl->config);
        m_impl->screen_height = logical_height_for(m_impl->config);
        ensure_transfer_buffer(*m_impl);
        m_impl->region_transfer_buffer.clear();
        m_impl->region_transfer_buffer.shrink_to_fit();
        {
            std::lock_guard<std::mutex> transfer_guard(m_impl->transfer_lock);
            if (m_impl->config.multi_buffering) {
                ensure_frame_buffers(*m_impl);
            } else {
                m_impl->frame_buffers.clear();
                m_impl->frame_buffers.shrink_to_fit();
                m_impl->transfer_pending = false;
                m_impl->pending_buffer_index = -1;
                m_impl->active_buffer_index = -1;
            }
        }
        m_impl->screen_dma.reset();
        m_impl->cached_src_dma.reset();
        m_impl->cached_gray_src_dma.reset();
        m_impl->cached_gray_scaled_dma.reset();

        if (spi_transport_is_open(*m_impl)) {
            std::lock_guard<std::mutex> spi_guard(m_impl->spi_lock);
            write_command(*m_impl, kCmdMadctl);
            write_data_u8(*m_impl, madctl_for_rotation(m_impl->config));
        }
    }

    if (restart_worker) {
        start_transfer_worker(*m_impl);
    }
}

int DisplaySPI::get_screen_width() const {
    return m_impl->screen_width;
}

int DisplaySPI::get_screen_height() const {
    return m_impl->screen_height;
}
