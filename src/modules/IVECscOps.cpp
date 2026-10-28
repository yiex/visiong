// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "modules/internal/ive_memory.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

bool is_yuv_layout(IVE_IMAGE_TYPE_E type) {
    return type == IVE_IMAGE_TYPE_YUV420SP || type == IVE_IMAGE_TYPE_YUV422SP;
}

bool is_color_layout(IVE_IMAGE_TYPE_E type) {
    return type == IVE_IMAGE_TYPE_U8C3_PACKAGE || type == IVE_IMAGE_TYPE_U8C3_PLANAR;
}

struct CscBytesMode {
    bool yuv_input;
    bool yuv_output;
    bool hsv_output;
};

CscBytesMode csc_bytes_mode(IVE_CSC_MODE_E mode) {
    const int value = static_cast<int>(mode);
    if (value < 0 || (value & ~0x33) || (value & 0x0f) > 3) {
        throw std::invalid_argument("Unsupported CSC mode.");
    }
    return {value < 0x20, value >= 0x20 && value < 0x30,
            (value >= 0x10 && value < 0x20) || value >= 0x30};
}

size_t csc_compact_size(IVE_IMAGE_TYPE_E type, int width, int height) {
    const size_t pixels = static_cast<size_t>(width) * height;
    if (type == IVE_IMAGE_TYPE_YUV420SP) {
        if ((width & 1) || (height & 1)) throw std::invalid_argument("YUV420SP requires even width and height.");
        return pixels + pixels / 2;
    }
    if (type == IVE_IMAGE_TYPE_YUV422SP) {
        if (width & 1) throw std::invalid_argument("YUV422SP requires even width.");
        return pixels * 2;
    }
    if (is_color_layout(type)) return pixels * 3;
    throw std::invalid_argument("CSC byte layout must be YUV420SP, YUV422SP, U8C3_PACKAGE or U8C3_PLANAR.");
}

class CscImage {
public:
    CscImage(int width, int height, bool yuv) {
        using visiong::ive_internal::calc_stride;
        image.enType = yuv ? IVE_IMAGE_TYPE_YUV420SP : IVE_IMAGE_TYPE_U8C3_PACKAGE;
        image.u32Width = width;
        image.u32Height = height;
        image.au32Stride[0] = calc_stride(width);
        const RK_U32 plane_size = image.au32Stride[0] * height;
        size = yuv ? plane_size + plane_size / 2 : plane_size * 3;
        void* address = nullptr;
        if (visiong::ive_internal::mmz_alloc(&image.au64PhyAddr[0], &address, size) != RK_SUCCESS) {
            throw std::runtime_error("CSC failed to allocate an image buffer.");
        }
        image.au64VirAddr[0] = reinterpret_cast<RK_U64>(address);
        if (yuv) {
            image.au32Stride[1] = image.au32Stride[0];
            image.au64PhyAddr[1] = image.au64PhyAddr[0] + plane_size;
            image.au64VirAddr[1] = image.au64VirAddr[0] + plane_size;
        }
    }
    ~CscImage() {
        visiong::ive_internal::free_mmz(image.au64PhyAddr[0], image.au64VirAddr[0]);
    }
    CscImage(const CscImage&) = delete;
    CscImage& operator=(const CscImage&) = delete;

    IVE_IMAGE_S image{};
    RK_U32 size = 0;
};

class CscCpuAccess {
public:
    explicit CscCpuAccess(RK_U64 address) : address_(reinterpret_cast<void*>(address)) {
        if (visiong::ive_internal::mmz_flush_start(address_) != RK_SUCCESS) {
            throw std::runtime_error("CSC failed to begin buffer access.");
        }
    }
    ~CscCpuAccess() {
        if (address_) visiong::ive_internal::mmz_flush_end(address_);
    }
    void finish() {
        void* address = address_;
        address_ = nullptr;
        if (visiong::ive_internal::mmz_flush_end(address) != RK_SUCCESS) {
            throw std::runtime_error("CSC failed to end buffer access.");
        }
    }
    CscCpuAccess(const CscCpuAccess&) = delete;
    CscCpuAccess& operator=(const CscCpuAccess&) = delete;

private:
    void* address_;
};

uint8_t compact_color(const std::vector<uint8_t>& bytes, IVE_IMAGE_TYPE_E type,
                      int width, int height, int x, int y, int channel) {
    const size_t index = static_cast<size_t>(y) * width + x;
    return type == IVE_IMAGE_TYPE_U8C3_PACKAGE ? bytes[index * 3 + channel] :
        bytes[static_cast<size_t>(channel) * width * height + index];
}

void set_compact_color(std::vector<uint8_t>& bytes, IVE_IMAGE_TYPE_E type,
                       int width, int height, int x, int y, int channel, uint8_t value) {
    const size_t index = static_cast<size_t>(y) * width + x;
    bytes[type == IVE_IMAGE_TYPE_U8C3_PACKAGE ? index * 3 + channel :
          static_cast<size_t>(channel) * width * height + index] = value;
}

void stage_csc_yuv(const std::vector<uint8_t>& bytes, IVE_IMAGE_TYPE_E type, int width, int height,
                   bool full_range, bool vu_order, int row_offset, CscImage& destination) {
    const bool double_rows = type == IVE_IMAGE_TYPE_YUV422SP;
    const int stride = destination.image.au32Stride[0];
    auto* output = reinterpret_cast<uint8_t*>(destination.image.au64VirAddr[0]);
    auto* chroma = reinterpret_cast<uint8_t*>(destination.image.au64VirAddr[1]);
    const size_t luma_size = static_cast<size_t>(width) * height;
    for (int y = 0; y < static_cast<int>(destination.image.u32Height); ++y) {
        const auto* source = bytes.data() + static_cast<size_t>(row_offset + (double_rows ? y / 2 : y)) * width;
        auto* row = output + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
            row[x] = full_range ? static_cast<uint8_t>(16 + (source[x] * 219u + 127) / 255) : source[x];
        }
    }
    const int chroma_rows = destination.image.u32Height / 2;
    const int chroma_offset = double_rows ? row_offset : row_offset / 2;
    for (int y = 0; y < chroma_rows; ++y) {
        const auto* source = bytes.data() + luma_size + static_cast<size_t>(chroma_offset + y) * width;
        auto* row = chroma + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
            const unsigned value = source[vu_order ? x : (x ^ 1)];
            row[x] = full_range ? static_cast<uint8_t>(16 + (value * 224u + 127) / 255) : static_cast<uint8_t>(value);
        }
    }
}

void stage_csc_color(const std::vector<uint8_t>& bytes, IVE_IMAGE_TYPE_E type, int width, int height,
                     bool bgr_order, bool double_rows, int row_offset, CscImage& destination) {
    const size_t row_stride = static_cast<size_t>(destination.image.au32Stride[0]) * 3;
    auto* output = reinterpret_cast<uint8_t*>(destination.image.au64VirAddr[0]);
    for (int y = 0; y < static_cast<int>(destination.image.u32Height); ++y) {
        auto* row = output + static_cast<size_t>(y) * row_stride;
        const int source_y = row_offset + (double_rows ? y / 2 : y);
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                row[x * 3 + channel] = compact_color(bytes, type, width, height, x, source_y,
                                                     bgr_order ? channel : 2 - channel);
            }
        }
    }
}

void read_csc_yuv(const CscImage& source, IVE_IMAGE_TYPE_E type, int width, int height,
                  bool vu_order, int row_offset, std::vector<uint8_t>& output) {
    const bool double_rows = type == IVE_IMAGE_TYPE_YUV422SP;
    const int stride = source.image.au32Stride[0];
    const auto* luma = reinterpret_cast<const uint8_t*>(source.image.au64VirAddr[0]);
    const auto* chroma = reinterpret_cast<const uint8_t*>(source.image.au64VirAddr[1]);
    const size_t luma_size = static_cast<size_t>(width) * height;
    const int block_height = double_rows ? source.image.u32Height / 2 : source.image.u32Height;
    for (int y = 0; y < block_height; ++y) {
        std::memcpy(output.data() + static_cast<size_t>(row_offset + y) * width,
                    luma + static_cast<size_t>(double_rows ? y * 2 : y) * stride, width);
    }
    const int chroma_rows = source.image.u32Height / 2;
    const int chroma_offset = double_rows ? row_offset : row_offset / 2;
    for (int y = 0; y < chroma_rows; ++y) {
        const auto* row = chroma + static_cast<size_t>(y) * stride;
        auto* target = output.data() + luma_size + static_cast<size_t>(chroma_offset + y) * width;
        for (int x = 0; x < width; ++x) target[x] = row[vu_order ? x : (x ^ 1)];
    }
}

void read_csc_color(const CscImage& source, IVE_IMAGE_TYPE_E type, int width, int height,
                    bool bgr_order, bool hsv_output, bool double_rows, int row_offset, std::vector<uint8_t>& output) {
    const size_t row_stride = static_cast<size_t>(source.image.au32Stride[0]) * 3;
    const auto* bytes = reinterpret_cast<const uint8_t*>(source.image.au64VirAddr[0]);
    const int block_height = double_rows ? source.image.u32Height / 2 : source.image.u32Height;
    for (int y = 0; y < block_height; ++y) {
        const auto* row = bytes + static_cast<size_t>(double_rows ? y * 2 : y) * row_stride;
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const int native_channel = hsv_output || bgr_order ? channel : 2 - channel;
                set_compact_color(output, type, width, height, x, row_offset + y, channel, row[x * 3 + native_channel]);
            }
        }
    }
}

}  // namespace

std::vector<uint8_t> IVE::csc_bytes(std::vector<uint8_t> data, int width, int height,
                                  IVE_IMAGE_TYPE_E src_type, IVE_IMAGE_TYPE_E dst_type, IVE_CSC_MODE_E mode,
                                  bool input_full_range, bool bgr_order, bool vu_order) {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        throw std::invalid_argument("CSC byte dimensions must be in [1,4096].");
    }
    const CscBytesMode traits = csc_bytes_mode(mode);
    const size_t input_size = csc_compact_size(src_type, width, height);
    const size_t output_size = csc_compact_size(dst_type, width, height);
    if (data.size() != input_size) throw std::invalid_argument("CSC input byte count does not match the compact layout.");
    if (traits.yuv_input != is_yuv_layout(src_type) || traits.yuv_output != is_yuv_layout(dst_type) ||
        (!traits.yuv_input && !is_color_layout(src_type)) || (!traits.yuv_output && !is_color_layout(dst_type))) {
        throw std::invalid_argument("CSC input and output layouts do not match the conversion mode.");
    }
    if (input_full_range && !traits.yuv_input) {
        throw std::invalid_argument("input_full_range is supported only for YUV input.");
    }
    if (vu_order && !traits.yuv_input && !traits.yuv_output) {
        throw std::invalid_argument("vu_order requires a YUV input or output.");
    }
    if (bgr_order && traits.yuv_input && traits.hsv_output) {
        throw std::invalid_argument("YUV-to-HSV has no RGB byte order to select.");
    }
    const bool double_rows = src_type == IVE_IMAGE_TYPE_YUV422SP || dst_type == IVE_IMAGE_TYPE_YUV422SP;
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<uint8_t> output(output_size);
    // Captured CSC output becomes incorrect at 2048 native rows; YUV blocks also need even heights.
    // 实测 CSC 输出在 2048 个原生行处发生异常；YUV 分块还须保持偶数高度。
    const bool yuv_conversion = traits.yuv_input || traits.yuv_output;
    const int maximum_rows = double_rows ? 1023 : (yuv_conversion ? 2046 : 2047);
    for (int row_offset = 0; row_offset < height;) {
        int block_height = std::min(maximum_rows, height - row_offset);
        if (!double_rows && !yuv_conversion && height - row_offset - block_height == 1 && block_height > 1) {
            --block_height;
        }
        const int native_height = double_rows ? block_height * 2 : block_height;
        CscImage source(width, native_height, traits.yuv_input);
        CscImage destination(width, native_height, traits.yuv_output);
        {
            CscCpuAccess access(source.image.au64VirAddr[0]);
            std::memset(reinterpret_cast<void*>(source.image.au64VirAddr[0]), 0, source.size);
            if (traits.yuv_input) stage_csc_yuv(data, src_type, width, height, input_full_range, vu_order, row_offset, source);
            else stage_csc_color(data, src_type, width, height, bgr_order, double_rows, row_offset, source);
            access.finish();
        }
        {
            CscCpuAccess access(destination.image.au64VirAddr[0]);
            std::memset(reinterpret_cast<void*>(destination.image.au64VirAddr[0]), 0, destination.size);
            access.finish();
        }
        // The runtime fixes YUV to 420SP. Each 422 block duplicates at most 1023 original rows around one hardware CSC.
        // 运行库将 YUV 固定为 420SP；每个 422 分块在一次硬件 CSC 前后重复、抽取至多 1023 个原始行。
        // Plane offsets use the complete image height so 422 chroma and planar channels remain continuous across blocks.
        // 平面偏移使用完整图像高度，使 422 色度和平面通道在块间保持连续。
        IVE_CSC_CTRL_S control{};
        control.enMode = mode;
        IVE_HANDLE handle{};
        if (RK_MPI_IVE_CSC(&handle, &source.image, &destination.image, &control, RK_TRUE) != RK_SUCCESS) {
            throw std::runtime_error("RK_MPI_IVE_CSC failed.");
        }
        {
            CscCpuAccess access(destination.image.au64VirAddr[0]);
            if (traits.yuv_output) read_csc_yuv(destination, dst_type, width, height, vu_order, row_offset, output);
            else read_csc_color(destination, dst_type, width, height, bgr_order, traits.hsv_output, double_rows, row_offset, output);
            access.finish();
        }
        row_offset += block_height;
    }
    return output;
}
