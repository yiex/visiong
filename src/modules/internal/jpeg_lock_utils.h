// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/common/pixel_format.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace visiong::jpeg_lock {

inline bool is_yuv_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU;
}

inline int get_color_priority(PIXEL_FORMAT_E format) {
    if (format == RK_FMT_BGR888) {
        return 3;
    }
    if (format == RK_FMT_RGB888) {
        return 2;
    }
    if (is_yuv_format(format)) {
        return 1;
    }
    return 0;
}

inline PIXEL_FORMAT_E choose_lock_format(PIXEL_FORMAT_E input_format) {
    if (input_format == RK_FMT_BGR888) {
        return RK_FMT_BGR888;
    }
    if (input_format == RK_FMT_RGB888) {
        return RK_FMT_RGB888;
    }
    if (is_yuv_format(input_format)) {
        return RK_FMT_YUV420SP;
    }
    return RK_FMT_YUV420SP;
}

inline void normalize_lock_size(PIXEL_FORMAT_E lock_format, int* width, int* height) {
    if (!width || !height) {
        return;
    }
    *width = std::max(*width, 1);
    *height = std::max(*height, 1);
    if (is_yuv_format(lock_format)) {
        *width = std::max(2, (*width + 1) & ~1);
        *height = std::max(2, (*height + 1) & ~1);
    }
}

inline bool should_expand_lock_size(int locked_width,
                                    int locked_height,
                                    int candidate_width,
                                    int candidate_height) {
    return candidate_width > locked_width || candidate_height > locked_height;
}

inline ImageBuffer pad_frame_without_scaling(const ImageBuffer& src,
                                             PIXEL_FORMAT_E lock_format,
                                             int target_width,
                                             int target_height) {
    if (!src.is_valid()) {
        throw std::runtime_error("JPEG lock pad: invalid source image.");
    }
    if (src.width == target_width && src.height == target_height) {
        return src.copy();
    }
    if (src.width > target_width || src.height > target_height) {
        throw std::runtime_error("JPEG lock pad: source is larger than the locked canvas.");
    }

    const int src_x_stride = std::max(src.w_stride, src.width);
    const int pad_x_raw = (target_width - src.width) / 2;
    const int pad_y_raw = (target_height - src.height) / 2;

    if (lock_format == RK_FMT_BGR888 || lock_format == RK_FMT_RGB888) {
        std::vector<unsigned char> padded(static_cast<size_t>(target_width) * target_height * 3U, 0U);
        const int pad_x = pad_x_raw;
        const int pad_y = pad_y_raw;
        const size_t src_row_bytes = static_cast<size_t>(src.width) * 3U;
        const size_t src_stride_bytes = static_cast<size_t>(src_x_stride) * 3U;
        const size_t dst_stride_bytes = static_cast<size_t>(target_width) * 3U;
        visiong::bufstate::prepare_cpu_read(src);
        const auto* src_ptr = static_cast<const unsigned char*>(src.get_data());
        auto* dst_ptr = padded.data();
        for (int y = 0; y < src.height; ++y) {
            std::memcpy(dst_ptr + static_cast<size_t>(pad_y + y) * dst_stride_bytes +
                            static_cast<size_t>(pad_x) * 3U,
                        src_ptr + static_cast<size_t>(y) * src_stride_bytes,
                        src_row_bytes);
        }
        return ImageBuffer(target_width, target_height, lock_format, std::move(padded));
    }

    if (lock_format == RK_FMT_YUV420SP) {
        const int pad_x = pad_x_raw & ~1;
        const int pad_y = pad_y_raw & ~1;
        std::vector<unsigned char> padded(static_cast<size_t>(target_width) * target_height * 3U / 2U, 0U);
        std::fill(padded.begin() + static_cast<ptrdiff_t>(target_width) * target_height, padded.end(), 128U);

        visiong::bufstate::prepare_cpu_read(src);
        const auto* src_ptr = static_cast<const unsigned char*>(src.get_data());
        const size_t src_y_plane_bytes = static_cast<size_t>(src_x_stride) * src.height;
        const auto* src_y = src_ptr;
        const auto* src_uv = src_ptr + src_y_plane_bytes;

        auto* dst_y = padded.data();
        auto* dst_uv = padded.data() + static_cast<size_t>(target_width) * target_height;
        for (int y = 0; y < src.height; ++y) {
            std::memcpy(dst_y + static_cast<size_t>(pad_y + y) * target_width + pad_x,
                        src_y + static_cast<size_t>(y) * src_x_stride,
                        static_cast<size_t>(src.width));
        }
        for (int y = 0; y < src.height / 2; ++y) {
            std::memcpy(dst_uv + static_cast<size_t>(pad_y / 2 + y) * target_width + pad_x,
                        src_uv + static_cast<size_t>(y) * src_x_stride,
                        static_cast<size_t>(src.width));
        }
        return ImageBuffer(target_width, target_height, RK_FMT_YUV420SP, std::move(padded));
    }

    throw std::runtime_error("JPEG lock pad: unsupported locked format.");
}

}  // namespace visiong::jpeg_lock
