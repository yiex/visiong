// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"

#include <stdexcept>
#include <unordered_map>

namespace visiong {
namespace {

std::string normalize_key(const std::string& key) {
    return visiong::to_lower_copy(key);
}

const std::unordered_map<std::string, PIXEL_FORMAT_E>& pixel_format_aliases() {
    static const std::unordered_map<std::string, PIXEL_FORMAT_E> aliases = {
        {"rgb888", RK_FMT_RGB888},
        {"rgba8888", RK_FMT_RGBA8888},
        {"rgb565", RK_FMT_RGB565},
        {"bgr565", RK_FMT_BGR565},
        {"bgr888", RK_FMT_BGR888},
        {"bgra8888", RK_FMT_BGRA8888},
        {"yuv420sp", RK_FMT_YUV420SP},
        {"yuv420sp_vu", RK_FMT_YUV420SP_VU},
        {"gray8", kGray8Format},
        {"gray", kGray8Format},
        {"rgb", RK_FMT_RGB888},
        {"bgr", RK_FMT_BGR888},
        {"yuv", RK_FMT_YUV420SP},
        {"yuv420", RK_FMT_YUV420SP},
        {"nv12", RK_FMT_YUV420SP},
    };
    return aliases;
}

}  // namespace

PIXEL_FORMAT_E parse_pixel_format(const std::string& format_string) {
    const auto it = pixel_format_aliases().find(normalize_key(format_string));
    if (it == pixel_format_aliases().end()) {
        throw std::invalid_argument("Unsupported pixel format: '" + format_string + "'.");
    }
    return it->second;
}

PIXEL_FORMAT_E parse_camera_pixel_format(const std::string& format_string) {
    const PIXEL_FORMAT_E format = parse_pixel_format(format_string);
    if (format == RK_FMT_RGB888 || format == RK_FMT_BGR888 || format == RK_FMT_YUV420SP ||
        format == kGray8Format) {
        return format;
    }
    throw std::invalid_argument(
        "Unsupported camera pixel format '" + format_string +
        "'. Supported formats: rgb, bgr, yuv, gray.");
}

const char* pixel_format_name(PIXEL_FORMAT_E format) {
    switch (format) {
        case RK_FMT_RGB888:
            return "RGB888";
        case RK_FMT_RGBA8888:
            return "RGBA8888";
        case RK_FMT_RGB565:
            return "RGB565";
        case RK_FMT_BGR565:
            return "BGR565";
        case RK_FMT_BGR888:
            return "BGR888";
        case RK_FMT_BGRA8888:
            return "BGRA8888";
        case RK_FMT_YUV420SP:
            return "YUV420SP";
        case RK_FMT_YUV420SP_VU:
            return "YUV420SP_VU";
        default:
            break;
    }
    if (format == kGray8Format) {
        return "GRAY8";
    }
    return "Unknown";
}

bool is_gray8_format(PIXEL_FORMAT_E format) {
    return format == kGray8Format;
}

bool is_yuv420sp_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU;
}

}  // namespace visiong

