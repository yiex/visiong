// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_VENC_UTILS_H_
#define VISIONG_MODULES_VENC_UTILS_H_

#include "common/internal/string_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace visiong {
namespace venc {

inline int clamp_quality(int quality) {
    return std::clamp(quality, 1, 100);
}

inline int clamp_non_negative_fps(int fps) {
    return std::max(0, fps);
}

inline int clamp_record_fps(int fps) {
    const int normalized = (fps <= 0) ? 30 : fps;
    return std::clamp(normalized, 1, 120);
}

inline std::string normalize_rc_mode(const std::string& rc_mode) {
    const std::string mode = to_lower_copy(rc_mode);
    if (mode == "cbr") {
        return "cbr";
    }
    if (mode == "vbr" || mode == "avbr") {
        return "vbr";
    }
    throw std::invalid_argument("rc_mode must be 'cbr' or 'vbr'.");
}

} // namespace venc
} // namespace visiong

#endif // VISIONG_MODULES_VENC_UTILS_H_

