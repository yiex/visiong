// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_COMMON_PIXEL_FORMAT_H
#define VISIONG_COMMON_PIXEL_FORMAT_H

#include <string>

#include "rk_comm_video.h"

namespace visiong {

constexpr PIXEL_FORMAT_E kGray8Format = static_cast<PIXEL_FORMAT_E>(0x1000);

PIXEL_FORMAT_E parse_pixel_format(const std::string& format_string);
PIXEL_FORMAT_E parse_camera_pixel_format(const std::string& format_string);
const char* pixel_format_name(PIXEL_FORMAT_E format);
bool is_gray8_format(PIXEL_FORMAT_E format);
bool is_yuv420sp_format(PIXEL_FORMAT_E format);

} // namespace visiong

#endif  // VISIONG_COMMON_PIXEL_FORMAT_H

