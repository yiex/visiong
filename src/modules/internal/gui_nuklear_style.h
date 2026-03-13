// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_STYLE_H
#define VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_STYLE_H

#include "modules/internal/gui_nuklear_config.h"

#include <cstdint>
#include <string>
#include <tuple>

namespace visiong::gui {

struct TouchUiProfile {
    float ui_scale;
    float font_size;
    float min_row_height;
    float window_padding;
    float content_padding;
    float small_padding;
    float touch_padding;
    float stacked_touch_padding_y;
    float chrome_touch_padding_y;
    float rounding;
    float scrollbar_size;
    float slider_cursor;
    float long_press_seconds;
    float touch_slop;
    float min_fling_velocity;
    float max_fling_velocity;
    float momentum_deceleration;
    float scroll_pixels_per_unit;
};

nk_flags parse_window_flags(const std::string& flags_str);
nk_text_alignment parse_text_align(const std::string& align_str);
nk_color tuple_to_nuklear_color(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> value);
TouchUiProfile make_touch_ui_profile(int width, int height);
void apply_touch_friendly_style(nk_context* ctx, const TouchUiProfile& profile);
float pixels_to_nuklear_scroll_units(float pixel_delta, float pixels_per_unit);
nk_color* get_style_color_ptr(nk_context* ctx, const std::string& name);

}  // namespace visiong::gui

#endif  // VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_STYLE_H
