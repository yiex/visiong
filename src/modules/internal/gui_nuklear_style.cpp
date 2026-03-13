// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/gui_nuklear_style.h"
#include "common/internal/string_utils.h"

#include <algorithm>
#include <cmath>

namespace visiong::gui {
namespace {

float compute_touch_ui_scale(int width, int height) {
    const float short_side = static_cast<float>(std::min(width, height));
    return std::clamp(short_side / 240.0f, 1.0f, 1.4f);
}

void set_touch_button_style(nk_style_button* button,
                            float padding_x,
                            float padding_y,
                            float touch_padding_x,
                            float touch_padding_y,
                            float rounding) {
    if (button == nullptr) {
        return;
    }
    button->padding = nk_vec2(padding_x, padding_y);
    button->touch_padding = nk_vec2(touch_padding_x, touch_padding_y);
    button->image_padding = nk_vec2(std::max(4.0f, padding_x * 0.35f),
                                    std::max(4.0f, padding_y * 0.35f));
    button->rounding = rounding;
}

nk_color mix_color(const nk_color& a, const nk_color& b, float t) {
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    return nk_rgba(
        static_cast<nk_byte>(std::lround(a.r + (b.r - a.r) * clamped)),
        static_cast<nk_byte>(std::lround(a.g + (b.g - a.g) * clamped)),
        static_cast<nk_byte>(std::lround(a.b + (b.b - a.b) * clamped)),
        static_cast<nk_byte>(std::lround(a.a + (b.a - a.a) * clamped)));
}

void apply_touch_color_system(nk_context* ctx) {
    if (ctx == nullptr) {
        return;
    }

    const nk_color surface_0 = nk_rgb(28, 32, 36);
    const nk_color surface_1 = nk_rgb(40, 46, 52);
    const nk_color surface_2 = nk_rgb(53, 61, 69);
    const nk_color surface_3 = nk_rgb(71, 81, 92);
    const nk_color border = nk_rgb(95, 107, 119);
    const nk_color text = nk_rgb(236, 240, 244);
    const nk_color text_muted = nk_rgb(191, 199, 207);
    const nk_color accent = nk_rgb(92, 165, 242);
    const nk_color accent_strong = nk_rgb(66, 145, 232);
    const nk_color accent_soft = mix_color(surface_2, accent, 0.42f);
    const nk_color accent_soft_hover = mix_color(surface_3, accent, 0.50f);
    const nk_color accent_fill = mix_color(surface_1, accent, 0.68f);
    const nk_color selected = mix_color(surface_1, accent, 0.36f);
    const nk_color selected_hover = mix_color(surface_2, accent, 0.42f);
    const nk_color transparent = nk_rgba(0, 0, 0, 0);

    auto set_button_colors = [&](nk_style_button* button,
                                 nk_color normal_bg,
                                 nk_color hover_bg,
                                 nk_color active_bg,
                                 nk_color fg) {
        if (button == nullptr) {
            return;
        }
        button->normal = nk_style_item_color(normal_bg);
        button->hover = nk_style_item_color(hover_bg);
        button->active = nk_style_item_color(active_bg);
        button->border_color = border;
        button->text_background = transparent;
        button->text_normal = fg;
        button->text_hover = fg;
        button->text_active = fg;
        button->color_factor_background = 1.0f;
        button->color_factor_text = 1.0f;
        button->disabled_factor = 0.46f;
        button->border = 1.0f;
    };

    ctx->style.text.color = text;
    ctx->style.cursor_visible = 0;

    ctx->style.window.background = surface_0;
    ctx->style.window.fixed_background = nk_style_item_color(surface_0);
    ctx->style.window.border_color = border;
    ctx->style.window.popup_border_color = border;
    ctx->style.window.combo_border_color = border;
    ctx->style.window.contextual_border_color = border;
    ctx->style.window.menu_border_color = border;
    ctx->style.window.group_border_color = border;
    ctx->style.window.tooltip_border_color = border;
    ctx->style.window.scaler = nk_style_item_color(accent_soft);
    ctx->style.window.border = 1.0f;
    ctx->style.window.combo_border = 1.0f;
    ctx->style.window.contextual_border = 1.0f;
    ctx->style.window.menu_border = 1.0f;
    ctx->style.window.group_border = 1.0f;
    ctx->style.window.tooltip_border = 1.0f;
    ctx->style.window.popup_border = 1.0f;
    ctx->style.window.header.normal = nk_style_item_color(surface_1);
    ctx->style.window.header.hover = nk_style_item_color(surface_1);
    ctx->style.window.header.active = nk_style_item_color(surface_2);
    ctx->style.window.header.label_normal = text;
    ctx->style.window.header.label_hover = text;
    ctx->style.window.header.label_active = text;

    set_button_colors(&ctx->style.button, surface_1, surface_1, accent_soft, text);
    set_button_colors(&ctx->style.contextual_button, surface_1, surface_1, accent_soft, text);
    set_button_colors(&ctx->style.menu_button, surface_1, surface_1, accent_soft, text);
    set_button_colors(&ctx->style.combo.button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.property.inc_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.property.dec_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.window.header.close_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.window.header.minimize_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.tab.tab_maximize_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.tab.tab_minimize_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.tab.node_maximize_button, surface_2, surface_2, accent_soft_hover, text);
    set_button_colors(&ctx->style.tab.node_minimize_button, surface_2, surface_2, accent_soft_hover, text);

    ctx->style.checkbox.normal = nk_style_item_color(surface_1);
    ctx->style.checkbox.hover = nk_style_item_color(surface_1);
    ctx->style.checkbox.active = nk_style_item_color(selected);
    ctx->style.checkbox.cursor_normal = nk_style_item_color(accent_fill);
    ctx->style.checkbox.cursor_hover = nk_style_item_color(accent_strong);
    ctx->style.checkbox.border_color = border;
    ctx->style.checkbox.text_normal = text;
    ctx->style.checkbox.text_hover = text;
    ctx->style.checkbox.text_active = text;
    ctx->style.checkbox.text_background = transparent;
    ctx->style.checkbox.color_factor = 1.0f;
    ctx->style.checkbox.disabled_factor = 0.46f;
    ctx->style.checkbox.border = 1.0f;

    ctx->style.option.normal = nk_style_item_color(surface_1);
    ctx->style.option.hover = nk_style_item_color(surface_1);
    ctx->style.option.active = nk_style_item_color(selected);
    ctx->style.option.cursor_normal = nk_style_item_color(accent_fill);
    ctx->style.option.cursor_hover = nk_style_item_color(accent_strong);
    ctx->style.option.border_color = border;
    ctx->style.option.text_normal = text;
    ctx->style.option.text_hover = text;
    ctx->style.option.text_active = text;
    ctx->style.option.text_background = transparent;
    ctx->style.option.color_factor = 1.0f;
    ctx->style.option.disabled_factor = 0.46f;
    ctx->style.option.border = 1.0f;

    ctx->style.selectable.normal = nk_style_item_color(surface_1);
    ctx->style.selectable.hover = nk_style_item_color(surface_1);
    ctx->style.selectable.pressed = nk_style_item_color(accent_soft);
    ctx->style.selectable.normal_active = nk_style_item_color(selected);
    ctx->style.selectable.hover_active = nk_style_item_color(selected);
    ctx->style.selectable.pressed_active = nk_style_item_color(accent_soft_hover);
    ctx->style.selectable.text_normal = text;
    ctx->style.selectable.text_hover = text;
    ctx->style.selectable.text_pressed = text;
    ctx->style.selectable.text_normal_active = text;
    ctx->style.selectable.text_hover_active = text;
    ctx->style.selectable.text_pressed_active = text;
    ctx->style.selectable.text_background = transparent;
    ctx->style.selectable.color_factor = 1.0f;
    ctx->style.selectable.disabled_factor = 0.46f;

    ctx->style.slider.normal = nk_style_item_color(surface_1);
    ctx->style.slider.hover = nk_style_item_color(surface_1);
    ctx->style.slider.active = nk_style_item_color(surface_2);
    ctx->style.slider.border_color = border;
    ctx->style.slider.bar_normal = surface_2;
    ctx->style.slider.bar_hover = surface_2;
    ctx->style.slider.bar_active = surface_3;
    ctx->style.slider.bar_filled = accent_fill;
    ctx->style.slider.cursor_normal = nk_style_item_color(accent_soft);
    ctx->style.slider.cursor_hover = nk_style_item_color(accent_fill);
    ctx->style.slider.cursor_active = nk_style_item_color(accent_strong);
    ctx->style.slider.color_factor = 1.0f;
    ctx->style.slider.disabled_factor = 0.46f;
    ctx->style.slider.border = 1.0f;

    ctx->style.progress.normal = nk_style_item_color(surface_1);
    ctx->style.progress.hover = nk_style_item_color(surface_1);
    ctx->style.progress.active = nk_style_item_color(surface_2);
    ctx->style.progress.border_color = border;
    ctx->style.progress.cursor_normal = nk_style_item_color(accent_soft);
    ctx->style.progress.cursor_hover = nk_style_item_color(accent_fill);
    ctx->style.progress.cursor_active = nk_style_item_color(accent_strong);
    ctx->style.progress.cursor_border_color = accent_strong;
    ctx->style.progress.color_factor = 1.0f;
    ctx->style.progress.disabled_factor = 0.46f;
    ctx->style.progress.border = 1.0f;
    ctx->style.progress.cursor_border = 1.0f;

    ctx->style.scrollv.normal = nk_style_item_color(surface_1);
    ctx->style.scrollv.hover = nk_style_item_color(surface_1);
    ctx->style.scrollv.active = nk_style_item_color(surface_2);
    ctx->style.scrollv.border_color = border;
    ctx->style.scrollv.cursor_normal = nk_style_item_color(surface_3);
    ctx->style.scrollv.cursor_hover = nk_style_item_color(surface_3);
    ctx->style.scrollv.cursor_active = nk_style_item_color(accent_fill);
    ctx->style.scrollv.cursor_border_color = border;
    ctx->style.scrollv.color_factor = 1.0f;
    ctx->style.scrollv.disabled_factor = 0.46f;
    ctx->style.scrollv.border = 1.0f;
    ctx->style.scrollv.border_cursor = 1.0f;

    ctx->style.scrollh = ctx->style.scrollv;

    ctx->style.edit.normal = nk_style_item_color(surface_1);
    ctx->style.edit.hover = nk_style_item_color(surface_1);
    ctx->style.edit.active = nk_style_item_color(surface_2);
    ctx->style.edit.border_color = border;
    ctx->style.edit.cursor_normal = text;
    ctx->style.edit.cursor_hover = text;
    ctx->style.edit.cursor_text_normal = surface_0;
    ctx->style.edit.cursor_text_hover = surface_0;
    ctx->style.edit.text_normal = text;
    ctx->style.edit.text_hover = text;
    ctx->style.edit.text_active = text;
    ctx->style.edit.selected_normal = accent_soft;
    ctx->style.edit.selected_hover = accent_fill;
    ctx->style.edit.selected_text_normal = text;
    ctx->style.edit.selected_text_hover = text;
    ctx->style.edit.color_factor = 1.0f;
    ctx->style.edit.disabled_factor = 0.46f;
    ctx->style.edit.border = 1.0f;
    ctx->style.edit.scrollbar = ctx->style.scrollv;
    ctx->style.property.edit = ctx->style.edit;

    ctx->style.property.normal = nk_style_item_color(surface_1);
    ctx->style.property.hover = nk_style_item_color(surface_1);
    ctx->style.property.active = nk_style_item_color(surface_2);
    ctx->style.property.border_color = border;
    ctx->style.property.label_normal = text_muted;
    ctx->style.property.label_hover = text;
    ctx->style.property.label_active = text;
    ctx->style.property.color_factor = 1.0f;
    ctx->style.property.disabled_factor = 0.46f;
    ctx->style.property.border = 1.0f;

    ctx->style.combo.normal = nk_style_item_color(surface_1);
    ctx->style.combo.hover = nk_style_item_color(surface_1);
    ctx->style.combo.active = nk_style_item_color(surface_2);
    ctx->style.combo.border_color = border;
    ctx->style.combo.label_normal = text;
    ctx->style.combo.label_hover = text;
    ctx->style.combo.label_active = text;
    ctx->style.combo.symbol_normal = text_muted;
    ctx->style.combo.symbol_hover = text_muted;
    ctx->style.combo.symbol_active = accent_fill;
    ctx->style.combo.color_factor = 1.0f;
    ctx->style.combo.disabled_factor = 0.46f;
    ctx->style.combo.border = 1.0f;

    ctx->style.chart.background = nk_style_item_color(surface_1);
    ctx->style.chart.border_color = border;
    ctx->style.chart.color = accent_soft;
    ctx->style.chart.selected_color = accent_strong;
    ctx->style.chart.color_factor = 1.0f;
    ctx->style.chart.disabled_factor = 0.46f;
    ctx->style.chart.border = 1.0f;

    ctx->style.knob.normal = nk_style_item_color(surface_1);
    ctx->style.knob.hover = nk_style_item_color(surface_1);
    ctx->style.knob.active = nk_style_item_color(surface_2);
    ctx->style.knob.border_color = border;
    ctx->style.knob.knob_normal = surface_2;
    ctx->style.knob.knob_hover = surface_2;
    ctx->style.knob.knob_active = selected_hover;
    ctx->style.knob.knob_border_color = border;
    ctx->style.knob.cursor_normal = accent_soft;
    ctx->style.knob.cursor_hover = accent_soft;
    ctx->style.knob.cursor_active = accent_strong;
    ctx->style.knob.color_factor = 1.0f;
    ctx->style.knob.disabled_factor = 0.46f;
    ctx->style.knob.border = 1.0f;
    ctx->style.knob.knob_border = 1.0f;

    ctx->style.tab.background = nk_style_item_color(surface_1);
    ctx->style.tab.border_color = border;
    ctx->style.tab.text = text;
    ctx->style.tab.color_factor = 1.0f;
    ctx->style.tab.disabled_factor = 0.46f;
    ctx->style.tab.border = 1.0f;
}

}  // namespace

nk_flags parse_window_flags(const std::string& flags_str) {
    nk_flags flags = 0;
    const std::string lower = visiong::to_lower_copy(flags_str);
    if (lower.find("border") != std::string::npos) flags |= NK_WINDOW_BORDER;
    if (lower.find("movable") != std::string::npos) flags |= NK_WINDOW_MOVABLE;
    if (lower.find("scalable") != std::string::npos) flags |= NK_WINDOW_SCALABLE;
    if (lower.find("closable") != std::string::npos) flags |= NK_WINDOW_CLOSABLE;
    if (lower.find("minimizable") != std::string::npos) flags |= NK_WINDOW_MINIMIZABLE;
    if (lower.find("title") != std::string::npos) flags |= NK_WINDOW_TITLE;
    if (lower.find("no_background") != std::string::npos) flags |= NK_WINDOW_BACKGROUND;
    return flags;
}

nk_text_alignment parse_text_align(const std::string& align_str) {
    const std::string mode = visiong::to_lower_copy(align_str);
    if (mode == "left") return NK_TEXT_LEFT;
    if (mode == "center" || mode == "centered") return NK_TEXT_CENTERED;
    if (mode == "right") return NK_TEXT_RIGHT;
    return NK_TEXT_LEFT;
}

nk_color tuple_to_nuklear_color(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> value) {
    return nk_rgba(std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value));
}

TouchUiProfile make_touch_ui_profile(int width, int height) {
    const int long_side = std::max(width, height);
    const int short_side = std::min(width, height);
    const float ui_scale = compute_touch_ui_scale(width, height);
    const TouchUiProfile generic{
        ui_scale, 20.0f * ui_scale, 46.0f * ui_scale, 12.0f * ui_scale, 10.0f * ui_scale, 6.0f * ui_scale,
        8.0f * ui_scale, 0.0f, 4.0f * ui_scale, 12.0f * ui_scale, 18.0f * ui_scale, 28.0f * ui_scale,
        0.45f, 10.0f * ui_scale, 240.0f * ui_scale, 3200.0f * ui_scale, 3600.0f * ui_scale, 24.0f * ui_scale};
    const TouchUiProfile compact_anchor{0.92f, 17.0f, 38.0f, 4.0f, 6.0f, 4.0f, 6.0f, 0.0f, 2.0f,
                                        9.0f, 14.0f, 20.0f, 0.38f, 7.0f, 180.0f, 2600.0f, 3000.0f, 18.0f};

    const float compact_short_bias =
        1.0f - std::clamp((static_cast<float>(short_side) - 240.0f) / 80.0f, 0.0f, 1.0f);
    const float compact_long_bias =
        1.0f - std::clamp((static_cast<float>(long_side) - 320.0f) / 120.0f, 0.0f, 1.0f);
    const float compact_bias = std::clamp(std::min(compact_short_bias, compact_long_bias), 0.0f, 1.0f);

    auto mix_scalar = [compact_bias](float generic_value, float compact_value) {
        return generic_value + (compact_value - generic_value) * compact_bias;
    };

    return TouchUiProfile{
        mix_scalar(generic.ui_scale, compact_anchor.ui_scale),
        mix_scalar(generic.font_size, compact_anchor.font_size),
        mix_scalar(generic.min_row_height, compact_anchor.min_row_height),
        mix_scalar(generic.window_padding, compact_anchor.window_padding),
        mix_scalar(generic.content_padding, compact_anchor.content_padding),
        mix_scalar(generic.small_padding, compact_anchor.small_padding),
        mix_scalar(generic.touch_padding, compact_anchor.touch_padding),
        mix_scalar(generic.stacked_touch_padding_y, compact_anchor.stacked_touch_padding_y),
        mix_scalar(generic.chrome_touch_padding_y, compact_anchor.chrome_touch_padding_y),
        mix_scalar(generic.rounding, compact_anchor.rounding),
        mix_scalar(generic.scrollbar_size, compact_anchor.scrollbar_size),
        mix_scalar(generic.slider_cursor, compact_anchor.slider_cursor),
        mix_scalar(generic.long_press_seconds, compact_anchor.long_press_seconds),
        mix_scalar(generic.touch_slop, compact_anchor.touch_slop),
        mix_scalar(generic.min_fling_velocity, compact_anchor.min_fling_velocity),
        mix_scalar(generic.max_fling_velocity, compact_anchor.max_fling_velocity),
        mix_scalar(generic.momentum_deceleration, compact_anchor.momentum_deceleration),
        mix_scalar(generic.scroll_pixels_per_unit, compact_anchor.scroll_pixels_per_unit)};
}

void apply_touch_friendly_style(nk_context* ctx, const TouchUiProfile& profile) {
    if (ctx == nullptr) {
        return;
    }

    const float ui_scale = profile.ui_scale;
    const float window_padding = profile.window_padding;
    const float content_padding = profile.content_padding;
    const float small_padding = profile.small_padding;
    const float touch_padding = profile.touch_padding;
    const float stacked_touch_padding_y = profile.stacked_touch_padding_y;
    const float chrome_touch_padding_y = profile.chrome_touch_padding_y;
    const float rounding = profile.rounding;
    const float scrollbar_size = profile.scrollbar_size;
    const float slider_cursor = profile.slider_cursor;
    const float primary_button_pad_x = std::max(8.0f, content_padding + 4.0f);
    const float primary_button_pad_y = std::max(4.0f, content_padding - 1.0f);
    const float compact_button_pad_x = std::max(4.0f, small_padding + 2.0f);
    const float compact_button_pad_y = std::max(3.0f, small_padding);

    ctx->style.window.rounding = rounding;
    ctx->style.window.spacing = nk_vec2(8.0f * ui_scale, 8.0f * ui_scale);
    ctx->style.window.padding = nk_vec2(window_padding, window_padding);
    ctx->style.window.group_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.popup_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.combo_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.contextual_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.menu_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.tooltip_padding = nk_vec2(content_padding, content_padding);
    ctx->style.window.scrollbar_size = nk_vec2(scrollbar_size, scrollbar_size);
    ctx->style.window.min_row_height_padding = 8.0f * ui_scale;
    ctx->style.window.header.padding = nk_vec2(window_padding, window_padding);
    ctx->style.window.header.label_padding = nk_vec2(small_padding, 0.0f);
    ctx->style.window.header.spacing = nk_vec2(small_padding, 0.0f);

    set_touch_button_style(&ctx->style.button, primary_button_pad_x, primary_button_pad_y,
                           touch_padding, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.contextual_button, primary_button_pad_x, primary_button_pad_y,
                           touch_padding, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.menu_button, primary_button_pad_x, primary_button_pad_y,
                           touch_padding, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.combo.button, compact_button_pad_x, compact_button_pad_y,
                           touch_padding * 0.75f, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.property.inc_button, compact_button_pad_x, compact_button_pad_y,
                           touch_padding * 0.75f, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.property.dec_button, compact_button_pad_x, compact_button_pad_y,
                           touch_padding * 0.75f, stacked_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.window.header.close_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.window.header.minimize_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.tab.tab_maximize_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.tab.tab_minimize_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.tab.node_maximize_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);
    set_touch_button_style(&ctx->style.tab.node_minimize_button, small_padding, small_padding,
                           touch_padding, chrome_touch_padding_y, rounding);

    ctx->style.checkbox.padding = nk_vec2(small_padding, small_padding);
    ctx->style.checkbox.touch_padding = nk_vec2(touch_padding, stacked_touch_padding_y);
    ctx->style.checkbox.spacing = 12.0f * ui_scale;

    ctx->style.option.padding = nk_vec2(small_padding, small_padding);
    ctx->style.option.touch_padding = nk_vec2(touch_padding, stacked_touch_padding_y);
    ctx->style.option.spacing = 12.0f * ui_scale;

    ctx->style.selectable.rounding = rounding;
    ctx->style.selectable.padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.selectable.touch_padding = nk_vec2(touch_padding, stacked_touch_padding_y);

    ctx->style.slider.rounding = rounding;
    ctx->style.slider.padding = nk_vec2(small_padding, 0.0f);
    ctx->style.slider.spacing = nk_vec2(10.0f * ui_scale, 0.0f);
    ctx->style.slider.bar_height = 12.0f * ui_scale;
    ctx->style.slider.cursor_size = nk_vec2(slider_cursor, slider_cursor);

    ctx->style.progress.rounding = rounding;
    ctx->style.progress.cursor_rounding = rounding;
    ctx->style.progress.padding = nk_vec2(4.0f * ui_scale, 4.0f * ui_scale);

    ctx->style.property.rounding = rounding;
    ctx->style.property.padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.property.edit.rounding = rounding;
    ctx->style.property.edit.padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.property.edit.row_padding = 8.0f * ui_scale;
    ctx->style.property.edit.scrollbar_size = nk_vec2(scrollbar_size, scrollbar_size);

    ctx->style.edit.rounding = rounding;
    ctx->style.edit.padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.edit.row_padding = 8.0f * ui_scale;
    ctx->style.edit.scrollbar_size = nk_vec2(scrollbar_size, scrollbar_size);
    ctx->style.edit.scrollbar.padding = nk_vec2(2.0f * ui_scale, 2.0f * ui_scale);
    ctx->style.edit.scrollbar.rounding = rounding;
    ctx->style.edit.scrollbar.rounding_cursor = rounding;
    ctx->style.edit.scrollbar.show_buttons = 0;

    ctx->style.scrollv.padding = nk_vec2(2.0f * ui_scale, 2.0f * ui_scale);
    ctx->style.scrollv.rounding = rounding;
    ctx->style.scrollv.rounding_cursor = rounding;
    ctx->style.scrollv.show_buttons = 0;

    ctx->style.scrollh.padding = nk_vec2(2.0f * ui_scale, 2.0f * ui_scale);
    ctx->style.scrollh.rounding = rounding;
    ctx->style.scrollh.rounding_cursor = rounding;
    ctx->style.scrollh.show_buttons = 0;

    ctx->style.combo.rounding = rounding;
    ctx->style.combo.content_padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.combo.button_padding = nk_vec2(small_padding, small_padding);
    ctx->style.combo.spacing = nk_vec2(6.0f * ui_scale, 4.0f * ui_scale);

    ctx->style.tab.rounding = rounding;
    ctx->style.tab.padding = nk_vec2(content_padding, 8.0f * ui_scale);
    ctx->style.tab.spacing = nk_vec2(6.0f * ui_scale, 4.0f * ui_scale);

    apply_touch_color_system(ctx);
}

float pixels_to_nuklear_scroll_units(float pixel_delta, float pixels_per_unit) {
    return pixel_delta / std::max(pixels_per_unit, 1.0f);
}

nk_color* get_style_color_ptr(nk_context* ctx, const std::string& name) {
    if (ctx == nullptr) {
        return nullptr;
    }
    if (name == "text") return &ctx->style.text.color;
    if (name == "window_bg") return &ctx->style.window.fixed_background.data.color;
    if (name == "header_bg") return &ctx->style.window.header.normal.data.color;
    if (name == "button_normal") return &ctx->style.button.normal.data.color;
    if (name == "button_hover") return &ctx->style.button.hover.data.color;
    if (name == "button_active") return &ctx->style.button.active.data.color;
    if (name == "slider_fill") return &ctx->style.slider.bar_filled;
    if (name == "progress_fill") return &ctx->style.progress.cursor_normal.data.color;
    if (name == "select_active") return &ctx->style.selectable.normal_active.data.color;
    return nullptr;
}

}  // namespace visiong::gui
