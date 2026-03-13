// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/core/ImageBuffer.h"
#include "modules/internal/gui_command_renderer.h"
#include "modules/internal/gui_manager_impl.h"
#include "modules/internal/gui_nuklear_style.h"

namespace gui_style = visiong::gui;
namespace gui_render = visiong::gui::render;

namespace {

nk_layout_format layout_format_from_string(const std::string& format) {
    return format == "static" ? NK_STATIC : NK_DYNAMIC;
}

nk_chart_type chart_type_from_string(const std::string& type) {
    return type == "columns" ? NK_CHART_COLUMN : NK_CHART_LINES;
}

}  // namespace

void GUIManager::Impl::layoutRowDynamic(float height, int cols) {
    nk_layout_row_dynamic(m_ctx, clampRowHeight(height), cols);
}

void GUIManager::Impl::layoutRowStatic(float height, int item_width, int cols) {
    nk_layout_row_static(m_ctx, clampRowHeight(height), item_width, cols);
}

void GUIManager::Impl::layoutRowBegin(const std::string& format, float row_height, int cols) {
    nk_layout_row_begin(m_ctx, layout_format_from_string(format), clampRowHeight(row_height), cols);
}

void GUIManager::Impl::layoutRowPush(float value) {
    nk_layout_row_push(m_ctx, value);
}

void GUIManager::Impl::layoutRowEnd() {
    nk_layout_row_end(m_ctx);
}

bool GUIManager::Impl::groupBegin(const std::string& title, const std::string& flags_str) {
    return nk_group_begin(m_ctx, title.c_str(), gui_style::parse_window_flags(flags_str));
}

void GUIManager::Impl::groupEnd() {
    nk_group_end(m_ctx);
}

void GUIManager::Impl::label(const std::string& text, const std::string& align) {
    nk_label(m_ctx, text.c_str(), gui_style::parse_text_align(align));
}

void GUIManager::Impl::labelWrap(const std::string& text) {
    nk_label_wrap(m_ctx, text.c_str());
}

bool GUIManager::Impl::button(const std::string& label) {
    return nk_button_label(m_ctx, label.c_str());
}

float GUIManager::Impl::slider(const std::string& label, float value, float min, float max, float step) {
    float current = value;
    nk_layout_row_dynamic(m_ctx, m_min_row_height, 2);
    nk_label(m_ctx, label.c_str(), NK_TEXT_LEFT);
    nk_slider_float(m_ctx, min, &current, max, step);
    return current;
}

bool GUIManager::Impl::checkbox(const std::string& label, bool active) {
    int state = active;
    nk_checkbox_label(m_ctx, label.c_str(), &state);
    return state != 0;
}

bool GUIManager::Impl::option(const std::string& label, bool active) {
    return nk_option_label(m_ctx, label.c_str(), active);
}

std::tuple<bool, std::string> GUIManager::Impl::editString(const std::string& text, int max_len) {
    m_edit_buffer.assign(text.begin(), text.end());
    m_edit_buffer.resize(max_len + 1, '\0');

    const int buffer_size = static_cast<int>(m_edit_buffer.size());
    const nk_flags event = nk_edit_string_zero_terminated(
        m_ctx,
        NK_EDIT_SIMPLE,
        m_edit_buffer.data(),
        buffer_size,
        nk_filter_default);
    const bool changed = (event & NK_EDIT_COMMITED) || (event & NK_EDIT_DEACTIVATED);
    return {changed, std::string(m_edit_buffer.data())};
}

int GUIManager::Impl::progress(int current, int max, bool is_modifyable) {
    size_t value = static_cast<size_t>(current);
    nk_progress(m_ctx, &value, static_cast<nk_size>(max), is_modifyable);
    return static_cast<int>(value);
}

bool GUIManager::Impl::buttonImage(const ImageBuffer& img) {
    if (!img.is_valid()) {
        return false;
    }
    return nk_button_image(m_ctx, gui_render::image_from_buffer(&img));
}

bool GUIManager::Impl::treeNode(const std::string& title, bool is_expanded) {
    nk_collapse_states state = is_expanded ? NK_MAXIMIZED : NK_MINIMIZED;
    return nk_tree_state_push(m_ctx, NK_TREE_TAB, title.c_str(), &state);
}

void GUIManager::Impl::treePop() {
    nk_tree_pop(m_ctx);
}

int GUIManager::Impl::propertyInt(const std::string& name,
                                  int val,
                                  int min,
                                  int max,
                                  int step,
                                  float inc_per_pixel) {
    int value = val;
    nk_property_int(m_ctx, name.c_str(), min, &value, max, step, inc_per_pixel);
    return value;
}

float GUIManager::Impl::propertyFloat(const std::string& name,
                                      float val,
                                      float min,
                                      float max,
                                      float step,
                                      float inc_per_pixel) {
    float value = val;
    nk_property_float(m_ctx, name.c_str(), min, &value, max, step, inc_per_pixel);
    return value;
}

bool GUIManager::Impl::comboBegin(const std::string& text, float w, float h) {
    return nk_combo_begin_label(m_ctx, text.c_str(), nk_vec2(w, h));
}

bool GUIManager::Impl::comboItem(const std::string& text) {
    return nk_combo_item_label(m_ctx, text.c_str(), NK_TEXT_LEFT);
}

void GUIManager::Impl::comboEnd() {
    nk_combo_end(m_ctx);
}

bool GUIManager::Impl::contextualBegin(float w, float h) {
    return nk_contextual_begin(m_ctx, 0, nk_vec2(w, h), nk_window_get_bounds(m_ctx));
}

bool GUIManager::Impl::contextualItem(const std::string& text) {
    return nk_contextual_item_label(m_ctx, text.c_str(), NK_TEXT_LEFT);
}

void GUIManager::Impl::contextualEnd() {
    nk_contextual_end(m_ctx);
}

void GUIManager::Impl::strokeLine(nk_command_buffer* canvas,
                                  float x0,
                                  float y0,
                                  float x1,
                                  float y1,
                                  float thickness,
                                  std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    nk_stroke_line(canvas, x0, y0, x1, y1, thickness, gui_style::tuple_to_nuklear_color(color));
}

void GUIManager::Impl::strokeRect(nk_command_buffer* canvas,
                                  float x,
                                  float y,
                                  float w,
                                  float h,
                                  float rounding,
                                  float thickness,
                                  std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    nk_stroke_rect(
        canvas,
        nk_rect(x, y, w, h),
        rounding,
        thickness,
        gui_style::tuple_to_nuklear_color(color));
}

void GUIManager::Impl::fillRect(nk_command_buffer* canvas,
                                float x,
                                float y,
                                float w,
                                float h,
                                float rounding,
                                std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    nk_fill_rect(canvas, nk_rect(x, y, w, h), rounding, gui_style::tuple_to_nuklear_color(color));
}

void GUIManager::Impl::drawText(nk_command_buffer* canvas,
                                float x,
                                float y,
                                const std::string& text,
                                std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    nk_draw_text(
        canvas,
        nk_rect(x, y, m_width, m_height),
        text.c_str(),
        text.length(),
        m_ctx->style.font,
        m_ctx->style.window.background,
        gui_style::tuple_to_nuklear_color(color));
}

void GUIManager::Impl::setStyleColor(const std::string& property_name,
                                     std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    nk_color* style_color = gui_style::get_style_color_ptr(m_ctx, property_name);
    if (style_color != nullptr) {
        *style_color = gui_style::tuple_to_nuklear_color(color);
    }
}

void GUIManager::Impl::setStyleButtonRounding(float rounding) {
    m_ctx->style.button.rounding = rounding;
}

void GUIManager::Impl::setStyleWindowRounding(float rounding) {
    m_ctx->style.window.rounding = rounding;
}

void GUIManager::Impl::setWindowBackgroundColor(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_ctx->style.window.fixed_background = nk_style_item_color(gui_style::tuple_to_nuklear_color(color));
}

bool GUIManager::Impl::chartBegin(const std::string& type, int count, float min_val, float max_val) {
    return nk_chart_begin(m_ctx, chart_type_from_string(type), count, min_val, max_val);
}

void GUIManager::Impl::chartPush(float value) {
    nk_chart_push(m_ctx, value);
}

void GUIManager::Impl::chartEnd() {
    nk_chart_end(m_ctx);
}

void GUIManager::Impl::menubarBegin() {
    nk_menubar_begin(m_ctx);
}

void GUIManager::Impl::menubarEnd() {
    nk_menubar_end(m_ctx);
}

bool GUIManager::Impl::menuBegin(const std::string& label, float width, float height) {
    return nk_menu_begin_label(m_ctx, label.c_str(), NK_TEXT_LEFT, nk_vec2(width, height));
}

bool GUIManager::Impl::menuItem(const std::string& label) {
    return nk_menu_item_label(m_ctx, label.c_str(), NK_TEXT_LEFT);
}

void GUIManager::Impl::menuEnd() {
    nk_menu_end(m_ctx);
}

void GUIManager::Impl::tooltip(const std::string& text) {
    if (m_has_touch_input) {
        if (m_touch_long_press_fired && nk_widget_is_hovered(m_ctx)) {
            nk_tooltip(m_ctx, text.c_str());
        }
        return;
    }

    if (nk_widget_is_hovered(m_ctx)) {
        nk_tooltip(m_ctx, text.c_str());
    }
}

nk_command_buffer* GUIManager::Impl::getCanvas() {
    return nk_window_get_canvas(m_ctx);
}

std::tuple<float, float, float, float> GUIManager::Impl::widget_bounds(nk_command_buffer* canvas) {
    (void)canvas;
    struct nk_rect bounds;
    nk_widget(&bounds, m_ctx);
    return {bounds.x, bounds.y, bounds.w, bounds.h};
}

void GUIManager::Impl::window_set_focus(const std::string& name) {
    nk_window_set_focus(m_ctx, name.c_str());
}

void GUIManager::Impl::window_set_scroll(float scroll_y) {
    if (m_ctx != nullptr && m_ctx->current != nullptr) {
        m_ctx->current->scrollbar.y = scroll_y;
    }
}

float GUIManager::Impl::get_content_height() {
    if (m_ctx == nullptr || m_ctx->current == nullptr || m_ctx->current->layout == nullptr) {
        return 0.0f;
    }
    return m_ctx->current->layout->at_y - m_ctx->current->layout->bounds.y;
}

void GUIManager::Impl::push_style_vec2(const std::string& name, float x, float y) {
    if (name == "padding") {
        nk_style_push_vec2(m_ctx, &m_ctx->style.window.padding, nk_vec2(x, y));
    } else if (name == "spacing") {
        nk_style_push_vec2(m_ctx, &m_ctx->style.window.spacing, nk_vec2(x, y));
    }
}

void GUIManager::Impl::pop_style() {
    nk_style_pop_vec2(m_ctx);
}
