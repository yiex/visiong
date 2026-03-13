// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/GUIManager.h"

void GUIManager::begin_frame(TouchDevice* touch) { beginFrame(touch); }

bool GUIManager::begin_window(const std::string& title,
                              float x,
                              float y,
                              float w,
                              float h,
                              const std::string& flags_str) {
    return beginWindow(title, x, y, w, h, flags_str);
}

void GUIManager::end_window() { endWindow(); }
void GUIManager::end_frame(ImageBuffer& target) { endFrame(target); }
void GUIManager::layout_row_dynamic(float height, int cols) { layoutRowDynamic(height, cols); }
void GUIManager::layout_row_static(float height, int item_width, int cols) {
    layoutRowStatic(height, item_width, cols);
}
void GUIManager::layout_row_begin(const std::string& format, float row_height, int cols) {
    layoutRowBegin(format, row_height, cols);
}
void GUIManager::layout_row_push(float value) { layoutRowPush(value); }
void GUIManager::layout_row_end() { layoutRowEnd(); }
bool GUIManager::group_begin(const std::string& title, const std::string& flags_str) {
    return groupBegin(title, flags_str);
}
void GUIManager::group_end() { groupEnd(); }
void GUIManager::label_wrap(const std::string& text) { labelWrap(text); }
std::tuple<bool, std::string> GUIManager::edit_string(const std::string& text, int max_len) {
    return editString(text, max_len);
}
bool GUIManager::button_image(const ImageBuffer& img) { return buttonImage(img); }
bool GUIManager::tree_node(const std::string& title, bool is_expanded) { return treeNode(title, is_expanded); }
void GUIManager::tree_pop() { treePop(); }
int GUIManager::property_int(const std::string& name, int val, int min, int max, int step, float inc_per_pixel) {
    return propertyInt(name, val, min, max, step, inc_per_pixel);
}
float GUIManager::property_float(const std::string& name,
                                 float val,
                                 float min,
                                 float max,
                                 float step,
                                 float inc_per_pixel) {
    return propertyFloat(name, val, min, max, step, inc_per_pixel);
}

bool GUIManager::combo_begin(const std::string& text, float w, float h) { return comboBegin(text, w, h); }
bool GUIManager::combo_item(const std::string& text) { return comboItem(text); }
void GUIManager::combo_end() { comboEnd(); }
bool GUIManager::contextual_begin(float w, float h) { return contextualBegin(w, h); }
bool GUIManager::contextual_item(const std::string& text) { return contextualItem(text); }
void GUIManager::contextual_end() { contextualEnd(); }

bool GUIManager::chart_begin(const std::string& type, int count, float min_val, float max_val) {
    return chartBegin(type, count, min_val, max_val);
}
void GUIManager::chart_push(float value) { chartPush(value); }
void GUIManager::chart_end() { chartEnd(); }
void GUIManager::menubar_begin() { menubarBegin(); }
void GUIManager::menubar_end() { menubarEnd(); }
bool GUIManager::menu_begin(const std::string& label, float width, float height) {
    return menuBegin(label, width, height);
}
bool GUIManager::menu_item(const std::string& label) { return menuItem(label); }
void GUIManager::menu_end() { menuEnd(); }

bool GUIManager::input_is_pointer_down_in_rect(std::tuple<float, float, float, float> rect, bool primary_pointer) {
    return inputIsPointerDownInRect(rect, primary_pointer);
}

std::tuple<bool, float, std::tuple<float, float, float, float>> GUIManager::input_is_pointer_dragging_in_rect() {
    return inputIsPointerDraggingInRect();
}

bool GUIManager::is_title_bar_active() { return isTitleBarActive(); }
float GUIManager::get_scroll_delta_y() { return getScrollDeltaY(); }
nk_command_buffer* GUIManager::get_canvas() { return getCanvas(); }

void GUIManager::stroke_line(nk_command_buffer* canvas,
                             float x0,
                             float y0,
                             float x1,
                             float y1,
                             float thickness,
                             std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    strokeLine(canvas, x0, y0, x1, y1, thickness, color);
}

void GUIManager::stroke_rect(nk_command_buffer* canvas,
                             float x,
                             float y,
                             float w,
                             float h,
                             float rounding,
                             float thickness,
                             std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    strokeRect(canvas, x, y, w, h, rounding, thickness, color);
}

void GUIManager::fill_rect(nk_command_buffer* canvas,
                           float x,
                           float y,
                           float w,
                           float h,
                           float rounding,
                           std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    fillRect(canvas, x, y, w, h, rounding, color);
}

void GUIManager::draw_text(nk_command_buffer* canvas,
                           float x,
                           float y,
                           const std::string& text,
                           std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    drawText(canvas, x, y, text, color);
}

void GUIManager::set_style_color(const std::string& property_name,
                                 std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    setStyleColor(property_name, color);
}
void GUIManager::set_style_button_rounding(float rounding) { setStyleButtonRounding(rounding); }
void GUIManager::set_style_window_rounding(float rounding) { setStyleWindowRounding(rounding); }
void GUIManager::set_window_background_color(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    setWindowBackgroundColor(color);
}
