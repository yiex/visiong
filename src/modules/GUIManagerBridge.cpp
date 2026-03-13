// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/GUIManager.h"
#include "modules/internal/gui_manager_impl.h"

nk_command_buffer* GUIManager::getCanvas() { return m_impl->getCanvas(); }

std::tuple<float, float, float, float> GUIManager::widget_bounds(nk_command_buffer* canvas) {
    return m_impl->widget_bounds(canvas);
}

bool GUIManager::input_is_mouse_down_in_rect(std::tuple<float, float, float, float> rect, bool left_mouse) {
    return m_impl->input_is_mouse_down_in_rect(rect, left_mouse);
}

void GUIManager::window_set_focus(const std::string& name) { m_impl->window_set_focus(name); }
void GUIManager::window_drag_from_pos(nk_command_buffer* canvas) { m_impl->window_drag_from_pos(canvas); }
void GUIManager::window_set_scroll(float scroll_y) { m_impl->window_set_scroll(scroll_y); }

std::tuple<bool, float, std::tuple<float, float, float, float>> GUIManager::input_is_mouse_dragging_in_rect() {
    return m_impl->input_is_mouse_dragging_in_rect();
}

bool GUIManager::is_title_bar_pressed() { return m_impl->is_title_bar_pressed(); }
float GUIManager::get_content_height() { return m_impl->get_content_height(); }
void GUIManager::push_style_vec2(const std::string& name, float x, float y) {
    m_impl->push_style_vec2(name, x, y);
}
void GUIManager::pop_style() { m_impl->pop_style(); }
float GUIManager::get_smart_scroll_dy() { return m_impl->get_smart_scroll_dy(); }

bool GUIManager::inputIsPointerDownInRect(std::tuple<float, float, float, float> rect, bool primary_pointer) {
    return input_is_mouse_down_in_rect(rect, primary_pointer);
}

std::tuple<bool, float, std::tuple<float, float, float, float>> GUIManager::inputIsPointerDraggingInRect() {
    return input_is_mouse_dragging_in_rect();
}

bool GUIManager::isTitleBarActive() { return is_title_bar_pressed(); }
float GUIManager::getScrollDeltaY() { return get_smart_scroll_dy(); }
