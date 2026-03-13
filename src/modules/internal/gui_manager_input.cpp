// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/gui_manager_impl.h"

#include "modules/internal/gui_nuklear_style.h"
#include "visiong/modules/Touch.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gui_style = visiong::gui;

void GUIManager::Impl::beginFrame(TouchDevice* touch) {
    constexpr float kVerticalScrollBias = 1.35f;

    updateFrameTiming();
    m_has_touch_input = (touch != nullptr);
    nk_input_begin(m_ctx);
    m_touch_scroll_delta_y = 0.0f;

    std::vector<TouchPoint> points;
    if (touch) {
        points = touch->get_touch_points();
    }

    if (!points.empty()) {
        m_momentum_active = false;
        m_momentum_velocity_y = 0.0f;

        const float x = static_cast<float>(points[0].x);
        const float y = static_cast<float>(points[0].y);
        const float frame_dy = y - m_touch_last_y;

        nk_input_motion(m_ctx, points[0].x, points[0].y);
        if (!m_touch_is_down) {
            m_touch_is_down = true;
            m_touch_button_down = (points.size() == 1);
            m_touch_gesture = (points.size() == 1) ? TouchGestureMode::TapCandidate
                                                    : TouchGestureMode::VerticalScroll;
            m_touch_start_x = x;
            m_touch_start_y = y;
            m_touch_last_x = x;
            m_touch_last_y = y;
            m_touch_press_duration = 0.0f;
            m_touch_long_press_fired = false;
            m_touch_velocity_y = 0.0f;
            if (m_touch_button_down) {
                nk_input_button(m_ctx, NK_BUTTON_LEFT, points[0].x, points[0].y, 1);
            }
        } else {
            const float total_dx = x - m_touch_start_x;
            const float total_dy = y - m_touch_start_y;
            const float touch_slop_sq = m_touch_slop * m_touch_slop;

            if (points.size() > 1) {
                m_touch_gesture = TouchGestureMode::VerticalScroll;
                releaseTouchButton(true);
            }

            if (m_touch_gesture == TouchGestureMode::TapCandidate &&
                (total_dx * total_dx + total_dy * total_dy) > touch_slop_sq) {
                if (std::abs(total_dy) > std::abs(total_dx) * kVerticalScrollBias) {
                    m_touch_gesture = TouchGestureMode::VerticalScroll;
                    releaseTouchButton(true);
                } else {
                    m_touch_gesture = TouchGestureMode::HorizontalDrag;
                }
            }

            if (m_touch_gesture == TouchGestureMode::TapCandidate) {
                m_touch_press_duration += m_frame_dt_seconds;
                if (!m_touch_long_press_fired && m_touch_press_duration >= m_long_press_seconds) {
                    m_touch_long_press_fired = true;
                    m_touch_gesture = TouchGestureMode::LongPress;
                    releaseTouchButton(true);
                    nk_input_button(m_ctx, NK_BUTTON_RIGHT, points[0].x, points[0].y, 1);
                    nk_input_button(m_ctx, NK_BUTTON_RIGHT, points[0].x, points[0].y, 0);
                }
            }

            if (m_touch_gesture == TouchGestureMode::VerticalScroll) {
                m_touch_scroll_delta_y = frame_dy;
                if (std::abs(frame_dy) > 0.5f) {
                    const float instant_velocity = frame_dy / std::max(m_frame_dt_seconds, 1.0e-3f);
                    m_touch_velocity_y = m_touch_velocity_y * 0.65f + instant_velocity * 0.35f;
                    m_touch_velocity_y =
                        std::clamp(m_touch_velocity_y,
                                   -m_max_fling_velocity,
                                   m_max_fling_velocity);
                }
            } else if (m_touch_gesture != TouchGestureMode::HorizontalDrag) {
                m_touch_velocity_y = 0.0f;
            }

            m_touch_last_x = x;
            m_touch_last_y = y;
        }
    } else if (m_touch_is_down) {
        if (m_touch_gesture == TouchGestureMode::VerticalScroll) {
            const float launch_velocity =
                std::clamp(m_touch_velocity_y,
                           -m_max_fling_velocity,
                           m_max_fling_velocity);
            if (std::abs(launch_velocity) >= m_min_fling_velocity) {
                m_momentum_active = true;
                m_momentum_velocity_y = launch_velocity;
            } else {
                m_momentum_active = false;
                m_momentum_velocity_y = 0.0f;
            }
        } else {
            m_momentum_active = false;
            m_momentum_velocity_y = 0.0f;
        }
        releaseTouchButton(m_touch_gesture == TouchGestureMode::VerticalScroll);
        resetTouchState();
    } else if (m_momentum_active) {
        const float speed = std::abs(m_momentum_velocity_y);
        const float new_speed =
            std::max(0.0f, speed - m_momentum_deceleration * m_frame_dt_seconds);
        const float average_speed = 0.5f * (speed + new_speed);
        const float direction = (m_momentum_velocity_y >= 0.0f) ? 1.0f : -1.0f;
        m_touch_scroll_delta_y = direction * average_speed * m_frame_dt_seconds;
        m_momentum_velocity_y = direction * new_speed;
        if (new_speed <= 1.0f) {
            m_momentum_active = false;
            m_momentum_velocity_y = 0.0f;
            m_touch_scroll_delta_y = 0.0f;
        }
    }

    if (std::abs(m_touch_scroll_delta_y) > 0.01f) {
        injectTouchScroll(m_touch_scroll_delta_y);
    }
    nk_input_end(m_ctx);
}

bool GUIManager::Impl::input_is_mouse_down_in_rect(std::tuple<float, float, float, float> rect, bool left_mouse) {
    struct nk_rect r = nk_rect(std::get<0>(rect), std::get<1>(rect), std::get<2>(rect), std::get<3>(rect));
    if (left_mouse && m_has_touch_input) {
        return m_touch_is_down && nk_input_is_mouse_hovering_rect(&m_ctx->input, r);
    }
    return nk_input_is_mouse_down(&m_ctx->input, left_mouse ? NK_BUTTON_LEFT : NK_BUTTON_RIGHT) &&
           nk_input_is_mouse_hovering_rect(&m_ctx->input, r);
}

void GUIManager::Impl::window_drag_from_pos(nk_command_buffer* canvas) {
    (void)canvas;
    const char* window_name = currentWindowName();
    if (!window_name || window_name[0] == '\0') {
        return;
    }

    const struct nk_rect header_bounds = currentHeaderBounds();
    const bool pointer_down = m_has_touch_input ? m_touch_is_down
                                                : nk_input_is_mouse_down(&m_ctx->input, NK_BUTTON_LEFT);
    const bool pointer_over_header =
        nk_input_is_mouse_hovering_rect(&m_ctx->input, header_bounds) ||
        nk_input_is_mouse_prev_hovering_rect(&m_ctx->input, header_bounds);

    if (!pointer_down) {
        m_window_drag_active = false;
        m_window_drag_name.clear();
        return;
    }

    if (!m_window_drag_active && pointer_over_header) {
        m_window_drag_active = true;
        m_window_drag_name = window_name;
    }

    if (!m_window_drag_active || m_window_drag_name != window_name) {
        return;
    }

    const struct nk_rect bounds = nk_window_get_bounds(m_ctx);
    const struct nk_vec2 delta = m_ctx->input.mouse.delta;
    if (std::abs(delta.x) <= 0.01f && std::abs(delta.y) <= 0.01f) {
        return;
    }
    nk_window_set_position(m_ctx, window_name, nk_vec2(bounds.x + delta.x, bounds.y + delta.y));
}

std::tuple<bool, float, std::tuple<float, float, float, float>> GUIManager::Impl::input_is_mouse_dragging_in_rect() {
    struct nk_rect content_bounds = nk_window_get_content_region(m_ctx);
    const bool is_touch_scroll =
        m_touch_gesture == TouchGestureMode::VerticalScroll &&
        (nk_input_is_mouse_hovering_rect(&m_ctx->input, content_bounds) ||
         nk_input_is_mouse_prev_hovering_rect(&m_ctx->input, content_bounds));
    bool is_dragging = is_touch_scroll ||
                       (nk_input_is_mouse_prev_hovering_rect(&m_ctx->input, content_bounds) &&
                        nk_input_is_mouse_down(&m_ctx->input, NK_BUTTON_LEFT));
    float dy = is_touch_scroll ? m_touch_scroll_delta_y :
               (is_dragging ? m_ctx->input.mouse.delta.y : 0.0f);
    return {is_dragging, dy, {content_bounds.x, content_bounds.y, content_bounds.w, content_bounds.h}};
}

bool GUIManager::Impl::is_title_bar_pressed() {
    if (!m_ctx || !m_ctx->current) return false;
    const struct nk_rect header_bounds = currentHeaderBounds();
    const char* current_name = currentWindowName();
    const bool pointer_down = m_has_touch_input ? m_touch_is_down
                                                : nk_input_is_mouse_down(&m_ctx->input, NK_BUTTON_LEFT);
    return pointer_down &&
           (nk_input_is_mouse_hovering_rect(&m_ctx->input, header_bounds) ||
            (m_window_drag_active && current_name && m_window_drag_name == current_name));
}

float GUIManager::Impl::get_smart_scroll_dy() {
    if (std::abs(m_touch_scroll_delta_y) <= 0.01f) {
        return 0.0f;
    }

    const struct nk_rect content_bounds = nk_window_get_content_region(m_ctx);
    const bool is_in_content =
        nk_input_is_mouse_hovering_rect(&m_ctx->input, content_bounds) ||
        nk_input_is_mouse_prev_hovering_rect(&m_ctx->input, content_bounds);
    return is_in_content ? m_touch_scroll_delta_y : 0.0f;
}

void GUIManager::Impl::releaseTouchButton(bool cancel) {
    if (!m_touch_button_down) {
        return;
    }

    const int release_x = cancel ? -4096 : static_cast<int>(std::lround(m_touch_last_x));
    const int release_y = cancel ? -4096 : static_cast<int>(std::lround(m_touch_last_y));
    nk_input_button(m_ctx, NK_BUTTON_LEFT, release_x, release_y, 0);
    m_touch_button_down = false;
}

void GUIManager::Impl::resetTouchState() {
    m_touch_is_down = false;
    m_touch_button_down = false;
    m_touch_gesture = TouchGestureMode::None;
    m_touch_start_x = 0.0f;
    m_touch_start_y = 0.0f;
    m_touch_scroll_delta_y = 0.0f;
    m_touch_press_duration = 0.0f;
    m_touch_long_press_fired = false;
    m_touch_velocity_y = 0.0f;
    m_window_drag_active = false;
    m_window_drag_name.clear();
}

struct nk_rect GUIManager::Impl::currentHeaderBounds() const {
    if (!m_ctx || !m_ctx->current || !m_ctx->current->layout) {
        return nk_rect(0, 0, 0, 0);
    }

    const struct nk_rect bounds = nk_window_get_bounds(m_ctx);
    float header_height = m_ctx->current->layout->header_height;
    if (header_height <= 0.0f) {
        header_height = m_ctx->style.font ? (m_ctx->style.font->height + 2.0f * m_ctx->style.window.header.padding.y)
                                          : m_min_row_height;
    }
    return nk_rect(bounds.x, bounds.y, bounds.w, header_height);
}

const char* GUIManager::Impl::currentWindowName() const {
    if (!m_ctx || !m_ctx->current) {
        return nullptr;
    }
    return m_ctx->current->name_string;
}

float GUIManager::Impl::clampRowHeight(float requested_height) const {
    if (requested_height <= 0.0f) {
        return requested_height;
    }
    return requested_height;
}

void GUIManager::Impl::injectTouchScroll(float pixel_delta_y) {
    nk_input_scroll(m_ctx, nk_vec2(0.0f, gui_style::pixels_to_nuklear_scroll_units(pixel_delta_y, m_scroll_pixels_per_unit)));
}

void GUIManager::Impl::updateFrameTiming() {
    const Clock::time_point now = Clock::now();
    if (m_has_frame_timestamp) {
        m_frame_dt_seconds =
            std::clamp(std::chrono::duration<float>(now - m_last_frame_timestamp).count(),
                       1.0f / 240.0f,
                       1.0f / 20.0f);
    } else {
        m_frame_dt_seconds = 1.0f / 60.0f;
        m_has_frame_timestamp = true;
    }
    m_last_frame_timestamp = now;
    m_ctx->delta_time_seconds = m_frame_dt_seconds;
}
