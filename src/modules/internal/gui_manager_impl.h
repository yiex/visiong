// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_GUI_MANAGER_IMPL_H
#define VISIONG_MODULES_INTERNAL_GUI_MANAGER_IMPL_H

#include "visiong/modules/GUIManager.h"
#include "modules/internal/gui_nuklear_config.h"

#include <chrono>
#include <string>
#include <tuple>
#include <vector>

struct GUIManager::Impl {
public:
    Impl(int width, int height, const std::string& font_path, const std::string& pre_chars);
    ~Impl();
    void beginFrame(TouchDevice* touch);
    bool beginWindow(const std::string& title, float x, float y, float w, float h, const std::string& flags_str);
    void endWindow();
    void endFrame(ImageBuffer& target);
    void layoutRowDynamic(float height, int cols);
    void layoutRowStatic(float height, int item_width, int cols);
    void layoutRowBegin(const std::string& format, float row_height, int cols);
    void layoutRowPush(float value);
    void layoutRowEnd();
    bool groupBegin(const std::string& title, const std::string& flags_str);
    void groupEnd();
    void label(const std::string& text, const std::string& align);
    void labelWrap(const std::string& text);
    bool button(const std::string& label);
    float slider(const std::string& label, float value, float min, float max, float step);
    bool checkbox(const std::string& label, bool active);
    bool option(const std::string& label, bool active);
    std::tuple<bool, std::string> editString(const std::string& text, int max_len);
    int progress(int current, int max, bool is_modifyable);
    bool buttonImage(const ImageBuffer& img);
    bool treeNode(const std::string& title, bool is_expanded);
    void treePop();
    int propertyInt(const std::string& name, int val, int min, int max, int step, float inc_per_pixel);
    float propertyFloat(const std::string& name, float val, float min, float max, float step, float inc_per_pixel);
    bool comboBegin(const std::string& text, float w, float h);
    bool comboItem(const std::string& text);
    void comboEnd();
    bool contextualBegin(float w, float h);
    bool contextualItem(const std::string& text);
    void contextualEnd();
    nk_command_buffer* getCanvas();
    std::tuple<float, float, float, float> widget_bounds(nk_command_buffer* canvas);
    bool input_is_mouse_down_in_rect(std::tuple<float, float, float, float> rect, bool left_mouse);
    void window_set_focus(const std::string& name);
    void window_drag_from_pos(nk_command_buffer* canvas);
    void window_set_scroll(float scroll_y);
    std::tuple<bool, float, std::tuple<float, float, float, float>> input_is_mouse_dragging_in_rect();
    bool is_title_bar_pressed();
    float get_content_height();
    void push_style_vec2(const std::string& name, float x, float y);
    void pop_style();
    float get_smart_scroll_dy();
    void strokeLine(nk_command_buffer* c, float x0, float y0, float x1, float y1, float thickness, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void strokeRect(nk_command_buffer* c, float x, float y, float w, float h, float rounding, float thickness, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void fillRect(nk_command_buffer* c, float x, float y, float w, float h, float rounding, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void drawText(nk_command_buffer* c, float x, float y, const std::string& text, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void setStyleColor(const std::string& property_name, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void setStyleButtonRounding(float rounding);
    void setStyleWindowRounding(float rounding);
    void setWindowBackgroundColor(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    bool chartBegin(const std::string& type, int count, float min_val, float max_val);
    void chartPush(float value);
    void chartEnd();
    void menubarBegin();
    void menubarEnd();
    bool menuBegin(const std::string& label, float width, float height);
    bool menuItem(const std::string& label);
    void menuEnd();
    void tooltip(const std::string& text);
    void releaseTouchButton(bool cancel);
    void resetTouchState();
    float clampRowHeight(float requested_height) const;
    void injectTouchScroll(float pixel_delta_y);
    void updateFrameTiming();
    struct nk_rect currentHeaderBounds() const;
    const char* currentWindowName() const;

private:
    enum class TouchGestureMode {
        None,
        TapCandidate,
        HorizontalDrag,
        VerticalScroll,
        LongPress
    };

    using Clock = std::chrono::steady_clock;

    void render(ImageBuffer& target);
    struct nk_context* m_ctx;
    struct nk_font_atlas* m_atlas;
    struct nk_buffer* m_cmds;
    std::vector<uint8_t> m_font_atlas_data;
    std::vector<nk_rune> m_rune_ranges;
    int m_atlas_width, m_atlas_height;
    int m_width, m_height;
    std::vector<char> m_edit_buffer;

    float m_ui_scale = 1.0f;
    float m_min_row_height = 46.0f;
    float m_touch_slop = 10.0f;
    float m_long_press_seconds = 0.45f;
    float m_min_fling_velocity = 240.0f;
    float m_max_fling_velocity = 3200.0f;
    float m_momentum_deceleration = 3600.0f;
    float m_scroll_pixels_per_unit = 24.0f;
    bool m_touch_is_down = false;
    bool m_touch_button_down = false;
    TouchGestureMode m_touch_gesture = TouchGestureMode::None;
    float m_touch_start_x = 0.0f;
    float m_touch_start_y = 0.0f;
    float m_touch_last_x = 0.0f;
    float m_touch_last_y = 0.0f;
    float m_touch_scroll_delta_y = 0.0f;
    float m_touch_press_duration = 0.0f;
    float m_touch_velocity_y = 0.0f;
    float m_momentum_velocity_y = 0.0f;
    float m_frame_dt_seconds = 1.0f / 60.0f;
    bool m_momentum_active = false;
    bool m_has_touch_input = false;
    bool m_has_frame_timestamp = false;
    bool m_touch_long_press_fired = false;
    bool m_window_drag_active = false;
    std::string m_window_drag_name;
    Clock::time_point m_last_frame_timestamp{};

    std::string m_custom_font_path;
    std::string m_user_defined_chars_utf8;
    bool m_is_font_baked = false;
};

#endif  // VISIONG_MODULES_INTERNAL_GUI_MANAGER_IMPL_H
