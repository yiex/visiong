// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_GUIMANAGER_H
#define VISIONG_MODULES_GUIMANAGER_H

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

class ImageBuffer;
class TouchDevice;
struct nk_command_buffer;

class GUIManager {
public:
    GUIManager(int width, int height);
    GUIManager(int width, int height, const std::string& font_path, const std::string& pre_chars);
    ~GUIManager();

    GUIManager(const GUIManager&) = delete;
    GUIManager& operator=(const GUIManager&) = delete;
    GUIManager(GUIManager&&) noexcept;
    GUIManager& operator=(GUIManager&&) noexcept;

    // Frame lifecycle and layout. / Frame lifecycle 与 layout.
    void beginFrame(TouchDevice* touch);
    bool beginWindow(const std::string& title, float x, float y, float w, float h,
                     const std::string& flags_str);
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
    float propertyFloat(const std::string& name, float val, float min, float max, float step,
                        float inc_per_pixel);

    bool comboBegin(const std::string& text, float w, float h);
    bool comboItem(const std::string& text);
    void comboEnd();

    bool contextualBegin(float w, float h);
    bool contextualItem(const std::string& text);
    void contextualEnd();

    bool chartBegin(const std::string& type, int count, float min_val, float max_val);
    void chartPush(float value);
    void chartEnd();
    void menubarBegin();
    void menubarEnd();
    bool menuBegin(const std::string& label, float width, float height);
    bool menuItem(const std::string& label);
    void menuEnd();
    void tooltip(const std::string& text);

    void strokeLine(nk_command_buffer* canvas, float x0, float y0, float x1, float y1, float thickness,
                    std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void strokeRect(nk_command_buffer* canvas, float x, float y, float w, float h, float rounding,
                    float thickness, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void fillRect(nk_command_buffer* canvas, float x, float y, float w, float h, float rounding,
                  std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void drawText(nk_command_buffer* canvas, float x, float y, const std::string& text,
                  std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);

    void setStyleColor(const std::string& property_name,
                       std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void setStyleButtonRounding(float rounding);
    void setStyleWindowRounding(float rounding);
    void setWindowBackgroundColor(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);

    // Low-level canvas and interaction bridge. / Low-level canvas 与 interaction bridge.
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
    bool inputIsPointerDownInRect(std::tuple<float, float, float, float> rect, bool primary_pointer = true);
    std::tuple<bool, float, std::tuple<float, float, float, float>> inputIsPointerDraggingInRect();
    bool isTitleBarActive();
    float getScrollDeltaY();

    // Compatibility wrappers kept in sync with the Python-facing naming scheme. / Compatibility 包装层 kept 在 sync 与 Python-facing naming scheme.
    void begin_frame(TouchDevice* touch);
    bool begin_window(const std::string& title, float x, float y, float w, float h,
                      const std::string& flags_str);
    void end_window();
    void end_frame(ImageBuffer& target);
    void layout_row_dynamic(float height, int cols);
    void layout_row_static(float height, int item_width, int cols);
    void layout_row_begin(const std::string& format, float row_height, int cols);
    void layout_row_push(float value);
    void layout_row_end();
    bool group_begin(const std::string& title, const std::string& flags_str);
    void group_end();
    void label_wrap(const std::string& text);
    std::tuple<bool, std::string> edit_string(const std::string& text, int max_len);
    bool button_image(const ImageBuffer& img);
    bool tree_node(const std::string& title, bool is_expanded);
    void tree_pop();
    int property_int(const std::string& name, int val, int min, int max, int step, float inc_per_pixel);
    float property_float(const std::string& name, float val, float min, float max, float step,
                         float inc_per_pixel);
    bool combo_begin(const std::string& text, float w, float h);
    bool combo_item(const std::string& text);
    void combo_end();
    bool contextual_begin(float w, float h);
    bool contextual_item(const std::string& text);
    void contextual_end();
    bool chart_begin(const std::string& type, int count, float min_val, float max_val);
    void chart_push(float value);
    void chart_end();
    void menubar_begin();
    void menubar_end();
    bool menu_begin(const std::string& label, float width, float height);
    bool menu_item(const std::string& label);
    void menu_end();
    bool input_is_pointer_down_in_rect(std::tuple<float, float, float, float> rect, bool primary_pointer = true);
    std::tuple<bool, float, std::tuple<float, float, float, float>> input_is_pointer_dragging_in_rect();
    bool is_title_bar_active();
    float get_scroll_delta_y();
    nk_command_buffer* get_canvas();
    void stroke_line(nk_command_buffer* canvas, float x0, float y0, float x1, float y1, float thickness,
                     std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void stroke_rect(nk_command_buffer* canvas, float x, float y, float w, float h, float rounding,
                     float thickness, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void fill_rect(nk_command_buffer* canvas, float x, float y, float w, float h, float rounding,
                   std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void draw_text(nk_command_buffer* canvas, float x, float y, const std::string& text,
                   std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void set_style_color(const std::string& property_name,
                         std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);
    void set_style_button_rounding(float rounding);
    void set_style_window_rounding(float rounding);
    void set_window_background_color(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_GUIMANAGER_H
