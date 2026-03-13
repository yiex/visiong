// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/GUIManager.h"
#include "modules/internal/gui_manager_impl.h"

GUIManager::~GUIManager() = default;
GUIManager::GUIManager(GUIManager&&) noexcept = default;
GUIManager& GUIManager::operator=(GUIManager&&) noexcept = default;

void GUIManager::beginFrame(TouchDevice* touch) { m_impl->beginFrame(touch); }

bool GUIManager::beginWindow(const std::string& title,
                             float x,
                             float y,
                             float w,
                             float h,
                             const std::string& flags_str) {
    return m_impl->beginWindow(title, x, y, w, h, flags_str);
}

void GUIManager::endWindow() { m_impl->endWindow(); }
void GUIManager::endFrame(ImageBuffer& target) { m_impl->endFrame(target); }
void GUIManager::layoutRowDynamic(float height, int cols) { m_impl->layoutRowDynamic(height, cols); }
void GUIManager::layoutRowStatic(float height, int item_width, int cols) {
    m_impl->layoutRowStatic(height, item_width, cols);
}
void GUIManager::layoutRowBegin(const std::string& format, float row_height, int cols) {
    m_impl->layoutRowBegin(format, row_height, cols);
}
void GUIManager::layoutRowPush(float value) { m_impl->layoutRowPush(value); }
void GUIManager::layoutRowEnd() { m_impl->layoutRowEnd(); }
bool GUIManager::groupBegin(const std::string& title, const std::string& flags_str) {
    return m_impl->groupBegin(title, flags_str);
}
void GUIManager::groupEnd() { m_impl->groupEnd(); }

void GUIManager::label(const std::string& text, const std::string& align) { m_impl->label(text, align); }
void GUIManager::labelWrap(const std::string& text) { m_impl->labelWrap(text); }
bool GUIManager::button(const std::string& label) { return m_impl->button(label); }
float GUIManager::slider(const std::string& label, float value, float min, float max, float step) {
    return m_impl->slider(label, value, min, max, step);
}
bool GUIManager::checkbox(const std::string& label, bool active) { return m_impl->checkbox(label, active); }
bool GUIManager::option(const std::string& label, bool active) { return m_impl->option(label, active); }
std::tuple<bool, std::string> GUIManager::editString(const std::string& text, int max_len) {
    return m_impl->editString(text, max_len);
}
int GUIManager::progress(int current, int max, bool is_modifyable) {
    return m_impl->progress(current, max, is_modifyable);
}
bool GUIManager::buttonImage(const ImageBuffer& img) { return m_impl->buttonImage(img); }

bool GUIManager::treeNode(const std::string& title, bool is_expanded) {
    return m_impl->treeNode(title, is_expanded);
}
void GUIManager::treePop() { m_impl->treePop(); }
int GUIManager::propertyInt(const std::string& name, int val, int min, int max, int step, float inc_per_pixel) {
    return m_impl->propertyInt(name, val, min, max, step, inc_per_pixel);
}
float GUIManager::propertyFloat(const std::string& name,
                                float val,
                                float min,
                                float max,
                                float step,
                                float inc_per_pixel) {
    return m_impl->propertyFloat(name, val, min, max, step, inc_per_pixel);
}

bool GUIManager::comboBegin(const std::string& text, float w, float h) { return m_impl->comboBegin(text, w, h); }
bool GUIManager::comboItem(const std::string& text) { return m_impl->comboItem(text); }
void GUIManager::comboEnd() { m_impl->comboEnd(); }
bool GUIManager::contextualBegin(float w, float h) { return m_impl->contextualBegin(w, h); }
bool GUIManager::contextualItem(const std::string& text) { return m_impl->contextualItem(text); }
void GUIManager::contextualEnd() { m_impl->contextualEnd(); }

bool GUIManager::chartBegin(const std::string& type, int count, float min_val, float max_val) {
    return m_impl->chartBegin(type, count, min_val, max_val);
}
void GUIManager::chartPush(float value) { m_impl->chartPush(value); }
void GUIManager::chartEnd() { m_impl->chartEnd(); }
void GUIManager::menubarBegin() { m_impl->menubarBegin(); }
void GUIManager::menubarEnd() { m_impl->menubarEnd(); }
bool GUIManager::menuBegin(const std::string& label, float width, float height) {
    return m_impl->menuBegin(label, width, height);
}
bool GUIManager::menuItem(const std::string& label) { return m_impl->menuItem(label); }
void GUIManager::menuEnd() { m_impl->menuEnd(); }
void GUIManager::tooltip(const std::string& text) { m_impl->tooltip(text); }

void GUIManager::strokeLine(nk_command_buffer* canvas,
                            float x0,
                            float y0,
                            float x1,
                            float y1,
                            float thickness,
                            std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->strokeLine(canvas, x0, y0, x1, y1, thickness, color);
}

void GUIManager::strokeRect(nk_command_buffer* canvas,
                            float x,
                            float y,
                            float w,
                            float h,
                            float rounding,
                            float thickness,
                            std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->strokeRect(canvas, x, y, w, h, rounding, thickness, color);
}

void GUIManager::fillRect(nk_command_buffer* canvas,
                          float x,
                          float y,
                          float w,
                          float h,
                          float rounding,
                          std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->fillRect(canvas, x, y, w, h, rounding, color);
}

void GUIManager::drawText(nk_command_buffer* canvas,
                          float x,
                          float y,
                          const std::string& text,
                          std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->drawText(canvas, x, y, text, color);
}

void GUIManager::setStyleColor(const std::string& property_name,
                               std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->setStyleColor(property_name, color);
}
void GUIManager::setStyleButtonRounding(float rounding) { m_impl->setStyleButtonRounding(rounding); }
void GUIManager::setStyleWindowRounding(float rounding) { m_impl->setStyleWindowRounding(rounding); }
void GUIManager::setWindowBackgroundColor(std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color) {
    m_impl->setWindowBackgroundColor(color);
}
