// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/ISPController.h"

#include <stdexcept>
#include <string>

struct ISPControllerImpl {
    rk_aiq_sys_ctx_t* aiq_ctx = nullptr;
    int saturation = 128;
    int contrast = 128;
    int brightness = 128;
    int sharpness = 50;
    int hue = 128;
    int white_balance_temperature = 5000;
    int exposure_gain = 1;
    float exposure_time = 0.01f;
    int focus_position = 0;
    int spatial_denoise_level = 0;
    int temporal_denoise_level = 0;
    int frame_rate = 25;
    std::string white_balance_mode = "auto";
    std::string exposure_mode = "auto";
    std::string focus_mode = "continuous";
    std::string power_line_mode = "50hz";
    bool flip = false;
    bool mirror = false;
};

namespace {

void ensure_range(int value, int min_value, int max_value, const char* field) {
    if (value < min_value || value > max_value) {
        throw std::invalid_argument(std::string(field) + " value is out of range.");
    }
}

}  // namespace

ISPController::ISPController(rk_aiq_sys_ctx_t* aiq_ctx) : m_impl(std::make_unique<ISPControllerImpl>()) {
    if (!aiq_ctx) {
        throw std::runtime_error("ISPController Error: Provided AIQ context is null.");
    }
    m_impl->aiq_ctx = aiq_ctx;
}

ISPController::~ISPController() = default;

void ISPController::set_saturation(int value) {
    ensure_range(value, 0, 255, "Saturation");
    m_impl->saturation = value;
}

int ISPController::get_saturation() {
    return m_impl->saturation;
}

void ISPController::set_contrast(int value) {
    ensure_range(value, 0, 255, "Contrast");
    m_impl->contrast = value;
}

int ISPController::get_contrast() {
    return m_impl->contrast;
}

void ISPController::set_brightness(int value) {
    ensure_range(value, 0, 255, "Brightness");
    m_impl->brightness = value;
}

int ISPController::get_brightness() {
    return m_impl->brightness;
}

void ISPController::set_hue(int value) {
    ensure_range(value, 0, 255, "Hue");
    m_impl->hue = value;
}

int ISPController::get_hue() {
    return m_impl->hue;
}

void ISPController::set_sharpness(int value) {
    ensure_range(value, 0, 100, "Sharpness");
    m_impl->sharpness = value;
}

int ISPController::get_sharpness() {
    return m_impl->sharpness;
}

void ISPController::set_white_balance_mode(const std::string& mode) {
    if (mode != "auto" && mode != "manual") {
        throw std::invalid_argument("Invalid white balance mode.");
    }
    m_impl->white_balance_mode = mode;
}

std::string ISPController::get_white_balance_mode() {
    return m_impl->white_balance_mode;
}

void ISPController::set_white_balance_temperature(int temp) {
    if (temp < 0) {
        throw std::invalid_argument("White balance temperature must be non-negative.");
    }
    if (m_impl->white_balance_mode != "manual") {
        throw std::runtime_error("Must be in manual white balance mode to set temperature.");
    }
    m_impl->white_balance_temperature = temp;
}

int ISPController::get_white_balance_temperature() {
    return m_impl->white_balance_temperature;
}

void ISPController::set_exposure_mode(const std::string& mode) {
    if (mode != "auto" && mode != "manual") {
        throw std::invalid_argument("Invalid exposure mode.");
    }
    m_impl->exposure_mode = mode;
}

std::string ISPController::get_exposure_mode() {
    return m_impl->exposure_mode;
}

void ISPController::set_exposure_time(float time_s) {
    if (time_s <= 0.0f) {
        throw std::invalid_argument("Exposure time must be positive.");
    }
    if (m_impl->exposure_mode != "manual") {
        throw std::runtime_error("Must be in manual exposure mode to set exposure time.");
    }
    m_impl->exposure_time = time_s;
}

float ISPController::get_exposure_time() {
    return m_impl->exposure_time;
}

void ISPController::set_exposure_gain(int gain) {
    ensure_range(gain, 0, 127, "Exposure gain");
    if (m_impl->exposure_mode != "manual") {
        throw std::runtime_error("Must be in manual exposure mode to set exposure gain.");
    }
    m_impl->exposure_gain = gain;
}

int ISPController::get_exposure_gain() {
    return m_impl->exposure_gain;
}

void ISPController::lock_focus() {}

void ISPController::unlock_focus() {}

void ISPController::trigger_focus() {}

void ISPController::set_focus_mode(const std::string& mode) {
    if (mode != "continuous-picture" && mode != "continuous" && mode != "manual") {
        throw std::invalid_argument("Invalid focus mode.");
    }
    m_impl->focus_mode = mode;
}

void ISPController::set_manual_focus_position(int position) {
    m_impl->focus_mode = "manual";
    m_impl->focus_position = position;
}

int ISPController::get_focus_position() {
    return m_impl->focus_position;
}

void ISPController::set_spatial_denoise_level(int level) {
    ensure_range(level, 0, 100, "Spatial denoise");
    m_impl->spatial_denoise_level = level;
}

void ISPController::set_temporal_denoise_level(int level) {
    ensure_range(level, 0, 100, "Temporal denoise");
    m_impl->temporal_denoise_level = level;
}

void ISPController::set_frame_rate(int fps) {
    if (fps != 0 && (fps < 10 || fps > 60)) {
        throw std::invalid_argument("Frame rate is out of range.");
    }
    m_impl->frame_rate = (fps > 0) ? fps : 25;
}

void ISPController::set_power_line_frequency(const std::string& mode) {
    if (mode != "50hz" && mode != "60hz" && mode != "off") {
        throw std::invalid_argument("Invalid power line frequency mode.");
    }
    m_impl->power_line_mode = mode;
}

void ISPController::set_flip(bool flip, bool mirror) {
    m_impl->flip = flip;
    m_impl->mirror = mirror;
}
