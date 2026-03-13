// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/ISPController.h"
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "rk_aiq_user_api2_imgproc.h"
#include "af/rk_aiq_uapi_af_int.h"

struct ISPControllerImpl {
    rk_aiq_sys_ctx_t* aiq_ctx = nullptr;
};

namespace {

void configure_fixed_focus(rk_aiq_af_attrib_t& attr, short focus_code) {
    attr.AfMode = RKAIQ_AF_MODE_FIXED;
    attr.fixedModeDefCode = focus_code;
    attr.h_offs = 0;
    attr.v_offs = 0;
    attr.h_size = 0;
    attr.v_size = 0;
}

}  // namespace

ISPController::ISPController(rk_aiq_sys_ctx_t* aiq_ctx) : m_impl(std::make_unique<ISPControllerImpl>()) {
    if (!aiq_ctx) {
        throw std::runtime_error("ISPController Error: Provided AIQ context is null.");
    }
    m_impl->aiq_ctx = aiq_ctx;
}

ISPController::~ISPController() = default;

// --- Image Adjustment --- / --- 图像调节 ---

void ISPController::set_saturation(int value) {
    if (value < 0 || value > 255) {
        throw std::invalid_argument("Saturation value " + std::to_string(value) + " is out of the valid range [0, 255].");
    }
    if (rk_aiq_uapi2_setSaturation(m_impl->aiq_ctx, static_cast<unsigned int>(value)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set saturation via rk_aiq API.");
    }
}

int ISPController::get_saturation() {
    unsigned int level;
    rk_aiq_uapi2_getSaturation(m_impl->aiq_ctx, &level);
    return static_cast<int>(level);
}

void ISPController::set_contrast(int value) {
    if (value < 0 || value > 255) {
        throw std::invalid_argument("Contrast value " + std::to_string(value) + " is out of the valid range [0, 255].");
    }
    if (rk_aiq_uapi2_setContrast(m_impl->aiq_ctx, static_cast<unsigned int>(value)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set contrast via rk_aiq API.");
    }
}

int ISPController::get_contrast() {
    unsigned int level;
    rk_aiq_uapi2_getContrast(m_impl->aiq_ctx, &level);
    return static_cast<int>(level);
}

void ISPController::set_brightness(int value) {
    if (value < 0 || value > 255) {
        throw std::invalid_argument("Brightness value " + std::to_string(value) + " is out of the valid range [0, 255].");
    }
    if (rk_aiq_uapi2_setBrightness(m_impl->aiq_ctx, static_cast<unsigned int>(value)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set brightness via rk_aiq API.");
    }
}

int ISPController::get_brightness() {
    unsigned int level;
    rk_aiq_uapi2_getBrightness(m_impl->aiq_ctx, &level);
    return static_cast<int>(level);
}

void ISPController::set_hue(int value) {
    if (value < 0 || value > 255) {
        throw std::invalid_argument("Hue value " + std::to_string(value) + " is out of the valid range [0, 255].");
    }
    if (rk_aiq_uapi2_setHue(m_impl->aiq_ctx, static_cast<unsigned int>(value)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set hue via rk_aiq API.");
    }
}

int ISPController::get_hue() {
    unsigned int level;
    rk_aiq_uapi2_getHue(m_impl->aiq_ctx, &level);
    return static_cast<int>(level);
}

void ISPController::set_sharpness(int value) {
    if (value < 0 || value > 100) {
        throw std::invalid_argument("Sharpness value " + std::to_string(value) + " is out of the valid range [0, 100].");
    }
    if (rk_aiq_uapi2_setSharpness(m_impl->aiq_ctx, static_cast<unsigned int>(value)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set sharpness via rk_aiq API.");
    }
}

int ISPController::get_sharpness() {
    unsigned int level;
    rk_aiq_uapi2_getSharpness(m_impl->aiq_ctx, &level);
    return static_cast<int>(level);
}

// --- White Balance --- / --- 白平衡 ---

void ISPController::set_white_balance_mode(const std::string& mode) {
    if (mode != "auto" && mode != "manual") {
        throw std::invalid_argument("Invalid white balance mode '" + mode + "'. Use 'auto' or 'manual'.");
    }
    opMode_t wb_mode = (mode == "manual") ? OP_MANUAL : OP_AUTO;
    if (rk_aiq_uapi2_setWBMode(m_impl->aiq_ctx, wb_mode) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set white balance mode via rk_aiq API.");
    }
}

std::string ISPController::get_white_balance_mode() {
    opMode_t wb_mode;
    if (rk_aiq_uapi2_getWBMode(m_impl->aiq_ctx, &wb_mode) == XCAM_RETURN_NO_ERROR) {
        return (wb_mode == OP_MANUAL) ? "manual" : "auto";
    }
    return "unknown";
}

void ISPController::set_white_balance_temperature(int temp) {
    if (temp < 0) {
        throw std::invalid_argument("White balance temperature " + std::to_string(temp) + " is out of the valid range.");
    }
    if (get_white_balance_mode() != "manual") {
        throw std::runtime_error("Must be in 'manual' white balance mode to set temperature.");
    }
    if (rk_aiq_uapi2_setMWBCT(m_impl->aiq_ctx, static_cast<unsigned int>(temp)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set white balance temperature via rk_aiq API.");
    }
}

int ISPController::get_white_balance_temperature() {
    unsigned int cct;
    rk_aiq_uapi2_getWBCT(m_impl->aiq_ctx, &cct);
    return static_cast<int>(cct);
}

// --- Exposure --- / --- 曝光 ---

void ISPController::set_exposure_mode(const std::string& mode) {
    if (mode != "auto" && mode != "manual") {
        throw std::invalid_argument("Invalid exposure mode '" + mode + "'. Use 'auto' or 'manual'.");
    }
    opMode_t exp_mode = (mode == "manual") ? OP_MANUAL : OP_AUTO;
    if (rk_aiq_uapi2_setExpMode(m_impl->aiq_ctx, exp_mode) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set exposure mode via rk_aiq API.");
    }
}

std::string ISPController::get_exposure_mode() {
    opMode_t exp_mode;
    if (rk_aiq_uapi2_getExpMode(m_impl->aiq_ctx, &exp_mode) == XCAM_RETURN_NO_ERROR) {
        return (exp_mode == OP_MANUAL) ? "manual" : "auto";
    }
    return "unknown";
}

void ISPController::set_exposure_time(float time_s) {
    if (time_s <= 0) {
        throw std::invalid_argument("Exposure time " + std::to_string(time_s) + " must be positive.");
    }
    if (get_exposure_mode() != "manual") {
        throw std::runtime_error("Must be in 'manual' exposure mode to set exposure time.");
    }
    Uapi_ExpSwAttrV2_t expSwAttr;
    rk_aiq_user_api2_ae_getExpSwAttr(m_impl->aiq_ctx, &expSwAttr);
    
    // V4L2 implementation does not easily expose WDR mode, assume NORMAL / V4L2 实现不容易直接暴露 WDR 模式，这里假定为 NORMAL。
    expSwAttr.stManual.LinearAE.ManualTimeEn = true;
    expSwAttr.stManual.LinearAE.TimeValue = time_s;

    if (rk_aiq_user_api2_ae_setExpSwAttr(m_impl->aiq_ctx, expSwAttr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set exposure time via rk_aiq API.");
    }
}

float ISPController::get_exposure_time() {
    Uapi_ExpQueryInfo_t exp_query_info;
    if (rk_aiq_user_api2_ae_queryExpResInfo(m_impl->aiq_ctx, &exp_query_info) == XCAM_RETURN_NO_ERROR) {
        // Assume NORMAL mode / 假定为 NORMAL 模式。
        return exp_query_info.LinAeInfo.LinearExp.integration_time;
    }
    return -1.0f;
}

void ISPController::set_exposure_gain(int gain) {
    if (gain < 0 || gain > 127) { // A typical upper bound
        throw std::invalid_argument("Exposure gain " + std::to_string(gain) + " is out of the typical range [0, 127].");
    }
    if (get_exposure_mode() != "manual") {
        throw std::runtime_error("Must be in 'manual' exposure mode to set exposure gain.");
    }
    Uapi_ExpSwAttrV2_t expSwAttr;
    rk_aiq_user_api2_ae_getExpSwAttr(m_impl->aiq_ctx, &expSwAttr);
    float gain_f = static_cast<float>(gain);

    // Assume NORMAL mode / 假定为 NORMAL 模式。
    expSwAttr.stManual.LinearAE.ManualGainEn = true;
    expSwAttr.stManual.LinearAE.GainValue = gain_f;
    
    if (rk_aiq_user_api2_ae_setExpSwAttr(m_impl->aiq_ctx, expSwAttr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set exposure gain via rk_aiq API.");
    }
}

int ISPController::get_exposure_gain() {
    Uapi_ExpQueryInfo_t exp_query_info;
    if (rk_aiq_user_api2_ae_queryExpResInfo(m_impl->aiq_ctx, &exp_query_info) == XCAM_RETURN_NO_ERROR) {
        // Assume NORMAL mode / 假定为 NORMAL 模式。
        return static_cast<int>(exp_query_info.LinAeInfo.LinearExp.analog_gain);
    }
    return -1;
}

void ISPController::lock_focus() {
    if (rk_aiq_user_api2_af_Lock(m_impl->aiq_ctx) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to lock focus via rk_aiq API.");
    }
    std::cout << "ISPController: Focus locked." << std::endl;
}

void ISPController::unlock_focus() {
    if (rk_aiq_user_api2_af_Unlock(m_impl->aiq_ctx) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to unlock focus via rk_aiq API.");
    }
    std::cout << "ISPController: Focus unlocked." << std::endl;
}

void ISPController::trigger_focus() {
    if (rk_aiq_user_api2_af_Oneshot(m_impl->aiq_ctx) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to trigger one-shot focus via rk_aiq API.");
    }
    std::cout << "ISPController: One-shot focus triggered." << std::endl;
}

void ISPController::set_focus_mode(const std::string& mode) {
    rk_aiq_af_attrib_t attr;
    if (rk_aiq_user_api2_af_GetAttrib(m_impl->aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to get current AF attributes.");
    }

    if (mode == "continuous-picture" || mode == "continuous") {
        attr.AfMode = RKAIQ_AF_MODE_CONTINUOUS_PICTURE;
    } else if (mode == "manual") {
        configure_fixed_focus(attr, 1023);
    } else {
        throw std::invalid_argument("Invalid focus mode '" + mode + "'. Use 'continuous' or 'manual'.");
    }

    if (rk_aiq_user_api2_af_SetAttrib(m_impl->aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set focus mode via rk_aiq API.");
    }
}

void ISPController::set_manual_focus_position(int position) {
    rk_aiq_af_attrib_t attr;
    if (rk_aiq_user_api2_af_GetAttrib(m_impl->aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to get current AF attributes.");
    }

    configure_fixed_focus(attr, static_cast<short>(position));

    if (rk_aiq_user_api2_af_SetAttrib(m_impl->aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set manual focus position via SetAttrib API.");
    }
    std::cout << "ISPController: Manual focus position set to " << position << " via SetAttrib." << std::endl;
}

int ISPController::get_focus_position() {
    int pos = -1;
    if (rk_aiq_user_api2_af_GetFocusPos(m_impl->aiq_ctx, &pos) != XCAM_RETURN_NO_ERROR) {
        return -1;
    }
    return pos;
}
// --- Enhancement --- / --- 增强 ---

void ISPController::set_spatial_denoise_level(int level) {
    if (level < 0 || level > 100) {
        throw std::invalid_argument("Spatial denoise level " + std::to_string(level) + " is out of the valid range [0, 100].");
    }
    if (rk_aiq_uapi2_setMSpaNRStrth(m_impl->aiq_ctx, true, static_cast<unsigned int>(level)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set spatial denoise level via rk_aiq API.");
    }
}

void ISPController::set_temporal_denoise_level(int level) {
    if (level < 0 || level > 100) {
        throw std::invalid_argument("Temporal denoise level " + std::to_string(level) + " is out of the valid range [0, 100].");
    }
    if (rk_aiq_uapi2_setMTNRStrth(m_impl->aiq_ctx, true, static_cast<unsigned int>(level)) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set temporal denoise level via rk_aiq API.");
    }
}

// --- Misc --- / --- 其他 ---
void ISPController::set_frame_rate(int fps) {
    if (fps != 0 && (fps < 10 || fps > 60)) {
        throw std::invalid_argument("Frame rate " + std::to_string(fps) + " is out of the valid range [10, 60] (or 0 for auto).");
    }
    Uapi_ExpSwAttrV2_t expSwAttr;
    rk_aiq_user_api2_ae_getExpSwAttr(m_impl->aiq_ctx, &expSwAttr);
    expSwAttr.stAuto.stFrmRate.isFpsFix = (fps > 0);
    expSwAttr.stAuto.stFrmRate.FpsValue = (fps > 0) ? static_cast<float>(fps) : 25.0f; // Default to 25 if auto
    if (rk_aiq_user_api2_ae_setExpSwAttr(m_impl->aiq_ctx, expSwAttr) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set frame rate via rk_aiq API.");
    }
}

void ISPController::set_power_line_frequency(const std::string& mode) {
    if (mode != "50hz" && mode != "60hz" && mode != "off") {
        throw std::invalid_argument("Invalid power line frequency mode '" + mode + "'. Use '50hz', '60hz', or 'off'.");
    }
    expPwrLineFreq_t freq = EXP_PWR_LINE_FREQ_50HZ;
    if (mode == "60hz") {
        freq = EXP_PWR_LINE_FREQ_60HZ;
    } else if (mode == "off") {
        freq = EXP_PWR_LINE_FREQ_DIS;
    }
    if (rk_aiq_uapi2_setExpPwrLineFreqMode(m_impl->aiq_ctx, freq) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set power line frequency via rk_aiq API.");
    }
}

void ISPController::set_flip(bool flip, bool mirror) {
    if (rk_aiq_uapi2_setMirrorFlip(m_impl->aiq_ctx, mirror, flip, 4) != XCAM_RETURN_NO_ERROR) {
        throw std::runtime_error("Failed to set flip/mirror via rk_aiq API.");
    }
}

