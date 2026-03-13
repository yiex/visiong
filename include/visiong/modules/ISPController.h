// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_ISPCONTROLLER_H
#define VISIONG_MODULES_ISPCONTROLLER_H

#include <memory>
#include <string>

typedef struct rk_aiq_sys_ctx_s rk_aiq_sys_ctx_t;

struct ISPControllerImpl;

class ISPController {
  public:
    explicit ISPController(rk_aiq_sys_ctx_t* aiq_ctx);
    ~ISPController();

    ISPController(const ISPController&) = delete;
    ISPController& operator=(const ISPController&) = delete;

    void set_saturation(int value);
    void set_contrast(int value);
    void set_brightness(int value);
    void set_sharpness(int value);
    void set_hue(int value);

    void set_white_balance_mode(const std::string& mode);
    void set_white_balance_temperature(int temp);

    void set_exposure_mode(const std::string& mode);
    void set_exposure_time(float time_s);
    void set_exposure_gain(int gain);

    void lock_focus();
    void unlock_focus();
    void trigger_focus();
    void set_focus_mode(const std::string& mode);
    void set_manual_focus_position(int position);
    int get_focus_position();

    void set_spatial_denoise_level(int level);
    void set_temporal_denoise_level(int level);

    void set_frame_rate(int fps);
    void set_power_line_frequency(const std::string& mode);
    void set_flip(bool flip, bool mirror);

    int get_saturation();
    int get_contrast();
    int get_brightness();
    int get_sharpness();
    int get_hue();
    std::string get_white_balance_mode();
    int get_white_balance_temperature();
    std::string get_exposure_mode();
    float get_exposure_time();
    int get_exposure_gain();

  private:
    std::unique_ptr<ISPControllerImpl> m_impl;
};

#endif  // VISIONG_MODULES_ISPCONTROLLER_H

