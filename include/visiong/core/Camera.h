// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_CAMERA_H
#define VISIONG_CORE_CAMERA_H

#include <memory>
#include <string>

#include "visiong/core/ImageBuffer.h"

struct CameraImpl;

class Camera {
public:
    Camera(int target_width,
           int target_height,
           const std::string& format = "yuv",
           bool hdr_enabled = false,
           const std::string& crop_mode = "auto");
    Camera();
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    bool init(int target_width,
              int target_height,
              const std::string& format = "yuv",
              bool hdr_enabled = false,
              const std::string& crop_mode = "auto");
    ImageBuffer snapshot();
    void skip_frames(int num_frames);
    void skip(int num_frames = 10) { skip_frames(num_frames); }
    void release();
    bool is_initialized() const;

    int get_target_width() const;
    int get_target_height() const;
    int target_width() const { return get_target_width(); }
    int target_height() const { return get_target_height(); }
    int get_actual_capture_width() const;
    int get_actual_capture_height() const;
    int actual_width() const { return get_actual_capture_width(); }
    int actual_height() const { return get_actual_capture_height(); }
    std::string get_crop_mode() const;
    std::string crop_mode() const { return get_crop_mode(); }

    int get_capture_width() const { return get_actual_capture_width(); }
    int get_capture_height() const { return get_actual_capture_height(); }

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
    void set_frame_rate(int fps);
    void set_power_line_frequency(const std::string& mode);
    void set_flip(bool flip, bool mirror);
    void set_spatial_denoise_level(int level);
    void set_temporal_denoise_level(int level);

    void lock_focus();
    void unlock_focus();
    void trigger_focus();
    void set_focus_mode(const std::string& mode);
    void set_manual_focus(int position);
    int get_focus_position();

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
    std::unique_ptr<CameraImpl> m_impl;
};

#endif  // VISIONG_CORE_CAMERA_H
