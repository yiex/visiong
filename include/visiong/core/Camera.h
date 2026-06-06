// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_CAMERA_H
#define VISIONG_CORE_CAMERA_H

#include <memory>
#include <string>
#include <utility>

#include "visiong/core/ImageBuffer.h"

struct CameraImpl;

class Camera {
public:
    Camera(int target_width,
           int target_height,
           const std::string& format = "yuv",
           bool hdr_enabled = false,
           const std::string& crop_mode = "auto",
           const std::string& isp_path = "auto");
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
              const std::string& crop_mode = "auto",
              const std::string& isp_path = "auto");
    ImageBuffer snapshot();
    Camera& sub(int target_width = 640,
                int target_height = 360,
                const std::string& format = "auto",
                const std::string& crop_mode = "auto",
                const std::string& isp_path = "auto");
    std::pair<ImageBuffer, ImageBuffer> snapshots();
    std::unique_ptr<Camera> open_stream(int target_width = 640,
                                        int target_height = 360,
                                        const std::string& format = "auto",
                                        const std::string& crop_mode = "auto",
                                        const std::string& isp_path = "auto");
    bool has_sub() const;
    void close_sub();
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
    std::string get_format() const;
    std::string format() const { return get_format(); }
    bool is_hdr_enabled() const;
    bool hdr_enabled() const { return is_hdr_enabled(); }
    std::string get_isp_path() const;
    std::string isp_path() const { return get_isp_path(); }
    std::string get_device_path() const;
    std::string device_path() const { return get_device_path(); }

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
