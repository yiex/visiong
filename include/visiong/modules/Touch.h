// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_TOUCH_H
#define VISIONG_MODULES_TOUCH_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TouchPoint {
    int x;
    int y;
};

class TouchDevice {
public:
    virtual ~TouchDevice() = default;

    virtual void release() = 0;
    virtual bool is_pressed() = 0;
    virtual std::vector<TouchPoint> get_touch_points() = 0;
    virtual std::vector<TouchPoint> read() { return get_touch_points(); }
    virtual void configure_geometry(int original_width, int original_height, int rotation_degrees) = 0;
};

class FT6336U_Touch : public TouchDevice {
public:
    FT6336U_Touch(const std::string& i2c_bus_path, uint8_t device_address);
    ~FT6336U_Touch() override;

    FT6336U_Touch(const FT6336U_Touch&) = delete;
    FT6336U_Touch& operator=(const FT6336U_Touch&) = delete;

    void release() override;
    bool is_pressed() override;
    std::vector<TouchPoint> get_touch_points() override;
    void configure_geometry(int original_width, int original_height, int rotation_degrees) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

std::unique_ptr<TouchDevice> create_touch_device(const std::string& chip_model,
                                                 const std::string& i2c_bus_path);

#endif  // VISIONG_MODULES_TOUCH_H
