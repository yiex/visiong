// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/Touch.h"
#include "common/internal/string_utils.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint8_t kFt6336uChipIdRegister = 0xA8;
constexpr uint8_t kFt6336uTouchCountRegister = 0x02;
constexpr uint8_t kFt6336uPoint1Register = 0x03;
constexpr uint8_t kFt6336uPoint2Register = 0x09;
constexpr uint8_t kFt6336uExpectedChipId = 0x11;

bool read_registers(int fd, uint8_t reg_start, uint8_t* buffer, int len) {
    if (fd < 0) {
        return false;
    }
    if (write(fd, &reg_start, 1) != 1) {
        return false;
    }
    if (::read(fd, buffer, len) != len) {
        return false;
    }
    return true;
}

TouchPoint apply_rotation(int raw_x,
                          int raw_y,
                          int original_width,
                          int original_height,
                          int rotation_degrees) {
    TouchPoint point{raw_x, raw_y};
    int target_width = original_height;
    int target_height = original_width;

    if (rotation_degrees == 270) {
        point.x = (original_height - 1) - raw_y;
        point.y = raw_x;
    } else if (rotation_degrees == 90) {
        point.x = raw_y;
        point.y = (original_width - 1) - raw_x;
    } else if (rotation_degrees == 180) {
        point.x = (original_width - 1) - raw_x;
        point.y = (original_height - 1) - raw_y;
        target_width = original_width;
        target_height = original_height;
    }

    if (rotation_degrees != 0) {
        point.x = std::max(0, std::min(point.x, target_width - 1));
        point.y = std::max(0, std::min(point.y, target_height - 1));
    }
    return point;
}

}  // namespace

struct FT6336U_Touch::Impl {
    std::string i2c_bus_path;
    uint8_t device_address = 0;
    int i2c_fd = -1;
    int original_width = 240;
    int original_height = 320;
    int rotation_degrees = 270;
};

std::unique_ptr<TouchDevice> create_touch_device(const std::string& chip_model,
                                                 const std::string& i2c_bus_path) {
    const std::string model = visiong::to_lower_copy(chip_model);
    if (model == "ft6336u") {
        return std::make_unique<FT6336U_Touch>(i2c_bus_path, 0x38);
    }

    std::cerr << "[Touch Factory] Unsupported chip model: " << chip_model << std::endl;
    return nullptr;
}

FT6336U_Touch::FT6336U_Touch(const std::string& i2c_bus_path, uint8_t device_address)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->i2c_bus_path = i2c_bus_path;
    m_impl->device_address = device_address;
    configure_geometry(240, 320, 270);

    m_impl->i2c_fd = open(m_impl->i2c_bus_path.c_str(), O_RDWR);
    if (m_impl->i2c_fd < 0) {
        throw std::runtime_error("[Touch] Failed to open I2C bus " + m_impl->i2c_bus_path + ": " +
                                 std::strerror(errno));
    }

    if (ioctl(m_impl->i2c_fd, I2C_SLAVE, m_impl->device_address) < 0) {
        release();
        throw std::runtime_error("[Touch] Failed to set I2C slave address: " +
                                 std::to_string(m_impl->device_address));
    }

    uint8_t chip_id = 0;
    if (!read_registers(m_impl->i2c_fd, kFt6336uChipIdRegister, &chip_id, 1)) {
        release();
        throw std::runtime_error("[Touch] Failed to read FT6336U chip ID.");
    }
    if (chip_id != kFt6336uExpectedChipId) {
        release();
        throw std::runtime_error("[Touch] Unexpected FT6336U chip ID: " + std::to_string(chip_id));
    }

    std::cout << "[Touch] FT6336U initialized on " << m_impl->i2c_bus_path << " (0x" << std::hex
              << static_cast<int>(m_impl->device_address) << std::dec << ")" << std::endl;
}

FT6336U_Touch::~FT6336U_Touch() {
    release();
}

void FT6336U_Touch::configure_geometry(int original_width,
                                       int original_height,
                                       int rotation_degrees) {
    m_impl->original_width = original_width;
    m_impl->original_height = original_height;
    m_impl->rotation_degrees = rotation_degrees;
}

void FT6336U_Touch::release() {
    if (m_impl->i2c_fd >= 0) {
        close(m_impl->i2c_fd);
        m_impl->i2c_fd = -1;
    }
}

bool FT6336U_Touch::is_pressed() {
    uint8_t touch_count = 0;
    if (!read_registers(m_impl->i2c_fd, kFt6336uTouchCountRegister, &touch_count, 1)) {
        return false;
    }
    return (touch_count & 0x0F) > 0;
}

std::vector<TouchPoint> FT6336U_Touch::get_touch_points() {
    std::vector<TouchPoint> points;
    if (m_impl->i2c_fd < 0) {
        return points;
    }

    uint8_t status = 0;
    if (!read_registers(m_impl->i2c_fd, kFt6336uTouchCountRegister, &status, 1)) {
        return points;
    }

    const uint8_t touch_count = status & 0x0F;
    if (touch_count == 0 || touch_count > 2) {
        return points;
    }

    std::vector<uint8_t> p1(6);
    if (read_registers(m_impl->i2c_fd, kFt6336uPoint1Register, p1.data(), static_cast<int>(p1.size()))) {
        const int raw_x = ((p1[0] & 0x0F) << 8) | p1[1];
        const int raw_y = ((p1[2] & 0x0F) << 8) | p1[3];
        points.push_back(apply_rotation(raw_x, raw_y, m_impl->original_width, m_impl->original_height,
                                        m_impl->rotation_degrees));
    }

    if (touch_count > 1) {
        std::vector<uint8_t> p2(6);
        if (read_registers(m_impl->i2c_fd, kFt6336uPoint2Register, p2.data(), static_cast<int>(p2.size()))) {
            const int raw_x = ((p2[0] & 0x0F) << 8) | p2[1];
            const int raw_y = ((p2[2] & 0x0F) << 8) | p2[3];
            points.push_back(apply_rotation(raw_x, raw_y, m_impl->original_width, m_impl->original_height,
                                            m_impl->rotation_degrees));
        }
    }

    return points;
}
