// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/Touch.h"
#include "common/internal/string_utils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint8_t kFt6336uChipIdRegister = 0xA8;
constexpr uint8_t kFt6336uTouchCountRegister = 0x02;
constexpr uint8_t kFt6336uPoint1Register = 0x03;
constexpr uint8_t kFt6336uPoint2Register = 0x09;
constexpr uint8_t kFt6336uExpectedChipId = 0x11;

constexpr uint16_t kGt911CommandRegister = 0x8040;
constexpr uint16_t kGt911ConfigVersion = 0x8047;
constexpr uint16_t kGt911StatusRegister = 0x814E;
constexpr uint16_t kGt911Point1Register = 0x814F;
constexpr uint8_t kGt911MaxTouchPoints = 5;
constexpr uint8_t kGt911PointSize = 8;
constexpr uint8_t kGt911DefaultAddress = 0x5D;
constexpr int kFakeTouchDefaultPort = 8765;
constexpr int kFakeTouchDefaultHoldMs = 250;
constexpr int kFakeTouchDefaultStaleMs = 2000;

bool read_registers_8bit(int fd, uint8_t reg_start, uint8_t* buffer, int len) {
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

bool read_registers_16bit(int fd, uint16_t reg_start, uint8_t* buffer, int len) {
    if (fd < 0) {
        return false;
    }
    uint8_t addr[2] = {static_cast<uint8_t>(reg_start >> 8), static_cast<uint8_t>(reg_start & 0xFF)};
    if (write(fd, addr, sizeof(addr)) != static_cast<ssize_t>(sizeof(addr))) {
        return false;
    }
    if (::read(fd, buffer, len) != len) {
        return false;
    }
    return true;
}

bool write_register_16bit(int fd, uint16_t reg, uint8_t value) {
    if (fd < 0) {
        return false;
    }
    uint8_t buf[3] = {static_cast<uint8_t>(reg >> 8), static_cast<uint8_t>(reg & 0xFF), value};
    if (write(fd, buf, sizeof(buf)) != static_cast<ssize_t>(sizeof(buf))) {
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

std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool parse_int(const std::string& value, int& out) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(trimmed.c_str(), &end, 10);
    if (errno != 0 || end == trimmed.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_positive_int(const std::string& value, int& out) {
    int parsed = 0;
    if (!parse_int(value, parsed) || parsed <= 0) {
        return false;
    }
    out = parsed;
    return true;
}

struct FakeTouchEndpoint {
    std::string bind_ip = "0.0.0.0";
    int port = kFakeTouchDefaultPort;
    int hold_ms = kFakeTouchDefaultHoldMs;
    int stale_ms = kFakeTouchDefaultStaleMs;
};

bool is_default_i2c_spec(const std::string& spec) {
    return spec.empty() || spec.rfind("/dev/", 0) == 0;
}

void apply_fake_touch_endpoint_token(FakeTouchEndpoint& endpoint, const std::string& raw_token) {
    const std::string token = trim_copy(raw_token);
    if (token.empty()) {
        return;
    }

    const size_t eq = token.find('=');
    if (eq != std::string::npos) {
        const std::string key = visiong::to_lower_copy(trim_copy(token.substr(0, eq)));
        const std::string value = trim_copy(token.substr(eq + 1));
        int parsed = 0;
        if (key == "port" && parse_positive_int(value, parsed) && parsed <= 65535) {
            endpoint.port = parsed;
        } else if (key == "bind" && !value.empty()) {
            endpoint.bind_ip = value;
        } else if (key == "hold_ms" && parse_positive_int(value, parsed)) {
            endpoint.hold_ms = parsed;
        } else if (key == "stale_ms" && parse_positive_int(value, parsed)) {
            endpoint.stale_ms = parsed;
        }
        return;
    }

    int parsed_port = 0;
    if (parse_positive_int(token, parsed_port) && parsed_port <= 65535) {
        endpoint.port = parsed_port;
        return;
    }

    const size_t colon = token.rfind(':');
    if (colon != std::string::npos && colon + 1 < token.size()) {
        int port = 0;
        if (parse_positive_int(token.substr(colon + 1), port) && port <= 65535) {
            endpoint.bind_ip = token.substr(0, colon);
            endpoint.port = port;
        }
    }
}

FakeTouchEndpoint parse_fake_touch_endpoint(const std::string& endpoint_spec) {
    FakeTouchEndpoint endpoint;
    const std::string spec = trim_copy(endpoint_spec);
    if (is_default_i2c_spec(spec)) {
        return endpoint;
    }

    std::stringstream ss(spec);
    std::string token;
    bool saw_token = false;
    while (std::getline(ss, token, ',')) {
        saw_token = true;
        apply_fake_touch_endpoint_token(endpoint, token);
    }
    if (!saw_token) {
        apply_fake_touch_endpoint_token(endpoint, spec);
    }
    return endpoint;
}

enum class FakeTouchAction {
    Down,
    Move,
    Up,
    Cancel,
};

struct FakeTouchEvent {
    FakeTouchAction action = FakeTouchAction::Move;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

bool parse_fake_touch_packet(const char* data, ssize_t len, FakeTouchEvent& event) {
    if (!data || len <= 0) {
        return false;
    }

    const std::string text = trim_copy(std::string(data, static_cast<size_t>(len)));
    std::stringstream ss(text);
    std::vector<std::string> parts;
    std::string part;
    while (std::getline(ss, part, ',')) {
        parts.push_back(trim_copy(part));
    }

    if (parts.size() < 4) {
        return false;
    }
    const std::string header = visiong::to_lower_copy(parts[0]);
    if (header != "faketouch") {
        return false;
    }

    const std::string action = visiong::to_lower_copy(parts[1]);
    if (action == "down") {
        event.action = FakeTouchAction::Down;
    } else if (action == "move") {
        event.action = FakeTouchAction::Move;
    } else if (action == "up") {
        event.action = FakeTouchAction::Up;
    } else if (action == "cancel") {
        event.action = FakeTouchAction::Cancel;
    } else {
        return false;
    }

    if (!parse_int(parts[2], event.x) || !parse_int(parts[3], event.y)) {
        return false;
    }
    if (parts.size() >= 6) {
        parse_positive_int(parts[4], event.width);
        parse_positive_int(parts[5], event.height);
    }
    return true;
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

struct GT911_Touch::Impl {
    std::string i2c_bus_path;
    uint8_t device_address = kGt911DefaultAddress;
    int i2c_fd = -1;
    int original_width = 240;
    int original_height = 320;
    int rotation_degrees = 270;
    int max_x = 480;
    int max_y = 320;
    bool swap_xy = false;
};

struct FakeTouch::Impl {
    using Clock = std::chrono::steady_clock;

    int socket_fd = -1;
    std::string bind_ip = "0.0.0.0";
    int port = kFakeTouchDefaultPort;
    int hold_ms = kFakeTouchDefaultHoldMs;
    int stale_ms = kFakeTouchDefaultStaleMs;
    int original_width = 640;
    int original_height = 360;
    int rotation_degrees = 0;
    bool geometry_configured = false;
    bool pressed = false;
    bool release_pending = false;
    bool has_point = false;
    TouchPoint point{0, 0};
    Clock::time_point last_event_at = Clock::now();
    Clock::time_point release_at = Clock::now();
    std::deque<FakeTouchEvent> pending_events;

    void open_socket() {
        socket_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            throw std::runtime_error("[Touch] Fake touch failed to create UDP socket: " +
                                     std::string(std::strerror(errno)));
        }

        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        const int flags = fcntl(socket_fd, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (bind_ip.empty() || bind_ip == "*" || bind_ip == "0.0.0.0") {
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
        } else if (inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
            const std::string bad_ip = bind_ip;
            close_socket();
            throw std::runtime_error("[Touch] Fake touch invalid bind IP: " + bad_ip);
        }

        if (bind(socket_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            const int saved_errno = errno;
            close_socket();
            throw std::runtime_error("[Touch] Fake touch failed to bind UDP " + bind_ip + ":" +
                                     std::to_string(port) + ": " + std::strerror(saved_errno));
        }
    }

    void close_socket() {
        if (socket_fd >= 0) {
            close(socket_fd);
            socket_fd = -1;
        }
    }

    void apply_event(const FakeTouchEvent& event) {
        const auto now = Clock::now();
        if (!geometry_configured && event.width > 0 && event.height > 0) {
            original_width = event.width;
            original_height = event.height;
        }

        const int max_x = std::max(1, original_width) - 1;
        const int max_y = std::max(1, original_height) - 1;
        point.x = std::max(0, std::min(event.x, max_x));
        point.y = std::max(0, std::min(event.y, max_y));
        has_point = true;
        last_event_at = now;

        if (event.action == FakeTouchAction::Cancel) {
            pressed = false;
            release_pending = false;
            return;
        }

        pressed = true;
        if (event.action == FakeTouchAction::Up) {
            release_pending = true;
            release_at = now + std::chrono::milliseconds(std::max(1, hold_ms));
        } else {
            release_pending = false;
        }
    }

    void enqueue_event(const FakeTouchEvent& event) {
        if (event.action == FakeTouchAction::Move &&
            !pending_events.empty() &&
            pending_events.back().action == FakeTouchAction::Move) {
            pending_events.back() = event;
            return;
        }

        constexpr size_t kMaxPendingEvents = 16;
        while (pending_events.size() >= kMaxPendingEvents) {
            auto first_move = std::find_if(pending_events.begin(), pending_events.end(), [](const FakeTouchEvent& item) {
                return item.action == FakeTouchAction::Move;
            });
            if (first_move != pending_events.end()) {
                pending_events.erase(first_move);
            } else {
                pending_events.pop_front();
            }
        }
        pending_events.push_back(event);
    }

    void apply_one_pending_event() {
        if (pending_events.empty()) {
            return;
        }

        const FakeTouchEvent event = pending_events.front();
        pending_events.pop_front();
        apply_event(event);
    }

    void apply_all_pending_events() {
        while (!pending_events.empty()) {
            apply_one_pending_event();
        }
    }

    void update_timeouts() {
        const auto now = Clock::now();
        if (release_pending && now >= release_at) {
            pressed = false;
            release_pending = false;
            return;
        }

        if (pressed && !release_pending && stale_ms > 0 &&
            now - last_event_at > std::chrono::milliseconds(stale_ms)) {
            pressed = false;
        }
    }

    void drain_events() {
        if (socket_fd < 0) {
            return;
        }

        char buffer[256];
        while (true) {
            const ssize_t received = recvfrom(socket_fd, buffer, sizeof(buffer), 0, nullptr, nullptr);
            if (received < 0) {
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                break;
            }
            if (received == 0) {
                continue;
            }

            FakeTouchEvent event;
            if (parse_fake_touch_packet(buffer, received, event)) {
                enqueue_event(event);
            }
        }
    }

    void refresh_for_read() {
        drain_events();
        apply_one_pending_event();
        update_timeouts();
    }

    void refresh_for_state_query() {
        drain_events();
        apply_all_pending_events();
        update_timeouts();
    }
};

std::unique_ptr<TouchDevice> create_touch_device(const std::string& chip_model,
                                                 const std::string& i2c_bus_path) {
    const std::string model = visiong::to_lower_copy(chip_model);
    if (model == "ft6336u") {
        return std::make_unique<FT6336U_Touch>(i2c_bus_path, 0x38);
    }
    if (model == "gt911") {
        return std::make_unique<GT911_Touch>(i2c_bus_path, kGt911DefaultAddress, 320, 240, false);
    }
    if (model == "fake") {
        return std::make_unique<FakeTouch>(i2c_bus_path);
    }

    std::cerr << "[Touch Factory] Unsupported chip model: " << chip_model << std::endl;
    return nullptr;
}

FakeTouch::FakeTouch(const std::string& endpoint_spec)
    : m_impl(std::make_unique<Impl>()) {
    const FakeTouchEndpoint endpoint = parse_fake_touch_endpoint(endpoint_spec);
    m_impl->bind_ip = endpoint.bind_ip;
    m_impl->port = endpoint.port;
    m_impl->hold_ms = endpoint.hold_ms;
    m_impl->stale_ms = endpoint.stale_ms;
    m_impl->open_socket();
}

FakeTouch::~FakeTouch() {
    release();
}

void FakeTouch::configure_geometry(int original_width,
                                   int original_height,
                                   int rotation_degrees) {
    m_impl->original_width = std::max(1, original_width);
    m_impl->original_height = std::max(1, original_height);
    m_impl->rotation_degrees = rotation_degrees;
    m_impl->geometry_configured = true;
}

void FakeTouch::release() {
    m_impl->close_socket();
}

bool FakeTouch::is_pressed() {
    m_impl->refresh_for_state_query();
    return m_impl->pressed;
}

std::vector<TouchPoint> FakeTouch::get_touch_points() {
    std::vector<TouchPoint> points;
    m_impl->refresh_for_read();
    if (!m_impl->pressed || !m_impl->has_point) {
        return points;
    }

    points.push_back(apply_rotation(m_impl->point.x, m_impl->point.y,
                                    m_impl->original_width, m_impl->original_height,
                                    m_impl->rotation_degrees));
    return points;
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
    if (!read_registers_8bit(m_impl->i2c_fd, kFt6336uChipIdRegister, &chip_id, 1)) {
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
    if (!read_registers_8bit(m_impl->i2c_fd, kFt6336uTouchCountRegister, &touch_count, 1)) {
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
    if (!read_registers_8bit(m_impl->i2c_fd, kFt6336uTouchCountRegister, &status, 1)) {
        return points;
    }

    const uint8_t touch_count = status & 0x0F;
    if (touch_count == 0 || touch_count > 2) {
        return points;
    }

    std::vector<uint8_t> p1(6);
    if (read_registers_8bit(m_impl->i2c_fd, kFt6336uPoint1Register, p1.data(), static_cast<int>(p1.size()))) {
        const int raw_x = ((p1[0] & 0x0F) << 8) | p1[1];
        const int raw_y = ((p1[2] & 0x0F) << 8) | p1[3];
        points.push_back(apply_rotation(raw_x, raw_y, m_impl->original_width, m_impl->original_height,
                                        m_impl->rotation_degrees));
    }

    if (touch_count > 1) {
        std::vector<uint8_t> p2(6);
        if (read_registers_8bit(m_impl->i2c_fd, kFt6336uPoint2Register, p2.data(), static_cast<int>(p2.size()))) {
            const int raw_x = ((p2[0] & 0x0F) << 8) | p2[1];
            const int raw_y = ((p2[2] & 0x0F) << 8) | p2[3];
            points.push_back(apply_rotation(raw_x, raw_y, m_impl->original_width, m_impl->original_height,
                                            m_impl->rotation_degrees));
        }
    }

    return points;
}

GT911_Touch::GT911_Touch(const std::string& i2c_bus_path,
                         uint8_t device_address,
                         int max_x,
                         int max_y,
                         bool swap_xy)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->i2c_bus_path = i2c_bus_path;
    m_impl->device_address = device_address;
    m_impl->max_x = max_x > 0 ? max_x : 480;
    m_impl->max_y = max_y > 0 ? max_y : 320;
    m_impl->swap_xy = swap_xy;
    configure_geometry(480, 320, 0);

    m_impl->i2c_fd = open(m_impl->i2c_bus_path.c_str(), O_RDWR);
    if (m_impl->i2c_fd < 0) {
        throw std::runtime_error("[Touch] GT911 failed to open I2C bus " + m_impl->i2c_bus_path + ": " +
                                 std::strerror(errno));
    }

    if (ioctl(m_impl->i2c_fd, I2C_SLAVE, m_impl->device_address) < 0) {
        release();
        throw std::runtime_error("[Touch] GT911 failed to set I2C slave address: " +
                                 std::to_string(m_impl->device_address));
    }

    uint8_t config_version = 0;
    if (!read_registers_16bit(m_impl->i2c_fd, kGt911ConfigVersion, &config_version, 1)) {
        release();
        throw std::runtime_error("[Touch] GT911 not responding at address 0x" +
                                 std::to_string(m_impl->device_address) + " on " + m_impl->i2c_bus_path);
    }

    write_register_16bit(m_impl->i2c_fd, kGt911CommandRegister, 0x00);

    std::cout << "[Touch] GT911 initialized on " << m_impl->i2c_bus_path << " (0x" << std::hex
              << static_cast<int>(m_impl->device_address) << std::dec << ", config v"
              << static_cast<int>(config_version) << ", max " << m_impl->max_x << "x" << m_impl->max_y
              << ")" << std::endl;
}

GT911_Touch::~GT911_Touch() {
    release();
}

void GT911_Touch::configure_geometry(int original_width,
                                     int original_height,
                                     int rotation_degrees) {
    m_impl->original_width = original_width;
    m_impl->original_height = original_height;
    m_impl->rotation_degrees = rotation_degrees;
}

void GT911_Touch::release() {
    if (m_impl->i2c_fd >= 0) {
        close(m_impl->i2c_fd);
        m_impl->i2c_fd = -1;
    }
}

bool GT911_Touch::is_pressed() {
    uint8_t status = 0;
    if (!read_registers_16bit(m_impl->i2c_fd, kGt911StatusRegister, &status, 1)) {
        return false;
    }
    if (status & 0x80) {
        return false;
    }
    return (status & 0x0F) > 0;
}

std::vector<TouchPoint> GT911_Touch::get_touch_points() {
    std::vector<TouchPoint> points;
    if (m_impl->i2c_fd < 0) {
        return points;
    }

    uint8_t status = 0;
    if (!read_registers_16bit(m_impl->i2c_fd, kGt911StatusRegister, &status, 1)) {
        return points;
    }
    if (status & 0x80) {
        return points;
    }

    const uint8_t touch_count = std::min<uint8_t>(status & 0x0F, kGt911MaxTouchPoints);
    if (touch_count == 0) {
        return points;
    }

    for (uint8_t i = 0; i < touch_count; ++i) {
        const uint16_t point_reg = static_cast<uint16_t>(kGt911Point1Register + i * kGt911PointSize);
        uint8_t buf[kGt911PointSize] = {};
        if (!read_registers_16bit(m_impl->i2c_fd, point_reg, buf, sizeof(buf))) {
            continue;
        }

        const uint16_t raw_x = static_cast<uint16_t>(buf[1]) | (static_cast<uint16_t>(buf[2]) << 8);
        const uint16_t raw_y = static_cast<uint16_t>(buf[3]) | (static_cast<uint16_t>(buf[4]) << 8);

        int display_x = 0;
        int display_y = 0;
        if (m_impl->swap_xy) {
            display_x = (raw_y * m_impl->original_width) / m_impl->max_y;
            display_y = (raw_x * m_impl->original_height) / m_impl->max_x;
        } else {
            display_x = (raw_x * m_impl->original_width) / m_impl->max_x;
            display_y = (raw_y * m_impl->original_height) / m_impl->max_y;
        }

        int clamped_x = std::max(0, std::min(display_x, m_impl->original_width - 1));
        int clamped_y = std::max(0, std::min(display_y, m_impl->original_height - 1));

        points.push_back(apply_rotation(clamped_x, clamped_y,
                                        m_impl->original_width, m_impl->original_height,
                                        m_impl->rotation_degrees));
    }

    write_register_16bit(m_impl->i2c_fd, kGt911StatusRegister, 0x00);

    return points;
}
