// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/pinmux.h"

#include "internal/rv1106_pinmux_table.h"
#include "visiong/uapi/visiong_hw.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

#include <fcntl.h>
#include <linux/gpio.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <unistd.h>

namespace visiong::pinmux {

namespace {

constexpr off_t kIocBaseAddr = 0xff538000;
constexpr size_t kIocMapSize = 0x40000;

constexpr off_t kPmuiocBaseAddr = 0xff388000;
constexpr size_t kPmuiocMapSize = 0x1000;

constexpr size_t kGpioMapSize = 0x1000;
constexpr std::array<off_t, 5> kGpioBaseAddrs{{
    static_cast<off_t>(0xff380000u),
    static_cast<off_t>(0xff530000u),
    static_cast<off_t>(0xff540000u),
    static_cast<off_t>(0xff550000u),
    static_cast<off_t>(0xff560000u)}};

struct BankLayout {
    int pin_count = 0;
    bool use_pmuioc = false;
    std::array<int, 4> iomux_offsets{};
};

constexpr std::array<BankLayout, 5> kRv1106IomuxLayout{{
    {32, true, {0x00000, 0x00008, 0x00010, 0x00018}},
    {32, false, {0x00000, 0x00008, 0x00010, 0x00018}},
    // RV1106 vendor kernel table only defines GPIO2A/B mux registers. / RV1106 厂商内核表只定义了 GPIO2A/B 的 mux 寄存器。
    // GPIO2C/D are treated as GPIO-only here to avoid unsafe alias writes.
    {32, false, {0x10020, 0x10028, -1, -1}},
    {32, false, {0x20040, 0x20048, 0x20050, 0x20058}},
    {24, false, {0x30000, 0x30008, 0x30010, -1}},
}};

constexpr std::array<uint32_t, 5> kRv1106DriveBaseOffsets{{
    0x00010,   // GPIO0 (PMUIOC)
    0x00080,   // GPIO1
    0x100c0,   // GPIO2
    0x20100,   // GPIO3
    0x30020,   // GPIO4
}};

constexpr std::array<uint32_t, 5> kRv1106PullBaseOffsets{{
    0x00038,   // GPIO0 (PMUIOC)
    0x001c0,   // GPIO1
    0x101d0,   // GPIO2
    0x201e0,   // GPIO3
    0x30070,   // GPIO4
}};

constexpr std::array<uint32_t, 5> kRv1106SchmittBaseOffsets{{
    0x00040,   // GPIO0 (PMUIOC)
    0x00280,   // GPIO1
    0x10290,   // GPIO2
    0x202a0,   // GPIO3
    0x300a0,   // GPIO4
}};

inline std::string join_options(const std::vector<PinAltFunction>& entries) {
    std::string text;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) {
            text += ", ";
        }
        text += entries[i].function;
        text += "(mux=";
        text += std::to_string(entries[i].mux);
        text += ", group=";
        text += entries[i].group;
        text += ")";
    }
    return text;
}

inline std::string join_csv(const std::vector<std::string>& items) {
    std::string text;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            text += ", ";
        }
        text += items[i];
    }
    return text;
}

inline std::string trim_copy(std::string text) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

inline void append_unique(std::vector<std::string>* items, const std::string& value) {
    if (!items || value.empty()) {
        return;
    }
    if (std::find(items->begin(), items->end(), value) == items->end()) {
        items->push_back(value);
    }
}

struct SavedPinMuxState {
    int bank = 0;
    int pin = 0;
    uint32_t mux = 0;
    std::string mux_owner;
    std::string gpio_owner;
    std::string function;
    std::string group;
    size_t ref_count = 0;
};

using SavedPinMuxKey = std::pair<int, int>;

std::mutex& saved_pinmux_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<SavedPinMuxKey, SavedPinMuxState>& saved_pinmux_states() {
    static std::map<SavedPinMuxKey, SavedPinMuxState> states;
    return states;
}

bool& saved_pinmux_loaded() {
    static bool loaded = false;
    return loaded;
}

std::vector<std::filesystem::path> saved_pinmux_file_candidates() {
    std::vector<std::filesystem::path> paths;
    const char* env_path = std::getenv("VISIONG_PINMUX_STATE_FILE");
    if (env_path && *env_path) {
        paths.emplace_back(env_path);
    }
    paths.emplace_back("/var/lib/visiong/pinmux_restore.tsv");
    paths.emplace_back("/tmp/visiong_pinmux_restore.tsv");
    return paths;
}

std::filesystem::path active_saved_pinmux_file() {
    const auto candidates = saved_pinmux_file_candidates();
    for (const auto& path : candidates) {
        const auto parent = path.parent_path();
        if (parent.empty()) {
            return path;
        }
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (!ec) {
            return path;
        }
    }
    return "/tmp/visiong_pinmux_restore.tsv";
}

void load_saved_pinmux_states_locked() {
    if (saved_pinmux_loaded()) {
        return;
    }
    saved_pinmux_loaded() = true;

    auto& states = saved_pinmux_states();
    for (const auto& path : saved_pinmux_file_candidates()) {
        std::ifstream in(path);
        if (!in) {
            continue;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream input(line);
            std::string tag;
            SavedPinMuxState state;
            if (!(input >> tag) || tag != "pin") {
                continue;
            }
            if (!(input >> state.bank >> state.pin >> state.mux >>
                  std::quoted(state.mux_owner) >>
                  std::quoted(state.gpio_owner) >>
                  std::quoted(state.function) >>
                  std::quoted(state.group))) {
                continue;
            }
            states[{state.bank, state.pin}] = state;
        }
    }
}

void save_saved_pinmux_states_locked() {
    const auto active_path = active_saved_pinmux_file();
    auto& states = saved_pinmux_states();
    std::error_code ec;

    if (states.empty()) {
        for (const auto& path : saved_pinmux_file_candidates()) {
            std::filesystem::remove(path, ec);
        }
        return;
    }

    std::filesystem::create_directories(active_path.parent_path(), ec);
    const auto tmp_path = std::filesystem::path(active_path.string() + ".tmp");
    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out) {
            std::cerr << "[PinMux] Warning: failed to write persistent pinmux state: "
                      << tmp_path << std::endl;
            return;
        }
        out << "# visiong pinmux restore state v1\n";
        for (const auto& kv : states) {
            const auto& state = kv.second;
            out << "pin " << state.bank << ' ' << state.pin << ' ' << state.mux << ' '
                << std::quoted(state.mux_owner) << ' '
                << std::quoted(state.gpio_owner) << ' '
                << std::quoted(state.function) << ' '
                << std::quoted(state.group) << '\n';
        }
    }

    std::filesystem::rename(tmp_path, active_path, ec);
    if (ec) {
        std::filesystem::copy_file(tmp_path, active_path, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp_path, ec);
    }

    for (const auto& path : saved_pinmux_file_candidates()) {
        if (path != active_path) {
            std::filesystem::remove(path, ec);
        }
    }
}

class ScopedFd {
public:
    explicit ScopedFd(int fd = -1) : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ScopedFd(ScopedFd&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }
    ScopedFd& operator=(ScopedFd&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        other.fd_ = -1;
        return *this;
    }
    int get() const noexcept { return fd_; }
    explicit operator bool() const noexcept { return fd_ >= 0; }

private:
    int fd_ = -1;
};

inline std::string read_file_flatten_nuls(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (char& c : text) {
        if (c == '\0') {
            c = ' ';
        }
    }
    return trim_copy(text);
}

inline uint32_t read_be32_local(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

inline std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::string basename_from_symlink(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return "";
    }
    std::filesystem::path target = std::filesystem::read_symlink(path, ec);
    if (ec || target.empty()) {
        target = std::filesystem::canonical(path, ec);
    }
    if (ec || target.empty()) {
        return "";
    }
    return target.filename().string();
}

inline bool read_dt_reg(const std::filesystem::path& node, uint64_t* base, uint64_t* size) {
    const std::vector<uint8_t> reg = read_binary_file(node / "reg");
    if (reg.size() < 8 || !base || !size) {
        return false;
    }
    *base = read_be32_local(reg.data());
    *size = read_be32_local(reg.data() + 4);
    return *base != 0 && *size != 0;
}

inline std::string dt_node_from_alias(const std::string& alias) {
    for (const auto& root : {std::filesystem::path("/sys/firmware/devicetree/base/aliases"),
                             std::filesystem::path("/proc/device-tree/aliases")}) {
        const std::string target = read_file_flatten_nuls(root / alias);
        if (!target.empty()) {
            return target;
        }
    }
    return "";
}

inline std::filesystem::path dt_node_path_from_target(const std::string& target) {
    if (target.empty() || target.front() != '/') {
        return {};
    }
    for (const auto& root : {std::filesystem::path("/sys/firmware/devicetree/base"),
                             std::filesystem::path("/proc/device-tree")}) {
        std::filesystem::path node = root;
        std::string suffix = target.substr(1);
        while (!suffix.empty() && suffix.front() == '/') {
            suffix.erase(suffix.begin());
        }
        node /= suffix;
        std::error_code ec;
        if (std::filesystem::exists(node, ec)) {
            return node;
        }
    }
    return {};
}

inline std::string platform_suffix_for_alias(const std::string& alias) {
    if (alias.rfind("spi", 0) == 0) {
        return ".spi";
    }
    if (alias.rfind("i2c", 0) == 0) {
        return ".i2c";
    }
    if (alias.rfind("uart", 0) == 0 || alias.rfind("serial", 0) == 0) {
        return ".serial";
    }
    if (alias.rfind("pwm", 0) == 0) {
        return ".pwm";
    }
    if (alias.rfind("gpio", 0) == 0) {
        return ".gpio";
    }
    if (alias == "dmac" || alias.rfind("dma", 0) == 0) {
        return ".dma-controller";
    }
    return "";
}

inline std::string device_name_from_base(uint64_t base, const std::string& alias) {
    if (base == 0) {
        return "";
    }
    std::ostringstream ss;
    ss << std::hex << std::nouppercase << base << platform_suffix_for_alias(alias);
    return ss.str();
}

inline PeripheralRegisterInfo known_peripheral_info(const std::string& alias) {
    struct Known {
        const char* alias;
        uint64_t base;
        uint64_t size;
        const char* device;
        const char* compatible;
    };
    static const std::array<Known, 42> kKnown{{
        {"spi0", 0xff500000, 0x1000, "ff500000.spi", "rockchip,rv1106-spi"},
        {"spi1", 0xff510000, 0x1000, "ff510000.spi", "rockchip,rv1106-spi"},
        {"sfc", 0xffac0000, 0x4000, "ffac0000.spi", "rockchip,sfc"},
        {"uart0", 0xff4a0000, 0x100, "ff4a0000.serial", "rockchip,rv1106-uart"},
        {"uart1", 0xff4b0000, 0x100, "ff4b0000.serial", "rockchip,rv1106-uart"},
        {"uart2", 0xff4c0000, 0x100, "ff4c0000.serial", "rockchip,rv1106-uart"},
        {"uart3", 0xff4d0000, 0x100, "ff4d0000.serial", "rockchip,rv1106-uart"},
        {"uart4", 0xff4e0000, 0x100, "ff4e0000.serial", "rockchip,rv1106-uart"},
        {"uart5", 0xff4f0000, 0x100, "ff4f0000.serial", "rockchip,rv1106-uart"},
        {"serial0", 0xff4a0000, 0x100, "ff4a0000.serial", "rockchip,rv1106-uart"},
        {"serial1", 0xff4b0000, 0x100, "ff4b0000.serial", "rockchip,rv1106-uart"},
        {"serial2", 0xff4c0000, 0x100, "ff4c0000.serial", "rockchip,rv1106-uart"},
        {"serial3", 0xff4d0000, 0x100, "ff4d0000.serial", "rockchip,rv1106-uart"},
        {"serial4", 0xff4e0000, 0x100, "ff4e0000.serial", "rockchip,rv1106-uart"},
        {"serial5", 0xff4f0000, 0x100, "ff4f0000.serial", "rockchip,rv1106-uart"},
        {"i2c0", 0xff310000, 0x1000, "ff310000.i2c", "rockchip,rv1106-i2c"},
        {"i2c1", 0xff320000, 0x1000, "ff320000.i2c", "rockchip,rv1106-i2c"},
        {"i2c2", 0xff450000, 0x1000, "ff450000.i2c", "rockchip,rv1106-i2c"},
        {"i2c3", 0xff460000, 0x1000, "ff460000.i2c", "rockchip,rv1106-i2c"},
        {"i2c4", 0xff470000, 0x1000, "ff470000.i2c", "rockchip,rv1106-i2c"},
        {"pwm0", 0xff350000, 0x10, "ff350000.pwm", "rockchip,rk3328-pwm"},
        {"pwm1", 0xff350010, 0x10, "ff350010.pwm", "rockchip,rk3328-pwm"},
        {"pwm2", 0xff350020, 0x10, "ff350020.pwm", "rockchip,rk3328-pwm"},
        {"pwm3", 0xff350030, 0x10, "ff350030.pwm", "rockchip,rk3328-pwm"},
        {"pwm4", 0xff360000, 0x10, "ff360000.pwm", "rockchip,rk3328-pwm"},
        {"pwm5", 0xff360010, 0x10, "ff360010.pwm", "rockchip,rk3328-pwm"},
        {"pwm6", 0xff360020, 0x10, "ff360020.pwm", "rockchip,rk3328-pwm"},
        {"pwm7", 0xff360030, 0x10, "ff360030.pwm", "rockchip,rk3328-pwm"},
        {"pwm8", 0xff490000, 0x10, "ff490000.pwm", "rockchip,rk3328-pwm"},
        {"pwm9", 0xff490010, 0x10, "ff490010.pwm", "rockchip,rk3328-pwm"},
        {"pwm10", 0xff490020, 0x10, "ff490020.pwm", "rockchip,rk3328-pwm"},
        {"pwm11", 0xff490030, 0x10, "ff490030.pwm", "rockchip,rk3328-pwm"},
        {"cru", 0xff3a0000, 0x20000, "ff3a0000.clock-controller", "rockchip,rv1106-cru"},
        {"dmac", 0xff420000, 0x4000, "ff420000.dma-controller", "arm,pl330"},
        {"gic", 0xff1f1000, 0x1000, "ff1f0000.interrupt-controller", "arm,gic-400"},
        {"gicd", 0xff1f1000, 0x1000, "ff1f0000.interrupt-controller", "arm,gic-400"},
        {"gpio0", 0xff380000, 0x8000, "ff380000.gpio", "rockchip,gpio-bank"},
        {"gpio1", 0xff530000, 0x8000, "ff530000.gpio", "rockchip,gpio-bank"},
        {"gpio2", 0xff540000, 0x8000, "ff540000.gpio", "rockchip,gpio-bank"},
        {"gpio3", 0xff550000, 0x8000, "ff550000.gpio", "rockchip,gpio-bank"},
        {"gpio4", 0xff560000, 0x8000, "ff560000.gpio", "rockchip,gpio-bank"},
        {"gpio", 0xff538000ULL, kIocMapSize, "pinctrl", "rockchip,rv1106-pinctrl"},
    }};
    for (const auto& item : kKnown) {
        if (alias == item.alias) {
            PeripheralRegisterInfo info;
            info.available = true;
            info.request = alias;
            info.alias = alias;
            info.base_addr = item.base;
            info.size = item.size;
            info.device = item.device;
            info.compatible = item.compatible;
            info.note = "Resolved from built-in RV1103/RV1106 register table.";
            return info;
        }
    }
    return {};
}

inline PeripheralRegisterInfo resolve_peripheral_register_info(const std::string& request) {
    PeripheralRegisterInfo info;
    info.request = request;
    std::string token = trim_copy(request);
    std::replace(token.begin(), token.end(), '\\', '/');
    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (token.rfind("spidev", 0) == 0) {
        token = "spi" + token.substr(6);
    }
    const size_t dot = token.find('.');
    if (dot != std::string::npos && std::all_of(token.begin() + dot + 1, token.end(), [](unsigned char c) {
            return std::isdigit(c);
        })) {
        token = token.substr(0, dot);
    }
    info.alias = token;

    const std::string target = dt_node_from_alias(token);
    const std::filesystem::path dt_node = dt_node_path_from_target(target);
    if (!dt_node.empty() && read_dt_reg(dt_node, &info.base_addr, &info.size)) {
        info.available = true;
        info.compatible = read_file_flatten_nuls(dt_node / "compatible");
        info.device = device_name_from_base(info.base_addr, token);
        info.note = "Resolved from device-tree alias " + token + ".";
    } else {
        info = known_peripheral_info(token);
        if (!info.available) {
            info.request = request;
            info.alias = token;
            info.note = "Unknown peripheral alias. Expected names like spi0, uart4, i2c3, pwm0..pwm11, gpio0..gpio4, cru, dmac, gic.";
            return info;
        }
        info.request = request;
    }

    if (!info.device.empty()) {
        info.bound_driver = basename_from_symlink("/sys/bus/platform/devices/" + info.device + "/driver");
    }
    return info;
}

}  // namespace

RegisterBlock::RegisterBlock(const std::string& peripheral, size_t map_size) {
    info_ = resolve_peripheral_register_info(peripheral);
    if (!info_.available || info_.base_addr == 0 || info_.size == 0) {
        throw std::invalid_argument("Unknown peripheral register block '" + peripheral + "': " + info_.note);
    }

    const long page_size_long = ::sysconf(_SC_PAGESIZE);
    const uint64_t page_size = page_size_long > 0 ? static_cast<uint64_t>(page_size_long) : 4096;
    map_base_ = info_.base_addr & ~(page_size - 1);
    page_offset_ = static_cast<size_t>(info_.base_addr - map_base_);
    const size_t requested_size = map_size > 0 ? map_size : static_cast<size_t>(info_.size);
    map_size_ = page_offset_ + requested_size;
    map_size_ = static_cast<size_t>((map_size_ + page_size - 1) & ~(page_size - 1));

    fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open /dev/mem for " + peripheral + ": " + std::strerror(errno));
    }
    map_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, static_cast<off_t>(map_base_));
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        const std::string err = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to mmap " + peripheral + " at 0x" +
                                  [&]() {
                                      std::ostringstream ss;
                                      ss << std::hex << info_.base_addr;
                                      return ss.str();
                                  }() + ": " + err);
    }
}

RegisterBlock::~RegisterBlock() {
    close();
}

RegisterBlock::RegisterBlock(RegisterBlock&& other) noexcept {
    *this = std::move(other);
}

RegisterBlock& RegisterBlock::operator=(RegisterBlock&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    info_ = std::move(other.info_);
    fd_ = other.fd_;
    map_ = other.map_;
    map_base_ = other.map_base_;
    map_size_ = other.map_size_;
    page_offset_ = other.page_offset_;
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_base_ = 0;
    other.map_size_ = 0;
    other.page_offset_ = 0;
    return *this;
}

bool RegisterBlock::is_open() const noexcept {
    return fd_ >= 0 && map_ != nullptr;
}

void RegisterBlock::close() {
    if (map_) {
        ::munmap(map_, map_size_);
        map_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

const PeripheralRegisterInfo& RegisterBlock::info() const noexcept {
    return info_;
}

uint8_t RegisterBlock::read8(uint32_t offset) const {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint8_t) > map_size_) {
        throw std::out_of_range("Register read offset out of mapped range.");
    }
    volatile uint8_t* reg = reinterpret_cast<volatile uint8_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    return *reg;
}

void RegisterBlock::write8(uint32_t offset, uint8_t value) {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint8_t) > map_size_) {
        throw std::out_of_range("Register write offset out of mapped range.");
    }
    volatile uint8_t* reg = reinterpret_cast<volatile uint8_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    *reg = value;
    __sync_synchronize();
}

void RegisterBlock::write8_repeat(uint32_t offset, const void* data, size_t size) {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint8_t) > map_size_) {
        throw std::out_of_range("Register write offset out of mapped range.");
    }
    if (size == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("Register write buffer is null.");
    }
    volatile uint8_t* reg = reinterpret_cast<volatile uint8_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        *reg = bytes[i];
    }
    __sync_synchronize();
}

std::vector<uint8_t> RegisterBlock::read8_repeat(uint32_t offset, size_t size) const {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint8_t) > map_size_) {
        throw std::out_of_range("Register read offset out of mapped range.");
    }
    std::vector<uint8_t> out(size);
    if (size == 0) {
        return out;
    }
    volatile uint8_t* reg = reinterpret_cast<volatile uint8_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    for (size_t i = 0; i < size; ++i) {
        out[i] = *reg;
    }
    return out;
}

uint16_t RegisterBlock::read16(uint32_t offset) const {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint16_t) > map_size_) {
        throw std::out_of_range("Register read offset out of mapped range.");
    }
    volatile uint16_t* reg = reinterpret_cast<volatile uint16_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    return *reg;
}

void RegisterBlock::write16(uint32_t offset, uint16_t value) {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint16_t) > map_size_) {
        throw std::out_of_range("Register write offset out of mapped range.");
    }
    volatile uint16_t* reg = reinterpret_cast<volatile uint16_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    *reg = value;
    __sync_synchronize();
}

uint32_t RegisterBlock::read32(uint32_t offset) const {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint32_t) > map_size_) {
        throw std::out_of_range("Register read offset out of mapped range.");
    }
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    return *reg;
}

void RegisterBlock::write32(uint32_t offset, uint32_t value) {
    if (!is_open()) {
        throw std::runtime_error("Register block is closed.");
    }
    if (page_offset_ + offset + sizeof(uint32_t) > map_size_) {
        throw std::out_of_range("Register write offset out of mapped range.");
    }
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(map_) + page_offset_ + offset);
    *reg = value;
    __sync_synchronize();
}

void RegisterBlock::update32(uint32_t offset, uint32_t mask, uint32_t value) {
    const uint32_t current = read32(offset);
    write32(offset, (current & ~mask) | (value & mask));
}

void RegisterBlock::set_bits(uint32_t offset, uint32_t mask) {
    update32(offset, mask, mask);
}

void RegisterBlock::clear_bits(uint32_t offset, uint32_t mask) {
    update32(offset, mask, 0);
}

size_t Controller::PinKeyHash::operator()(const PinKey& key) const noexcept {
    return static_cast<size_t>((key.bank << 8) ^ key.pin);
}

Controller::Controller() {
    hw_fd_ = ::open("/dev/visiong-hw", O_RDWR | O_CLOEXEC);
    if (hw_fd_ >= 0) {
        return;
    }

    fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open /dev/mem: " + std::string(std::strerror(errno)));
    }

    ioc_map_ = ::mmap(nullptr, kIocMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, kIocBaseAddr);
    if (ioc_map_ == MAP_FAILED) {
        ioc_map_ = nullptr;
        const std::string err = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to mmap IOC (0xff538000): " + err);
    }

    pmuioc_map_ = ::mmap(nullptr, kPmuiocMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, kPmuiocBaseAddr);
    if (pmuioc_map_ == MAP_FAILED) {
        pmuioc_map_ = nullptr;
        ::munmap(ioc_map_, kIocMapSize);
        ioc_map_ = nullptr;
        const std::string err = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to mmap PMUIOC (0xff388000): " + err);
    }
}

Controller::~Controller() {
    close();
}

bool Controller::is_open() const noexcept {
    return hw_fd_ >= 0 || (fd_ >= 0 && ioc_map_ != nullptr && pmuioc_map_ != nullptr);
}

void Controller::close() {
    std::vector<PinKey> claims;

    {
        std::lock_guard<std::mutex> guard(lock_);

        for (auto& kv : gpio_line_handles_) {
            if (kv.second.fd >= 0) {
                ::close(kv.second.fd);
                kv.second.fd = -1;
            }
        }
        gpio_line_handles_.clear();
        claims.swap(pin_claims_);
    }

    release_pin_claims(claims);

    {
        std::lock_guard<std::mutex> guard(lock_);

        if (pmuioc_map_ != nullptr) {
            ::munmap(pmuioc_map_, kPmuiocMapSize);
            pmuioc_map_ = nullptr;
        }
        if (ioc_map_ != nullptr) {
            ::munmap(ioc_map_, kIocMapSize);
            ioc_map_ = nullptr;
        }
        for (void*& map : gpio_maps_) {
            if (map != nullptr) {
                ::munmap(map, kGpioMapSize);
                map = nullptr;
            }
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (hw_fd_ >= 0) {
            ::close(hw_fd_);
            hw_fd_ = -1;
        }
    }
}

uint32_t Controller::hw_read32_unsafe(uint32_t block, uint32_t offset) const {
    if (hw_fd_ < 0) {
        throw std::runtime_error("visiong-hw register backend is unavailable.");
    }
    visiong_hw_reg_access access{};
    access.size = sizeof(access);
    access.block = block;
    access.offset = offset;
    if (::ioctl(hw_fd_, VISIONG_HW_REG_READ, &access) < 0) {
        throw std::runtime_error("visiong-hw register read failed: " + std::string(std::strerror(errno)));
    }
    return access.value;
}

void Controller::hw_write32_unsafe(uint32_t block,
                                   uint32_t offset,
                                   uint32_t value,
                                   uint32_t mask,
                                   bool hiword) const {
    if (hw_fd_ < 0) {
        throw std::runtime_error("visiong-hw register backend is unavailable.");
    }
    visiong_hw_reg_access access{};
    access.size = sizeof(access);
    access.block = block;
    access.offset = offset;
    access.value = value;
    access.mask = mask;
    access.flags = hiword ? VISIONG_HW_REG_FLAG_HIWORD_UPDATE : 0;
    if (::ioctl(hw_fd_, VISIONG_HW_REG_WRITE, &access) < 0) {
        throw std::runtime_error("visiong-hw register write failed: " + std::string(std::strerror(errno)));
    }
}

uint32_t Controller::ioc_read32_unsafe(bool use_pmuioc, uint32_t offset) const {
    if (hw_fd_ >= 0) {
        return hw_read32_unsafe(use_pmuioc ? VISIONG_HW_REG_BLOCK_PMUIOC : VISIONG_HW_REG_BLOCK_IOC, offset);
    }
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(use_pmuioc ? pmuioc_map_ : ioc_map_) + offset);
    return *reg;
}

void Controller::ioc_hiword_update_unsafe(bool use_pmuioc, uint32_t offset, uint32_t mask, uint32_t value) const {
    if (hw_fd_ >= 0) {
        hw_write32_unsafe(use_pmuioc ? VISIONG_HW_REG_BLOCK_PMUIOC : VISIONG_HW_REG_BLOCK_IOC,
                          offset,
                          value & mask,
                          mask,
                          true);
        return;
    }
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(use_pmuioc ? pmuioc_map_ : ioc_map_) + offset);
    *reg = (mask << 16) | (value & mask);
}

uint32_t Controller::gpio_reg_read32_unsafe(int bank, uint32_t offset) const {
    if (bank < 0 || bank >= static_cast<int>(kGpioBaseAddrs.size())) {
        throw std::invalid_argument("GPIO bank out of range: " + std::to_string(bank));
    }
    if (hw_fd_ >= 0) {
        return hw_read32_unsafe(VISIONG_HW_REG_BLOCK_GPIO0 + static_cast<uint32_t>(bank), offset);
    }
    void* map = gpio_map_bank_unsafe(bank);
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(map) + offset);
    return *reg;
}

void Controller::gpio_reg_hiword_update_unsafe(int bank, uint32_t offset, uint32_t mask, uint32_t value) const {
    if (bank < 0 || bank >= static_cast<int>(kGpioBaseAddrs.size())) {
        throw std::invalid_argument("GPIO bank out of range: " + std::to_string(bank));
    }
    if (hw_fd_ >= 0) {
        hw_write32_unsafe(VISIONG_HW_REG_BLOCK_GPIO0 + static_cast<uint32_t>(bank),
                          offset,
                          value & mask,
                          mask,
                          true);
        return;
    }
    void* map = gpio_map_bank_unsafe(bank);
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(map) + offset);
    *reg = (mask << 16) | (value & mask);
}

Controller::ResolvedField Controller::resolve_field(int bank, int pin) const {
    if (bank < 0 || bank >= static_cast<int>(kRv1106IomuxLayout.size())) {
        throw std::invalid_argument("bank out of range: " + std::to_string(bank));
    }

    const BankLayout& bank_layout = kRv1106IomuxLayout[bank];
    if (pin < 0 || pin >= bank_layout.pin_count) {
        throw std::invalid_argument("pin out of range for gpio" + std::to_string(bank) + ": " + std::to_string(pin));
    }

    const int iomux_group = pin / 8;
    if (iomux_group < 0 || iomux_group >= 4) {
        throw std::invalid_argument("invalid iomux group for pin: " + std::to_string(pin));
    }

    ResolvedField field;
    field.use_pmuioc = bank_layout.use_pmuioc;
    field.info.domain = bank_layout.use_pmuioc ? "pmuioc" : "ioc";
    field.info.base_addr = bank_layout.use_pmuioc ? static_cast<uint32_t>(kPmuiocBaseAddr) : static_cast<uint32_t>(kIocBaseAddr);
    field.info.width = 4;
    field.info.mask = 0xF;

    const int group_offset = bank_layout.iomux_offsets[iomux_group];
    if (group_offset < 0) {
        field.info.gpio_only = true;
        return field;
    }

    const uint32_t reg_offset = static_cast<uint32_t>(group_offset + (((pin % 8) >= 4) ? 0x4 : 0x0));
    const uint8_t bit = static_cast<uint8_t>((pin % 4) * 4);

    const size_t map_size = bank_layout.use_pmuioc ? kPmuiocMapSize : kIocMapSize;
    if (static_cast<size_t>(reg_offset + sizeof(uint32_t)) > map_size) {
        throw std::runtime_error("computed register offset out of range: 0x" + std::to_string(reg_offset));
    }

    field.info.reg_offset = reg_offset;
    field.info.absolute_addr = field.info.base_addr + reg_offset;
    field.info.bit = bit;
    return field;
}

Controller::ResolvedField Controller::resolve_drive_field(int bank, int pin) const {
    const ResolvedField iomux_field = resolve_field(bank, pin);
    (void)iomux_field;

    ResolvedField field;
    field.use_pmuioc = (bank == 0);
    field.info.domain = field.use_pmuioc ? "pmuioc" : "ioc";
    field.info.base_addr = field.use_pmuioc ? static_cast<uint32_t>(kPmuiocBaseAddr) : static_cast<uint32_t>(kIocBaseAddr);
    field.info.width = 8;
    field.info.mask = 0xFF;

    const uint32_t bank_base = kRv1106DriveBaseOffsets[static_cast<size_t>(bank)];
    const uint32_t reg_offset = bank_base + static_cast<uint32_t>((pin / 2) * 4);
    const uint8_t bit = static_cast<uint8_t>((pin % 2) * 8);

    const size_t map_size = field.use_pmuioc ? kPmuiocMapSize : kIocMapSize;
    if (static_cast<size_t>(reg_offset + sizeof(uint32_t)) > map_size) {
        throw std::runtime_error("computed drive register offset out of range: 0x" + std::to_string(reg_offset));
    }

    field.info.reg_offset = reg_offset;
    field.info.absolute_addr = field.info.base_addr + reg_offset;
    field.info.bit = bit;
    return field;
}

Controller::ResolvedField Controller::resolve_pull_field(int bank, int pin) const {
    const ResolvedField iomux_field = resolve_field(bank, pin);
    (void)iomux_field;

    ResolvedField field;
    field.use_pmuioc = (bank == 0);
    field.info.domain = field.use_pmuioc ? "pmuioc" : "ioc";
    field.info.base_addr = field.use_pmuioc ? static_cast<uint32_t>(kPmuiocBaseAddr) : static_cast<uint32_t>(kIocBaseAddr);
    field.info.width = 2;
    field.info.mask = 0x3;

    const uint32_t bank_base = kRv1106PullBaseOffsets[static_cast<size_t>(bank)];
    const uint32_t reg_offset = bank_base + static_cast<uint32_t>((pin / 8) * 4);
    const uint8_t bit = static_cast<uint8_t>((pin % 8) * 2);

    const size_t map_size = field.use_pmuioc ? kPmuiocMapSize : kIocMapSize;
    if (static_cast<size_t>(reg_offset + sizeof(uint32_t)) > map_size) {
        throw std::runtime_error("computed pull register offset out of range: 0x" + std::to_string(reg_offset));
    }

    field.info.reg_offset = reg_offset;
    field.info.absolute_addr = field.info.base_addr + reg_offset;
    field.info.bit = bit;
    return field;
}

Controller::ResolvedField Controller::resolve_schmitt_field(int bank, int pin) const {
    const ResolvedField iomux_field = resolve_field(bank, pin);
    (void)iomux_field;

    ResolvedField field;
    field.use_pmuioc = (bank == 0);
    field.info.domain = field.use_pmuioc ? "pmuioc" : "ioc";
    field.info.base_addr = field.use_pmuioc ? static_cast<uint32_t>(kPmuiocBaseAddr) : static_cast<uint32_t>(kIocBaseAddr);
    field.info.width = 1;
    field.info.mask = 0x1;

    const uint32_t bank_base = kRv1106SchmittBaseOffsets[static_cast<size_t>(bank)];
    const uint32_t reg_offset = bank_base + static_cast<uint32_t>((pin / 8) * 4);
    const uint8_t bit = static_cast<uint8_t>(pin % 8);

    const size_t map_size = field.use_pmuioc ? kPmuiocMapSize : kIocMapSize;
    if (static_cast<size_t>(reg_offset + sizeof(uint32_t)) > map_size) {
        throw std::runtime_error("computed schmitt register offset out of range: 0x" + std::to_string(reg_offset));
    }

    field.info.reg_offset = reg_offset;
    field.info.absolute_addr = field.info.base_addr + reg_offset;
    field.info.bit = bit;
    return field;
}

uint32_t Controller::get_mux(int bank, int pin) const {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    const ResolvedField field = resolve_field(bank, pin);
    if (field.info.gpio_only) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(lock_);
    const uint32_t value = ioc_read32_unsafe(field.use_pmuioc, field.info.reg_offset);
    return (value >> field.info.bit) & field.info.mask;
}

void Controller::write_mux_untracked(int bank, int pin, uint32_t mux) {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    const ResolvedField field = resolve_field(bank, pin);
    if (field.info.gpio_only) {
        if (mux != 0) {
            throw std::invalid_argument(
                "gpio" + std::to_string(bank) + "-" + std::to_string(pin) + " is GPIO-only and only supports mux=0.");
        }
        return;
    }

    if (mux > field.info.mask) {
        throw std::invalid_argument("mux value out of range for gpio" + std::to_string(bank) + "-" + std::to_string(pin) +
                                    ": " + std::to_string(mux));
    }

    std::lock_guard<std::mutex> guard(lock_);
    const uint32_t mask = field.info.mask << field.info.bit;
    const uint32_t value = (mux & field.info.mask) << field.info.bit;
    ioc_hiword_update_unsafe(field.use_pmuioc, field.info.reg_offset, mask, value);
}

void Controller::set_mux(int bank, int pin, uint32_t mux) {
    remember_pin_claim(bank, pin, "mux" + std::to_string(mux));
    const PinConflictReport conflict = check_conflict(bank, pin, "mux" + std::to_string(mux));
    if (conflict.conflict) {
        std::cerr << "[PinMux] Warning: overriding pin conflict on " << format_pin_label(bank, pin)
                  << ": " << conflict.reason << std::endl;
        (void)release_conflict(bank, pin);
    }
    write_mux_untracked(bank, pin, mux);
}

void* Controller::gpio_map_bank_unsafe(int bank) const {
    if (bank < 0 || bank >= static_cast<int>(kGpioBaseAddrs.size())) {
        throw std::invalid_argument("GPIO bank out of range: " + std::to_string(bank));
    }
    if (gpio_maps_[static_cast<size_t>(bank)] != nullptr) {
        return gpio_maps_[static_cast<size_t>(bank)];
    }
    if (fd_ < 0) {
        throw std::runtime_error("PinMux controller is closed.");
    }
    void* map = ::mmap(nullptr,
                       kGpioMapSize,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED,
                       fd_,
                       kGpioBaseAddrs[static_cast<size_t>(bank)]);
    if (map == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap GPIO bank " + std::to_string(bank) + ": " +
                                 std::string(std::strerror(errno)));
    }
    gpio_maps_[static_cast<size_t>(bank)] = map;
    return map;
}

void Controller::gpio_reg_set_direction_unsafe(int bank, int pin, bool output) const {
    const uint32_t offset = (pin < 16) ? 0x08 : 0x0c;
    const uint32_t bit = static_cast<uint32_t>(pin & 0x0f);
    const uint32_t mask = 1u << bit;
    gpio_reg_hiword_update_unsafe(bank, offset, mask, (output ? mask : 0u));
}

void Controller::gpio_reg_set_output_unsafe(int bank, int pin, int value) const {
    const uint32_t offset = (pin < 16) ? 0x00 : 0x04;
    const uint32_t bit = static_cast<uint32_t>(pin & 0x0f);
    const uint32_t mask = 1u << bit;
    gpio_reg_hiword_update_unsafe(bank, offset, mask, (value ? mask : 0u));
}

int Controller::gpio_reg_get_value_unsafe(int bank, int pin) const {
    return (gpio_reg_read32_unsafe(bank, 0x70) & (1u << static_cast<uint32_t>(pin))) ? 1 : 0;
}

std::string Controller::normalize_token(std::string token) {
    std::string normalized;
    normalized.reserve(token.size());
    for (unsigned char c : token) {
        if (std::isspace(c)) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(c)));
    }
    return normalized;
}

bool Controller::parse_pinmux_debug_line(const std::string& line, PinRuntimeStatus* status) {
    if (!status) {
        return false;
    }

    static const std::regex kLinePatten(
        R"(^pin\s+\d+\s+\(gpio([0-9]+)-([0-9]+)\):\s+(.+?)\s+\(([^)]*)\)(?:\s+function\s+([^\s]+)\s+group\s+([^\s]+))?.*$)");

    std::smatch match;
    if (!std::regex_match(line, match, kLinePatten)) {
        return false;
    }

    status->found = true;
    status->bank = std::stoi(match[1].str());
    status->pin = std::stoi(match[2].str());
    status->mux_owner = trim_copy(match[3].str());
    status->gpio_owner = trim_copy(match[4].str());
    status->function = match[5].matched ? trim_copy(match[5].str()) : "";
    status->group = match[6].matched ? trim_copy(match[6].str()) : "";
    return true;
}

bool Controller::is_unclaimed_mux_owner(const std::string& owner) {
    const std::string token = normalize_token(owner);
    return token.empty() || token == "muxunclaimed" || token == "(muxunclaimed)";
}

bool Controller::is_unclaimed_gpio_owner(const std::string& owner) {
    const std::string token = normalize_token(owner);
    return token.empty() || token == "gpiounclaimed" || token == "(gpiounclaimed)";
}

bool Controller::write_text_file(const std::string& path, const std::string& text) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const ssize_t n = ::write(fd, text.data(), text.size());
    const int saved_errno = errno;
    ::close(fd);
    errno = saved_errno;
    return n == static_cast<ssize_t>(text.size());
}

std::string Controller::format_pin_label(int bank, int pin) {
    return "gpio" + std::to_string(bank) + "-" + std::to_string(pin);
}

std::string Controller::find_gpiochip_name_for_bank(int bank) {
    static const std::array<std::string, 5> kGpioDeviceName{
        "ff380000.gpio", "ff530000.gpio", "ff540000.gpio", "ff550000.gpio", "ff560000.gpio"};
    if (bank < 0 || bank >= static_cast<int>(kGpioDeviceName.size())) {
        return "";
    }

    std::error_code ec;
    for (const auto& root : {std::filesystem::path("/sys/bus/platform/devices"),
                             std::filesystem::path("/sys/devices/platform/pinctrl")}) {
        const std::filesystem::path dev_path = root / kGpioDeviceName[bank];
        if (!std::filesystem::exists(dev_path, ec)) {
            continue;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dev_path, ec)) {
            if (ec) {
                break;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("gpiochip", 0) == 0) {
                return name;
            }
        }
    }
    return "";
}

uint64_t Controller::build_gpio_v2_flags(const GpioLineConfig& config) {
    const std::string direction = normalize_token(config.direction);
    const std::string bias = normalize_token(config.bias);
    const std::string drive = normalize_token(config.drive);
    const std::string edge = normalize_token(config.edge);

    uint64_t flags = 0;
    if (direction == "output" || direction == "out") {
        flags |= GPIO_V2_LINE_FLAG_OUTPUT;
    } else {
        flags |= GPIO_V2_LINE_FLAG_INPUT;
        if (edge == "rising" || edge == "rise") {
            flags |= GPIO_V2_LINE_FLAG_EDGE_RISING;
        } else if (edge == "falling" || edge == "fall") {
            flags |= GPIO_V2_LINE_FLAG_EDGE_FALLING;
        } else if (edge == "both" || edge == "any") {
            flags |= GPIO_V2_LINE_FLAG_EDGE_RISING | GPIO_V2_LINE_FLAG_EDGE_FALLING;
        }
    }

    if (bias == "pullup" || bias == "pull_up" || bias == "up") {
        flags |= GPIO_V2_LINE_FLAG_BIAS_PULL_UP;
    } else if (bias == "pulldown" || bias == "pull_down" || bias == "down") {
        flags |= GPIO_V2_LINE_FLAG_BIAS_PULL_DOWN;
    } else if (bias == "disable" || bias == "disabled") {
        flags |= GPIO_V2_LINE_FLAG_BIAS_DISABLED;
    }

    if (drive == "opendrain" || drive == "open_drain") {
        flags |= GPIO_V2_LINE_FLAG_OPEN_DRAIN;
    } else if (drive == "opensource" || drive == "open_source") {
        flags |= GPIO_V2_LINE_FLAG_OPEN_SOURCE;
    }

    if (config.active_low) {
        flags |= GPIO_V2_LINE_FLAG_ACTIVE_LOW;
    }
    return flags;
}

uint32_t Controller::build_gpio_v1_flags(const GpioLineConfig& config) {
    const std::string direction = normalize_token(config.direction);
    const std::string bias = normalize_token(config.bias);
    const std::string drive = normalize_token(config.drive);

    uint32_t flags = 0;
    if (direction == "output" || direction == "out") {
        flags |= GPIOHANDLE_REQUEST_OUTPUT;
    } else {
        flags |= GPIOHANDLE_REQUEST_INPUT;
    }

    if (bias == "pullup" || bias == "pull_up" || bias == "up") {
#ifdef GPIOHANDLE_REQUEST_BIAS_PULL_UP
        flags |= GPIOHANDLE_REQUEST_BIAS_PULL_UP;
#endif
    } else if (bias == "pulldown" || bias == "pull_down" || bias == "down") {
#ifdef GPIOHANDLE_REQUEST_BIAS_PULL_DOWN
        flags |= GPIOHANDLE_REQUEST_BIAS_PULL_DOWN;
#endif
    } else if (bias == "disable" || bias == "disabled") {
#ifdef GPIOHANDLE_REQUEST_BIAS_DISABLE
        flags |= GPIOHANDLE_REQUEST_BIAS_DISABLE;
#endif
    }

    if (drive == "opendrain" || drive == "open_drain") {
#ifdef GPIOHANDLE_REQUEST_OPEN_DRAIN
        flags |= GPIOHANDLE_REQUEST_OPEN_DRAIN;
#endif
    } else if (drive == "opensource" || drive == "open_source") {
#ifdef GPIOHANDLE_REQUEST_OPEN_SOURCE
        flags |= GPIOHANDLE_REQUEST_OPEN_SOURCE;
#endif
    }

    if (config.active_low) {
        flags |= GPIOHANDLE_REQUEST_ACTIVE_LOW;
    }
    return flags;
}

int Controller::parse_adc_channel_token(const std::string& token) {
    const std::string normalized = normalize_token(token);
    if (normalized.empty()) {
        return -1;
    }

    std::smatch match;
    static const std::regex kAdcPatten(R"((?:adc|saradc|saradc_in|saradcin)([0-9]+))");
    if (std::regex_match(normalized, match, kAdcPatten)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

std::vector<AdcChannelStatus> Controller::scan_adc_channels() {
    std::vector<AdcChannelStatus> channels;
    const std::filesystem::path root("/sys/bus/iio/devices");
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        return channels;
    }

    static const std::regex kRawPatten(R"(in_voltage([0-9]+)_raw)");
    for (const auto& dev_entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        if (!dev_entry.is_directory(ec) || ec) {
            continue;
        }
        const std::string dev_name = dev_entry.path().filename().string();
        if (dev_name.rfind("iio:device", 0) != 0) {
            continue;
        }

        const std::string driver_name = read_file_flatten_nuls(dev_entry.path() / "name");
        const std::string normalized_driver = normalize_token(driver_name);
        if (!normalized_driver.empty() && normalized_driver.find("saradc") == std::string::npos) {
            continue;
        }

        for (const auto& f : std::filesystem::directory_iterator(dev_entry.path(), ec)) {
            if (ec) {
                break;
            }
            if (!f.is_regular_file(ec) || ec) {
                continue;
            }
            const std::string base = f.path().filename().string();
            std::smatch match;
            if (!std::regex_match(base, match, kRawPatten)) {
                continue;
            }

            AdcChannelStatus status;
            status.available = true;
            status.channel = std::stoi(match[1].str());
            status.device = dev_name;
            status.raw_path = f.path().string();
            status.scale_path = (dev_entry.path() / "in_voltage_scale").string();
            if (status.channel == 0) {
                status.pin_hint = "GPIO4_C0";
            } else if (status.channel == 1) {
                status.pin_hint = "GPIO4_C1";
            }
            channels.push_back(std::move(status));
        }
    }

    std::sort(channels.begin(), channels.end(), [](const AdcChannelStatus& lhs, const AdcChannelStatus& rhs) {
        if (lhs.channel != rhs.channel) {
            return lhs.channel < rhs.channel;
        }
        return lhs.device < rhs.device;
    });
    return channels;
}

bool Controller::release_gpio_handle_unsafe(const PinKey& key) {
    auto it = gpio_line_handles_.find(key);
    if (it == gpio_line_handles_.end()) {
        return true;
    }
    if (it->second.fd >= 0) {
        ::close(it->second.fd);
    }
    gpio_line_handles_.erase(it);
    return true;
}

PinId Controller::parse_pin_name(const std::string& pin_name) const {
    const std::string token = normalize_token(pin_name);
    std::smatch match;

    static const std::regex kNumericWithGpio(R"(gpio([0-4])[-_:]?([0-9]{1,2}))");
    static const std::regex kAlphaWithGpio(R"(gpio([0-4])[_-]?([abcd])([0-7]))");
    static const std::regex kAlphaBare(R"(([0-4])[_-]?([abcd])[_-]?([0-7]))");
    static const std::regex kNumericBare(R"(([0-4])[-_:]([0-9]{1,2}))");

    PinId pin_id;
    if (std::regex_match(token, match, kNumericWithGpio)) {
        pin_id.bank = std::stoi(match[1].str());
        pin_id.pin = std::stoi(match[2].str());
    } else if (std::regex_match(token, match, kAlphaWithGpio)) {
        pin_id.bank = std::stoi(match[1].str());
        const int group = static_cast<int>(match[2].str()[0] - 'a');
        const int idx = std::stoi(match[3].str());
        pin_id.pin = group * 8 + idx;
    } else if (std::regex_match(token, match, kAlphaBare)) {
        pin_id.bank = std::stoi(match[1].str());
        const int group = static_cast<int>(match[2].str()[0] - 'a');
        const int idx = std::stoi(match[3].str());
        pin_id.pin = group * 8 + idx;
    } else if (std::regex_match(token, match, kNumericBare)) {
        pin_id.bank = std::stoi(match[1].str());
        pin_id.pin = std::stoi(match[2].str());
    } else {
        throw std::invalid_argument("Unsupported pin name: " + pin_name +
                                    ". Expected gpio1-20 / GPIO1_C4 / 1:20 / 1D3 formats.");
    }

    (void)resolve_field(pin_id.bank, pin_id.pin);
    return pin_id;
}

uint32_t Controller::get_mux(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_mux(pin_id.bank, pin_id.pin);
}

void Controller::set_mux(const std::string& pin_name, uint32_t mux) {
    const PinId pin_id = parse_pin_name(pin_name);
    set_mux(pin_id.bank, pin_id.pin, mux);
}

RegisterInfo Controller::get_register_info(int bank, int pin) const {
    return resolve_field(bank, pin).info;
}

RegisterInfo Controller::get_register_info(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_register_info(pin_id.bank, pin_id.pin);
}

void Controller::set_drive_strength(int bank, int pin, int level) {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }
    if (level < 0 || level > 7) {
        throw std::invalid_argument("drive strength level out of range (expected 0..7): " + std::to_string(level));
    }

    const ResolvedField field = resolve_drive_field(bank, pin);
    const uint32_t encoded = static_cast<uint32_t>((1u << (level + 1)) - 1u);

    std::lock_guard<std::mutex> guard(lock_);
    const uint32_t mask = field.info.mask << field.info.bit;
    const uint32_t value = (encoded & field.info.mask) << field.info.bit;
    ioc_hiword_update_unsafe(field.use_pmuioc, field.info.reg_offset, mask, value);
}

void Controller::set_drive_strength(const std::string& pin_name, int level) {
    const PinId pin_id = parse_pin_name(pin_name);
    set_drive_strength(pin_id.bank, pin_id.pin, level);
}

DriveStrengthStatus Controller::get_drive_strength(int bank, int pin) const {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    DriveStrengthStatus status;
    const ResolvedField field = resolve_drive_field(bank, pin);
    status.available = true;
    status.reg_offset = field.info.reg_offset;
    status.absolute_addr = field.info.absolute_addr;
    status.domain = field.info.domain;

    uint32_t reg_value = 0;
    {
        std::lock_guard<std::mutex> guard(lock_);
        reg_value = ioc_read32_unsafe(field.use_pmuioc, field.info.reg_offset);
    }

    status.raw = (reg_value >> field.info.bit) & field.info.mask;

    int decoded_level = -1;
    for (int level = 0; level <= 7; ++level) {
        const uint32_t encoded = static_cast<uint32_t>((1u << (level + 1)) - 1u);
        if (status.raw == encoded) {
            decoded_level = level;
            break;
        }
    }
    status.level = decoded_level;
    if (decoded_level < 0) {
        if (status.raw == 0) {
            status.note = "Drive raw value is 0 (unsupported pin or firmware keeps this field disabled).";
        } else {
            status.note = "Drive raw value is not a standard RV1106 level encoding.";
        }
    } else {
        status.note = "Drive strength level read ok.";
    }
    return status;
}

DriveStrengthStatus Controller::get_drive_strength(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_drive_strength(pin_id.bank, pin_id.pin);
}

void Controller::set_pull(int bank, int pin, const std::string& mode) {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    const std::string token = normalize_token(mode);
    int raw = -1;
    if (token == "0" || token == "disable" || token == "none" || token == "pullnone" || token == "pull_none" ||
        token == "default") {
        raw = 0;
    } else if (token == "1" || token == "pullup" || token == "pull_up" || token == "up") {
        raw = 1;
    } else if (token == "2" || token == "pulldown" || token == "pull_down" || token == "down") {
        raw = 2;
    } else if (token == "3" || token == "bushold" || token == "bus_hold" || token == "hold" || token == "keeper") {
        raw = 3;
    } else {
        throw std::invalid_argument("Unsupported pull mode '" + mode + "'. Use disable/pull_up/pull_down/bus_hold or 0..3.");
    }

    const ResolvedField field = resolve_pull_field(bank, pin);
    const uint32_t value = static_cast<uint32_t>(raw) & field.info.mask;

    std::lock_guard<std::mutex> guard(lock_);
    const uint32_t mask = field.info.mask << field.info.bit;
    ioc_hiword_update_unsafe(field.use_pmuioc, field.info.reg_offset, mask, value << field.info.bit);
}

void Controller::set_pull(const std::string& pin_name, const std::string& mode) {
    const PinId pin_id = parse_pin_name(pin_name);
    set_pull(pin_id.bank, pin_id.pin, mode);
}

PullStatus Controller::get_pull(int bank, int pin) const {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    PullStatus status;
    const ResolvedField field = resolve_pull_field(bank, pin);
    status.available = true;
    status.reg_offset = field.info.reg_offset;
    status.absolute_addr = field.info.absolute_addr;
    status.domain = field.info.domain;

    uint32_t reg_value = 0;
    {
        std::lock_guard<std::mutex> guard(lock_);
        reg_value = ioc_read32_unsafe(field.use_pmuioc, field.info.reg_offset);
    }

    status.raw = (reg_value >> field.info.bit) & field.info.mask;
    switch (status.raw) {
    case 0:
        status.mode = "disable";
        break;
    case 1:
        status.mode = "pull_up";
        break;
    case 2:
        status.mode = "pull_down";
        break;
    case 3:
        status.mode = "bus_hold";
        break;
    default:
        status.mode = "unknown";
        break;
    }
    status.note = "Pull status read ok.";
    return status;
}

PullStatus Controller::get_pull(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_pull(pin_id.bank, pin_id.pin);
}

void Controller::set_input_schmitt(int bank, int pin, bool enable) {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    const ResolvedField field = resolve_schmitt_field(bank, pin);
    const uint32_t value = enable ? 1u : 0u;

    std::lock_guard<std::mutex> guard(lock_);
    const uint32_t mask = field.info.mask << field.info.bit;
    ioc_hiword_update_unsafe(field.use_pmuioc, field.info.reg_offset, mask, value << field.info.bit);
}

void Controller::set_input_schmitt(const std::string& pin_name, bool enable) {
    const PinId pin_id = parse_pin_name(pin_name);
    set_input_schmitt(pin_id.bank, pin_id.pin, enable);
}

SchmittStatus Controller::get_input_schmitt(int bank, int pin) const {
    if (!is_open()) {
        throw std::runtime_error("PinMux controller is closed.");
    }

    SchmittStatus status;
    const ResolvedField field = resolve_schmitt_field(bank, pin);
    status.available = true;
    status.reg_offset = field.info.reg_offset;
    status.absolute_addr = field.info.absolute_addr;
    status.domain = field.info.domain;

    uint32_t reg_value = 0;
    {
        std::lock_guard<std::mutex> guard(lock_);
        reg_value = ioc_read32_unsafe(field.use_pmuioc, field.info.reg_offset);
    }

    status.raw = (reg_value >> field.info.bit) & field.info.mask;
    status.enabled = (status.raw & 0x1u) != 0;
    status.note = "Schmitt status read ok.";
    return status;
}

SchmittStatus Controller::get_input_schmitt(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_input_schmitt(pin_id.bank, pin_id.pin);
}

PinElectricalCapability Controller::probe_electrical_capability(int bank, int pin, bool active_test) {
    PinElectricalCapability cap;
    cap.bank = bank;
    cap.pin = pin;

    const auto write_raw_field = [&](const ResolvedField& field, uint32_t raw) {
        const uint32_t value = raw & field.info.mask;
        std::lock_guard<std::mutex> guard(lock_);
        const uint32_t mask = field.info.mask << field.info.bit;
        ioc_hiword_update_unsafe(field.use_pmuioc, field.info.reg_offset, mask, value << field.info.bit);
    };

    const auto read_raw_field = [&](const ResolvedField& field) -> uint32_t {
        std::lock_guard<std::mutex> guard(lock_);
        return (ioc_read32_unsafe(field.use_pmuioc, field.info.reg_offset) >> field.info.bit) & field.info.mask;
    };

    const DriveStrengthStatus drv = get_drive_strength(bank, pin);
    const PullStatus pull = get_pull(bank, pin);
    const SchmittStatus smt = get_input_schmitt(bank, pin);

    const bool likely_adc_pin = (bank == 4 && (pin == 16 || pin == 17));
    if (!active_test) {
        cap.drive_supported = !likely_adc_pin && (drv.level >= 0 || drv.raw != 0);
        cap.pull_supported = !likely_adc_pin;
        cap.schmitt_supported = !likely_adc_pin;
        cap.note = likely_adc_pin
                       ? "Passive probe: SARADC pin likely does not support GPIO electrical tuning."
                       : "Passive probe complete. Use active_test=True for strict write/read validation.";
        return cap;
    }

    const ResolvedField drive_field = resolve_drive_field(bank, pin);
    const ResolvedField pull_field = resolve_pull_field(bank, pin);
    const ResolvedField schmitt_field = resolve_schmitt_field(bank, pin);

    const uint32_t drive_orig = read_raw_field(drive_field);
    const uint32_t pull_orig = read_raw_field(pull_field);
    const uint32_t schmitt_orig = read_raw_field(schmitt_field);

    const uint32_t drive_test = (drive_orig == 0x03) ? 0x01 : 0x03;
    const uint32_t pull_test = (pull_orig == 0x01) ? 0x02 : 0x01;
    const uint32_t schmitt_test = (schmitt_orig == 0x1) ? 0x0 : 0x1;

    write_raw_field(drive_field, drive_test);
    write_raw_field(pull_field, pull_test);
    write_raw_field(schmitt_field, schmitt_test);

    const uint32_t drive_after = read_raw_field(drive_field);
    const uint32_t pull_after = read_raw_field(pull_field);
    const uint32_t schmitt_after = read_raw_field(schmitt_field);

    write_raw_field(drive_field, drive_orig);
    write_raw_field(pull_field, pull_orig);
    write_raw_field(schmitt_field, schmitt_orig);

    cap.drive_supported = (drive_after == (drive_test & drive_field.info.mask)) || (drv.level >= 0 || drv.raw != 0);
    cap.pull_supported = (pull_after == (pull_test & pull_field.info.mask)) || (pull.raw != 0);
    cap.schmitt_supported = (schmitt_after == (schmitt_test & schmitt_field.info.mask)) || (smt.raw != 0);
    cap.note = "Active electrical probe complete.";
    return cap;
}

PinElectricalCapability Controller::probe_electrical_capability(const std::string& pin_name, bool active_test) {
    const PinId pin_id = parse_pin_name(pin_name);
    return probe_electrical_capability(pin_id.bank, pin_id.pin, active_test);
}

std::vector<PinElectricalCapability> Controller::probe_electrical_capabilities(bool active_test) {
    std::vector<PinElectricalCapability> caps;
    for (int bank = 0; bank < static_cast<int>(kRv1106IomuxLayout.size()); ++bank) {
        const int pin_count = kRv1106IomuxLayout[bank].pin_count;
        for (int pin = 0; pin < pin_count; ++pin) {
            try {
                caps.push_back(probe_electrical_capability(bank, pin, active_test));
            } catch (...) {
                PinElectricalCapability cap;
                cap.bank = bank;
                cap.pin = pin;
                cap.note = "Probe failed.";
                caps.push_back(std::move(cap));
            }
        }
    }
    return caps;
}

AdcChannelStatus Controller::read_adc_by_channel(int channel) const {
    AdcChannelStatus status;
    status.channel = channel;
    if (channel < 0) {
        status.note = "ADC channel must be >= 0.";
        return status;
    }

    const auto channels = scan_adc_channels();
    auto it = std::find_if(channels.begin(), channels.end(), [&](const AdcChannelStatus& item) {
        return item.channel == channel;
    });
    if (it == channels.end()) {
        status.note = "ADC channel not found in IIO sysfs.";
        return status;
    }

    status = *it;
    {
        std::ifstream in(status.raw_path);
        if (!(in >> status.raw)) {
            status.available = false;
            status.note = "Failed to read ADC raw value: " + status.raw_path;
            return status;
        }
    }

    {
        std::ifstream in(status.scale_path);
        if (!(in >> status.scale)) {
            status.scale = 0.0;
        }
    }
    status.millivolts = static_cast<double>(status.raw) * status.scale;
    status.note = "ADC read ok.";
    return status;
}

std::vector<AdcChannelStatus> Controller::list_adc_channels() const {
    std::vector<AdcChannelStatus> channels = scan_adc_channels();
    for (auto& item : channels) {
        AdcChannelStatus live = read_adc_by_channel(item.channel);
        item.raw = live.raw;
        item.scale = live.scale;
        item.millivolts = live.millivolts;
        item.available = live.available;
        item.note = live.note;
    }
    return channels;
}

AdcChannelStatus Controller::read_adc(int channel) const {
    return read_adc_by_channel(channel);
}

AdcChannelStatus Controller::read_adc(const std::string& channel_or_pin) const {
    int channel = parse_adc_channel_token(channel_or_pin);
    if (channel < 0) {
        const PinId pin_id = parse_pin_name(channel_or_pin);
        if (pin_id.bank == 4 && pin_id.pin == 16) {
            channel = 0;
        } else if (pin_id.bank == 4 && pin_id.pin == 17) {
            channel = 1;
        } else {
            throw std::invalid_argument("Pin " + channel_or_pin + " is not mapped to SARADC channel.");
        }
    }
    return read_adc_by_channel(channel);
}

bool Controller::gpio_request_line(int bank, int pin, const GpioLineConfig& config) {
    (void)resolve_field(bank, pin);

    set_function(bank, pin, "gpio");

    GpioLineConfig effective_config = config;
    int drive_level = effective_config.drive_strength_level;
    if (drive_level < 0 && effective_config.drive_strength_ma >= 0 && effective_config.drive_strength_ma <= 7) {
        drive_level = effective_config.drive_strength_ma;
    }
    if (drive_level >= 0) {
        set_drive_strength(bank, pin, drive_level);
        effective_config.drive_strength_level = drive_level;
        if (effective_config.drive_strength_ma < 0) {
            effective_config.drive_strength_ma = drive_level;
        }
    }

    const std::string chip = find_gpiochip_name_for_bank(bank);
    int chip_fd = -1;
    if (!chip.empty()) {
        const std::string dev = "/dev/" + chip;
        chip_fd = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
    }

    int line_fd = -1;
    bool use_v2 = false;
    bool has_events = false;
    const std::string edge = normalize_token(effective_config.edge);
    const bool wants_events = edge == "rising" || edge == "rise" ||
                              edge == "falling" || edge == "fall" ||
                              edge == "both" || edge == "any";

    if (chip_fd >= 0) {
        gpio_v2_line_request request{};
        request.offsets[0] = static_cast<__u32>(pin);
        request.num_lines = 1;

        const std::string consumer = effective_config.consumer.empty() ? "visiong-pinmux" : effective_config.consumer;
        std::strncpy(request.consumer, consumer.c_str(), sizeof(request.consumer) - 1);
        request.consumer[sizeof(request.consumer) - 1] = '\0';

        const uint64_t flags = build_gpio_v2_flags(effective_config);
        request.config.flags = flags;
        if ((flags & GPIO_V2_LINE_FLAG_OUTPUT) != 0) {
            request.config.num_attrs = 1;
            request.config.attrs[0].mask = 0x1;
            request.config.attrs[0].attr.id = GPIO_V2_LINE_ATTR_ID_OUTPUT_VALUES;
            request.config.attrs[0].attr.values = (effective_config.default_value != 0) ? 0x1 : 0x0;
        }

        if (::ioctl(chip_fd, GPIO_V2_GET_LINE_IOCTL, &request) == 0) {
            line_fd = request.fd;
            use_v2 = true;
            has_events = wants_events;
        }
    }

    if (chip_fd >= 0 && line_fd < 0) {
        if (wants_events) {
            gpioevent_request request{};
            request.lineoffset = static_cast<__u32>(pin);
            request.handleflags = build_gpio_v1_flags(effective_config);
            if (edge == "rising" || edge == "rise") {
                request.eventflags = GPIOEVENT_REQUEST_RISING_EDGE;
            } else if (edge == "falling" || edge == "fall") {
                request.eventflags = GPIOEVENT_REQUEST_FALLING_EDGE;
            } else {
                request.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES;
            }
            const std::string consumer = effective_config.consumer.empty() ? "visiong-pinmux" : effective_config.consumer;
            std::strncpy(request.consumer_label, consumer.c_str(), sizeof(request.consumer_label) - 1);
            request.consumer_label[sizeof(request.consumer_label) - 1] = '\0';

            if (::ioctl(chip_fd, GPIO_GET_LINEEVENT_IOCTL, &request) == 0) {
                line_fd = request.fd;
                use_v2 = false;
                has_events = true;
            }
        } else {
            gpiohandle_request request{};
            request.lineoffsets[0] = static_cast<__u32>(pin);
            request.lines = 1;
            request.flags = build_gpio_v1_flags(effective_config);
            request.default_values[0] = (effective_config.default_value != 0) ? 1 : 0;
            const std::string consumer = effective_config.consumer.empty() ? "visiong-pinmux" : effective_config.consumer;
            std::strncpy(request.consumer_label, consumer.c_str(), sizeof(request.consumer_label) - 1);
            request.consumer_label[sizeof(request.consumer_label) - 1] = '\0';

            if (::ioctl(chip_fd, GPIO_GET_LINEHANDLE_IOCTL, &request) == 0) {
                line_fd = request.fd;
                use_v2 = false;
            }
        }
    }

    if (line_fd < 0) {
        if (chip_fd >= 0) {
            ::close(chip_fd);
        }
        if (wants_events) {
            return false;
        }

        std::lock_guard<std::mutex> guard(lock_);
        const PinKey key{bank, pin};
        release_gpio_handle_unsafe(key);
        GpioLineHandle handle;
        handle.fd = -1;
        handle.reg_backed = true;
        handle.chip = chip.empty() ? ("gpio-reg" + std::to_string(bank)) : chip;
        handle.offset = pin;
        handle.config = effective_config;
        const std::string direction = normalize_token(effective_config.direction);
        if (direction == "output" || direction == "out") {
            gpio_reg_set_output_unsafe(bank, pin, effective_config.default_value);
            gpio_reg_set_direction_unsafe(bank, pin, true);
        } else {
            gpio_reg_set_direction_unsafe(bank, pin, false);
        }
        gpio_line_handles_[key] = std::move(handle);
        std::cerr << "[PinMux] Warning: GPIO chardev request failed for "
                  << format_pin_label(bank, pin)
                  << "; using register-backed GPIO control." << std::endl;
        return true;
    }
    if (chip_fd >= 0) {
        ::close(chip_fd);
    }

    std::lock_guard<std::mutex> guard(lock_);
    const PinKey key{bank, pin};
    release_gpio_handle_unsafe(key);
    GpioLineHandle handle;
    handle.fd = line_fd;
    handle.use_v2 = use_v2;
    handle.has_events = has_events;
    handle.chip = chip;
    handle.offset = pin;
    handle.config = effective_config;
    gpio_line_handles_[key] = std::move(handle);
    return true;
}

bool Controller::gpio_request_line(const std::string& pin_name, const GpioLineConfig& config) {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_request_line(pin_id.bank, pin_id.pin, config);
}

bool Controller::gpio_release_line(int bank, int pin) {
    std::lock_guard<std::mutex> guard(lock_);
    return release_gpio_handle_unsafe(PinKey{bank, pin});
}

bool Controller::gpio_release_line(const std::string& pin_name) {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_release_line(pin_id.bank, pin_id.pin);
}

void Controller::gpio_set_value(int bank, int pin, int value) const {
    std::lock_guard<std::mutex> guard(lock_);
    const PinKey key{bank, pin};
    auto it = gpio_line_handles_.find(key);
    if (it == gpio_line_handles_.end() || (it->second.fd < 0 && !it->second.reg_backed)) {
        throw std::runtime_error("GPIO line is not requested: " + format_pin_label(bank, pin));
    }
    if (it->second.has_events) {
        throw std::runtime_error("GPIO event line cannot be used as output: " + format_pin_label(bank, pin));
    }
    if (it->second.reg_backed) {
        gpio_reg_set_output_unsafe(bank, pin, value);
        return;
    }

    if (it->second.use_v2) {
        gpio_v2_line_values values{};
        values.mask = 0x1;
        values.bits = (value != 0) ? 0x1 : 0x0;
        if (::ioctl(it->second.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, &values) < 0) {
            throw std::runtime_error("Failed to set GPIO value: " + std::string(std::strerror(errno)));
        }
    } else {
        gpiohandle_data values{};
        values.values[0] = (value != 0) ? 1 : 0;
        if (::ioctl(it->second.fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &values) < 0) {
            throw std::runtime_error("Failed to set GPIO value: " + std::string(std::strerror(errno)));
        }
    }
}

void Controller::gpio_set_value(const std::string& pin_name, int value) const {
    const PinId pin_id = parse_pin_name(pin_name);
    gpio_set_value(pin_id.bank, pin_id.pin, value);
}

int Controller::gpio_get_value(int bank, int pin) const {
    std::lock_guard<std::mutex> guard(lock_);
    const PinKey key{bank, pin};
    auto it = gpio_line_handles_.find(key);
    if (it == gpio_line_handles_.end() || (it->second.fd < 0 && !it->second.reg_backed)) {
        throw std::runtime_error("GPIO line is not requested: " + format_pin_label(bank, pin));
    }
    if (it->second.has_events && !it->second.use_v2) {
        throw std::runtime_error("GPIO v1 event line does not support value reads; request it without edge or use gpio_wait_event().");
    }
    if (it->second.reg_backed) {
        return gpio_reg_get_value_unsafe(bank, pin);
    }

    if (it->second.use_v2) {
        gpio_v2_line_values values{};
        values.mask = 0x1;
        if (::ioctl(it->second.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &values) < 0) {
            throw std::runtime_error("Failed to read GPIO value: " + std::string(std::strerror(errno)));
        }
        return (values.bits & 0x1) ? 1 : 0;
    }

    gpiohandle_data values{};
    if (::ioctl(it->second.fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &values) < 0) {
        throw std::runtime_error("Failed to read GPIO value: " + std::string(std::strerror(errno)));
    }
    return values.values[0] ? 1 : 0;
}

int Controller::gpio_get_value(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_get_value(pin_id.bank, pin_id.pin);
}

GpioLineStatus Controller::gpio_get_status(int bank, int pin) const {
    GpioLineStatus status;
    status.bank = bank;
    status.pin = pin;

    std::lock_guard<std::mutex> guard(lock_);
    const PinKey key{bank, pin};
    auto it = gpio_line_handles_.find(key);
    if (it == gpio_line_handles_.end() || (it->second.fd < 0 && !it->second.reg_backed)) {
        status.requested = false;
        status.note = "GPIO line is not requested.";
        return status;
    }

    status.requested = true;
    status.gpiochip = it->second.chip;
    status.config = it->second.config;
    if (it->second.has_events && !it->second.use_v2) {
        status.note = "GPIO v1 event line is requested; current value is not readable through this fd.";
        return status;
    }

    if (it->second.reg_backed) {
        status.value = gpio_reg_get_value_unsafe(bank, pin);
        status.note = "GPIO line requested through register-backed control.";
    } else if (it->second.use_v2) {
        gpio_v2_line_values values{};
        values.mask = 0x1;
        if (::ioctl(it->second.fd, GPIO_V2_LINE_GET_VALUES_IOCTL, &values) < 0) {
            status.note = "Failed to read current GPIO line value.";
            return status;
        }
        status.value = (values.bits & 0x1) ? 1 : 0;
    } else {
        gpiohandle_data values{};
        if (::ioctl(it->second.fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &values) < 0) {
            status.note = "Failed to read current GPIO line value.";
            return status;
        }
        status.value = values.values[0] ? 1 : 0;
    }
    if (status.config.drive_strength_level >= 0) {
        const ResolvedField drive_field = resolve_drive_field(bank, pin);
        const uint32_t drive_raw =
            (ioc_read32_unsafe(drive_field.use_pmuioc, drive_field.info.reg_offset) >> drive_field.info.bit) &
            drive_field.info.mask;

        int drive_level = -1;
        for (int level = 0; level <= 7; ++level) {
            const uint32_t encoded = static_cast<uint32_t>((1u << (level + 1)) - 1u);
            if (drive_raw == encoded) {
                drive_level = level;
                break;
            }
        }
        if (drive_level == status.config.drive_strength_level) {
            status.note = "GPIO line requested. Applied drive strength level " +
                          std::to_string(status.config.drive_strength_level) + ".";
        } else {
            status.note = "GPIO line requested. Drive strength readback mismatch (raw=" + std::to_string(drive_raw) +
                          "), pin may not support this setting.";
        }
    } else if (status.config.drive_strength_ma > 7) {
        status.note = "GPIO line requested. drive_strength_ma is out of RV1106 range and was ignored.";
    } else {
        if (status.note.empty()) {
            status.note = "GPIO line requested.";
        }
    }
    return status;
}

GpioLineStatus Controller::gpio_get_status(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_get_status(pin_id.bank, pin_id.pin);
}

namespace {

void populate_gpio_event_from_fd(int fd, bool use_v2, GpioLineEvent* event) {
    if (!event) {
        throw std::invalid_argument("GPIO event destination is null.");
    }
    if (use_v2) {
        gpio_v2_line_event raw{};
        ssize_t got = -1;
        do {
            got = ::read(fd, &raw, sizeof(raw));
        } while (got < 0 && errno == EINTR);
        if (got != static_cast<ssize_t>(sizeof(raw))) {
            throw std::runtime_error("GPIO v2 event read failed: " + std::string(std::strerror(errno)));
        }
        event->valid = true;
        event->timestamp_ns = raw.timestamp_ns;
        event->offset = static_cast<int>(raw.offset);
        event->sequence = raw.seqno;
        event->line_sequence = raw.line_seqno;
        if (raw.id == GPIO_V2_LINE_EVENT_RISING_EDGE) {
            event->edge = "rising";
        } else if (raw.id == GPIO_V2_LINE_EVENT_FALLING_EDGE) {
            event->edge = "falling";
        } else {
            event->edge = "unknown";
        }
        return;
    }

    gpioevent_data raw{};
    ssize_t got = -1;
    do {
        got = ::read(fd, &raw, sizeof(raw));
    } while (got < 0 && errno == EINTR);
    if (got != static_cast<ssize_t>(sizeof(raw))) {
        throw std::runtime_error("GPIO v1 event read failed: " + std::string(std::strerror(errno)));
    }
    event->valid = true;
    event->timestamp_ns = raw.timestamp;
    if (raw.id == GPIOEVENT_EVENT_RISING_EDGE) {
        event->edge = "rising";
    } else if (raw.id == GPIOEVENT_EVENT_FALLING_EDGE) {
        event->edge = "falling";
    } else {
        event->edge = "unknown";
    }
}

}  // namespace

GpioLineEvent Controller::gpio_wait_event_cancelable(int bank, int pin, int cancel_fd, int timeout_ms) const {
    GpioLineEvent event;
    event.bank = bank;
    event.pin = pin;
    event.offset = pin;

    ScopedFd wait_fd;
    bool use_v2 = false;
    {
        std::lock_guard<std::mutex> guard(lock_);
        const PinKey key{bank, pin};
        auto it = gpio_line_handles_.find(key);
        if (it == gpio_line_handles_.end() || it->second.fd < 0) {
            throw std::runtime_error("GPIO line is not requested: " + format_pin_label(bank, pin));
        }
        if (!it->second.has_events) {
            throw std::runtime_error("GPIO line was not requested with edge events: " + format_pin_label(bank, pin));
        }
        const int duplicated = ::dup(it->second.fd);
        if (duplicated < 0) {
            throw std::runtime_error("Failed to duplicate GPIO event fd: " + std::string(std::strerror(errno)));
        }
        wait_fd = ScopedFd(duplicated);
        use_v2 = it->second.use_v2;
    }

    ScopedFd cancel_wait_fd;
    if (cancel_fd >= 0) {
        const int duplicated = ::dup(cancel_fd);
        if (duplicated < 0) {
            throw std::runtime_error("Failed to duplicate GPIO cancel fd: " + std::string(std::strerror(errno)));
        }
        cancel_wait_fd = ScopedFd(duplicated);
    }

    std::array<pollfd, 2> pfds{};
    pfds[0].fd = wait_fd.get();
    pfds[0].events = POLLIN | POLLPRI;
    nfds_t nfds = 1;
    if (cancel_wait_fd) {
        pfds[1].fd = cancel_wait_fd.get();
        pfds[1].events = POLLIN;
        nfds = 2;
    }

    int poll_result = 0;
    do {
        poll_result = ::poll(pfds.data(), nfds, timeout_ms < 0 ? -1 : timeout_ms);
    } while (poll_result < 0 && errno == EINTR);

    if (poll_result == 0) {
        event.timed_out = true;
        event.note = "timeout";
        return event;
    }
    if (poll_result < 0) {
        throw std::runtime_error("GPIO event poll failed: " + std::string(std::strerror(errno)));
    }

    if (nfds > 1 && (pfds[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
        event.cancelled = true;
        event.note = "cancelled";
        return event;
    }

    if ((pfds[0].revents & POLLNVAL) != 0) {
        throw std::runtime_error("GPIO event fd became invalid while waiting.");
    }
    if ((pfds[0].revents & (POLLIN | POLLPRI)) == 0) {
        throw std::runtime_error("GPIO event fd signalled an error while waiting.");
    }

    populate_gpio_event_from_fd(wait_fd.get(), use_v2, &event);
    return event;
}

GpioLineEvent Controller::gpio_wait_event(int bank, int pin, int timeout_ms) const {
    return gpio_wait_event_cancelable(bank, pin, -1, timeout_ms);
}

GpioLineEvent Controller::gpio_wait_event(const std::string& pin_name, int timeout_ms) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_wait_event(pin_id.bank, pin_id.pin, timeout_ms);
}

GpioLineEvent Controller::gpio_wait_event_cancelable(const std::string& pin_name,
                                                     int cancel_fd,
                                                     int timeout_ms) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return gpio_wait_event_cancelable(pin_id.bank, pin_id.pin, cancel_fd, timeout_ms);
}

std::vector<PinRuntimeStatus> Controller::read_runtime_rows() const {
    std::vector<PinRuntimeStatus> rows;
    std::ifstream in("/sys/kernel/debug/pinctrl/pinctrl-rockchip-pinctrl/pinmux-pins");
    if (!in) {
        return rows;
    }

    std::string line;
    while (std::getline(in, line)) {
        PinRuntimeStatus row;
        if (parse_pinmux_debug_line(line, &row)) {
            rows.push_back(std::move(row));
        }
    }

    {
        std::lock_guard<std::mutex> guard(lock_);
        for (const auto& row : rows) {
            if (is_unclaimed_mux_owner(row.mux_owner)) {
                continue;
            }
            const std::string fn = normalize_token(row.function);
            const std::string grp = normalize_token(row.group);
            if (!fn.empty()) {
                owner_hint_cache_[fn] = row.mux_owner;
            }
            if (!grp.empty()) {
                owner_hint_cache_[grp] = row.mux_owner;
            }
        }
    }
    return rows;
}

PinRuntimeStatus Controller::get_runtime_status(int bank, int pin) const {
    (void)resolve_field(bank, pin);
    const auto rows = read_runtime_rows();
    for (const auto& row : rows) {
        if (row.bank == bank && row.pin == pin) {
            return row;
        }
    }
    PinRuntimeStatus fallback;
    fallback.bank = bank;
    fallback.pin = pin;
    return fallback;
}

PinRuntimeStatus Controller::get_runtime_status(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_runtime_status(pin_id.bank, pin_id.pin);
}

PinConflictReport Controller::check_conflict(int bank, int pin, const std::string& target_function_or_group) const {
    PinConflictReport report;
    report.runtime = get_runtime_status(bank, pin);
    if (!report.runtime.found) {
        report.conflict = false;
        report.reason = "No runtime pinmux row found; debugfs pinctrl may be unavailable.";
        return report;
    }

    const std::string normalized_target = normalize_token(target_function_or_group);
    const std::string normalized_runtime_function = normalize_token(report.runtime.function);
    const std::string normalized_runtime_group = normalize_token(report.runtime.group);

    const bool mux_claimed = !is_unclaimed_mux_owner(report.runtime.mux_owner);
    const bool gpio_claimed = !is_unclaimed_gpio_owner(report.runtime.gpio_owner);

    if (mux_claimed) {
        if (!normalized_target.empty() &&
            (normalized_target == normalized_runtime_function || normalized_target == normalized_runtime_group)) {
            // Already on requested function/group and owned by a driver: not considered conflict. / 已经处于请求的功能/分组且被驱动占用：不视为冲突。
        } else {
            report.conflict = true;
            report.reason = "MUX owner is " + report.runtime.mux_owner +
                            ". Release or unbind the owner before switching this pin.";
            return report;
        }
    }

    if (gpio_claimed) {
        if (normalized_target == "gpio" || normalized_target == "default" || normalized_target.empty()) {
            report.conflict = true;
            report.reason = "GPIO line is claimed by " + report.runtime.gpio_owner + ".";
            return report;
        }
        report.conflict = true;
        report.reason = "GPIO line is claimed by " + report.runtime.gpio_owner +
                        ". Free the GPIO consumer before switching to altenate function.";
        return report;
    }

    report.conflict = false;
    report.reason = "No conflict detected.";
    return report;
}

PinConflictReport Controller::check_conflict(const std::string& pin_name,
                                             const std::string& target_function_or_group) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return check_conflict(pin_id.bank, pin_id.pin, target_function_or_group);
}

bool Controller::unbind_owner_device(const std::string& owner) {
    if (owner.empty() || is_unclaimed_mux_owner(owner) || is_unclaimed_gpio_owner(owner)) {
        return true;
    }

    const std::array<std::string, 3> buses{"platform", "i2c", "spi"};
    for (const std::string& bus : buses) {
        const std::string direct = "/sys/bus/" + bus + "/devices/" + owner + "/driver/unbind";
        if (write_text_file(direct, owner + "\n")) {
            return true;
        }
    }

    for (const std::string& bus : buses) {
        const std::filesystem::path drivers_root("/sys/bus/" + bus + "/drivers");
        std::error_code ec;
        if (!std::filesystem::exists(drivers_root, ec)) {
            continue;
        }
        for (const auto& entry : std::filesystem::directory_iterator(drivers_root, ec)) {
            if (ec) {
                break;
            }
            const std::filesystem::path unbind_path = entry.path() / "unbind";
            if (write_text_file(unbind_path.string(), owner + "\n")) {
                return true;
            }
        }
    }
    return false;
}

bool Controller::bind_owner_device(const std::string& owner) {
    if (owner.empty() || is_unclaimed_mux_owner(owner) || is_unclaimed_gpio_owner(owner)) {
        return false;
    }

    const std::array<std::string, 3> buses{"platform", "i2c", "spi"};
    for (const std::string& bus : buses) {
        const std::string direct = "/sys/bus/" + bus + "/devices/" + owner + "/driver/bind";
        if (write_text_file(direct, owner + "\n")) {
            return true;
        }
    }

    for (const std::string& bus : buses) {
        const std::filesystem::path drivers_root("/sys/bus/" + bus + "/drivers");
        std::error_code ec;
        if (!std::filesystem::exists(drivers_root, ec)) {
            continue;
        }
        for (const auto& entry : std::filesystem::directory_iterator(drivers_root, ec)) {
            if (ec) {
                break;
            }
            const std::filesystem::path bind_path = entry.path() / "bind";
            if (write_text_file(bind_path.string(), owner + "\n")) {
                return true;
            }
        }
    }
    return false;
}

std::string Controller::normalize_linux_bus_arg(const std::string& bus) {
    std::string token = normalize_token(bus);
    if (token == "spidev") {
        return "spi";
    }
    return token;
}

std::string Controller::normalize_linux_driver_arg(const std::string& driver) {
    std::string token = trim_copy(driver);
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (token == "rockchip_spi") {
        return "rockchip-spi";
    }
    if (token == "spi-dev" || token == "spi_dev") {
        return "spidev";
    }
    return token;
}

std::string Controller::normalize_linux_device_arg(const std::string& bus, const std::string& device) {
    const std::string bus_token = normalize_linux_bus_arg(bus);
    std::string token = trim_copy(device);
    if (token.empty()) {
        return "";
    }

    std::replace(token.begin(), token.end(), '\\', '/');
    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (bus_token == "spi") {
        std::smatch match;
        static const std::regex kSpidevPatten(R"(spidev([0-9]+)\.([0-9]+))");
        static const std::regex kSpiDevPatten(R"(spi([0-9]+)\.([0-9]+))");
        static const std::regex kBarePatten(R"(([0-9]+)\.([0-9]+))");
        if (std::regex_match(token, match, kSpidevPatten) || std::regex_match(token, match, kSpiDevPatten) ||
            std::regex_match(token, match, kBarePatten)) {
            return "spi" + match[1].str() + "." + match[2].str();
        }
        return token;
    }

    if (bus_token != "platform") {
        return token;
    }

    if (token.find('.') != std::string::npos) {
        return token;
    }

    const auto owner_from_dt_alias = [&](const std::string& alias) -> std::string {
        if (alias.empty()) {
            return "";
        }
        std::string target;
        for (const auto& root :
             {std::filesystem::path("/sys/firmware/devicetree/base/aliases"),
              std::filesystem::path("/proc/device-tree/aliases")}) {
            target = read_file_flatten_nuls(root / alias);
            if (!target.empty()) {
                break;
            }
        }
        if (target.empty()) {
            return "";
        }

        const size_t at = target.find('@');
        if (at == std::string::npos || at + 1 >= target.size()) {
            return "";
        }
        size_t end = at + 1;
        while (end < target.size() && std::isxdigit(static_cast<unsigned char>(target[end]))) {
            ++end;
        }
        if (end == at + 1) {
            return "";
        }

        std::string addr = target.substr(at + 1, end - (at + 1));
        std::transform(addr.begin(), addr.end(), addr.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (alias.rfind("spi", 0) == 0) {
            return addr + ".spi";
        }
        if (alias.rfind("i2c", 0) == 0) {
            return addr + ".i2c";
        }
        if (alias.rfind("pwm", 0) == 0) {
            return addr + ".pwm";
        }
        if (alias.rfind("uart", 0) == 0 || alias.rfind("serial", 0) == 0) {
            return addr + ".serial";
        }
        return "";
    };

    std::vector<std::string> aliases;
    append_unique(&aliases, token);
    std::smatch match;
    static const std::regex kUartPatten(R"(uart([0-9]+))");
    static const std::regex kSerialPatten(R"(serial([0-9]+))");
    if (std::regex_match(token, match, kUartPatten)) {
        append_unique(&aliases, "serial" + match[1].str());
    } else if (std::regex_match(token, match, kSerialPatten)) {
        append_unique(&aliases, "uart" + match[1].str());
    }

    for (const auto& alias : aliases) {
        const std::string owner = owner_from_dt_alias(alias);
        if (!owner.empty()) {
            return owner;
        }
    }

    static const std::array<std::pair<const char*, const char*>, 33> kRv1106PlatformAliases{{
        {"spi0", "ff500000.spi"},
        {"spi1", "ff510000.spi"},
        {"spi2", "ffac0000.spi"},
        {"uart0", "ff4a0000.serial"},
        {"serial0", "ff4a0000.serial"},
        {"uart1", "ff4b0000.serial"},
        {"serial1", "ff4b0000.serial"},
        {"uart2", "ff4c0000.serial"},
        {"serial2", "ff4c0000.serial"},
        {"uart3", "ff4d0000.serial"},
        {"serial3", "ff4d0000.serial"},
        {"uart4", "ff4e0000.serial"},
        {"serial4", "ff4e0000.serial"},
        {"uart5", "ff4f0000.serial"},
        {"serial5", "ff4f0000.serial"},
        {"i2c0", "ff310000.i2c"},
        {"i2c1", "ff320000.i2c"},
        {"i2c2", "ff450000.i2c"},
        {"i2c3", "ff460000.i2c"},
        {"i2c4", "ff470000.i2c"},
        {"pwm0", "ff350000.pwm"},
        {"pwm1", "ff350010.pwm"},
        {"pwm2", "ff350020.pwm"},
        {"pwm3", "ff350030.pwm"},
        {"pwm4", "ff360000.pwm"},
        {"pwm5", "ff360010.pwm"},
        {"pwm6", "ff360020.pwm"},
        {"pwm7", "ff360030.pwm"},
        {"pwm8", "ff490000.pwm"},
        {"pwm9", "ff490010.pwm"},
        {"pwm10", "ff490020.pwm"},
        {"pwm11", "ff490030.pwm"},
        {"npu", "ff660000.npu"},
    }};
    for (const auto& item : kRv1106PlatformAliases) {
        if (token == item.first) {
            return item.second;
        }
    }
    return token;
}

std::string Controller::read_bound_driver_name(const std::filesystem::path& device_path) {
    std::error_code ec;
    const std::filesystem::path driver_link = device_path / "driver";
    if (!std::filesystem::exists(driver_link, ec)) {
        return "";
    }
    std::filesystem::path target = std::filesystem::read_symlink(driver_link, ec);
    if (ec || target.empty()) {
        target = std::filesystem::canonical(driver_link, ec);
    }
    if (ec || target.empty()) {
        return "";
    }
    return target.filename().string();
}

int Controller::parse_spi_bus_number(const std::string& spi_device) {
    std::smatch match;
    static const std::regex kSpiDevPatten(R"(spi([0-9]+)\.([0-9]+))");
    if (!std::regex_match(spi_device, match, kSpiDevPatten)) {
        return -1;
    }
    return std::stoi(match[1].str());
}

bool Controller::parse_spi_device_parts(const std::string& spi_device, int* bus, int* chip_select) {
    if (!bus || !chip_select) {
        return false;
    }
    std::smatch match;
    static const std::regex kSpiDevPatten(R"(spi([0-9]+)\.([0-9]+))");
    if (!std::regex_match(spi_device, match, kSpiDevPatten)) {
        return false;
    }
    *bus = std::stoi(match[1].str());
    *chip_select = std::stoi(match[2].str());
    return true;
}

int Controller::parse_numbered_peripheral(const std::string& token, const std::string& prefix) {
    const std::string normalized = normalize_token(token);
    const std::string wanted = normalize_token(prefix);
    if (normalized.rfind(wanted, 0) != 0) {
        return -1;
    }
    const std::string suffix = normalized.substr(wanted.size());
    if (suffix.empty() ||
        !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return -1;
    }
    return std::stoi(suffix);
}

std::vector<PeripheralPinAssignment> Controller::infer_spi_pin_group(
    int bus,
    int chip_select,
    const std::vector<std::string>& pins) const {
    if (bus < 0) {
        throw std::invalid_argument("SPI bus must be >= 0.");
    }
    if (chip_select < 0) {
        throw std::invalid_argument("SPI chip_select must be >= 0.");
    }
    if (pins.size() < 3) {
        throw std::invalid_argument("SPI pin list must include at least clk, cs, and one data pin (mosi or miso).");
    }

    struct Candidate {
        PeripheralPinAssignment assignment;
        std::string group_base;
        int cs = -1;
    };

    const std::string expected_function = "spi" + std::to_string(bus);
    std::vector<std::vector<Candidate>> all_candidates;
    all_candidates.reserve(pins.size());

    static const std::regex kSpiGroupPattern(R"(^((spi[0-9]+m[0-9]+))-(clk|sck|mosi|miso|cs([0-9]+))$)");
    for (const std::string& pin_name : pins) {
        const PinId pin_id = parse_pin_name(pin_name);
        const std::string label = format_pin_label(pin_id.bank, pin_id.pin);
        std::vector<Candidate> candidates;
        for (const PinAltFunction& entry : list_functions(pin_id.bank, pin_id.pin)) {
            if (normalize_token(entry.function) != expected_function) {
                continue;
            }
            const std::string group = normalize_token(entry.group);
            std::smatch match;
            if (!std::regex_match(group, match, kSpiGroupPattern)) {
                continue;
            }

            std::string role = match[3].str();
            int candidate_cs = -1;
            if (role.rfind("cs", 0) == 0) {
                candidate_cs = std::stoi(match[4].str());
                if (candidate_cs != chip_select) {
                    continue;
                }
            }
            if (role == "sck") {
                role = "clk";
            }

            Candidate candidate;
            candidate.assignment.pin = label;
            candidate.assignment.function = entry.function;
            candidate.assignment.group = entry.group;
            candidate.assignment.role = role;
            candidate.assignment.mux = entry.mux;
            candidate.group_base = match[1].str();
            candidate.cs = candidate_cs;
            candidates.push_back(std::move(candidate));
        }

        if (candidates.empty()) {
            throw std::invalid_argument(label + " has no " + expected_function +
                                        " alternate function matching chip-select " + std::to_string(chip_select) + ".");
        }
        all_candidates.push_back(std::move(candidates));
    }

    std::vector<std::string> valid_group_bases;
    for (const Candidate& first : all_candidates.front()) {
        bool all_have_group = true;
        for (const auto& pin_candidates : all_candidates) {
            const bool found = std::any_of(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
                return item.group_base == first.group_base;
            });
            if (!found) {
                all_have_group = false;
                break;
            }
        }
        if (all_have_group) {
            append_unique(&valid_group_bases, first.group_base);
        }
    }

    if (valid_group_bases.empty()) {
        throw std::invalid_argument("SPI pins do not belong to one common " + expected_function +
                                    " mux group. Check that all pins are from the same spiXmY set.");
    }
    if (valid_group_bases.size() > 1) {
        throw std::invalid_argument("SPI pins are ambiguous across groups: " + join_csv(valid_group_bases) +
                                    ". Add or remove pins so exactly one spiXmY group matches.");
    }

    const std::string selected_group_base = valid_group_bases.front();
    std::vector<PeripheralPinAssignment> selected;
    std::unordered_set<std::string> roles;
    selected.reserve(pins.size());
    for (const auto& pin_candidates : all_candidates) {
        auto it = std::find_if(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
            return item.group_base == selected_group_base;
        });
        if (it == pin_candidates.end()) {
            throw std::logic_error("Internal SPI pin inference mismatch.");
        }
        if (!roles.insert(it->assignment.role).second) {
            throw std::invalid_argument("Duplicate SPI role '" + it->assignment.role +
                                        "' in pin list. Use one pin per role.");
        }
        selected.push_back(it->assignment);
    }

    const bool has_clk = roles.count("clk") > 0;
    const bool has_cs = roles.count("cs" + std::to_string(chip_select)) > 0 || roles.count("cs") > 0;
    const bool has_mosi = roles.count("mosi") > 0;
    const bool has_miso = roles.count("miso") > 0;
    if (!has_clk || !has_cs || (!has_mosi && !has_miso)) {
        throw std::invalid_argument("SPI pin list must contain clk, cs" + std::to_string(chip_select) +
                                    ", and at least one of mosi/miso from the same group.");
    }

    std::sort(selected.begin(), selected.end(), [](const PeripheralPinAssignment& lhs,
                                                   const PeripheralPinAssignment& rhs) {
        static const std::array<const char*, 4> order{{"clk", "mosi", "miso", "cs"}};
        const auto index_of = [&](const std::string& role) {
            for (size_t i = 0; i < order.size(); ++i) {
                if (role == order[i] || (role.rfind("cs", 0) == 0 && std::string(order[i]) == "cs")) {
                    return i;
                }
            }
            return order.size();
        };
        const size_t li = index_of(lhs.role);
        const size_t ri = index_of(rhs.role);
        if (li != ri) {
            return li < ri;
        }
        return lhs.pin < rhs.pin;
    });
    return selected;
}

std::vector<PeripheralPinAssignment> Controller::infer_uart_pin_group(
    int bus,
    const std::vector<std::string>& pins) const {
    if (bus < 0) {
        throw std::invalid_argument("UART bus must be >= 0.");
    }
    if (pins.empty()) {
        throw std::invalid_argument("UART pin list must include at least tx or rx.");
    }

    struct Candidate {
        PeripheralPinAssignment assignment;
        std::string group_base;
    };

    const std::string expected_function = "uart" + std::to_string(bus);
    std::vector<std::vector<Candidate>> all_candidates;
    all_candidates.reserve(pins.size());
    static const std::regex kUartGroupPattern(R"(^((uart[0-9]+m[0-9]+))-(tx|rx|rts|cts|xfer)$)");

    for (const std::string& pin_name : pins) {
        const PinId pin_id = parse_pin_name(pin_name);
        const std::string label = format_pin_label(pin_id.bank, pin_id.pin);
        std::vector<Candidate> candidates;
        for (const PinAltFunction& entry : list_functions(pin_id.bank, pin_id.pin)) {
            if (normalize_token(entry.function) != expected_function) {
                continue;
            }
            const std::string group = normalize_token(entry.group);
            std::smatch match;
            if (!std::regex_match(group, match, kUartGroupPattern)) {
                continue;
            }
            std::string role = match[3].str();
            if (role == "xfer") {
                role = "txrx";
            }
            Candidate candidate;
            candidate.assignment.pin = label;
            candidate.assignment.function = entry.function;
            candidate.assignment.group = entry.group;
            candidate.assignment.role = role;
            candidate.assignment.mux = entry.mux;
            candidate.group_base = match[1].str();
            candidates.push_back(std::move(candidate));
        }
        if (candidates.empty()) {
            throw std::invalid_argument(label + " has no " + expected_function + " alternate function.");
        }
        all_candidates.push_back(std::move(candidates));
    }

    std::vector<std::string> valid_group_bases;
    for (const Candidate& first : all_candidates.front()) {
        bool all_have_group = true;
        for (const auto& pin_candidates : all_candidates) {
            const bool found = std::any_of(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
                return item.group_base == first.group_base;
            });
            if (!found) {
                all_have_group = false;
                break;
            }
        }
        if (all_have_group) {
            append_unique(&valid_group_bases, first.group_base);
        }
    }

    if (valid_group_bases.empty()) {
        throw std::invalid_argument("UART pins do not belong to one common " + expected_function +
                                    " mux group.");
    }
    if (valid_group_bases.size() > 1) {
        throw std::invalid_argument("UART pins are ambiguous across groups: " + join_csv(valid_group_bases) + ".");
    }

    const std::string selected_group_base = valid_group_bases.front();
    std::vector<PeripheralPinAssignment> selected;
    std::unordered_set<std::string> roles;
    for (const auto& pin_candidates : all_candidates) {
        auto it = std::find_if(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
            return item.group_base == selected_group_base && item.assignment.role != "txrx";
        });
        if (it == pin_candidates.end()) {
            it = std::find_if(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
                return item.group_base == selected_group_base;
            });
        }
        if (it == pin_candidates.end()) {
            throw std::logic_error("Internal UART pin inference mismatch.");
        }
        if (!roles.insert(it->assignment.role).second) {
            throw std::invalid_argument("Duplicate UART role '" + it->assignment.role +
                                        "' in pin list. Use one pin per role.");
        }
        selected.push_back(it->assignment);
    }
    if (roles.count("tx") == 0 && roles.count("rx") == 0 && roles.count("txrx") == 0) {
        throw std::invalid_argument("UART pin list must include at least tx or rx.");
    }
    return selected;
}

std::vector<PeripheralPinAssignment> Controller::infer_i2c_pin_group(
    int bus,
    const std::vector<std::string>& pins) const {
    if (bus < 0) {
        throw std::invalid_argument("I2C bus must be >= 0.");
    }
    if (pins.size() < 2) {
        throw std::invalid_argument("I2C pin list must include scl and sda.");
    }

    struct Candidate {
        PeripheralPinAssignment assignment;
        std::string group_base;
    };

    const std::string expected_function = "i2c" + std::to_string(bus);
    std::vector<std::vector<Candidate>> all_candidates;
    all_candidates.reserve(pins.size());
    static const std::regex kI2cGroupPattern(R"(^((i2c[0-9]+m[0-9]+))-(scl|sda|xfer)$)");

    for (const std::string& pin_name : pins) {
        const PinId pin_id = parse_pin_name(pin_name);
        const std::string label = format_pin_label(pin_id.bank, pin_id.pin);
        std::vector<Candidate> candidates;
        for (const PinAltFunction& entry : list_functions(pin_id.bank, pin_id.pin)) {
            if (normalize_token(entry.function) != expected_function) {
                continue;
            }
            const std::string group = normalize_token(entry.group);
            std::smatch match;
            if (!std::regex_match(group, match, kI2cGroupPattern)) {
                continue;
            }

            Candidate candidate;
            candidate.assignment.pin = label;
            candidate.assignment.function = entry.function;
            candidate.assignment.group = entry.group;
            candidate.assignment.role = match[3].str();
            candidate.assignment.mux = entry.mux;
            candidate.group_base = match[1].str();
            candidates.push_back(std::move(candidate));
        }
        if (candidates.empty()) {
            throw std::invalid_argument(label + " has no " + expected_function + " alternate function.");
        }
        all_candidates.push_back(std::move(candidates));
    }

    std::vector<std::string> valid_group_bases;
    for (const Candidate& first : all_candidates.front()) {
        bool all_have_group = true;
        for (const auto& pin_candidates : all_candidates) {
            const bool found = std::any_of(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
                return item.group_base == first.group_base;
            });
            if (!found) {
                all_have_group = false;
                break;
            }
        }
        if (all_have_group) {
            append_unique(&valid_group_bases, first.group_base);
        }
    }

    if (valid_group_bases.empty()) {
        throw std::invalid_argument("I2C pins do not belong to one common " + expected_function +
                                    " mux group.");
    }
    if (valid_group_bases.size() > 1) {
        throw std::invalid_argument("I2C pins are ambiguous across groups: " + join_csv(valid_group_bases) + ".");
    }

    const std::string selected_group_base = valid_group_bases.front();
    std::vector<PeripheralPinAssignment> selected;
    std::unordered_set<std::string> roles;
    for (const auto& pin_candidates : all_candidates) {
        auto it = std::find_if(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
            return item.group_base == selected_group_base && item.assignment.role != "xfer";
        });
        if (it == pin_candidates.end()) {
            it = std::find_if(pin_candidates.begin(), pin_candidates.end(), [&](const Candidate& item) {
                return item.group_base == selected_group_base;
            });
        }
        if (it == pin_candidates.end()) {
            throw std::logic_error("Internal I2C pin inference mismatch.");
        }
        if (!roles.insert(it->assignment.role).second) {
            throw std::invalid_argument("Duplicate I2C role '" + it->assignment.role +
                                        "' in pin list. Use one pin per role.");
        }
        selected.push_back(it->assignment);
    }

    if (roles.count("scl") == 0 || roles.count("sda") == 0) {
        throw std::invalid_argument("I2C pin list must contain scl and sda from the same group.");
    }

    std::sort(selected.begin(), selected.end(), [](const PeripheralPinAssignment& lhs,
                                                   const PeripheralPinAssignment& rhs) {
        const auto index_of = [](const std::string& role) {
            if (role == "scl") {
                return 0;
            }
            if (role == "sda") {
                return 1;
            }
            return 2;
        };
        const int li = index_of(lhs.role);
        const int ri = index_of(rhs.role);
        if (li != ri) {
            return li < ri;
        }
        return lhs.pin < rhs.pin;
    });
    return selected;
}

std::vector<PeripheralPinAssignment> Controller::infer_pwm_pin_group(
    int channel,
    const std::vector<std::string>& pins) const {
    if (channel < 0) {
        throw std::invalid_argument("PWM channel must be >= 0.");
    }
    if (pins.size() != 1) {
        throw std::invalid_argument("PWM setup expects exactly one output pin.");
    }

    const PinId pin_id = parse_pin_name(pins.front());
    const std::string label = format_pin_label(pin_id.bank, pin_id.pin);
    const std::string expected_function = "pwm" + std::to_string(channel);
    static const std::regex kPwmGroupPattern(R"(^((pwm[0-9]+)(ir)?m[0-9]+)$)");

    std::vector<PeripheralPinAssignment> candidates;
    for (const PinAltFunction& entry : list_functions(pin_id.bank, pin_id.pin)) {
        if (normalize_token(entry.function) != expected_function) {
            continue;
        }
        const std::string group = normalize_token(entry.group);
        std::smatch match;
        if (!std::regex_match(group, match, kPwmGroupPattern)) {
            continue;
        }

        PeripheralPinAssignment assignment;
        assignment.pin = label;
        assignment.function = entry.function;
        assignment.group = entry.group;
        assignment.role = match[3].matched ? "ir" : "out";
        assignment.mux = entry.mux;
        candidates.push_back(std::move(assignment));
    }

    if (candidates.empty()) {
        throw std::invalid_argument(label + " has no " + expected_function + " alternate function.");
    }
    if (candidates.size() > 1) {
        std::vector<std::string> groups;
        for (const auto& item : candidates) {
            append_unique(&groups, item.group);
        }
        throw std::invalid_argument("PWM pin is ambiguous across groups: " + join_csv(groups) + ".");
    }
    return candidates;
}

bool Controller::create_spi_device_if_missing(const std::string& spi_device, const std::string& modalias) {
    std::error_code ec;
    const std::filesystem::path device_path("/sys/bus/spi/devices/" + spi_device);
    if (std::filesystem::exists(device_path, ec)) {
        return true;
    }

    std::smatch match;
    static const std::regex kSpiDevPatten(R"(spi([0-9]+)\.([0-9]+))");
    if (!std::regex_match(spi_device, match, kSpiDevPatten)) {
        return false;
    }
    const std::string bus = match[1].str();
    const std::string chip_select = match[2].str();
    const std::filesystem::path new_device("/sys/class/spi_master/spi" + bus + "/new_device");
    if (!std::filesystem::exists(new_device, ec)) {
        return false;
    }

    std::vector<std::string> payloads;
    append_unique(&payloads, modalias + " " + chip_select + "\n");
    append_unique(&payloads, "spidev " + chip_select + "\n");
    append_unique(&payloads, "rockchip,spidev " + chip_select + "\n");
    for (const auto& payload : payloads) {
        if (write_text_file(new_device.string(), payload)) {
            ::usleep(100000);
            if (std::filesystem::exists(device_path, ec)) {
                return true;
            }
        }
    }
    return std::filesystem::exists(device_path, ec);
}

bool Controller::release_owner(const std::string& owner) const {
    return unbind_owner_device(owner);
}

std::string Controller::get_bound_driver(const std::string& bus, const std::string& device) const {
    const std::string bus_token = normalize_linux_bus_arg(bus);
    const std::string dev_token = normalize_linux_device_arg(bus_token, device);
    if (bus_token.empty() || dev_token.empty()) {
        return "";
    }
    return read_bound_driver_name("/sys/bus/" + bus_token + "/devices/" + dev_token);
}

bool Controller::unbind_driver(const std::string& bus, const std::string& device) const {
    const std::string bus_token = normalize_linux_bus_arg(bus);
    const std::string dev_token = normalize_linux_device_arg(bus_token, device);
    if (bus_token.empty() || dev_token.empty()) {
        return false;
    }

    const std::filesystem::path device_path("/sys/bus/" + bus_token + "/devices/" + dev_token);
    const std::string current_driver = read_bound_driver_name(device_path);
    if (current_driver.empty()) {
        return true;
    }

    const std::filesystem::path direct_unbind = device_path / "driver" / "unbind";
    if (write_text_file(direct_unbind.string(), dev_token + "\n")) {
        return true;
    }
    const std::filesystem::path driver_unbind("/sys/bus/" + bus_token + "/drivers/" + current_driver + "/unbind");
    return write_text_file(driver_unbind.string(), dev_token + "\n");
}

bool Controller::bind_driver(const std::string& bus,
                             const std::string& device,
                             const std::string& driver,
                             bool unbind_current) const {
    const std::string bus_token = normalize_linux_bus_arg(bus);
    const std::string dev_token = normalize_linux_device_arg(bus_token, device);
    const std::string driver_token = normalize_linux_driver_arg(driver);
    if (bus_token.empty() || dev_token.empty() || driver_token.empty()) {
        return false;
    }

    std::error_code ec;
    const std::filesystem::path device_path("/sys/bus/" + bus_token + "/devices/" + dev_token);
    if (!std::filesystem::exists(device_path, ec)) {
        if (bus_token != "spi" || !create_spi_device_if_missing(dev_token, driver_token)) {
            return false;
        }
    }

    const std::string current_driver = read_bound_driver_name(device_path);
    if (normalize_linux_driver_arg(current_driver) == driver_token) {
        return true;
    }
    if (!current_driver.empty()) {
        if (!unbind_current) {
            return false;
        }
        if (!unbind_driver(bus_token, dev_token)) {
            return false;
        }
    }

    const std::filesystem::path override_path = device_path / "driver_override";
    if (std::filesystem::exists(override_path, ec)) {
        (void)write_text_file(override_path.string(), driver_token + "\n");
    }

    const std::filesystem::path bind_path("/sys/bus/" + bus_token + "/drivers/" + driver_token + "/bind");
    if (!write_text_file(bind_path.string(), dev_token + "\n")) {
        return normalize_linux_driver_arg(read_bound_driver_name(device_path)) == driver_token;
    }
    return normalize_linux_driver_arg(read_bound_driver_name(device_path)) == driver_token;
}

std::string Controller::spi_get_bound_driver(const std::string& spi_device) const {
    return get_bound_driver("spi", spi_device);
}

bool Controller::spi_bind_driver(const std::string& spi_device,
                                 const std::string& driver,
                                 bool unbind_current) const {
    return bind_driver("spi", spi_device, driver, unbind_current);
}

bool Controller::spi_bind_spidev(const std::string& spi_device) const {
    return spi_bind_driver(spi_device, "spidev", true);
}

SpiSetupStatus Controller::setup_spi(int bus,
                                     const std::vector<std::string>& pins,
                                     int chip_select,
                                     bool bind_spidev) {
    return setup_spi("spi" + std::to_string(bus) + "." + std::to_string(chip_select), pins, bind_spidev);
}

SpiSetupStatus Controller::setup_spi(const std::string& spi_device,
                                     const std::vector<std::string>& pins,
                                     bool bind_spidev) {
    SpiSetupStatus status;
    const std::string dev_token = normalize_linux_device_arg("spi", spi_device);
    int bus_number = -1;
    int chip_select = -1;
    if (!parse_spi_device_parts(dev_token, &bus_number, &chip_select)) {
        throw std::invalid_argument("Unsupported SPI device name: " + spi_device + ". Expected spi0.0 or /dev/spidev0.0.");
    }

    status.bus = bus_number;
    status.chip_select = chip_select;
    status.device = dev_token;
    status.dev_path = "/dev/spidev" + std::to_string(bus_number) + "." + std::to_string(chip_select);
    status.function = "spi" + std::to_string(bus_number);

    if (!pins.empty()) {
        status.pins = infer_spi_pin_group(bus_number, chip_select, pins);
        if (!status.pins.empty()) {
            const std::string group = normalize_token(status.pins.front().group);
            const size_t dash = group.find('-');
            status.group = dash == std::string::npos ? group : group.substr(0, dash);
        }
        for (const auto& pin : status.pins) {
            set_function(pin.pin, pin.group);
        }
    }

    std::error_code ec;
    const std::string master_name = "spi" + std::to_string(bus_number);
    const std::filesystem::path master_path("/sys/class/spi_master/" + master_name);
    status.controller_bound = std::filesystem::exists(master_path, ec);
    if (!status.controller_bound) {
        const std::string platform_alias = "spi" + std::to_string(bus_number);
        status.controller_bound = bind_driver("platform", platform_alias, "rockchip-spi", true);
        if (!status.controller_bound) {
            const std::string platform_dev = normalize_linux_device_arg("platform", platform_alias);
            status.note = "SPI controller " + platform_alias + " (" + platform_dev +
                          ") is not available. If its device-tree node is disabled, enable it in the boot DTB/overlay.";
        }
    }

    if (bind_spidev) {
        status.child_bound = spi_bind_spidev(dev_token);
        status.ok = status.child_bound && std::filesystem::exists(status.dev_path, ec);
        if (!status.ok && status.note.empty()) {
            status.note = "Failed to bind " + dev_token + " to spidev. Check /sys/class/spi_master/" +
                          master_name + " and /sys/bus/spi/devices/" + dev_token + ".";
        }
    } else {
        status.child_bound = !spi_get_bound_driver(dev_token).empty();
        status.ok = status.controller_bound || !status.pins.empty();
        if (status.ok && !status.controller_bound) {
            status.note = "SPI pinmux is configured. Linux SPI controller is unavailable, so use a direct register backend or enable the controller in DTB/overlay for spidev.";
        }
    }
    return status;
}

bool Controller::spi_prepare(const std::string& spi_device,
                             const std::vector<std::pair<std::string, std::string>>& pin_functions) {
    const std::string dev_token = normalize_linux_device_arg("spi", spi_device);
    const int bus_number = parse_spi_bus_number(dev_token);
    if (bus_number < 0) {
        throw std::invalid_argument("Unsupported SPI device name: " + spi_device + ". Expected spi0.0 or /dev/spidev0.0.");
    }

    for (const auto& item : pin_functions) {
        set_function(item.first, item.second);
    }

    std::error_code ec;
    const std::string master_name = "spi" + std::to_string(bus_number);
    const std::filesystem::path master_path("/sys/class/spi_master/" + master_name);
    if (!std::filesystem::exists(master_path, ec)) {
        const std::string platform_alias = "spi" + std::to_string(bus_number);
        if (!bind_driver("platform", platform_alias, "rockchip-spi", true)) {
            const std::string platform_dev = normalize_linux_device_arg("platform", platform_alias);
            std::cerr << "[PinMux] Warning: SPI controller " << platform_alias << " (" << platform_dev
                      << ") is not available. If its device-tree node is disabled, enable it in the boot DTB/overlay."
                      << std::endl;
        }
    }

    if (!spi_bind_spidev(dev_token)) {
        std::cerr << "[PinMux] Warning: failed to bind " << dev_token
                  << " to spidev. Check /sys/class/spi_master/" << master_name
                  << " and /sys/bus/spi/devices/" << dev_token << "." << std::endl;
        return false;
    }
    return std::filesystem::exists(std::string("/dev/spidev") + dev_token.substr(3), ec);
}

bool Controller::spi_prepare(const std::string& spi_device, const std::vector<std::string>& pins) {
    return setup_spi(spi_device, pins, true).ok;
}

UartSetupStatus Controller::setup_uart(int bus, const std::vector<std::string>& pins, bool bind_driver) {
    return setup_uart("uart" + std::to_string(bus), pins, bind_driver);
}

UartSetupStatus Controller::setup_uart(const std::string& uart_device,
                                       const std::vector<std::string>& pins,
                                       bool bind_driver) {
    UartSetupStatus status;
    int bus = parse_numbered_peripheral(uart_device, "uart");
    if (bus < 0) {
        bus = parse_numbered_peripheral(uart_device, "serial");
    }
    if (bus < 0) {
        throw std::invalid_argument("Unsupported UART name: " + uart_device + ". Expected uart0..uart5 or serial0..serial5.");
    }

    status.bus = bus;
    status.function = "uart" + std::to_string(bus);
    status.device = normalize_linux_device_arg("platform", status.function);
    status.dev_path = "/dev/ttyS" + std::to_string(bus);

    if (!pins.empty()) {
        status.pins = infer_uart_pin_group(bus, pins);
        if (!status.pins.empty()) {
            const std::string group = normalize_token(status.pins.front().group);
            const size_t dash = group.find('-');
            status.group = dash == std::string::npos ? group : group.substr(0, dash);
        }
        for (const auto& pin : status.pins) {
            set_function(pin.pin, pin.group);
        }
    }

    if (bind_driver) {
        status.driver_bound = bind_driver ? this->bind_driver("platform", status.function, "dw-apb-uart", true) : false;
        if (!status.driver_bound) {
            status.driver_bound = !get_bound_driver("platform", status.function).empty();
        }
    } else {
        status.driver_bound = !get_bound_driver("platform", status.function).empty();
    }
    std::error_code ec;
    status.ok = status.driver_bound || std::filesystem::exists(status.dev_path, ec);
    if (!bind_driver && !status.ok && !status.pins.empty()) {
        status.ok = true;
        status.note = "UART pinmux is configured. Linux UART device is unavailable, so use Reg('uart" +
                      std::to_string(bus) + "') or enable the controller in DTB/overlay for /dev/ttyS" +
                      std::to_string(bus) + ".";
    } else if (!status.ok) {
        status.note = "UART platform device " + status.device +
                      " is not available. If its device-tree node is disabled, enable it in the boot DTB/overlay.";
    }
    return status;
}

I2cSetupStatus Controller::setup_i2c(int bus, const std::vector<std::string>& pins, bool bind_driver) {
    return setup_i2c("i2c" + std::to_string(bus), pins, bind_driver);
}

I2cSetupStatus Controller::setup_i2c(const std::string& i2c_device,
                                     const std::vector<std::string>& pins,
                                     bool bind_driver) {
    std::string token = trim_copy(i2c_device);
    std::replace(token.begin(), token.end(), '\\', '/');
    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (token.rfind("i2c-", 0) == 0) {
        token = "i2c" + token.substr(4);
    }

    const int bus = parse_numbered_peripheral(token, "i2c");
    if (bus < 0) {
        throw std::invalid_argument("Unsupported I2C name: " + i2c_device + ". Expected i2c0..i2c4 or /dev/i2c-0.");
    }

    I2cSetupStatus status;
    status.bus = bus;
    status.function = "i2c" + std::to_string(bus);
    status.device = normalize_linux_device_arg("platform", status.function);
    status.dev_path = "/dev/i2c-" + std::to_string(bus);

    if (!pins.empty()) {
        status.pins = infer_i2c_pin_group(bus, pins);
        if (!status.pins.empty()) {
            const std::string group = normalize_token(status.pins.front().group);
            const size_t dash = group.find('-');
            status.group = dash == std::string::npos ? group : group.substr(0, dash);
        }
        for (const auto& pin : status.pins) {
            set_function(pin.pin, pin.group);
        }
    }

    if (bind_driver) {
        status.driver_bound = this->bind_driver("platform", status.function, "rk3x-i2c", true);
        if (!status.driver_bound) {
            status.driver_bound = !get_bound_driver("platform", status.function).empty();
        }
    } else {
        status.driver_bound = !get_bound_driver("platform", status.function).empty();
    }

    std::error_code ec;
    status.ok = status.driver_bound || std::filesystem::exists(status.dev_path, ec);
    if (!bind_driver && !status.ok && !status.pins.empty()) {
        status.ok = true;
        status.note = "I2C pinmux is configured. Linux I2C controller is unavailable, so use Reg('i2c" +
                      std::to_string(bus) + "') or enable the controller in DTB/overlay for /dev/i2c-" +
                      std::to_string(bus) + ".";
    } else if (!status.ok) {
        status.note = "I2C platform device " + status.device +
                      " is not available. If its device-tree node is disabled, enable it in the boot DTB/overlay.";
    }
    return status;
}

PwmSetupStatus Controller::setup_pwm(int channel, const std::vector<std::string>& pins, bool bind_driver) {
    return setup_pwm("pwm" + std::to_string(channel), pins, bind_driver);
}

PwmSetupStatus Controller::setup_pwm(const std::string& pwm_device,
                                     const std::vector<std::string>& pins,
                                     bool bind_driver) {
    std::string token = trim_copy(pwm_device);
    std::replace(token.begin(), token.end(), '\\', '/');
    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    const int channel = parse_numbered_peripheral(token, "pwm");
    if (channel < 0) {
        throw std::invalid_argument("Unsupported PWM name: " + pwm_device + ". Expected pwm0..pwm11.");
    }

    PwmSetupStatus status;
    status.channel = channel;
    status.function = "pwm" + std::to_string(channel);
    status.device = normalize_linux_device_arg("platform", status.function);
    status.dev_path = "/sys/bus/platform/devices/" + status.device;

    if (!pins.empty()) {
        status.pins = infer_pwm_pin_group(channel, pins);
        if (!status.pins.empty()) {
            status.group = normalize_token(status.pins.front().group);
        }
        for (const auto& pin : status.pins) {
            set_function(pin.pin, pin.group);
        }
    }

    if (bind_driver) {
        status.driver_bound = this->bind_driver("platform", status.function, "rockchip-pwm", true);
        if (!status.driver_bound) {
            status.driver_bound = !get_bound_driver("platform", status.function).empty();
        }
    } else {
        status.driver_bound = !get_bound_driver("platform", status.function).empty();
    }

    std::error_code ec;
    status.ok = status.driver_bound || std::filesystem::exists(status.dev_path, ec);
    if (!bind_driver && !status.ok && !status.pins.empty()) {
        status.ok = true;
        status.note = "PWM pinmux is configured. Linux PWM device is unavailable, so use Reg('pwm" +
                      std::to_string(channel) + "') or enable the controller in DTB/overlay for sysfs PWM.";
    } else if (!status.ok) {
        status.note = "PWM platform device " + status.device +
                      " is not available. If its device-tree node is disabled, enable it in the boot DTB/overlay.";
    }
    return status;
}

PeripheralRegisterInfo Controller::get_register_block_info(const std::string& peripheral) const {
    (void)this;
    return resolve_peripheral_register_info(peripheral);
}

RegisterBlock Controller::map_registers(const std::string& peripheral, size_t map_size) const {
    (void)this;
    return RegisterBlock(peripheral, map_size);
}

bool Controller::release_conflict(int bank, int pin) const {
    PinRuntimeStatus status = get_runtime_status(bank, pin);
    if (!status.found) {
        return false;
    }

    bool ok = true;
    if (!is_unclaimed_mux_owner(status.mux_owner)) {
        ok = unbind_owner_device(status.mux_owner) && ok;
    }

    if (!is_unclaimed_gpio_owner(status.gpio_owner)) {
        ok = unbind_owner_device(status.gpio_owner) && ok;
    }
    return ok;
}

bool Controller::release_conflict(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return release_conflict(pin_id.bank, pin_id.pin);
}

void Controller::remember_pin_claim(int bank, int pin, const std::string& target_function_or_group) {
    (void)resolve_field(bank, pin);
    const SavedPinMuxKey key{bank, pin};
    bool need_capture = false;

    {
        std::lock_guard<std::mutex> guard(saved_pinmux_mutex());
        load_saved_pinmux_states_locked();
        need_capture = saved_pinmux_states().find(key) == saved_pinmux_states().end();
    }

    uint32_t original_mux = 0;
    PinRuntimeStatus runtime;
    if (need_capture) {
        original_mux = get_mux(bank, pin);
        runtime = get_runtime_status(bank, pin);
    }

    {
        std::lock_guard<std::mutex> guard(saved_pinmux_mutex());
        load_saved_pinmux_states_locked();
        auto& states = saved_pinmux_states();
        auto it = states.find(key);
        if (it == states.end()) {
            SavedPinMuxState state;
            state.bank = bank;
            state.pin = pin;
            state.mux = original_mux;
            state.mux_owner = runtime.mux_owner;
            state.gpio_owner = runtime.gpio_owner;
            state.function = runtime.function;
            state.group = runtime.group;
            it = states.emplace(key, std::move(state)).first;
            save_saved_pinmux_states_locked();
            std::cerr << "[PinMux] Saved original state for " << format_pin_label(bank, pin)
                      << " before switching to " << target_function_or_group
                      << " (mux=" << original_mux
                      << ", owner=" << runtime.mux_owner << ")." << std::endl;
        }
        ++it->second.ref_count;
    }

    std::lock_guard<std::mutex> local_guard(lock_);
    pin_claims_.push_back(PinKey{bank, pin});
}

void Controller::release_pin_claims(const std::vector<PinKey>& claims) noexcept {
    for (auto it_claim = claims.rbegin(); it_claim != claims.rend(); ++it_claim) {
        try {
            std::lock_guard<std::mutex> guard(saved_pinmux_mutex());
            load_saved_pinmux_states_locked();
            auto& states = saved_pinmux_states();
            const SavedPinMuxKey key{it_claim->bank, it_claim->pin};
            auto it = states.find(key);
            if (it == states.end()) {
                continue;
            }

            if (it->second.ref_count > 0) {
                --it->second.ref_count;
            }
            if (it->second.ref_count > 0) {
                continue;
            }

            const SavedPinMuxState state = it->second;
            bool restored = false;
            try {
                write_mux_untracked(state.bank, state.pin, state.mux);
                restored = true;
            } catch (const std::exception& ex) {
                std::cerr << "[PinMux] Warning: failed to restore mux for "
                          << format_pin_label(state.bank, state.pin) << ": "
                          << ex.what() << std::endl;
            }

            bool owner_restored = true;
            std::vector<std::string> restore_owners;
            if (!is_unclaimed_mux_owner(state.mux_owner)) {
                append_unique(&restore_owners, state.mux_owner);
            }
            if (!is_unclaimed_gpio_owner(state.gpio_owner)) {
                append_unique(&restore_owners, state.gpio_owner);
            }
            for (const auto& owner : restore_owners) {
                if (!restored) {
                    break;
                }
                bool one_owner_restored = bind_owner_device(owner);
                if (!one_owner_restored) {
                    for (const std::string& bus : {"platform", "i2c", "spi"}) {
                        const std::filesystem::path driver_path(
                            "/sys/bus/" + std::string(bus) + "/devices/" + owner + "/driver");
                        std::error_code ec;
                        if (std::filesystem::exists(driver_path, ec)) {
                            one_owner_restored = true;
                            break;
                        }
                    }
                }
                owner_restored = one_owner_restored && owner_restored;
                if (!one_owner_restored) {
                    std::cerr << "[PinMux] Warning: restored mux for "
                              << format_pin_label(state.bank, state.pin)
                              << " but failed to rebind previous owner "
                              << owner << "." << std::endl;
                }
            }

            if (restored && owner_restored) {
                states.erase(it);
                save_saved_pinmux_states_locked();
                std::cerr << "[PinMux] Restored original state for "
                          << format_pin_label(state.bank, state.pin)
                          << " and cleared persistent record." << std::endl;
            } else {
                it->second.ref_count = 0;
                save_saved_pinmux_states_locked();
            }
        } catch (const std::exception& ex) {
            std::cerr << "[PinMux] Warning: pin restore failed: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "[PinMux] Warning: pin restore failed with unknown error." << std::endl;
        }
    }
}

uint32_t Controller::read_be32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) | (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

void Controller::load_function_table_if_needed() const {
    std::lock_guard<std::mutex> guard(lock_);
    if (function_table_loaded_) {
        return;
    }

    function_table_.clear();
    const std::filesystem::path pinctrl_root("/proc/device-tree/pinctrl");
    std::error_code ec;
    if (!std::filesystem::exists(pinctrl_root, ec)) {
        function_table_loaded_ = true;
        return;
    }

    std::filesystem::recursive_directory_iterator it(pinctrl_root, ec);
    std::filesystem::recursive_directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        if (!it->is_regular_file(ec) || ec) {
            continue;
        }
        if (it->path().filename() != "rockchip,pins") {
            continue;
        }

        const std::filesystem::path group_node = it->path().parent_path();
        const std::filesystem::path function_node = group_node.parent_path();
        if (function_node.empty()) {
            continue;
        }

        const std::string group_name = group_node.filename().string();
        const std::string function_name = function_node.filename().string();

        std::ifstream in(it->path(), std::ios::binary);
        if (!in) {
            continue;
        }

        std::vector<uint8_t> payload((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (payload.size() < 16 || (payload.size() % 16) != 0) {
            continue;
        }

        for (size_t offset = 0; offset + 16 <= payload.size(); offset += 16) {
            const uint32_t bank = read_be32(payload.data() + offset);
            const uint32_t pin = read_be32(payload.data() + offset + 4);
            const uint32_t mux = read_be32(payload.data() + offset + 8);

            if (bank >= kRv1106IomuxLayout.size()) {
                continue;
            }
            if (pin >= static_cast<uint32_t>(kRv1106IomuxLayout[bank].pin_count)) {
                continue;
            }

            PinKey key{static_cast<int>(bank), static_cast<int>(pin)};
            std::vector<PinAltFunction>& entries = function_table_[key];

            PinAltFunction candidate{function_name, group_name, mux};
            const bool duplicate = std::any_of(entries.begin(), entries.end(), [&](const PinAltFunction& value) {
                return value.function == candidate.function && value.group == candidate.group && value.mux == candidate.mux;
            });
            if (!duplicate) {
                entries.push_back(std::move(candidate));
            }
        }
    }

    // Some SDK DTBs omit valid altenatives that still exist in TRM/vendor pinmux tables. / 部分 SDK DTB 省略了仍存在于 TRM/厂商 pinmux 表中的有效备选项。
    // Add a minimal manual supplement for communication interfaces we rely on at runtime.
    const auto add_manual_alt = [&](int bank, int pin, const char* function, const char* group, uint32_t mux) {
        const PinKey key{bank, pin};
        std::vector<PinAltFunction>& entries = function_table_[key];
        PinAltFunction candidate{function, group, mux};
        const bool duplicate = std::any_of(entries.begin(), entries.end(), [&](const PinAltFunction& value) {
            return value.function == candidate.function && value.group == candidate.group && value.mux == candidate.mux;
        });
        if (!duplicate) {
            entries.push_back(std::move(candidate));
        }
    };

    // GPIO1_C2/GPIO1_C3 support I2C4 in RV1106 pinmux, but some DTBs don't expose this in pinctrl nodes. / GPIO1_C2/GPIO1_C3 在 RV1106 pinmux 中支持 I2C4，但有些 DTB 没有在 pinctrl 节点中暴露它。
    add_manual_alt(1, 18, "i2c4", "i2c4m1-xfer", 4);  // GPIO1_C2
    add_manual_alt(1, 19, "i2c4", "i2c4m1-xfer", 4);  // GPIO1_C3
    add_manual_alt(3, 23, "i2c4", "i2c4m2-scl", 3);   // GPIO3_C7
    add_manual_alt(3, 24, "i2c4", "i2c4m2-sda", 3);   // GPIO3_D0

    for (size_t i = 0; i < internal::kRv1106PinFunctionTableSize; ++i) {
        const auto& item = internal::kRv1106PinFunctionTable[i];
        add_manual_alt(static_cast<int>(item.bank),
                       static_cast<int>(item.pin),
                       item.function,
                       item.group,
                       item.mux);
    }

    for (auto& item : function_table_) {
        auto& entries = item.second;
        std::sort(entries.begin(), entries.end(), [](const PinAltFunction& lhs, const PinAltFunction& rhs) {
            if (lhs.mux != rhs.mux) {
                return lhs.mux < rhs.mux;
            }
            if (lhs.function != rhs.function) {
                return lhs.function < rhs.function;
            }
            return lhs.group < rhs.group;
        });
    }

    function_table_loaded_ = true;
}

std::vector<PinAltFunction> Controller::list_functions(int bank, int pin) const {
    (void)resolve_field(bank, pin);
    load_function_table_if_needed();

    std::vector<PinAltFunction> entries;
    entries.push_back(PinAltFunction{"gpio", "gpio", 0});

    std::lock_guard<std::mutex> guard(lock_);
    const PinKey key{bank, pin};
    const auto it = function_table_.find(key);
    if (it == function_table_.end()) {
        return entries;
    }

    for (const auto& value : it->second) {
        const bool duplicate = std::any_of(entries.begin(), entries.end(), [&](const PinAltFunction& existing) {
            return existing.function == value.function && existing.group == value.group && existing.mux == value.mux;
        });
        if (!duplicate) {
            entries.push_back(value);
        }
    }
    return entries;
}

std::vector<PinAltFunction> Controller::list_functions(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return list_functions(pin_id.bank, pin_id.pin);
}

std::pair<std::string, std::string> Controller::resolve_function_and_group(const std::string& function_or_group) const {
    const std::string target = normalize_token(function_or_group);
    if (target.empty()) {
        return {"", ""};
    }

    load_function_table_if_needed();
    std::lock_guard<std::mutex> guard(lock_);
    for (const auto& kv : function_table_) {
        for (const auto& entry : kv.second) {
            if (normalize_token(entry.function) == target || normalize_token(entry.group) == target) {
                return {entry.function, entry.group};
            }
        }
    }

    // Fallback: treat request as function name directly. / 回退路径：将请求直接当作功能名处理。
    return {function_or_group, function_or_group};
}

std::string Controller::find_owner_for_function(const std::vector<PinRuntimeStatus>& rows,
                                                const std::string& normalized_function,
                                                const std::string& normalized_group) {
    for (const auto& row : rows) {
        if (is_unclaimed_mux_owner(row.mux_owner)) {
            continue;
        }
        if ((!normalized_function.empty() && normalize_token(row.function) == normalized_function) ||
            (!normalized_group.empty() && normalize_token(row.group) == normalized_group)) {
            return row.mux_owner;
        }
    }

    for (const auto& row : rows) {
        if ((!normalized_function.empty() && normalize_token(row.function) == normalized_function) ||
            (!normalized_group.empty() && normalize_token(row.group) == normalized_group)) {
            return row.mux_owner;
        }
    }
    return "";
}

std::vector<std::string> Controller::collect_interfaces_for_owner(const std::string& owner) const {
    std::vector<std::string> interfaces;
    if (owner.empty() || is_unclaimed_mux_owner(owner)) {
        return interfaces;
    }

    std::vector<std::filesystem::path> roots;
    for (const std::string& bus : {"platform", "i2c", "spi"}) {
        const std::filesystem::path p("/sys/bus/" + std::string(bus) + "/devices/" + owner);
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            roots.push_back(p);
        }
    }
    if (roots.empty()) {
        return interfaces;
    }

    static const std::regex kI2cName(R"(i2c-([0-9]+))");
    static const std::regex kTtyName(R"(tty[A-Za-z0-9]+)");
    static const std::regex kSpidevName(R"(spidev[0-9]+\.[0-9]+)");
    static const std::regex kSpiMasterName(R"(spi[0-9]+)");
    static const std::regex kPwmChipName(R"(pwmchip[0-9]+)");

    for (const auto& root : roots) {
        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(root, ec);
        std::filesystem::recursive_directory_iterator end;
        for (; !ec && it != end; it.increment(ec)) {
            if (ec) {
                break;
            }
            const std::string name = it->path().filename().string();
            std::smatch match;
            if (std::regex_match(name, match, kI2cName)) {
                const std::string dev = "/dev/" + name;
                if (std::filesystem::exists(dev, ec)) {
                    append_unique(&interfaces, dev);
                }
                continue;
            }
            if (std::regex_match(name, kTtyName) || std::regex_match(name, kSpidevName)) {
                const std::string dev = "/dev/" + name;
                if (std::filesystem::exists(dev, ec)) {
                    append_unique(&interfaces, dev);
                }
                continue;
            }
            if (std::regex_match(name, kPwmChipName)) {
                const std::string path = "/sys/class/pwm/" + name;
                if (std::filesystem::exists(path, ec)) {
                    append_unique(&interfaces, path);
                }
                continue;
            }
            if (std::regex_match(name, kSpiMasterName)) {
                const std::string path = "/sys/class/spi_master/" + name;
                if (std::filesystem::exists(path, ec)) {
                    append_unique(&interfaces, path);
                }
                continue;
            }
        }
    }

    // Some kernels expose PWM controller by owner-name symlink under /sys/class/pwm. / 有些内核会通过 /sys/class/pwm 下基于 owner-name 的符号链接暴露 PWM 控制器。
    std::error_code ec;
    const std::filesystem::path pwm_owner_class("/sys/class/pwm/" + owner);
    if (std::filesystem::exists(pwm_owner_class, ec)) {
        append_unique(&interfaces, pwm_owner_class.string());
    }
    return interfaces;
}

FunctionInterfaceStatus Controller::get_interface_status(const std::string& function_or_group) const {
    FunctionInterfaceStatus status;
    status.request = function_or_group;

    const auto function_group = resolve_function_and_group(function_or_group);
    status.function = function_group.first;
    status.group = function_group.second;

    const auto rows = read_runtime_rows();
    const std::string normalized_request = normalize_token(status.request);
    const std::string normalized_function = normalize_token(status.function);
    const std::string normalized_group = normalize_token(status.group);
    status.owner = find_owner_for_function(rows, normalized_function, normalized_group);

    if (status.owner.empty() || is_unclaimed_mux_owner(status.owner)) {
        std::lock_guard<std::mutex> guard(lock_);
        auto it = owner_hint_cache_.find(normalized_function);
        if (it == owner_hint_cache_.end() && !normalized_group.empty()) {
            it = owner_hint_cache_.find(normalized_group);
        }
        if (it != owner_hint_cache_.end()) {
            status.owner = it->second;
        }
    }

    const auto infer_owner_from_alias = [&](const std::string& alias_token) -> std::string {
        if (alias_token.empty()) {
            return "";
        }

        std::vector<std::string> alias_candidates;
        append_unique(&alias_candidates, alias_token);

        std::smatch alias_match;
        static const std::regex kUartAlias(R"(uart([0-9]+))");
        static const std::regex kSerialAlias(R"(serial([0-9]+))");
        if (std::regex_match(alias_token, alias_match, kUartAlias)) {
            append_unique(&alias_candidates, "serial" + alias_match[1].str());
        } else if (std::regex_match(alias_token, alias_match, kSerialAlias)) {
            append_unique(&alias_candidates, "uart" + alias_match[1].str());
        }

        std::string alias_target;
        std::string alias_used;
        for (const auto& candidate : alias_candidates) {
            for (const auto& root :
                 {std::filesystem::path("/sys/firmware/devicetree/base/aliases"),
                  std::filesystem::path("/proc/device-tree/aliases")}) {
                alias_target = read_file_flatten_nuls(root / candidate);
                if (!alias_target.empty()) {
                    alias_used = candidate;
                    break;
                }
            }
            if (!alias_target.empty()) {
                break;
            }
        }
        if (alias_target.empty()) {
            return "";
        }

        std::error_code ec;
        const size_t at = alias_target.find('@');
        if (at == std::string::npos || at + 1 >= alias_target.size()) {
            return "";
        }

        size_t end = at + 1;
        while (end < alias_target.size() && std::isxdigit(static_cast<unsigned char>(alias_target[end]))) {
            ++end;
        }
        if (end == at + 1) {
            return "";
        }

        std::string addr = alias_target.substr(at + 1, end - (at + 1));
        std::transform(addr.begin(), addr.end(), addr.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        auto starts_with = [&](const std::string& prefix) {
            return alias_used.rfind(prefix, 0) == 0;
        };
        if (starts_with("pwm")) {
            return addr + ".pwm";
        }
        if (starts_with("uart") || starts_with("serial")) {
            return addr + ".serial";
        }
        if (starts_with("i2c")) {
            return addr + ".i2c";
        }
        if (starts_with("spi")) {
            return addr + ".spi";
        }

        // Fallback: probe existing platform device names by MMIO address prefix. / 回退路径：按 MMIO 地址前缀探测现有 platform device 名称。
        const std::filesystem::path platform_root("/sys/bus/platform/devices");
        for (const auto& entry : std::filesystem::directory_iterator(platform_root, ec)) {
            if (ec) {
                break;
            }
            const std::string name = entry.path().filename().string();
            std::string normalized_name = normalize_token(name);
            if (normalized_name.rfind(addr + ".", 0) == 0) {
                return name;
            }
        }
        return "";
    };

    if (status.owner.empty() || is_unclaimed_mux_owner(status.owner)) {
        status.owner = infer_owner_from_alias(normalized_request);
        if (status.owner.empty()) {
            status.owner = infer_owner_from_alias(normalized_function);
        }
        if (status.owner.empty() && !normalized_group.empty()) {
            status.owner = infer_owner_from_alias(normalized_group);
        }
        if (!status.owner.empty()) {
            std::lock_guard<std::mutex> guard(lock_);
            if (!normalized_request.empty()) {
                owner_hint_cache_[normalized_request] = status.owner;
            }
            if (!normalized_function.empty()) {
                owner_hint_cache_[normalized_function] = status.owner;
            }
            if (!normalized_group.empty()) {
                owner_hint_cache_[normalized_group] = status.owner;
            }
        }
    }

    const auto collect_interfaces_from_tokens =
        [&](const std::string& request_token, const std::string& function_token, const std::string& group_token) {
            std::vector<std::string> interfaces;
            std::vector<std::string> tokens;
            append_unique(&tokens, request_token);
            append_unique(&tokens, function_token);
            append_unique(&tokens, group_token);

            std::error_code ec;
            static const std::regex kUartToken(R"((?:uart|serial)([0-9]+))");
            static const std::regex kI2cToken(R"(i2c([0-9]+))");
            static const std::regex kSpiToken(R"(spi([0-9]+))");
            static const std::regex kPwmToken(R"(pwm([0-9]+))");

            for (const auto& token : tokens) {
                if (token.empty()) {
                    continue;
                }
                std::smatch match;
                if (std::regex_match(token, match, kUartToken)) {
                    const std::string idx = match[1].str();
                    for (const auto& dev : {"/dev/ttyS" + idx, "/dev/ttyAMA" + idx, "/dev/ttyFIQ" + idx}) {
                        if (std::filesystem::exists(dev, ec)) {
                            append_unique(&interfaces, dev);
                        }
                    }
                }
                if (std::regex_match(token, match, kI2cToken)) {
                    const std::string dev = "/dev/i2c-" + match[1].str();
                    if (std::filesystem::exists(dev, ec)) {
                        append_unique(&interfaces, dev);
                    }
                }
                if (std::regex_match(token, match, kSpiToken)) {
                    const std::string idx = match[1].str();
                    const std::string spi_master = "/sys/class/spi_master/spi" + idx;
                    if (std::filesystem::exists(spi_master, ec)) {
                        append_unique(&interfaces, spi_master);
                    }
                    std::filesystem::path dev_root("/dev");
                    if (std::filesystem::exists(dev_root, ec)) {
                        for (const auto& entry : std::filesystem::directory_iterator(dev_root, ec)) {
                            if (ec) {
                                break;
                            }
                            const std::string name = entry.path().filename().string();
                            if (name.rfind("spidev" + idx + ".", 0) == 0) {
                                append_unique(&interfaces, "/dev/" + name);
                            }
                        }
                    }
                }
                if (std::regex_match(token, match, kPwmToken)) {
                    const std::string path = "/sys/class/pwm/pwmchip" + match[1].str();
                    if (std::filesystem::exists(path, ec)) {
                        append_unique(&interfaces, path);
                    }
                }
            }
            return interfaces;
        };

    const auto direct_interfaces = collect_interfaces_from_tokens(normalized_request, normalized_function, normalized_group);
    for (const auto& item : direct_interfaces) {
        append_unique(&status.interfaces, item);
    }

    if (status.owner.empty() || is_unclaimed_mux_owner(status.owner)) {
        status.owner_bound = !status.interfaces.empty();
        if (status.interfaces.empty()) {
            status.note = "No active kernel owner was found for this function/group.";
        } else {
            status.note = "Userspace interface is visible, but kernel owner inference is unavailable.";
        }
        return status;
    }

    bool bound = false;
    for (const std::string& bus : {"platform", "i2c", "spi"}) {
        const std::filesystem::path driver_path("/sys/bus/" + std::string(bus) + "/devices/" + status.owner + "/driver");
        std::error_code ec;
        if (std::filesystem::exists(driver_path, ec)) {
            bound = true;
            break;
        }
    }
    status.owner_bound = bound;
    status.interfaces = collect_interfaces_for_owner(status.owner);
    for (const auto& item : direct_interfaces) {
        append_unique(&status.interfaces, item);
    }
    if (status.interfaces.empty()) {
        status.note = "Owner exists but no userspace interface node is visible.";
    }
    return status;
}

FunctionInterfaceStatus Controller::ensure_interface(const std::string& function_or_group) const {
    FunctionInterfaceStatus status = get_interface_status(function_or_group);
    if (status.owner.empty() || is_unclaimed_mux_owner(status.owner)) {
        if (!status.interfaces.empty()) {
            status.owner_bound = true;
            status.note = "Userspace interface is ready (owner inference unavailable).";
        } else {
            status.note = "Unable to infer kernel device owner. This usually needs DT overlay/static DTS support.";
        }
        return status;
    }

    const auto try_pwm_active_pinctrl_fix = [&](const std::string& owner, std::string* detail) -> bool {
        if (owner.empty()) {
            return false;
        }

        static const std::regex kPwmOwnerPatten(R"(([0-9a-fA-F]+)\.pwm$)");
        std::smatch match;
        if (!std::regex_match(owner, match, kPwmOwnerPatten)) {
            return false;
        }

        std::string addr = match[1].str();
        std::transform(addr.begin(), addr.end(), addr.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const std::string node_path = "/pwm@" + addr;
        const std::filesystem::path node_fs = std::filesystem::path("/proc/device-tree") / ("pwm@" + addr);
        std::error_code ec;
        if (!std::filesystem::exists(node_fs, ec)) {
            if (detail) {
                *detail = "PWM node not found in live device-tree: " + node_path;
            }
            return false;
        }

        const std::string pinctrl_names = normalize_token(read_file_flatten_nuls(node_fs / "pinctrl-names"));
        if (pinctrl_names.find("active") != std::string::npos) {
            if (detail) {
                *detail = "pinctrl-names already contains active state.";
            }
            return false;
        }

        const bool dtc_available = (::access("/usr/bin/dtc", X_OK) == 0) || (::access("/bin/dtc", X_OK) == 0) ||
                                   (::access("/sbin/dtc", X_OK) == 0);
        if (!dtc_available) {
            if (detail) {
                *detail = "dtc tool is not available on target system.";
            }
            return false;
        }

        const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::string suffix = std::to_string(static_cast<long long>(ts & 0x7fffff));
        const std::filesystem::path dts_path("/tmp/visiong_pinmux_pwmfix_" + addr + "_" + suffix + ".dts");
        const std::filesystem::path dtbo_path("/tmp/visiong_pinmux_pwmfix_" + addr + "_" + suffix + ".dtbo");

        {
            std::ofstream out(dts_path);
            if (!out) {
                if (detail) {
                    *detail = "Failed to create temporary DTS: " + dts_path.string();
                }
                return false;
            }
            out << "/dts-v1/;\n"
                << "/plugin/;\n\n"
                << "/ {\n"
                << "    fragment@0 {\n"
                << "        target-path = \"" << node_path << "\";\n"
                << "        __overlay__ {\n"
                << "            status = \"okay\";\n"
                << "            pinctrl-names = \"active\";\n"
                << "        };\n"
                << "    };\n"
                << "};\n";
        }

        const std::string compile_cmd =
            "dtc -@ -I dts -O dtb -o " + dtbo_path.string() + " " + dts_path.string() + " >/dev/null 2>&1";
        if (std::system(compile_cmd.c_str()) != 0) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = "Failed to compile temporary PWM overlay via dtc.";
            }
            return false;
        }

        std::string overlay_name;
        try {
            overlay_name = apply_overlay(dtbo_path.string(), "visiong_pwmfix_" + addr);
        } catch (const std::exception& ex) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = std::string("Failed to apply generated PWM overlay: ") + ex.what();
            }
            return false;
        }

        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        if (detail) {
            *detail = "Applied generated PWM overlay: " + overlay_name;
        }
        return true;
    };

    const auto try_enable_owner_node_overlay = [&](const std::string& owner, std::string* detail) -> bool {
        if (owner.empty()) {
            return false;
        }

        static const std::regex kOwnerPatten(R"(([0-9a-fA-F]+)\.([A-Za-z0-9_-]+)$)");
        std::smatch match;
        if (!std::regex_match(owner, match, kOwnerPatten)) {
            if (detail) {
                *detail = "Owner format is not recognized for auto-enable: " + owner;
            }
            return false;
        }

        std::string addr = match[1].str();
        std::string kind = normalize_token(match[2].str());
        std::transform(addr.begin(), addr.end(), addr.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        std::string node_path;
        if (kind == "serial") {
            node_path = "/serial@" + addr;
        } else if (kind == "i2c") {
            node_path = "/i2c@" + addr;
        } else if (kind == "spi") {
            node_path = "/spi@" + addr;
        } else if (kind == "pwm") {
            node_path = "/pwm@" + addr;
        } else {
            if (detail) {
                *detail = "Auto-enable is not implemented for owner kind: " + kind;
            }
            return false;
        }

        const std::filesystem::path node_fs = std::filesystem::path("/proc/device-tree") / (node_path.substr(1));
        std::error_code ec;
        if (!std::filesystem::exists(node_fs, ec)) {
            if (detail) {
                *detail = "Live DT node not found: " + node_path;
            }
            return false;
        }

        const std::string current_status = normalize_token(read_file_flatten_nuls(node_fs / "status"));
        if (current_status.empty() || current_status == "okay" || current_status == "ok") {
            if (detail) {
                *detail = "DT node already enabled: " + node_path;
            }
            return false;
        }

        const bool dtc_available = (::access("/usr/bin/dtc", X_OK) == 0) || (::access("/bin/dtc", X_OK) == 0) ||
                                   (::access("/sbin/dtc", X_OK) == 0);
        if (!dtc_available) {
            if (detail) {
                *detail = "dtc tool is not available on target system.";
            }
            return false;
        }

        const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::string suffix = std::to_string(static_cast<long long>(ts & 0x7fffff));
        const std::filesystem::path dts_path("/tmp/visiong_pinmux_enable_" + kind + "_" + addr + "_" + suffix + ".dts");
        const std::filesystem::path dtbo_path("/tmp/visiong_pinmux_enable_" + kind + "_" + addr + "_" + suffix + ".dtbo");

        {
            std::ofstream out(dts_path);
            if (!out) {
                if (detail) {
                    *detail = "Failed to create temporary DTS: " + dts_path.string();
                }
                return false;
            }
            out << "/dts-v1/;\n"
                << "/plugin/;\n\n"
                << "/ {\n"
                << "    fragment@0 {\n"
                << "        target-path = \"" << node_path << "\";\n"
                << "        __overlay__ {\n"
                << "            status = \"okay\";\n"
                << "        };\n"
                << "    };\n"
                << "};\n";
        }

        const std::string compile_cmd =
            "dtc -@ -I dts -O dtb -o " + dtbo_path.string() + " " + dts_path.string() + " >/dev/null 2>&1";
        if (std::system(compile_cmd.c_str()) != 0) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = "Failed to compile temporary DT enable overlay via dtc.";
            }
            return false;
        }

        std::string overlay_name;
        try {
            overlay_name = apply_overlay(dtbo_path.string(), "visiong_enable_" + kind + "_" + addr);
        } catch (const std::exception& ex) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = std::string("Failed to apply generated DT enable overlay: ") + ex.what();
            }
            return false;
        }

        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        if (detail) {
            *detail = "Applied generated DT enable overlay: " + overlay_name;
        }
        return true;
    };

    const auto try_add_spi_spidev_overlay = [&](const std::string& owner, std::string* detail) -> bool {
        if (owner.empty()) {
            return false;
        }

        static const std::regex kSpiOwnerPatten(R"(([0-9a-fA-F]+)\.spi$)");
        std::smatch match;
        if (!std::regex_match(owner, match, kSpiOwnerPatten)) {
            return false;
        }

        std::string addr = match[1].str();
        std::transform(addr.begin(), addr.end(), addr.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const std::string node_path = "/spi@" + addr;
        const std::filesystem::path node_fs = std::filesystem::path("/proc/device-tree") / ("spi@" + addr);
        std::error_code ec;
        if (!std::filesystem::exists(node_fs, ec)) {
            if (detail) {
                *detail = "SPI node not found in live device-tree: " + node_path;
            }
            return false;
        }

        const std::string compat = normalize_token(read_file_flatten_nuls(node_fs / "compatible"));
        if (compat.find("rv1106-spi") == std::string::npos && compat.find("rk3066-spi") == std::string::npos) {
            if (detail) {
                *detail = "SPI node is not a generic rockchip-spi controller: " + node_path;
            }
            return false;
        }

        std::unordered_set<uint32_t> used_cs;
        bool spidev_defined = false;
        for (const auto& entry : std::filesystem::directory_iterator(node_fs, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_directory(ec) || ec) {
                continue;
            }
            const std::filesystem::path reg_path = entry.path() / "reg";
            if (std::filesystem::exists(reg_path, ec)) {
                std::ifstream reg_in(reg_path, std::ios::binary);
                std::array<uint8_t, 4> bytes{};
                if (reg_in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
                    used_cs.insert(read_be32(bytes.data()));
                }
            }
            const std::string child_compat = normalize_token(read_file_flatten_nuls(entry.path() / "compatible"));
            if (child_compat.find("spidev") != std::string::npos || child_compat.find("dh2228fv") != std::string::npos) {
                spidev_defined = true;
            }
        }
        if (spidev_defined) {
            if (detail) {
                *detail = "SPI node already has a spidev-compatible child.";
            }
            return false;
        }

        int cs = -1;
        for (int candidate = 0; candidate <= 3; ++candidate) {
            if (used_cs.find(static_cast<uint32_t>(candidate)) == used_cs.end()) {
                cs = candidate;
                break;
            }
        }
        if (cs < 0) {
            if (detail) {
                *detail = "No free chip-select slot (0..3) for auto spidev node.";
            }
            return false;
        }

        const bool dtc_available = (::access("/usr/bin/dtc", X_OK) == 0) || (::access("/bin/dtc", X_OK) == 0) ||
                                   (::access("/sbin/dtc", X_OK) == 0);
        if (!dtc_available) {
            if (detail) {
                *detail = "dtc tool is not available on target system.";
            }
            return false;
        }

        const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::string suffix = std::to_string(static_cast<long long>(ts & 0x7fffff));
        const std::filesystem::path dts_path("/tmp/visiong_pinmux_spidev_" + addr + "_" + suffix + ".dts");
        const std::filesystem::path dtbo_path("/tmp/visiong_pinmux_spidev_" + addr + "_" + suffix + ".dtbo");

        {
            std::ofstream out(dts_path);
            if (!out) {
                if (detail) {
                    *detail = "Failed to create temporary SPI spidev DTS: " + dts_path.string();
                }
                return false;
            }
            out << "/dts-v1/;\n"
                << "/plugin/;\n\n"
                << "/ {\n"
                << "    fragment@0 {\n"
                << "        target-path = \"" << node_path << "\";\n"
                << "        __overlay__ {\n"
                << "            spidev_auto@" << cs << " {\n"
                << "                compatible = \"rohm,dh2228fv\";\n"
                << "                reg = <" << cs << ">;\n"
                << "                spi-max-frequency = <10000000>;\n"
                << "                status = \"okay\";\n"
                << "            };\n"
                << "        };\n"
                << "    };\n"
                << "};\n";
        }

        const std::string compile_cmd =
            "dtc -@ -I dts -O dtb -o " + dtbo_path.string() + " " + dts_path.string() + " >/dev/null 2>&1";
        if (std::system(compile_cmd.c_str()) != 0) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = "Failed to compile temporary SPI spidev overlay via dtc.";
            }
            return false;
        }

        std::string overlay_name;
        try {
            overlay_name = apply_overlay(dtbo_path.string(), "visiong_spidev_" + addr + "_cs" + std::to_string(cs));
        } catch (const std::exception& ex) {
            std::filesystem::remove(dts_path, ec);
            std::filesystem::remove(dtbo_path, ec);
            if (detail) {
                *detail = std::string("Failed to apply generated SPI spidev overlay: ") + ex.what();
            }
            return false;
        }

        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        if (detail) {
            *detail = "Applied generated SPI spidev overlay: " + overlay_name;
        }
        return true;
    };

    const std::string request_token = normalize_token(status.request);
    const std::string function_token = normalize_token(status.function);
    const std::string group_token = normalize_token(status.group);
    const bool pwm_related =
        request_token.rfind("pwm", 0) == 0 || function_token.rfind("pwm", 0) == 0 || group_token.rfind("pwm", 0) == 0;
    const bool spi_related =
        request_token.rfind("spi", 0) == 0 || function_token.rfind("spi", 0) == 0 || group_token.rfind("spi", 0) == 0;
    const std::string target_owner = status.owner;

    const auto collect_conflicting_mux_owners = [&](const std::string& owner) -> std::vector<std::string> {
        std::vector<std::string> blockers;
        if (owner.empty()) {
            return blockers;
        }

        load_function_table_if_needed();
        std::vector<PinKey> target_pins;
        {
            std::lock_guard<std::mutex> guard(lock_);
            for (const auto& kv : function_table_) {
                for (const auto& entry : kv.second) {
                    if (normalize_token(entry.function) == function_token || normalize_token(entry.group) == group_token) {
                        target_pins.push_back(kv.first);
                        break;
                    }
                }
            }
        }
        if (target_pins.empty()) {
            return blockers;
        }

        const auto rows = read_runtime_rows();
        for (const auto& pin : target_pins) {
            for (const auto& row : rows) {
                if (row.bank != pin.bank || row.pin != pin.pin) {
                    continue;
                }
                if (is_unclaimed_mux_owner(row.mux_owner)) {
                    continue;
                }
                if (normalize_token(row.mux_owner) == normalize_token(owner)) {
                    continue;
                }
                append_unique(&blockers, row.mux_owner);
            }
        }
        return blockers;
    };

    const auto collect_bind_conflict_owners_from_dmesg = [&](const std::string& owner) -> std::vector<std::string> {
        std::vector<std::string> blockers;
        if (owner.empty()) {
            return blockers;
        }

        // Some SDK kernels do not expose pinctrl debug rows; fall back to recent dmesg conflict diagnostics. / 部分 SDK 内核不会暴露 pinctrl 调试行；此时回退到最近的 dmesg 冲突诊断信息。
        std::unique_ptr<FILE, decltype(&pclose)> pipe(::popen("dmesg | tail -n 512", "r"), &pclose);
        if (!pipe) {
            return blockers;
        }

        static const std::regex kClaimConflict(
            R"(already requested by ([^;]+);\s*cannot claim for ([A-Za-z0-9_.-]+))");
        char line_buf[512];
        const std::string target_owner = normalize_token(owner);
        while (std::fgets(line_buf, static_cast<int>(sizeof(line_buf)), pipe.get()) != nullptr) {
            std::string line = trim_copy(line_buf);
            std::smatch match;
            if (!std::regex_search(line, match, kClaimConflict)) {
                continue;
            }
            const std::string blocker = trim_copy(match[1].str());
            const std::string target = normalize_token(match[2].str());
            if (target != target_owner) {
                continue;
            }
            if (blocker.empty() || is_unclaimed_mux_owner(blocker)) {
                continue;
            }
            append_unique(&blockers, blocker);
        }
        return blockers;
    };

    const auto try_release_conflicts_and_rebind = [&](std::string* detail) -> bool {
        const std::string owner_for_bind = status.owner.empty() ? target_owner : status.owner;
        if (owner_for_bind.empty()) {
            if (detail) {
                *detail = "Target owner is empty, cannot rebind.";
            }
            return false;
        }

        std::vector<std::string> blockers = collect_conflicting_mux_owners(owner_for_bind);
        const auto dmesg_blockers = collect_bind_conflict_owners_from_dmesg(owner_for_bind);
        for (const auto& item : dmesg_blockers) {
            append_unique(&blockers, item);
        }
        if (blockers.empty()) {
            if (detail) {
                *detail = "No pinctrl owner conflicts detected for target function/group.";
            }
            return false;
        }

        std::vector<std::string> released;
        for (const auto& blocker : blockers) {
            if (unbind_owner_device(blocker)) {
                append_unique(&released, blocker);
            }
        }

        if (released.empty()) {
            if (detail) {
                *detail = "Detected pinctrl conflicts but failed to unbind blockers: " + join_csv(blockers);
            }
            return false;
        }

        (void)bind_owner_device(owner_for_bind);
        status = get_interface_status(function_or_group);
        if (status.owner_bound) {
            if (detail) {
                *detail = "Released conflicting owners and rebound target: " + join_csv(released);
            }
            return true;
        }

        if (detail) {
            *detail = "Released blockers (" + join_csv(released) + ") but bind still failed.";
        }
        return false;
    };

    const std::string initial_owner = status.owner;
    if (!status.owner_bound) {
        const bool bound = bind_owner_device(status.owner);
        status = get_interface_status(function_or_group);
        if (!bound || !status.owner_bound) {
            if (status.owner.empty()) {
                status.owner = initial_owner;
            }

            if (pwm_related) {
                std::string fix_detail;
                if (try_pwm_active_pinctrl_fix(status.owner, &fix_detail)) {
                    (void)bind_owner_device(status.owner);
                    status = get_interface_status(function_or_group);
                    if (status.owner_bound) {
                        status.note = "Auto-fixed PWM pinctrl active state; kernel device is now bound.";
                    } else {
                        std::string release_detail;
                        if (try_release_conflicts_and_rebind(&release_detail)) {
                            status.note = "Auto-fixed PWM and released conflicts. " + release_detail;
                        } else {
                            status.note = "Applied PWM pinctrl fix, but device bind still failed. " + fix_detail;
                            return status;
                        }
                    }
                } else {
                    std::string release_detail;
                    if (try_release_conflicts_and_rebind(&release_detail)) {
                        status.note = "Resolved bind failure by releasing conflicts. " + release_detail;
                    } else {
                        status.note = "Failed to bind owner device automatically. " + fix_detail;
                        return status;
                    }
                }
            } else {
                std::string fix_detail;
                if (try_enable_owner_node_overlay(status.owner, &fix_detail)) {
                    (void)bind_owner_device(status.owner);
                    status = get_interface_status(function_or_group);
                    if (status.owner_bound) {
                        status.note = "Auto-enabled DT node; kernel device is now bound.";
                    } else {
                        std::string release_detail;
                        if (try_release_conflicts_and_rebind(&release_detail)) {
                            status.note = "Auto-enabled DT node and released conflicts. " + release_detail;
                        } else {
                            status.note = "Applied DT enable overlay, but device bind still failed. " + fix_detail;
                            return status;
                        }
                    }
                } else {
                    std::string release_detail;
                    if (try_release_conflicts_and_rebind(&release_detail)) {
                        status.note = "Resolved bind failure by releasing conflicts. " + release_detail;
                    } else {
                        status.note = "Failed to bind owner device automatically. " + fix_detail;
                        return status;
                    }
                }
            }
        }
    }

    if (status.interfaces.empty()) {
        if (pwm_related) {
            std::string fix_detail;
            if (try_pwm_active_pinctrl_fix(status.owner, &fix_detail)) {
                (void)bind_owner_device(status.owner);
                status = get_interface_status(function_or_group);
            }
        }
    }

    if (spi_related && status.owner_bound) {
        auto has_prefix = [&](const std::string& prefix) {
            for (const auto& item : status.interfaces) {
                if (item.rfind(prefix, 0) == 0) {
                    return true;
                }
            }
            return false;
        };

        bool has_spidev = has_prefix("/dev/spidev");
        bool has_spi_master = has_prefix("/sys/class/spi_master/");
        if (has_spi_master && !has_spidev) {
            std::string spidev_detail;
            if (try_add_spi_spidev_overlay(status.owner, &spidev_detail)) {
                status = get_interface_status(function_or_group);
                has_spidev = has_prefix("/dev/spidev");
                has_spi_master = has_prefix("/sys/class/spi_master/");
            }
            if (has_spidev) {
                status.note = "Userspace interface is ready.";
                return status;
            }
            status.note = "SPI controller is ready, but no /dev/spidev node is present (DTS child node required).";
            return status;
        }
    }

    if (status.interfaces.empty()) {
        status.note = "Driver is bound but no userspace node exposed. DTS child nodes may be missing.";
    } else {
        status.note = "Userspace interface is ready.";
    }
    return status;
}

std::vector<std::string> Controller::list_overlays() const {
    std::vector<std::string> overlays;
    const std::filesystem::path root("/sys/kernel/config/device-tree/overlays");
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        return overlays;
    }

    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_directory(ec)) {
            overlays.push_back(entry.path().filename().string());
        }
    }
    std::sort(overlays.begin(), overlays.end());
    return overlays;
}

std::string Controller::apply_overlay(const std::string& dtbo_path, const std::string& overlay_name) const {
    const std::filesystem::path dtbo(dtbo_path);
    std::error_code ec;
    if (!std::filesystem::exists(dtbo, ec)) {
        throw std::runtime_error("DTBO file not found: " + dtbo_path);
    }

    std::string name = overlay_name;
    if (name.empty()) {
        name = dtbo.stem().string();
    }
    if (name.empty()) {
        name = "overlay";
    }
    for (char& c : name) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
        if (!ok) {
            c = '_';
        }
    }
    if (name == "." || name == "..") {
        name = "overlay";
    }

    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    name += "_" + std::to_string(static_cast<long long>(ts & 0x7fffff));

    const std::filesystem::path root("/sys/kernel/config/device-tree/overlays");
    const std::filesystem::path ov_dir = root / name;
    if (!std::filesystem::exists(root, ec)) {
        throw std::runtime_error("Configfs overlay root not available: " + root.string());
    }
    if (!std::filesystem::create_directory(ov_dir, ec)) {
        throw std::runtime_error("Failed to create overlay dir " + ov_dir.string() + ": " + ec.message());
    }

    std::ifstream in(dtbo, std::ios::binary);
    if (!in) {
        std::filesystem::remove(ov_dir, ec);
        throw std::runtime_error("Failed to open DTBO file: " + dtbo_path);
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::filesystem::path dtbo_node = ov_dir / "dtbo";
    const int fd = ::open(dtbo_node.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        std::filesystem::remove(ov_dir, ec);
        throw std::runtime_error("Failed to open overlay dtbo node: " + dtbo_node.string() + " (" + std::strerror(errno) + ")");
    }

    ssize_t written = 0;
    while (written < static_cast<ssize_t>(blob.size())) {
        const ssize_t n = ::write(fd, blob.data() + written, blob.size() - static_cast<size_t>(written));
        if (n <= 0) {
            const int err = errno;
            ::close(fd);
            std::filesystem::remove(ov_dir, ec);
            throw std::runtime_error("Failed to apply overlay '" + name + "': " + std::strerror(err));
        }
        written += n;
    }
    ::close(fd);

    const std::filesystem::path status_node = ov_dir / "status";
    if (std::filesystem::exists(status_node, ec)) {
        const bool wrote = write_text_file(status_node.string(), "1\n") || write_text_file(status_node.string(), "1");
        if (!wrote) {
            const std::string err = std::strerror(errno);
            std::filesystem::remove(ov_dir, ec);
            throw std::runtime_error("Failed to activate overlay '" + name + "' via status node: " + err);
        }
        const std::string active = normalize_token(read_file_flatten_nuls(status_node));
        if (!active.empty() && active != "1") {
            std::filesystem::remove(ov_dir, ec);
            throw std::runtime_error("Overlay '" + name + "' did not become active (status=" + active + ").");
        }
    }

    return name;
}

bool Controller::remove_overlay(const std::string& overlay_name) const {
    if (overlay_name.empty()) {
        return false;
    }
    const std::filesystem::path ov_dir("/sys/kernel/config/device-tree/overlays/" + overlay_name);
    std::error_code ec;
    if (!std::filesystem::exists(ov_dir, ec)) {
        return false;
    }
    return std::filesystem::remove(ov_dir, ec);
}

uint32_t Controller::resolve_function_mux(int bank, int pin, const std::string& function_or_group) const {
    const std::string token = normalize_token(function_or_group);
    if (token.empty()) {
        throw std::invalid_argument("function name cannot be empty.");
    }
    if (token == "gpio" || token == "gpio0" || token == "default") {
        return 0;
    }

    const std::vector<PinAltFunction> entries = list_functions(bank, pin);
    for (const auto& entry : entries) {
        if (normalize_token(entry.function) == token || normalize_token(entry.group) == token) {
            return entry.mux;
        }
    }

    const std::string label = format_pin_label(bank, pin);
    throw std::invalid_argument("Function/group '" + function_or_group + "' is not available for " + label +
                                ". Available: " + join_options(entries));
}

void Controller::set_function(int bank, int pin, const std::string& function_or_group) {
    const PinConflictReport conflict = check_conflict(bank, pin, function_or_group);
    remember_pin_claim(bank, pin, function_or_group);
    if (conflict.conflict) {
        std::cerr << "[PinMux] Warning: overriding pin conflict on " << format_pin_label(bank, pin)
                  << ": " << conflict.reason << std::endl;
        (void)release_conflict(bank, pin);
    }
    const uint32_t mux = resolve_function_mux(bank, pin, function_or_group);
    write_mux_untracked(bank, pin, mux);
}

void Controller::set_function(const std::string& pin_name, const std::string& function_or_group) {
    const PinId pin_id = parse_pin_name(pin_name);
    set_function(pin_id.bank, pin_id.pin, function_or_group);
}

void Controller::set_functions(const std::vector<std::pair<std::string, std::string>>& pin_functions) {
    for (const auto& item : pin_functions) {
        set_function(item.first, item.second);
    }
}

std::string Controller::get_function_name(int bank, int pin) const {
    const uint32_t current_mux = get_mux(bank, pin);
    if (current_mux == 0) {
        return "gpio";
    }

    const std::vector<PinAltFunction> entries = list_functions(bank, pin);
    for (const auto& entry : entries) {
        if (entry.mux == current_mux) {
            return entry.function;
        }
    }
    return "mux" + std::to_string(current_mux);
}

std::string Controller::get_function_name(const std::string& pin_name) const {
    const PinId pin_id = parse_pin_name(pin_name);
    return get_function_name(pin_id.bank, pin_id.pin);
}

}  // namespace visiong::pinmux
