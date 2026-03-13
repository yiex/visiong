// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/npu_clock.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace visiong::pinmux {

namespace {

constexpr const char* kNpuNodePath = "/sys/firmware/devicetree/base/npu@ff660000";
constexpr const char* kNpuAssignedRatesPath = "/sys/firmware/devicetree/base/npu@ff660000/assigned-clock-rates";
constexpr const char* kCruAssignedRatesPath =
    "/sys/firmware/devicetree/base/clock-controller@ff3a0000/assigned-clock-rates";
constexpr const char* kOverlayRootPath = "/sys/kernel/config/device-tree/overlays";

constexpr const char* kClkRknnRatePath = "/sys/kernel/debug/clk/aclk_rknn/clk_rate";
constexpr const char* kClkNpuRootRatePath = "/sys/kernel/debug/clk/aclk_npu_root/clk_rate";
constexpr const char* kClk500mSrcRatePath = "/sys/kernel/debug/clk/clk_500m_src/clk_rate";
constexpr const char* kDebugfsClkDir = "/sys/kernel/debug/clk";

constexpr const char* kRknpuUnbindPath = "/sys/bus/platform/drivers/RKNPU/unbind";
constexpr const char* kRknpuBindPath = "/sys/bus/platform/drivers/RKNPU/bind";
constexpr const char* kRknpuDeviceId = "ff660000.npu";

constexpr std::array<uint32_t, 7> kSupportedRatesHz{{
    200000000u,
    300000000u,
    400000000u,
    420000000u,
    500000000u,
    600000000u,
    700000000u,
}};


bool write_text_file(const std::string& path, const std::string& text) {
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

bool read_text_u32(const std::string& path, uint32_t* value) {
    if (!value) {
        return false;
    }
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    unsigned long long parsed = 0;
    in >> parsed;
    if (!in || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

std::vector<uint32_t> read_dt_be32_array(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty() || (bytes.size() % 4) != 0) {
        return {};
    }

    std::vector<uint32_t> values;
    values.reserve(bytes.size() / 4);
    for (size_t i = 0; i < bytes.size(); i += 4) {
        const uint32_t v = (static_cast<uint32_t>(bytes[i]) << 24) |
                           (static_cast<uint32_t>(bytes[i + 1]) << 16) |
                           (static_cast<uint32_t>(bytes[i + 2]) << 8) |
                           static_cast<uint32_t>(bytes[i + 3]);
        values.push_back(v);
    }
    return values;
}

bool ensure_debugfs_mounted() {
    std::error_code ec;
    if (std::filesystem::exists(kDebugfsClkDir, ec)) {
        return true;
    }
    (void)std::system("mount -t debugfs debugfs /sys/kernel/debug >/dev/null 2>&1");
    return std::filesystem::exists(kDebugfsClkDir, ec);
}

bool dtc_available() {
    return (::access("/usr/bin/dtc", X_OK) == 0) || (::access("/bin/dtc", X_OK) == 0) ||
           (::access("/sbin/dtc", X_OK) == 0);
}

std::string join_uints(const std::vector<uint32_t>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    return oss.str();
}

std::string make_overlay_dts(uint32_t rate_hz, bool with_cru_fragment, const std::vector<uint32_t>& cru_rates) {
    std::ostringstream oss;
    oss << "/dts-v1/;\n"
        << "/plugin/;\n\n"
        << "/ {\n"
        << "    fragment@0 {\n"
        << "        target-path = \"/npu@ff660000\";\n"
        << "        __overlay__ {\n"
        << "            assigned-clock-rates = <" << rate_hz << ">;\n"
        << "        };\n"
        << "    };\n";

    if (with_cru_fragment && !cru_rates.empty()) {
        oss << "    fragment@1 {\n"
            << "        target-path = \"/clock-controller@ff3a0000\";\n"
            << "        __overlay__ {\n"
            << "            assigned-clock-rates = <";
        for (size_t i = 0; i < cru_rates.size(); ++i) {
            if (i > 0) {
                oss << " ";
            }
            oss << cru_rates[i];
        }
        oss << ">;\n"
            << "        };\n"
            << "    };\n";
    }

    oss << "};\n";
    return oss.str();
}

bool compile_dts_to_dtbo(const std::filesystem::path& dts_path,
                         const std::filesystem::path& dtbo_path,
                         std::string* error_text) {
    const std::string cmd =
        "dtc -@ -I dts -O dtb -o " + dtbo_path.string() + " " + dts_path.string() + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0) {
        return true;
    }
    if (error_text) {
        *error_text = "Failed to compile DTS via dtc: " + dts_path.string();
    }
    return false;
}

bool apply_overlay_blob(const std::vector<char>& blob, const std::string& overlay_name, std::string* error_text) {
    if (blob.empty()) {
        if (error_text) {
            *error_text = "Overlay blob is empty.";
        }
        return false;
    }

    std::error_code ec;
    const std::filesystem::path root(kOverlayRootPath);
    if (!std::filesystem::exists(root, ec)) {
        if (error_text) {
            *error_text = "Overlay configfs path is not available: " + root.string();
        }
        return false;
    }

    std::filesystem::path dir = root / overlay_name;
    if (!std::filesystem::create_directory(dir, ec)) {
        if (error_text) {
            *error_text = "Failed to create overlay dir " + dir.string() + ": " + ec.message();
        }
        return false;
    }

    const std::filesystem::path dtbo_node = dir / "dtbo";
    const int fd = ::open(dtbo_node.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        if (error_text) {
            *error_text = "Failed to open overlay dtbo node: " + dtbo_node.string() + " (" + std::strerror(errno) + ")";
        }
        std::filesystem::remove(dir, ec);
        return false;
    }

    ssize_t written = 0;
    while (written < static_cast<ssize_t>(blob.size())) {
        const ssize_t n = ::write(fd, blob.data() + written, blob.size() - static_cast<size_t>(written));
        if (n <= 0) {
            const int err = errno;
            ::close(fd);
            std::filesystem::remove(dir, ec);
            if (error_text) {
                *error_text = "Failed writing overlay dtbo: " + std::string(std::strerror(err));
            }
            return false;
        }
        written += n;
    }
    ::close(fd);

    const std::filesystem::path status_node = dir / "status";
    if (std::filesystem::exists(status_node, ec)) {
        const bool ok = write_text_file(status_node.string(), "1\n") || write_text_file(status_node.string(), "1");
        if (!ok) {
            const std::string err = std::strerror(errno);
            std::filesystem::remove(dir, ec);
            if (error_text) {
                *error_text = "Failed to activate overlay via status node: " + err;
            }
            return false;
        }
    }
    return true;
}

bool try_rebind_rknpu(std::string* error_text) {
    const bool unbound = write_text_file(kRknpuUnbindPath, std::string(kRknpuDeviceId) + "\n") ||
                         write_text_file(kRknpuUnbindPath, kRknpuDeviceId);
    if (!unbound) {
        if (error_text) {
            *error_text = "Failed to unbind " + std::string(kRknpuDeviceId) + ": " + std::strerror(errno);
        }
        return false;
    }

    ::usleep(100 * 1000);

    const bool bound = write_text_file(kRknpuBindPath, std::string(kRknpuDeviceId) + "\n") ||
                       write_text_file(kRknpuBindPath, kRknpuDeviceId);
    if (!bound) {
        if (error_text) {
            *error_text = "Failed to bind " + std::string(kRknpuDeviceId) + ": " + std::strerror(errno);
        }
        return false;
    }
    return true;
}

}  // namespace

bool NpuClock::is_supported_rate(uint32_t rate_hz) {
    return std::find(kSupportedRatesHz.begin(), kSupportedRatesHz.end(), rate_hz) != kSupportedRatesHz.end();
}

NpuClockStatus NpuClock::status() const {
    NpuClockStatus s;
    std::error_code ec;

    s.npu_node_present = std::filesystem::exists(kNpuNodePath, ec);
    s.overlay_configfs_available = std::filesystem::exists(kOverlayRootPath, ec);

    const std::vector<uint32_t> assigned = read_dt_be32_array(kNpuAssignedRatesPath);
    if (!assigned.empty()) {
        s.assigned_rate_hz = assigned[0];
    }

    s.debugfs_available = ensure_debugfs_mounted();
    if (s.debugfs_available) {
        (void)read_text_u32(kClkRknnRatePath, &s.current_rate_hz);
        (void)read_text_u32(kClkNpuRootRatePath, &s.npu_root_rate_hz);
        (void)read_text_u32(kClk500mSrcRatePath, &s.clk500m_src_rate_hz);
    }

    if (!s.npu_node_present) {
        s.note = "NPU DT node /npu@ff660000 is missing.";
    } else if (!s.overlay_configfs_available) {
        s.note = "Configfs DT overlay path is missing; runtime DT patch is unavailable.";
    } else if (!s.debugfs_available) {
        s.note = "debugfs clock tree is unavailable; runtime clock readback is disabled.";
    } else if (s.current_rate_hz == 0) {
        s.note = "Clock readback is unavailable for aclk_rknn; assigned clock may require reboot to apply.";
    } else {
        s.note = "NPU clock status is available.";
    }

    return s;
}

std::vector<uint32_t> NpuClock::supported_rates_hz() const {
    return std::vector<uint32_t>(kSupportedRatesHz.begin(), kSupportedRatesHz.end());
}

std::vector<uint32_t> NpuClock::supported_rates_mhz() const {
    std::vector<uint32_t> out;
    out.reserve(kSupportedRatesHz.size());
    for (const uint32_t rate_hz : kSupportedRatesHz) {
        out.push_back(rate_hz / 1000000u);
    }
    return out;
}

std::vector<std::string> NpuClock::list_overlays(const std::string& prefix) const {
    std::vector<std::string> names;
    const std::filesystem::path root(kOverlayRootPath);
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        return names;
    }

    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_directory(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (prefix.empty() || name.rfind(prefix, 0) == 0) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool NpuClock::remove_overlay(const std::string& overlay_name) const {
    if (overlay_name.empty()) {
        return false;
    }
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(kOverlayRootPath) / overlay_name;
    if (!std::filesystem::exists(dir, ec)) {
        return false;
    }
    return std::filesystem::remove(dir, ec);
}

NpuClockApplyResult NpuClock::set_rate(uint32_t rate_hz,
                                       bool update_cru_clk500m_src,
                                       bool unbind_rebind_npu,
                                       bool allow_unsafe_rate) const {
    NpuClockApplyResult result;
    result.requested_rate_hz = rate_hz;

    if (rate_hz == 0) {
        throw std::invalid_argument("NPU rate must be > 0 Hz.");
    }

    if (!allow_unsafe_rate && !is_supported_rate(rate_hz)) {
        throw std::invalid_argument("Unsupported NPU rate " + std::to_string(rate_hz) +
                                    " Hz. Supported rates: " + join_uints(supported_rates_hz()));
    }

    NpuClockStatus before = status();
    if (!before.npu_node_present) {
        throw std::runtime_error("NPU DT node /npu@ff660000 is not present.");
    }
    if (!before.overlay_configfs_available) {
        throw std::runtime_error("Configfs DT overlay path is unavailable: " + std::string(kOverlayRootPath));
    }
    if (!dtc_available()) {
        throw std::runtime_error("dtc tool is not available on target system.");
    }

    std::vector<uint32_t> cru_rates = read_dt_be32_array(kCruAssignedRatesPath);
    bool with_cru_fragment = false;
    if (update_cru_clk500m_src && !cru_rates.empty()) {
        // On RV1106 SDK clock-controller node, the last item maps to CLK_500M_SRC. / 在 RV1106 SDK clock-controller node, last item 映射 以 CLK_500M_SRC.
        if (cru_rates.size() >= 11) {
            cru_rates.back() = rate_hz;
            with_cru_fragment = true;
        }
    }

    for (const std::string& old_overlay : list_overlays("visiong_npuclk_")) {
        (void)remove_overlay(old_overlay);
    }

    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string suffix = std::to_string(static_cast<long long>(ts & 0x7fffff));
    const std::string mhz = std::to_string(rate_hz / 1000000u);
    const std::string overlay_name = "visiong_npuclk_" + mhz + "m_" + suffix;

    const std::filesystem::path dts_path("/tmp/" + overlay_name + ".dts");
    const std::filesystem::path dtbo_path("/tmp/" + overlay_name + ".dtbo");

    {
        std::ofstream out(dts_path);
        if (!out) {
            throw std::runtime_error("Failed to create temporary DTS: " + dts_path.string());
        }
        out << make_overlay_dts(rate_hz, with_cru_fragment, cru_rates);
    }

    std::error_code ec;
    std::string error_text;
    if (!compile_dts_to_dtbo(dts_path, dtbo_path, &error_text)) {
        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        throw std::runtime_error(error_text);
    }

    std::ifstream in(dtbo_path, std::ios::binary);
    if (!in) {
        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        throw std::runtime_error("Failed to open temporary DTBO: " + dtbo_path.string());
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    if (!apply_overlay_blob(blob, overlay_name, &error_text)) {
        std::filesystem::remove(dts_path, ec);
        std::filesystem::remove(dtbo_path, ec);
        throw std::runtime_error(error_text);
    }

    std::filesystem::remove(dts_path, ec);
    std::filesystem::remove(dtbo_path, ec);

    result.overlay_name = overlay_name;

    if (unbind_rebind_npu) {
        result.rebind_attempted = true;
        std::string rebind_error;
        result.rebind_ok = try_rebind_rknpu(&rebind_error);
        if (!result.rebind_ok) {
            result.message = rebind_error;
        }
    }

    NpuClockStatus after = status();
    result.assigned_rate_hz = after.assigned_rate_hz;
    result.current_rate_hz = after.current_rate_hz;
    result.npu_root_rate_hz = after.npu_root_rate_hz;
    result.clk500m_src_rate_hz = after.clk500m_src_rate_hz;

    result.ok = (result.assigned_rate_hz == rate_hz);
    result.reboot_required = false;

    if (!result.ok) {
        result.message = "Overlay applied but assigned-clock-rates readback is " +
                         std::to_string(result.assigned_rate_hz) + " Hz (expected " + std::to_string(rate_hz) + " Hz).";
        return result;
    }

    if (result.current_rate_hz == rate_hz) {
        result.message = "NPU clock applied successfully (runtime clock is now " + std::to_string(rate_hz) + " Hz).";
    } else if (result.current_rate_hz == 0u) {
        result.message = "Assigned NPU rate updated to " + std::to_string(rate_hz) +
                         " Hz. Runtime clock readback is unavailable. Use unbind_rebind_npu=True for immediate effect (overlay is runtime-only).";
    } else {
        result.message = "Assigned NPU rate updated to " + std::to_string(rate_hz) +
                         " Hz, but runtime aclk_rknn is still " + std::to_string(result.current_rate_hz) +
                         " Hz. Rebind is required for immediate effect (overlay is runtime-only and does not persist across reboot).";
    }

    return result;
}

NpuClockApplyResult NpuClock::set_rate_mhz(uint32_t rate_mhz,
                                           bool update_cru_clk500m_src,
                                           bool unbind_rebind_npu,
                                           bool allow_unsafe_rate) const {
    if (rate_mhz == 0u) {
        throw std::invalid_argument("NPU rate in MHz must be > 0.");
    }
    if (rate_mhz > (std::numeric_limits<uint32_t>::max() / 1000000u)) {
        throw std::invalid_argument("NPU MHz value is too large.");
    }
    return set_rate(rate_mhz * 1000000u, update_cru_clk500m_src, unbind_rebind_npu, allow_unsafe_rate);
}

bool NpuClock::request_reboot() const {
    (void)std::system("sync");
    return std::system("reboot") == 0;
}

}  // namespace visiong::pinmux

