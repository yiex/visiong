// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_NPU_CLOCK_H
#define VISIONG_CORE_NPU_CLOCK_H

#include <cstdint>
#include <string>
#include <vector>

namespace visiong::pinmux {

struct NpuClockStatus {
    bool npu_node_present = false;
    bool debugfs_available = false;
    bool overlay_configfs_available = false;
    uint32_t assigned_rate_hz = 0;
    uint32_t current_rate_hz = 0;
    uint32_t npu_root_rate_hz = 0;
    uint32_t clk500m_src_rate_hz = 0;
    std::string note;
};

struct NpuClockApplyResult {
    bool ok = false;
    bool rebind_attempted = false;
    bool rebind_ok = false;
    bool reboot_required = false;
    uint32_t requested_rate_hz = 0;
    uint32_t assigned_rate_hz = 0;
    uint32_t current_rate_hz = 0;
    uint32_t npu_root_rate_hz = 0;
    uint32_t clk500m_src_rate_hz = 0;
    std::string overlay_name;
    std::string message;
};

class NpuClock final {
public:
    NpuClock() = default;

    NpuClockStatus status() const;

    std::vector<uint32_t> supported_rates_hz() const;
    std::vector<uint32_t> supported_rates_mhz() const;

    std::vector<std::string> list_overlays(const std::string& prefix = "visiong_npuclk_") const;
    bool remove_overlay(const std::string& overlay_name) const;

    NpuClockApplyResult set_rate(uint32_t rate_hz,
                                 bool update_cru_clk500m_src = true,
                                 bool unbind_rebind_npu = false,
                                 bool allow_unsafe_rate = false) const;

    NpuClockApplyResult set_rate_hz(uint32_t rate_hz,
                                    bool update_cru_clk500m_src = true,
                                    bool unbind_rebind_npu = false,
                                    bool allow_unsafe_rate = false) const {
        return set_rate(rate_hz, update_cru_clk500m_src, unbind_rebind_npu, allow_unsafe_rate);
    }

    NpuClockApplyResult set_rate_mhz(uint32_t rate_mhz,
                                     bool update_cru_clk500m_src = true,
                                     bool unbind_rebind_npu = false,
                                     bool allow_unsafe_rate = false) const;

    bool request_reboot() const;

private:
    static bool is_supported_rate(uint32_t rate_hz);
};

}  // namespace visiong::pinmux

#endif  // VISIONG_CORE_NPU_CLOCK_H

