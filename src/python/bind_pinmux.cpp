// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/core/pinmux.h"
#include "visiong/core/npu_clock.h"

namespace vp = visiong::pinmux;

void bind_pinmux(py::module_& m) {
    py::class_<vp::PinId>(m, "PinId", "Resolved pin identifier (bank + pin index).")
        .def_readonly("bank", &vp::PinId::bank)
        .def_readonly("pin", &vp::PinId::pin)
        .def("__repr__", [](const vp::PinId& id) {
            return "PinId(bank=" + std::to_string(id.bank) + ", pin=" + std::to_string(id.pin) + ")";
        });

    py::class_<vp::RegisterInfo>(m, "PinMuxRegisterInfo", "Raw IOMUX register field information for one pin.")
        .def_readonly("domain", &vp::RegisterInfo::domain, "Register domain: 'pmuioc' or 'ioc'.")
        .def_readonly("base_addr", &vp::RegisterInfo::base_addr, "Physical base address of the register domain.")
        .def_readonly("reg_offset", &vp::RegisterInfo::reg_offset, "Offset inside the register domain.")
        .def_readonly("absolute_addr", &vp::RegisterInfo::absolute_addr, "Absolute physical register address.")
        .def_readonly("bit", &vp::RegisterInfo::bit, "Bit index of mux field inside the register.")
        .def_readonly("width", &vp::RegisterInfo::width, "Mux field width in bits.")
        .def_readonly("mask", &vp::RegisterInfo::mask, "Mux field mask before bit-shift.")
        .def_readonly("gpio_only", &vp::RegisterInfo::gpio_only, "True if this pin is GPIO-only and has no mux register field.")
        .def("__repr__", [](const vp::RegisterInfo& info) {
            return "PinMuxRegisterInfo(domain='" + info.domain +
                   "', base=0x" + py::str("{:08x}").format(info.base_addr).cast<std::string>() +
                   ", offset=0x" + py::str("{:x}").format(info.reg_offset).cast<std::string>() +
                   ", bit=" + std::to_string(info.bit) + ", mask=0x" +
                   py::str("{:x}").format(info.mask).cast<std::string>() +
                   ", gpio_only=" + (info.gpio_only ? "True" : "False") + ")";
        });

    py::class_<vp::PinAltFunction>(m, "PinAltFunction", "Alternative function description for one pin.")
        .def_readonly("function", &vp::PinAltFunction::function)
        .def_readonly("group", &vp::PinAltFunction::group)
        .def_readonly("mux", &vp::PinAltFunction::mux)
        .def("__repr__", [](const vp::PinAltFunction& item) {
            return "PinAltFunction(function='" + item.function + "', group='" + item.group +
                   "', mux=" + std::to_string(item.mux) + ")";
        });

    py::class_<vp::PinRuntimeStatus>(m, "PinRuntimeStatus", "Runtime pin ownership status from debugfs pinctrl.")
        .def_readonly("found", &vp::PinRuntimeStatus::found)
        .def_readonly("bank", &vp::PinRuntimeStatus::bank)
        .def_readonly("pin", &vp::PinRuntimeStatus::pin)
        .def_readonly("mux_owner", &vp::PinRuntimeStatus::mux_owner)
        .def_readonly("gpio_owner", &vp::PinRuntimeStatus::gpio_owner)
        .def_readonly("function", &vp::PinRuntimeStatus::function)
        .def_readonly("group", &vp::PinRuntimeStatus::group);

    py::class_<vp::PinConflictReport>(m, "PinConflictReport", "Pin conflict detection report.")
        .def_readonly("conflict", &vp::PinConflictReport::conflict)
        .def_readonly("reason", &vp::PinConflictReport::reason)
        .def_readonly("runtime", &vp::PinConflictReport::runtime);

    py::class_<vp::FunctionInterfaceStatus>(m, "FunctionInterfaceStatus",
                                            "Kernel interface exposure status for one pin function.")
        .def_readonly("request", &vp::FunctionInterfaceStatus::request)
        .def_readonly("function", &vp::FunctionInterfaceStatus::function)
        .def_readonly("group", &vp::FunctionInterfaceStatus::group)
        .def_readonly("owner", &vp::FunctionInterfaceStatus::owner)
        .def_readonly("owner_bound", &vp::FunctionInterfaceStatus::owner_bound)
        .def_readonly("interfaces", &vp::FunctionInterfaceStatus::interfaces)
        .def_readonly("note", &vp::FunctionInterfaceStatus::note);

    py::class_<vp::AdcChannelStatus>(m, "AdcChannelStatus", "IIO ADC channel readout status.")
        .def_readonly("available", &vp::AdcChannelStatus::available)
        .def_readonly("channel", &vp::AdcChannelStatus::channel)
        .def_readonly("raw", &vp::AdcChannelStatus::raw)
        .def_readonly("scale", &vp::AdcChannelStatus::scale)
        .def_readonly("millivolts", &vp::AdcChannelStatus::millivolts)
        .def_readonly("device", &vp::AdcChannelStatus::device)
        .def_readonly("raw_path", &vp::AdcChannelStatus::raw_path)
        .def_readonly("scale_path", &vp::AdcChannelStatus::scale_path)
        .def_readonly("pin_hint", &vp::AdcChannelStatus::pin_hint)
        .def_readonly("note", &vp::AdcChannelStatus::note);

    py::class_<vp::GpioLineConfig>(m, "GpioLineConfig", "GPIO line request options (linux gpio-v2).")
        .def(py::init<>())
        .def_readwrite("direction", &vp::GpioLineConfig::direction)
        .def_readwrite("bias", &vp::GpioLineConfig::bias)
        .def_readwrite("drive", &vp::GpioLineConfig::drive)
        .def_readwrite("drive_strength_level", &vp::GpioLineConfig::drive_strength_level)
        .def_readwrite("drive_strength_ma", &vp::GpioLineConfig::drive_strength_ma)
        .def_readwrite("active_low", &vp::GpioLineConfig::active_low)
        .def_readwrite("default_value", &vp::GpioLineConfig::default_value)
        .def_readwrite("consumer", &vp::GpioLineConfig::consumer);

    py::class_<vp::GpioLineStatus>(m, "GpioLineStatus", "Requested GPIO line runtime status.")
        .def_readonly("requested", &vp::GpioLineStatus::requested)
        .def_readonly("value", &vp::GpioLineStatus::value)
        .def_readonly("bank", &vp::GpioLineStatus::bank)
        .def_readonly("pin", &vp::GpioLineStatus::pin)
        .def_readonly("gpiochip", &vp::GpioLineStatus::gpiochip)
        .def_readonly("config", &vp::GpioLineStatus::config)
        .def_readonly("note", &vp::GpioLineStatus::note);

    py::class_<vp::DriveStrengthStatus>(m, "DriveStrengthStatus", "RV1106 IOC drive strength register status.")
        .def_readonly("available", &vp::DriveStrengthStatus::available)
        .def_readonly("level", &vp::DriveStrengthStatus::level)
        .def_readonly("raw", &vp::DriveStrengthStatus::raw)
        .def_readonly("reg_offset", &vp::DriveStrengthStatus::reg_offset)
        .def_readonly("absolute_addr", &vp::DriveStrengthStatus::absolute_addr)
        .def_readonly("domain", &vp::DriveStrengthStatus::domain)
        .def_readonly("note", &vp::DriveStrengthStatus::note);

    py::class_<vp::PullStatus>(m, "PullStatus", "RV1106 IOC pull-up/down register status.")
        .def_readonly("available", &vp::PullStatus::available)
        .def_readonly("mode", &vp::PullStatus::mode)
        .def_readonly("raw", &vp::PullStatus::raw)
        .def_readonly("reg_offset", &vp::PullStatus::reg_offset)
        .def_readonly("absolute_addr", &vp::PullStatus::absolute_addr)
        .def_readonly("domain", &vp::PullStatus::domain)
        .def_readonly("note", &vp::PullStatus::note);

    py::class_<vp::SchmittStatus>(m, "SchmittStatus", "RV1106 IOC input schmitt register status.")
        .def_readonly("available", &vp::SchmittStatus::available)
        .def_readonly("enabled", &vp::SchmittStatus::enabled)
        .def_readonly("raw", &vp::SchmittStatus::raw)
        .def_readonly("reg_offset", &vp::SchmittStatus::reg_offset)
        .def_readonly("absolute_addr", &vp::SchmittStatus::absolute_addr)
        .def_readonly("domain", &vp::SchmittStatus::domain)
        .def_readonly("note", &vp::SchmittStatus::note);

    py::class_<vp::PinElectricalCapability>(
        m, "PinElectricalCapability", "Best-effort per-pin electrical capability probe result.")
        .def_readonly("bank", &vp::PinElectricalCapability::bank)
        .def_readonly("pin", &vp::PinElectricalCapability::pin)
        .def_readonly("drive_supported", &vp::PinElectricalCapability::drive_supported)
        .def_readonly("pull_supported", &vp::PinElectricalCapability::pull_supported)
        .def_readonly("schmitt_supported", &vp::PinElectricalCapability::schmitt_supported)
        .def_readonly("note", &vp::PinElectricalCapability::note);

    py::class_<vp::Controller>(m, "PinMux",
                               "Runtime RV1106 pin multiplexing controller via direct IOC/PMUIOC register access.")
        .def(py::init<>(),
             "Opens /dev/mem and maps IOC(0xff538000) + PMUIOC(0xff388000). Requires root privileges.")
        .def("is_open", &vp::Controller::is_open, "Returns True if memory mappings are active.")
        .def("close", &vp::Controller::close, "Closes /dev/mem mappings.")

        .def("parse_pin", &vp::Controller::parse_pin, "pin_name"_a,
             "Parses a pin string like 'GPIO1_C4', 'gpio1-20', or '1:20'.")

        .def("get_mux", py::overload_cast<int, int>(&vp::Controller::get_mux, py::const_),
             "bank"_a, "pin"_a,
             "Reads current mux value from register field.")
        .def("get_mux", py::overload_cast<const std::string&>(&vp::Controller::get_mux, py::const_),
             "pin_name"_a,
             "Reads current mux value by pin string.")

        .def("set_mux", py::overload_cast<int, int, uint32_t>(&vp::Controller::set_mux),
             "bank"_a, "pin"_a, "mux"_a,
             "Writes mux value using Rockchip write-mask semantics (no reboot required).")
        .def("set_mux", py::overload_cast<const std::string&, uint32_t>(&vp::Controller::set_mux),
             "pin_name"_a, "mux"_a,
             "Writes mux value by pin string.")

        .def("get_register_info", py::overload_cast<int, int>(&vp::Controller::get_register_info, py::const_),
             "bank"_a, "pin"_a,
             "Returns register address/bitfield info used for this pin.")
        .def("get_register_info", py::overload_cast<const std::string&>(&vp::Controller::get_register_info, py::const_),
             "pin_name"_a,
             "Returns register address/bitfield info by pin string.")

        .def("list_functions", py::overload_cast<int, int>(&vp::Controller::list_functions, py::const_),
             "bank"_a, "pin"_a,
             "Lists available alternate functions by parsing /proc/device-tree/pinctrl.")
        .def("list_functions", py::overload_cast<const std::string&>(&vp::Controller::list_functions, py::const_),
             "pin_name"_a,
             "Lists available alternate functions by pin string.")

        .def("get_runtime_status", py::overload_cast<int, int>(&vp::Controller::get_runtime_status, py::const_),
             "bank"_a, "pin"_a,
             "Reads mux/gpio owner and current function/group from debugfs pinctrl.")
        .def("get_runtime_status", py::overload_cast<const std::string&>(&vp::Controller::get_runtime_status, py::const_),
             "pin_name"_a,
             "Reads mux/gpio owner by pin string.")

        .def("check_conflict",
             py::overload_cast<int, int, const std::string&>(&vp::Controller::check_conflict, py::const_),
             "bank"_a, "pin"_a, "target_function_or_group"_a = "",
             "Checks whether switching this pin may conflict with current mux/gpio owners.")
        .def("check_conflict",
             py::overload_cast<const std::string&, const std::string&>(&vp::Controller::check_conflict, py::const_),
             "pin_name"_a, "target_function_or_group"_a = "",
             "Checks conflict by pin string.")

        .def("release_conflict", py::overload_cast<int, int>(&vp::Controller::release_conflict, py::const_),
             "bank"_a, "pin"_a,
             "Attempts to unbind current mux owner device. Returns False if release is incomplete.")
        .def("release_conflict", py::overload_cast<const std::string&>(&vp::Controller::release_conflict, py::const_),
             "pin_name"_a,
             "Attempts to release conflict by pin string.")

        .def("get_interface_status", &vp::Controller::get_interface_status, "function_or_group"_a,
             "Reports whether Linux has exposed usable interfaces (/dev/* or /sys/class/*) for the function.")
        .def("ensure_interface", &vp::Controller::ensure_interface, "function_or_group"_a,
             "Attempts to bind the inferred owner device and re-check userspace interface visibility.")
        .def("list_overlays", &vp::Controller::list_overlays,
             "Lists currently active device-tree overlays from configfs.")
        .def("apply_overlay", &vp::Controller::apply_overlay,
             "dtbo_path"_a, "overlay_name"_a = "",
             "Applies a DT overlay (.dtbo) through configfs and returns created overlay entry name.")
        .def("remove_overlay", &vp::Controller::remove_overlay,
             "overlay_name"_a,
             "Removes an applied configfs overlay by name.")
        .def("list_adc_channels", &vp::Controller::list_adc_channels,
             "Lists available SARADC channels from IIO sysfs and reads current values.")
        .def("read_adc", py::overload_cast<int>(&vp::Controller::read_adc, py::const_),
             "channel"_a,
             "Reads one ADC channel by numeric index.")
        .def("read_adc", py::overload_cast<const std::string&>(&vp::Controller::read_adc, py::const_),
             "channel_or_pin"_a,
             "Reads one ADC channel by token (e.g. adc0) or pin name (GPIO4_C0/GPIO4_C1).")
        .def("gpio_request_line", py::overload_cast<int, int, const vp::GpioLineConfig&>(&vp::Controller::gpio_request_line),
             "bank"_a, "pin"_a, "config"_a = vp::GpioLineConfig{},
             "Requests one GPIO line with direction/bias/drive options.")
        .def("gpio_request_line", py::overload_cast<const std::string&, const vp::GpioLineConfig&>(&vp::Controller::gpio_request_line),
             "pin_name"_a, "config"_a = vp::GpioLineConfig{},
             "Requests one GPIO line by pin name.")
        .def("gpio_release_line", py::overload_cast<int, int>(&vp::Controller::gpio_release_line),
             "bank"_a, "pin"_a,
             "Releases a previously requested GPIO line.")
        .def("gpio_release_line", py::overload_cast<const std::string&>(&vp::Controller::gpio_release_line),
             "pin_name"_a,
             "Releases a requested GPIO line by pin name.")
        .def("gpio_set_value", py::overload_cast<int, int, int>(&vp::Controller::gpio_set_value, py::const_),
             "bank"_a, "pin"_a, "value"_a,
             "Sets value on a requested GPIO output line.")
        .def("gpio_set_value", py::overload_cast<const std::string&, int>(&vp::Controller::gpio_set_value, py::const_),
             "pin_name"_a, "value"_a,
             "Sets value on a requested GPIO line by pin name.")
        .def("gpio_get_value", py::overload_cast<int, int>(&vp::Controller::gpio_get_value, py::const_),
             "bank"_a, "pin"_a,
             "Reads value from a requested GPIO line.")
        .def("gpio_get_value", py::overload_cast<const std::string&>(&vp::Controller::gpio_get_value, py::const_),
             "pin_name"_a,
             "Reads value from a requested GPIO line by pin name.")
        .def("gpio_get_status", py::overload_cast<int, int>(&vp::Controller::gpio_get_status, py::const_),
             "bank"_a, "pin"_a,
             "Returns runtime status of requested GPIO line.")
        .def("gpio_get_status", py::overload_cast<const std::string&>(&vp::Controller::gpio_get_status, py::const_),
             "pin_name"_a,
             "Returns runtime status of requested GPIO line by pin name.")
        .def("set_drive_strength", py::overload_cast<int, int, int>(&vp::Controller::set_drive_strength),
             "bank"_a, "pin"_a, "level"_a,
             "Sets RV1106 IOC drive strength level (0..7) for a pin.")
        .def("set_drive_strength", py::overload_cast<const std::string&, int>(&vp::Controller::set_drive_strength),
             "pin_name"_a, "level"_a,
             "Sets RV1106 IOC drive strength level (0..7) by pin name.")
        .def("get_drive_strength", py::overload_cast<int, int>(&vp::Controller::get_drive_strength, py::const_),
             "bank"_a, "pin"_a,
             "Reads RV1106 IOC drive strength level/raw register for a pin.")
        .def("get_drive_strength",
             py::overload_cast<const std::string&>(&vp::Controller::get_drive_strength, py::const_),
             "pin_name"_a,
             "Reads RV1106 IOC drive strength level/raw register by pin name.")
        .def("set_pull", py::overload_cast<int, int, const std::string&>(&vp::Controller::set_pull),
             "bank"_a, "pin"_a, "mode"_a,
             "Sets pull mode (disable/pull_up/pull_down/bus_hold or 0..3).")
        .def("set_pull", py::overload_cast<const std::string&, const std::string&>(&vp::Controller::set_pull),
             "pin_name"_a, "mode"_a,
             "Sets pull mode by pin name.")
        .def("get_pull", py::overload_cast<int, int>(&vp::Controller::get_pull, py::const_),
             "bank"_a, "pin"_a,
             "Reads pull mode/raw register for a pin.")
        .def("get_pull", py::overload_cast<const std::string&>(&vp::Controller::get_pull, py::const_),
             "pin_name"_a,
             "Reads pull mode/raw register by pin name.")
        .def("set_input_schmitt", py::overload_cast<int, int, bool>(&vp::Controller::set_input_schmitt),
             "bank"_a, "pin"_a, "enable"_a,
             "Enables/disables input schmitt for a pin.")
        .def("set_input_schmitt", py::overload_cast<const std::string&, bool>(&vp::Controller::set_input_schmitt),
             "pin_name"_a, "enable"_a,
             "Enables/disables input schmitt by pin name.")
        .def("get_input_schmitt", py::overload_cast<int, int>(&vp::Controller::get_input_schmitt, py::const_),
             "bank"_a, "pin"_a,
             "Reads input schmitt state/raw register for a pin.")
        .def("get_input_schmitt",
             py::overload_cast<const std::string&>(&vp::Controller::get_input_schmitt, py::const_),
             "pin_name"_a,
             "Reads input schmitt state/raw register by pin name.")
        .def("probe_electrical_capability",
             py::overload_cast<int, int, bool>(&vp::Controller::probe_electrical_capability),
             "bank"_a, "pin"_a, "active_test"_a = false,
             "Probes drive/pull/schmitt capability for one pin. active_test=True performs write-restore checks.")
        .def("probe_electrical_capability",
             py::overload_cast<const std::string&, bool>(&vp::Controller::probe_electrical_capability),
             "pin_name"_a, "active_test"_a = false,
             "Probes drive/pull/schmitt capability by pin name.")
        .def("probe_electrical_capabilities", &vp::Controller::probe_electrical_capabilities,
             "active_test"_a = false,
             "Probes drive/pull/schmitt capability for all pins.")

        .def("get_function_name", py::overload_cast<int, int>(&vp::Controller::get_function_name, py::const_),
             "bank"_a, "pin"_a,
             "Returns best-effort function name matching current mux.")
        .def("get_function_name", py::overload_cast<const std::string&>(&vp::Controller::get_function_name, py::const_),
             "pin_name"_a,
             "Returns best-effort function name matching current mux by pin string.")

        .def("set_function", py::overload_cast<int, int, const std::string&>(&vp::Controller::set_function),
             "bank"_a, "pin"_a, "function_or_group"_a,
             "Sets mux by function name (e.g. 'uart4', 'pwm1') or group name (e.g. 'uart4m1-xfer').")
        .def("set_function", py::overload_cast<const std::string&, const std::string&>(&vp::Controller::set_function),
             "pin_name"_a, "function_or_group"_a,
             "Sets mux by pin string + function/group name.");

    py::class_<vp::NpuClockStatus>(m, "NpuClockStatus", "NPU clock probe status.")
        .def_readonly("npu_node_present", &vp::NpuClockStatus::npu_node_present)
        .def_readonly("debugfs_available", &vp::NpuClockStatus::debugfs_available)
        .def_readonly("overlay_configfs_available", &vp::NpuClockStatus::overlay_configfs_available)
        .def_readonly("assigned_rate_hz", &vp::NpuClockStatus::assigned_rate_hz)
        .def_readonly("current_rate_hz", &vp::NpuClockStatus::current_rate_hz)
        .def_readonly("npu_root_rate_hz", &vp::NpuClockStatus::npu_root_rate_hz)
        .def_readonly("clk500m_src_rate_hz", &vp::NpuClockStatus::clk500m_src_rate_hz)
        .def_readonly("note", &vp::NpuClockStatus::note);

    py::class_<vp::NpuClockApplyResult>(m, "NpuClockApplyResult", "NPU clock apply result.")
        .def_readonly("ok", &vp::NpuClockApplyResult::ok)
        .def_readonly("rebind_attempted", &vp::NpuClockApplyResult::rebind_attempted)
        .def_readonly("rebind_ok", &vp::NpuClockApplyResult::rebind_ok)
        .def_readonly("reboot_required", &vp::NpuClockApplyResult::reboot_required)
        .def_readonly("requested_rate_hz", &vp::NpuClockApplyResult::requested_rate_hz)
        .def_readonly("assigned_rate_hz", &vp::NpuClockApplyResult::assigned_rate_hz)
        .def_readonly("current_rate_hz", &vp::NpuClockApplyResult::current_rate_hz)
        .def_readonly("npu_root_rate_hz", &vp::NpuClockApplyResult::npu_root_rate_hz)
        .def_readonly("clk500m_src_rate_hz", &vp::NpuClockApplyResult::clk500m_src_rate_hz)
        .def_readonly("overlay_name", &vp::NpuClockApplyResult::overlay_name)
        .def_readonly("message", &vp::NpuClockApplyResult::message);

    py::class_<vp::NpuClock>(m, "NpuClock", "RV1106 NPU clock helper via DT overlay and clock readback.")
        .def(py::init<>())
        .def("status", &vp::NpuClock::status,
             "Reads assigned/runtime NPU clock status.")
        .def("supported_rates_hz", &vp::NpuClock::supported_rates_hz,
             "Returns conservative validated NPU rates in Hz.")
        .def("supported_rates_mhz", &vp::NpuClock::supported_rates_mhz,
             "Returns conservative validated NPU rates in MHz.")
        .def("list_overlays", &vp::NpuClock::list_overlays,
             "prefix"_a = "visiong_npuclk_",
             "Lists active DT overlays with the given prefix.")
        .def("remove_overlay", &vp::NpuClock::remove_overlay,
             "overlay_name"_a,
             "Removes one DT overlay by name.")
        .def("set_rate_hz", &vp::NpuClock::set_rate_hz,
             "rate_hz"_a,
             "update_cru_clk500m_src"_a = true,
             "unbind_rebind_npu"_a = false,
             "allow_unsafe_rate"_a = false,
             "Applies NPU assigned-clock-rates in Hz. Can optionally update CRU CLK_500M_SRC and rebind NPU driver.")
        .def("set_rate_mhz", &vp::NpuClock::set_rate_mhz,
             "rate_mhz"_a,
             "update_cru_clk500m_src"_a = true,
             "unbind_rebind_npu"_a = false,
             "allow_unsafe_rate"_a = false,
             "Applies NPU assigned-clock-rates in MHz.")
        .def("request_reboot", &vp::NpuClock::request_reboot,
             "Requests immediate system reboot (sync + reboot).")
        ;
}

