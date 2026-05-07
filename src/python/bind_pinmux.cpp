// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/core/pinmux.h"
#include "visiong/core/npu_clock.h"

namespace vp = visiong::pinmux;

void bind_pinmux(py::module_& m) {
    const auto parse_pin_function_pairs = [](const py::object& value) {
        std::vector<std::pair<std::string, std::string>> pairs;
        if (value.is_none()) {
            return pairs;
        }
        if (py::isinstance<py::dict>(value)) {
            py::dict dict = value.cast<py::dict>();
            for (const auto& item : dict) {
                pairs.emplace_back(py::str(item.first).cast<std::string>(),
                                   py::str(item.second).cast<std::string>());
            }
            return pairs;
        }
        if (py::isinstance<py::str>(value) || !py::isinstance<py::sequence>(value)) {
            throw py::type_error("pin/function mapping must be a dict or a sequence of (pin, function) pairs.");
        }
        py::sequence seq = value.cast<py::sequence>();
        for (const auto& item : seq) {
            py::sequence pair = py::reinterpret_borrow<py::object>(item).cast<py::sequence>();
            if (pair.size() != 2) {
                throw py::value_error("pin/function pair must contain exactly two items.");
            }
            pairs.emplace_back(py::str(pair[0]).cast<std::string>(), py::str(pair[1]).cast<std::string>());
        }
        return pairs;
    };

    const auto parse_pin_names = [](const py::object& value) {
        std::vector<std::string> pins;
        if (value.is_none()) {
            return pins;
        }
        if (py::isinstance<py::dict>(value)) {
            py::dict dict = value.cast<py::dict>();
            for (const auto& item : dict) {
                pins.push_back(py::str(item.first).cast<std::string>());
            }
            return pins;
        }
        if (py::isinstance<py::str>(value)) {
            pins.push_back(py::str(value).cast<std::string>());
            return pins;
        }
        if (!py::isinstance<py::sequence>(value)) {
            throw py::type_error("pins must be a pin string, dict, or sequence of pin strings.");
        }
        py::sequence seq = value.cast<py::sequence>();
        for (const auto& item : seq) {
            pins.push_back(py::str(item).cast<std::string>());
        }
        return pins;
    };

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

    py::class_<vp::PinAltFunction>(m, "PinAltFunction", "Altenative function description for one pin.")
        .def_readonly("function", &vp::PinAltFunction::function)
        .def_readonly("group", &vp::PinAltFunction::group)
        .def_readonly("mux", &vp::PinAltFunction::mux)
        .def("__repr__", [](const vp::PinAltFunction& item) {
            return "PinAltFunction(function='" + item.function + "', group='" + item.group +
                   "', mux=" + std::to_string(item.mux) + ")";
        });

    py::class_<vp::PeripheralPinAssignment>(m, "PeripheralPin", "Resolved peripheral pin assignment.")
        .def_readonly("pin", &vp::PeripheralPinAssignment::pin)
        .def_readonly("function", &vp::PeripheralPinAssignment::function)
        .def_readonly("group", &vp::PeripheralPinAssignment::group)
        .def_readonly("role", &vp::PeripheralPinAssignment::role)
        .def_readonly("mux", &vp::PeripheralPinAssignment::mux)
        .def("__repr__", [](const vp::PeripheralPinAssignment& item) {
            return "PeripheralPin(pin='" + item.pin + "', function='" + item.function +
                   "', group='" + item.group + "', role='" + item.role +
                   "', mux=" + std::to_string(item.mux) + ")";
        });

    py::class_<vp::SpiSetupStatus>(m, "SPISetup", "SPI pinmux and Linux interface setup result.")
        .def_readonly("ok", &vp::SpiSetupStatus::ok)
        .def_readonly("bus", &vp::SpiSetupStatus::bus)
        .def_readonly("chip_select", &vp::SpiSetupStatus::chip_select)
        .def_readonly("device", &vp::SpiSetupStatus::device)
        .def_readonly("dev_path", &vp::SpiSetupStatus::dev_path)
        .def_readonly("function", &vp::SpiSetupStatus::function)
        .def_readonly("group", &vp::SpiSetupStatus::group)
        .def_readonly("controller_bound", &vp::SpiSetupStatus::controller_bound)
        .def_readonly("child_bound", &vp::SpiSetupStatus::child_bound)
        .def_readonly("pins", &vp::SpiSetupStatus::pins)
        .def_readonly("note", &vp::SpiSetupStatus::note)
        .def("__bool__", [](const vp::SpiSetupStatus& status) { return status.ok; })
        .def("__repr__", [](const vp::SpiSetupStatus& status) {
            return "SPISetup(ok=" + std::string(status.ok ? "True" : "False") +
                   ", device='" + status.device + "', group='" + status.group +
                   "', dev_path='" + status.dev_path + "')";
        });

    py::class_<vp::UartSetupStatus>(m, "UARTSetup", "UART pinmux and Linux interface setup result.")
        .def_readonly("ok", &vp::UartSetupStatus::ok)
        .def_readonly("bus", &vp::UartSetupStatus::bus)
        .def_readonly("device", &vp::UartSetupStatus::device)
        .def_readonly("dev_path", &vp::UartSetupStatus::dev_path)
        .def_readonly("function", &vp::UartSetupStatus::function)
        .def_readonly("group", &vp::UartSetupStatus::group)
        .def_readonly("driver_bound", &vp::UartSetupStatus::driver_bound)
        .def_readonly("pins", &vp::UartSetupStatus::pins)
        .def_readonly("note", &vp::UartSetupStatus::note)
        .def("__bool__", [](const vp::UartSetupStatus& status) { return status.ok; })
        .def("__repr__", [](const vp::UartSetupStatus& status) {
            return "UARTSetup(ok=" + std::string(status.ok ? "True" : "False") +
                   ", device='" + status.device + "', group='" + status.group +
                   "', dev_path='" + status.dev_path + "')";
        });

    py::class_<vp::I2cSetupStatus>(m, "I2CSetup", "I2C pinmux and Linux interface setup result.")
        .def_readonly("ok", &vp::I2cSetupStatus::ok)
        .def_readonly("bus", &vp::I2cSetupStatus::bus)
        .def_readonly("device", &vp::I2cSetupStatus::device)
        .def_readonly("dev_path", &vp::I2cSetupStatus::dev_path)
        .def_readonly("function", &vp::I2cSetupStatus::function)
        .def_readonly("group", &vp::I2cSetupStatus::group)
        .def_readonly("driver_bound", &vp::I2cSetupStatus::driver_bound)
        .def_readonly("pins", &vp::I2cSetupStatus::pins)
        .def_readonly("note", &vp::I2cSetupStatus::note)
        .def("__bool__", [](const vp::I2cSetupStatus& status) { return status.ok; })
        .def("__repr__", [](const vp::I2cSetupStatus& status) {
            return "I2CSetup(ok=" + std::string(status.ok ? "True" : "False") +
                   ", device='" + status.device + "', group='" + status.group +
                   "', dev_path='" + status.dev_path + "')";
        });

    py::class_<vp::PwmSetupStatus>(m, "PWMSetup", "PWM pinmux and Linux interface setup result.")
        .def_readonly("ok", &vp::PwmSetupStatus::ok)
        .def_readonly("channel", &vp::PwmSetupStatus::channel)
        .def_readonly("device", &vp::PwmSetupStatus::device)
        .def_readonly("dev_path", &vp::PwmSetupStatus::dev_path)
        .def_readonly("function", &vp::PwmSetupStatus::function)
        .def_readonly("group", &vp::PwmSetupStatus::group)
        .def_readonly("driver_bound", &vp::PwmSetupStatus::driver_bound)
        .def_readonly("pins", &vp::PwmSetupStatus::pins)
        .def_readonly("note", &vp::PwmSetupStatus::note)
        .def("__bool__", [](const vp::PwmSetupStatus& status) { return status.ok; })
        .def("__repr__", [](const vp::PwmSetupStatus& status) {
            return "PWMSetup(ok=" + std::string(status.ok ? "True" : "False") +
                   ", device='" + status.device + "', group='" + status.group +
                   "', dev_path='" + status.dev_path + "')";
        });

    py::class_<vp::PeripheralRegisterInfo>(m, "PeripheralRegisterInfo",
                                           "Resolved peripheral register block metadata.")
        .def_readonly("available", &vp::PeripheralRegisterInfo::available)
        .def_readonly("request", &vp::PeripheralRegisterInfo::request)
        .def_readonly("alias", &vp::PeripheralRegisterInfo::alias)
        .def_readonly("device", &vp::PeripheralRegisterInfo::device)
        .def_readonly("compatible", &vp::PeripheralRegisterInfo::compatible)
        .def_readonly("bound_driver", &vp::PeripheralRegisterInfo::bound_driver)
        .def_readonly("base_addr", &vp::PeripheralRegisterInfo::base_addr)
        .def_readonly("size", &vp::PeripheralRegisterInfo::size)
        .def_readonly("note", &vp::PeripheralRegisterInfo::note)
        .def("__bool__", [](const vp::PeripheralRegisterInfo& info) { return info.available; })
        .def("__repr__", [](const vp::PeripheralRegisterInfo& info) {
            return py::str("PeripheralRegisterInfo(alias='{}', base=0x{:x}, size=0x{:x}, device='{}')")
                .format(info.alias, info.base_addr, info.size, info.device)
                .cast<std::string>();
        });

    py::class_<vp::RegisterBlock>(m, "Reg", "Direct /dev/mem register block for RV1103/RV1106 peripherals.")
        .def(py::init<const std::string&, size_t>(), "peripheral"_a, "map_size"_a = 0)
        .def("is_open", &vp::RegisterBlock::is_open)
        .def("close", &vp::RegisterBlock::close)
        .def_property_readonly("info", &vp::RegisterBlock::info, py::return_value_policy::reference_internal)
        .def("read8", &vp::RegisterBlock::read8, "offset"_a)
        .def("write8", &vp::RegisterBlock::write8, "offset"_a, "value"_a)
        .def("write8_repeat",
             [](vp::RegisterBlock& self, uint32_t offset, py::buffer data) {
                 py::buffer_info info = data.request();
                 if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1) {
                     throw py::value_error("write8_repeat expects a contiguous 1-byte buffer.");
                 }
                 const size_t size = static_cast<size_t>(info.size);
                 self.write8_repeat(offset, info.ptr, size);
                 return size;
             },
             "offset"_a, "data"_a,
             "Writes each byte from data to the same 8-bit register offset.")
        .def("read8_repeat",
             [](const vp::RegisterBlock& self, uint32_t offset, size_t size) {
                 const auto data = self.read8_repeat(offset, size);
                 return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
             },
             "offset"_a, "size"_a,
             "Reads one 8-bit register offset repeatedly and returns bytes.")
        .def("read16", &vp::RegisterBlock::read16, "offset"_a)
        .def("write16", &vp::RegisterBlock::write16, "offset"_a, "value"_a)
        .def("read32", &vp::RegisterBlock::read32, "offset"_a)
        .def("write32", &vp::RegisterBlock::write32, "offset"_a, "value"_a)
        .def("update32", &vp::RegisterBlock::update32, "offset"_a, "mask"_a, "value"_a)
        .def("set_bits", &vp::RegisterBlock::set_bits, "offset"_a, "mask"_a)
        .def("clear_bits", &vp::RegisterBlock::clear_bits, "offset"_a, "mask"_a)
        .def("__enter__", [](vp::RegisterBlock& self) -> vp::RegisterBlock& { return self; },
             py::return_value_policy::reference_internal)
        .def("__exit__", [](vp::RegisterBlock& self, const py::object&, const py::object&, const py::object&) {
            self.close();
            return false;
        })
        .def("__repr__", [](const vp::RegisterBlock& self) {
            const auto& info = self.info();
            return py::str("Reg('{}', base=0x{:x}, size=0x{:x})")
                .format(info.alias, info.base_addr, info.size)
                .cast<std::string>();
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
                                            "Kenel interface exposure status for one pin function.")
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
        .def_readwrite("edge", &vp::GpioLineConfig::edge)
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

    py::class_<vp::GpioLineEvent>(m, "GpioLineEvent", "GPIO edge event returned by gpio_wait_event()/Pin.wait_irq().")
        .def_readonly("valid", &vp::GpioLineEvent::valid)
        .def_readonly("timed_out", &vp::GpioLineEvent::timed_out)
        .def_readonly("cancelled", &vp::GpioLineEvent::cancelled)
        .def_readonly("timestamp_ns", &vp::GpioLineEvent::timestamp_ns)
        .def_readonly("edge", &vp::GpioLineEvent::edge)
        .def_readonly("bank", &vp::GpioLineEvent::bank)
        .def_readonly("pin", &vp::GpioLineEvent::pin)
        .def_readonly("offset", &vp::GpioLineEvent::offset)
        .def_readonly("sequence", &vp::GpioLineEvent::sequence)
        .def_readonly("line_sequence", &vp::GpioLineEvent::line_sequence)
        .def_readonly("note", &vp::GpioLineEvent::note)
        .def("__bool__", [](const vp::GpioLineEvent& event) { return event.valid; })
        .def("__repr__", [](const vp::GpioLineEvent& event) {
            return py::str("GpioLineEvent(valid={}, edge='{}', timestamp_ns={}, pin='GPIO{}_{}')")
                .format(event.valid, event.edge, event.timestamp_ns, event.bank,
                        std::string(1, static_cast<char>('A' + event.pin / 8)) + std::to_string(event.pin % 8))
                .cast<std::string>();
        });

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
             "Lists available altenate functions by parsing /proc/device-tree/pinctrl.")
        .def("list_functions", py::overload_cast<const std::string&>(&vp::Controller::list_functions, py::const_),
             "pin_name"_a,
             "Lists available altenate functions by pin string.")

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
        .def("gpio_wait_event", py::overload_cast<int, int, int>(&vp::Controller::gpio_wait_event, py::const_),
             "bank"_a, "pin"_a, "timeout_ms"_a = -1, py::call_guard<py::gil_scoped_release>(),
             "Waits for one requested GPIO edge event. Returns GpioLineEvent; valid=False means timeout.")
        .def("gpio_wait_event", py::overload_cast<const std::string&, int>(&vp::Controller::gpio_wait_event, py::const_),
             "pin_name"_a, "timeout_ms"_a = -1, py::call_guard<py::gil_scoped_release>(),
             "Waits for one requested GPIO edge event by pin string.")
        .def("gpio_wait_event_cancelable",
             py::overload_cast<int, int, int, int>(&vp::Controller::gpio_wait_event_cancelable, py::const_),
             "bank"_a, "pin"_a, "cancel_fd"_a, "timeout_ms"_a = -1, py::call_guard<py::gil_scoped_release>(),
             "Waits for one GPIO edge event or a readable cancel fd. Returns cancelled=True when cancelled.")
        .def("gpio_wait_event_cancelable",
             py::overload_cast<const std::string&, int, int>(&vp::Controller::gpio_wait_event_cancelable, py::const_),
             "pin_name"_a, "cancel_fd"_a, "timeout_ms"_a = -1, py::call_guard<py::gil_scoped_release>(),
             "Waits for one GPIO edge event by pin string or a readable cancel fd.")
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
             "Sets mux by pin string + function/group name.")
        .def("func", py::overload_cast<const std::string&, const std::string&>(&vp::Controller::set_function),
             "pin_name"_a, "function_or_group"_a,
             "Short alias for set_function(pin_name, function_or_group).")
        .def("set_functions",
             [parse_pin_function_pairs](vp::Controller& self, const py::object& pin_functions) {
                 self.set_functions(parse_pin_function_pairs(pin_functions));
             },
             "pin_functions"_a,
             "Sets multiple pin functions from a dict or sequence of (pin, function) pairs.")
        .def("funcs",
             [parse_pin_function_pairs](vp::Controller& self, const py::object& pin_functions) {
                 self.set_functions(parse_pin_function_pairs(pin_functions));
             },
             "pin_functions"_a,
             "Short alias for set_functions().")
        .def("release_owner", &vp::Controller::release_owner,
             "owner"_a,
             "Unbinds a Linux owner device from platform/i2c/spi buses when possible.")
        .def("get_bound_driver", &vp::Controller::get_bound_driver,
             "bus"_a, "device"_a,
             "Returns the currently bound Linux driver for a bus device, or ''.")
        .def("driver", &vp::Controller::get_bound_driver,
             "bus"_a, "device"_a,
             "Short alias for get_bound_driver().")
        .def("bind_driver", &vp::Controller::bind_driver,
             "bus"_a, "device"_a, "driver"_a, "unbind_current"_a = true,
             "Binds a Linux bus device to a driver. Supports aliases like platform/spi0 and spi/spi0.0.")
        .def("bind", &vp::Controller::bind_driver,
             "bus"_a, "device"_a, "driver"_a, "unbind_current"_a = true,
             "Short alias for bind_driver().")
        .def("unbind_driver", &vp::Controller::unbind_driver,
             "bus"_a, "device"_a,
             "Unbinds the currently bound driver for a Linux bus device.")
        .def("unbind", &vp::Controller::unbind_driver,
             "bus"_a, "device"_a,
             "Short alias for unbind_driver().")
        .def("spi_get_bound_driver", &vp::Controller::spi_get_bound_driver,
             "spi_device"_a,
             "Returns the current driver for an SPI device such as spi0.0.")
        .def("spi_bind_driver", &vp::Controller::spi_bind_driver,
             "spi_device"_a, "driver"_a, "unbind_current"_a = true,
             "Binds an SPI child device to a driver.")
        .def("spi_bind_spidev", &vp::Controller::spi_bind_spidev,
             "spi_device"_a,
             "Binds an SPI child device to spidev.")
        .def("setup_spi",
             [parse_pin_names](vp::Controller& self,
                               const py::object& spi,
                               const py::object& pins,
                               int chip_select,
                               bool bind_spidev) {
                 if (py::isinstance<py::int_>(spi)) {
                     return self.setup_spi(spi.cast<int>(), parse_pin_names(pins), chip_select, bind_spidev);
                 }
                 return self.setup_spi(py::str(spi).cast<std::string>(), parse_pin_names(pins), bind_spidev);
             },
             "spi"_a, "pins"_a, "chip_select"_a = 0, "bind_spidev"_a = false,
             "Infers SPI pin roles from pin names, checks one common spiXmY group, sets mux, and optionally prepares spidev.")
        .def("spi",
             [parse_pin_names](vp::Controller& self,
                               const py::object& spi,
                               const py::object& pins,
                               int chip_select,
                               bool bind_spidev) {
                 if (py::isinstance<py::int_>(spi)) {
                     return self.setup_spi(spi.cast<int>(), parse_pin_names(pins), chip_select, bind_spidev);
                 }
                 return self.setup_spi(py::str(spi).cast<std::string>(), parse_pin_names(pins), bind_spidev);
             },
             "spi"_a, "pins"_a, "chip_select"_a = 0, "bind_spidev"_a = false,
             "MicroPython-style SPI pinmux setup. Example: p.spi(0, ['GPIO1_C1','GPIO1_C2','GPIO1_C0']).")
        .def("spi_prepare",
             [parse_pin_names](vp::Controller& self,
                               const py::object& spi,
                               const py::object& pins,
                               int chip_select,
                               bool bind_spidev) {
                 if (py::isinstance<py::int_>(spi)) {
                     return self.setup_spi(spi.cast<int>(), parse_pin_names(pins), chip_select, bind_spidev).ok;
                 }
                 return self.setup_spi(py::str(spi).cast<std::string>(), parse_pin_names(pins), bind_spidev).ok;
             },
             "spi"_a, "pins"_a, "chip_select"_a = 0, "bind_spidev"_a = false,
             "Compatibility alias returning bool. Prefer spi()/setup_spi() for detailed status.")
        .def("setup_uart",
             [parse_pin_names](vp::Controller& self,
                               const py::object& uart,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(uart)) {
                     return self.setup_uart(uart.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_uart(py::str(uart).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "uart"_a, "pins"_a, "bind_driver"_a = false,
             "Infers UART pin roles from pin names, checks one common uartXmY group, and sets mux.")
        .def("uart",
             [parse_pin_names](vp::Controller& self,
                               const py::object& uart,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(uart)) {
                     return self.setup_uart(uart.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_uart(py::str(uart).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "uart"_a, "pins"_a, "bind_driver"_a = false,
             "MicroPython-style UART pinmux setup. Example: p.uart(4, ['GPIO1_C4','GPIO1_C5']).")
        .def("setup_i2c",
             [parse_pin_names](vp::Controller& self,
                               const py::object& i2c,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(i2c)) {
                     return self.setup_i2c(i2c.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_i2c(py::str(i2c).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "i2c"_a, "pins"_a, "bind_driver"_a = false,
             "Infers I2C scl/sda roles from pin names, checks one common i2cXmY group, and sets mux.")
        .def("i2c",
             [parse_pin_names](vp::Controller& self,
                               const py::object& i2c,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(i2c)) {
                     return self.setup_i2c(i2c.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_i2c(py::str(i2c).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "i2c"_a, "pins"_a, "bind_driver"_a = false,
             "MicroPython-style I2C pinmux setup. Example: p.i2c(3, ['GPIO2_A6','GPIO2_A7']).")
        .def("setup_pwm",
             [parse_pin_names](vp::Controller& self,
                               const py::object& pwm,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(pwm)) {
                     return self.setup_pwm(pwm.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_pwm(py::str(pwm).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "pwm"_a, "pins"_a, "bind_driver"_a = false,
             "Infers a PWM output pin, checks that it supports pwmN, and sets mux.")
        .def("pwm",
             [parse_pin_names](vp::Controller& self,
                               const py::object& pwm,
                               const py::object& pins,
                               bool bind_driver) {
                 if (py::isinstance<py::int_>(pwm)) {
                     return self.setup_pwm(pwm.cast<int>(), parse_pin_names(pins), bind_driver);
                 }
                 return self.setup_pwm(py::str(pwm).cast<std::string>(), parse_pin_names(pins), bind_driver);
             },
             "pwm"_a, "pins"_a, "bind_driver"_a = false,
             "MicroPython-style PWM pinmux setup. Example: p.pwm(2, 'GPIO0_A1').")
        .def("reg_info", &vp::Controller::get_register_block_info,
             "peripheral"_a,
             "Resolves a peripheral short name such as spi0/uart4/i2c3/pwm0 to base address metadata.")
        .def("reg", &vp::Controller::map_registers,
             "peripheral"_a, "map_size"_a = 0,
             "Maps a peripheral register block through /dev/mem and returns Reg.");

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
