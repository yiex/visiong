// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_PINMUX_H
#define VISIONG_CORE_PINMUX_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace visiong::pinmux {

struct PinId {
    int bank = 0;
    int pin = 0;
};

struct RegisterInfo {
    uint32_t base_addr = 0;
    uint32_t reg_offset = 0;
    uint32_t absolute_addr = 0;
    uint8_t bit = 0;
    uint8_t width = 0;
    uint32_t mask = 0;
    bool gpio_only = false;
    std::string domain;
};

struct PinAltFunction {
    std::string function;
    std::string group;
    uint32_t mux = 0;
};

struct PeripheralPinAssignment {
    std::string pin;
    std::string function;
    std::string group;
    std::string role;
    uint32_t mux = 0;
};

struct SpiSetupStatus {
    bool ok = false;
    int bus = -1;
    int chip_select = -1;
    std::string device;
    std::string dev_path;
    std::string function;
    std::string group;
    bool controller_bound = false;
    bool child_bound = false;
    std::vector<PeripheralPinAssignment> pins;
    std::string note;
};

struct UartSetupStatus {
    bool ok = false;
    int bus = -1;
    std::string device;
    std::string dev_path;
    std::string function;
    std::string group;
    bool driver_bound = false;
    std::vector<PeripheralPinAssignment> pins;
    std::string note;
};

struct I2cSetupStatus {
    bool ok = false;
    int bus = -1;
    std::string device;
    std::string dev_path;
    std::string function;
    std::string group;
    bool driver_bound = false;
    std::vector<PeripheralPinAssignment> pins;
    std::string note;
};

struct PwmSetupStatus {
    bool ok = false;
    int channel = -1;
    std::string device;
    std::string dev_path;
    std::string function;
    std::string group;
    bool driver_bound = false;
    std::vector<PeripheralPinAssignment> pins;
    std::string note;
};

struct PeripheralRegisterInfo {
    bool available = false;
    std::string request;
    std::string alias;
    std::string device;
    std::string compatible;
    std::string bound_driver;
    uint64_t base_addr = 0;
    uint64_t size = 0;
    std::string note;
};

class RegisterBlock final {
public:
    explicit RegisterBlock(const std::string& peripheral, size_t map_size = 0);
    ~RegisterBlock();

    RegisterBlock(const RegisterBlock&) = delete;
    RegisterBlock& operator=(const RegisterBlock&) = delete;
    RegisterBlock(RegisterBlock&& other) noexcept;
    RegisterBlock& operator=(RegisterBlock&& other) noexcept;

    bool is_open() const noexcept;
    void close();

    const PeripheralRegisterInfo& info() const noexcept;
    uint8_t read8(uint32_t offset) const;
    void write8(uint32_t offset, uint8_t value);
    void write8_repeat(uint32_t offset, const void* data, size_t size);
    std::vector<uint8_t> read8_repeat(uint32_t offset, size_t size) const;
    uint16_t read16(uint32_t offset) const;
    void write16(uint32_t offset, uint16_t value);
    uint32_t read32(uint32_t offset) const;
    void write32(uint32_t offset, uint32_t value);
    void update32(uint32_t offset, uint32_t mask, uint32_t value);
    void set_bits(uint32_t offset, uint32_t mask);
    void clear_bits(uint32_t offset, uint32_t mask);

private:
    PeripheralRegisterInfo info_;
    int fd_ = -1;
    void* map_ = nullptr;
    uint64_t map_base_ = 0;
    size_t map_size_ = 0;
    size_t page_offset_ = 0;
};

struct PinRuntimeStatus {
    bool found = false;
    int bank = 0;
    int pin = 0;
    std::string mux_owner;
    std::string gpio_owner;
    std::string function;
    std::string group;
};

struct PinConflictReport {
    bool conflict = false;
    std::string reason;
    PinRuntimeStatus runtime;
};

struct FunctionInterfaceStatus {
    std::string request;
    std::string function;
    std::string group;
    std::string owner;
    bool owner_bound = false;
    std::vector<std::string> interfaces;
    std::string note;
};

struct AdcChannelStatus {
    bool available = false;
    int channel = -1;
    int raw = 0;
    double scale = 0.0;
    double millivolts = 0.0;
    std::string device;
    std::string raw_path;
    std::string scale_path;
    std::string pin_hint;
    std::string note;
};

struct GpioLineConfig {
    std::string direction = "input";   // input / output
    std::string bias = "default";      // default / pull_up / pull_down / disable
    std::string drive = "push_pull";   // push_pull / open_drain / open_source
    std::string edge = "none";         // none / rising / falling / both
    int drive_strength_level = -1;     // RV1106 IOC drive level (0..7)
    int drive_strength_ma = -1;        // backward-compatible alias; interpreted as level when 0..7
    bool active_low = false;
    int default_value = 0;             // used when direction=output
    std::string consumer = "visiong-pinmux";
};

struct GpioLineStatus {
    bool requested = false;
    int value = 0;
    int bank = 0;
    int pin = 0;
    std::string gpiochip;
    GpioLineConfig config;
    std::string note;
};

struct GpioLineEvent {
    bool valid = false;
    bool timed_out = false;
    bool cancelled = false;
    uint64_t timestamp_ns = 0;
    std::string edge;
    int bank = 0;
    int pin = 0;
    int offset = 0;
    uint32_t sequence = 0;
    uint32_t line_sequence = 0;
    std::string note;
};

struct DriveStrengthStatus {
    bool available = false;
    int level = -1;
    uint32_t raw = 0;
    uint32_t reg_offset = 0;
    uint32_t absolute_addr = 0;
    std::string domain;
    std::string note;
};

struct PullStatus {
    bool available = false;
    std::string mode;
    uint32_t raw = 0;
    uint32_t reg_offset = 0;
    uint32_t absolute_addr = 0;
    std::string domain;
    std::string note;
};

struct SchmittStatus {
    bool available = false;
    bool enabled = false;
    uint32_t raw = 0;
    uint32_t reg_offset = 0;
    uint32_t absolute_addr = 0;
    std::string domain;
    std::string note;
};

struct PinElectricalCapability {
    int bank = 0;
    int pin = 0;
    bool drive_supported = false;
    bool pull_supported = false;
    bool schmitt_supported = false;
    std::string note;
};

class Controller final {
public:
    Controller();
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;
    Controller(Controller&&) = delete;
    Controller& operator=(Controller&&) = delete;

    bool is_open() const noexcept;
    void close();

    uint32_t get_mux(int bank, int pin) const;
    void set_mux(int bank, int pin, uint32_t mux);

    PinId parse_pin_name(const std::string& pin_name) const;
    PinId parse_pin(const std::string& pin_name) const { return parse_pin_name(pin_name); }
    uint32_t get_mux(const std::string& pin_name) const;
    void set_mux(const std::string& pin_name, uint32_t mux);

    RegisterInfo get_register_info(int bank, int pin) const;
    RegisterInfo get_register_info(const std::string& pin_name) const;

    std::vector<PinAltFunction> list_functions(int bank, int pin) const;
    std::vector<PinAltFunction> list_functions(const std::string& pin_name) const;

    PinRuntimeStatus get_runtime_status(int bank, int pin) const;
    PinRuntimeStatus get_runtime_status(const std::string& pin_name) const;
    PinConflictReport check_conflict(int bank, int pin, const std::string& target_function_or_group = "") const;
    PinConflictReport check_conflict(const std::string& pin_name, const std::string& target_function_or_group = "") const;

    bool release_conflict(int bank, int pin) const;
    bool release_conflict(const std::string& pin_name) const;

    FunctionInterfaceStatus get_interface_status(const std::string& function_or_group) const;
    FunctionInterfaceStatus ensure_interface(const std::string& function_or_group) const;
    std::vector<std::string> list_overlays() const;
    std::string apply_overlay(const std::string& dtbo_path, const std::string& overlay_name = "") const;
    bool remove_overlay(const std::string& overlay_name) const;

    std::vector<AdcChannelStatus> list_adc_channels() const;
    AdcChannelStatus read_adc(int channel) const;
    AdcChannelStatus read_adc(const std::string& channel_or_pin) const;

    bool gpio_request_line(int bank, int pin, const GpioLineConfig& config = GpioLineConfig{});
    bool gpio_request_line(const std::string& pin_name, const GpioLineConfig& config = GpioLineConfig{});
    bool gpio_release_line(int bank, int pin);
    bool gpio_release_line(const std::string& pin_name);
    void gpio_set_value(int bank, int pin, int value) const;
    void gpio_set_value(const std::string& pin_name, int value) const;
    int gpio_get_value(int bank, int pin) const;
    int gpio_get_value(const std::string& pin_name) const;
    GpioLineStatus gpio_get_status(int bank, int pin) const;
    GpioLineStatus gpio_get_status(const std::string& pin_name) const;
    GpioLineEvent gpio_wait_event(int bank, int pin, int timeout_ms = -1) const;
    GpioLineEvent gpio_wait_event(const std::string& pin_name, int timeout_ms = -1) const;
    GpioLineEvent gpio_wait_event_cancelable(int bank, int pin, int cancel_fd, int timeout_ms = -1) const;
    GpioLineEvent gpio_wait_event_cancelable(const std::string& pin_name, int cancel_fd, int timeout_ms = -1) const;
    void set_drive_strength(int bank, int pin, int level);
    void set_drive_strength(const std::string& pin_name, int level);
    DriveStrengthStatus get_drive_strength(int bank, int pin) const;
    DriveStrengthStatus get_drive_strength(const std::string& pin_name) const;
    void set_pull(int bank, int pin, const std::string& mode);
    void set_pull(const std::string& pin_name, const std::string& mode);
    PullStatus get_pull(int bank, int pin) const;
    PullStatus get_pull(const std::string& pin_name) const;
    void set_input_schmitt(int bank, int pin, bool enable);
    void set_input_schmitt(const std::string& pin_name, bool enable);
    SchmittStatus get_input_schmitt(int bank, int pin) const;
    SchmittStatus get_input_schmitt(const std::string& pin_name) const;
    PinElectricalCapability probe_electrical_capability(int bank, int pin, bool active_test = false);
    PinElectricalCapability probe_electrical_capability(const std::string& pin_name, bool active_test = false);
    std::vector<PinElectricalCapability> probe_electrical_capabilities(bool active_test = false);

    std::string get_function_name(int bank, int pin) const;
    std::string get_function_name(const std::string& pin_name) const;

    void set_function(int bank, int pin, const std::string& function_or_group);
    void set_function(const std::string& pin_name, const std::string& function_or_group);
    void set_functions(const std::vector<std::pair<std::string, std::string>>& pin_functions);

    bool release_owner(const std::string& owner) const;

    std::string get_bound_driver(const std::string& bus, const std::string& device) const;
    bool bind_driver(const std::string& bus,
                     const std::string& device,
                     const std::string& driver,
                     bool unbind_current = true) const;
    bool unbind_driver(const std::string& bus, const std::string& device) const;

    std::string spi_get_bound_driver(const std::string& spi_device) const;
    bool spi_bind_driver(const std::string& spi_device,
                         const std::string& driver,
                         bool unbind_current = true) const;
    bool spi_bind_spidev(const std::string& spi_device) const;
    SpiSetupStatus setup_spi(const std::string& spi_device,
                             const std::vector<std::string>& pins,
                             bool bind_spidev = false);
    SpiSetupStatus setup_spi(int bus,
                             const std::vector<std::string>& pins,
                             int chip_select = 0,
                             bool bind_spidev = false);
    bool spi_prepare(const std::string& spi_device,
                     const std::vector<std::pair<std::string, std::string>>& pin_functions = {});
    bool spi_prepare(const std::string& spi_device, const std::vector<std::string>& pins);

    UartSetupStatus setup_uart(int bus,
                               const std::vector<std::string>& pins,
                               bool bind_driver = false);
    UartSetupStatus setup_uart(const std::string& uart_device,
                               const std::vector<std::string>& pins,
                               bool bind_driver = false);
    I2cSetupStatus setup_i2c(int bus,
                             const std::vector<std::string>& pins,
                             bool bind_driver = false);
    I2cSetupStatus setup_i2c(const std::string& i2c_device,
                             const std::vector<std::string>& pins,
                             bool bind_driver = false);
    PwmSetupStatus setup_pwm(int channel,
                             const std::vector<std::string>& pins,
                             bool bind_driver = false);
    PwmSetupStatus setup_pwm(const std::string& pwm_device,
                             const std::vector<std::string>& pins,
                             bool bind_driver = false);

    PeripheralRegisterInfo get_register_block_info(const std::string& peripheral) const;
    RegisterBlock map_registers(const std::string& peripheral, size_t map_size = 0) const;

private:
    struct PinKey {
        int bank = 0;
        int pin = 0;

        bool operator==(const PinKey& other) const noexcept {
            return bank == other.bank && pin == other.pin;
        }
    };

    struct PinKeyHash {
        size_t operator()(const PinKey& key) const noexcept;
    };

    struct ResolvedField {
        RegisterInfo info;
        bool use_pmuioc = false;
    };

    struct GpioLineHandle {
        int fd = -1;
        bool use_v2 = false;
        bool has_events = false;
        bool reg_backed = false;
        std::string chip;
        int offset = 0;
        GpioLineConfig config;
    };

    ResolvedField resolve_field(int bank, int pin) const;
    ResolvedField resolve_drive_field(int bank, int pin) const;
    ResolvedField resolve_pull_field(int bank, int pin) const;
    ResolvedField resolve_schmitt_field(int bank, int pin) const;
    static std::string normalize_token(std::string token);
    static std::string format_pin_label(int bank, int pin);
    static uint32_t read_be32(const uint8_t* data);
    static bool parse_pinmux_debug_line(const std::string& line, PinRuntimeStatus* status);
    static bool is_unclaimed_mux_owner(const std::string& owner);
    static bool is_unclaimed_gpio_owner(const std::string& owner);
    static bool write_text_file(const std::string& path, const std::string& text);

    void load_function_table_if_needed() const;
    uint32_t resolve_function_mux(int bank, int pin, const std::string& function_or_group) const;
    std::vector<std::string> collect_interfaces_for_owner(const std::string& owner) const;
    static std::string find_owner_for_function(const std::vector<PinRuntimeStatus>& rows,
                                               const std::string& normalized_function,
                                               const std::string& normalized_group);
    static bool unbind_owner_device(const std::string& owner);
    static bool bind_owner_device(const std::string& owner);
    static std::string normalize_linux_bus_arg(const std::string& bus);
    static std::string normalize_linux_device_arg(const std::string& bus, const std::string& device);
    static std::string normalize_linux_driver_arg(const std::string& driver);
    static bool create_spi_device_if_missing(const std::string& spi_device, const std::string& modalias);
    static int parse_spi_bus_number(const std::string& spi_device);
    static bool parse_spi_device_parts(const std::string& spi_device, int* bus, int* chip_select);
    static int parse_numbered_peripheral(const std::string& token, const std::string& prefix);
    static std::string read_bound_driver_name(const std::filesystem::path& device_path);
    std::vector<PeripheralPinAssignment> infer_spi_pin_group(int bus,
                                                             int chip_select,
                                                             const std::vector<std::string>& pins) const;
    std::vector<PeripheralPinAssignment> infer_uart_pin_group(int bus,
                                                              const std::vector<std::string>& pins) const;
    std::vector<PeripheralPinAssignment> infer_i2c_pin_group(int bus,
                                                             const std::vector<std::string>& pins) const;
    std::vector<PeripheralPinAssignment> infer_pwm_pin_group(int channel,
                                                             const std::vector<std::string>& pins) const;
    std::vector<PinRuntimeStatus> read_runtime_rows() const;
    std::pair<std::string, std::string> resolve_function_and_group(const std::string& function_or_group) const;
    static std::string find_gpiochip_name_for_bank(int bank);
    static uint64_t build_gpio_v2_flags(const GpioLineConfig& config);
    static uint32_t build_gpio_v1_flags(const GpioLineConfig& config);
    static int parse_adc_channel_token(const std::string& token);
    static std::vector<AdcChannelStatus> scan_adc_channels();
    AdcChannelStatus read_adc_by_channel(int channel) const;
    bool release_gpio_handle_unsafe(const PinKey& key);
    uint32_t hw_read32_unsafe(uint32_t block, uint32_t offset) const;
    void hw_write32_unsafe(uint32_t block, uint32_t offset, uint32_t value, uint32_t mask, bool hiword) const;
    uint32_t ioc_read32_unsafe(bool use_pmuioc, uint32_t offset) const;
    void ioc_hiword_update_unsafe(bool use_pmuioc, uint32_t offset, uint32_t mask, uint32_t value) const;
    uint32_t gpio_reg_read32_unsafe(int bank, uint32_t offset) const;
    void gpio_reg_hiword_update_unsafe(int bank, uint32_t offset, uint32_t mask, uint32_t value) const;
    void write_mux_untracked(int bank, int pin, uint32_t mux);
    void remember_pin_claim(int bank, int pin, const std::string& target_function_or_group);
    void release_pin_claims(const std::vector<PinKey>& claims) noexcept;
    void* gpio_map_bank_unsafe(int bank) const;
    void gpio_reg_set_direction_unsafe(int bank, int pin, bool output) const;
    void gpio_reg_set_output_unsafe(int bank, int pin, int value) const;
    int gpio_reg_get_value_unsafe(int bank, int pin) const;

    int hw_fd_ = -1;
    int fd_ = -1;
    void* ioc_map_ = nullptr;
    void* pmuioc_map_ = nullptr;
    mutable std::array<void*, 5> gpio_maps_{};

    mutable bool function_table_loaded_ = false;
    mutable std::unordered_map<PinKey, std::vector<PinAltFunction>, PinKeyHash> function_table_;
    mutable std::unordered_map<std::string, std::string> owner_hint_cache_;
    mutable std::unordered_map<PinKey, GpioLineHandle, PinKeyHash> gpio_line_handles_;
    std::vector<PinKey> pin_claims_;
    mutable std::mutex lock_;
};

using PinMux = Controller;

}  // namespace visiong::pinmux

#endif  // VISIONG_CORE_PINMUX_H
