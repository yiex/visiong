// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_PINMUX_H
#define VISIONG_CORE_PINMUX_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
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
    std::vector<PinRuntimeStatus> read_runtime_rows() const;
    std::pair<std::string, std::string> resolve_function_and_group(const std::string& function_or_group) const;
    static std::string find_gpiochip_name_for_bank(int bank);
    static uint64_t build_gpio_v2_flags(const GpioLineConfig& config);
    static uint32_t build_gpio_v1_flags(const GpioLineConfig& config);
    static int parse_adc_channel_token(const std::string& token);
    static std::vector<AdcChannelStatus> scan_adc_channels();
    AdcChannelStatus read_adc_by_channel(int channel) const;
    bool release_gpio_handle_unsafe(const PinKey& key);

    int fd_ = -1;
    void* ioc_map_ = nullptr;
    void* pmuioc_map_ = nullptr;

    mutable bool function_table_loaded_ = false;
    mutable std::unordered_map<PinKey, std::vector<PinAltFunction>, PinKeyHash> function_table_;
    mutable std::unordered_map<std::string, std::string> owner_hint_cache_;
    mutable std::unordered_map<PinKey, GpioLineHandle, PinKeyHash> gpio_line_handles_;
    mutable std::mutex lock_;
};

using PinMux = Controller;

}  // namespace visiong::pinmux

#endif  // VISIONG_CORE_PINMUX_H

